"""This module is a tutorial on how to use torch.fx to optimize a model on the graph level
for more efficient inference.

However, it doesn't make sense to fuse Conv2d and BatchNorm2d if you run the model in
CUDA with cudnn enabled, because cudnn already fuses Conv2d and BatchNorm2d on the kernel
level. It's only useful if you want to make your model to run on the custom devices.

References: 
https://github.com/pytorch/pytorch/blob/main/torch/fx/experimental/optimization.py
https://pytorch.org/docs/stable/_modules/torch/nn/utils/fusion.html#fuse_conv_bn_eval

"""
import copy
import torch
import torch.nn as nn
import torch.fx as fx
from torch.profiler import profile, record_function, ProfilerActivity
    
class BadModel(nn.Module):
    def __init__(self):
        super(BadModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
    

    def forward(self, x):
        """We intentionally premute the input tensor x to NCHW format
        and then permute it back to NHWC to make it inefficient.
        We do see this issue in some code when they have to switch between NCHW and NHWC
        only for using certain libraires that can only be performed in NHWC or NHWC format
        on CPU or GPU kernel."""
        # input to be in NCHW format
        x = x.permute(0,2,3,1) # convert to NHWC
        x = self.conv1(x.permute(0,3,1,2)) # convert to NCHW
        x = self.bn1(x)
        x = x.permute(0,2,3,1) # convert to NHWC
        x = self.relu(x)
        x = self.conv2(x.permute(0,3,1,2)) # convert to NHWC
        x = self.bn2(x)
        x = self.relu(x)

        return x


class RemoveRedundantPermute(fx.Transformer):
    """Remove redundant permute operations in the model."""
    def call_method(self, target, args, kwargs):
        if target == 'permute':
            if args[1:] == (0, 2, 3, 1) or args[1:] == (0, 3, 1, 2):
                # I just return args[0] whcih is just the input tensor x
                return args[0]
        return super().call_method(target, args, kwargs)


class FuseConv2dBatchNorm2d(fx.Transformer):
    """Fuse Conv2D with BatchNorm2d in-place.
    
    A simplified version of torch.fx.experimental.optimization.fuse
    """
    def __init__(self, fx_model: fx.GraphModule):
        super().__init__(fx_model)
        self.fx_model = fx_model 
        self.modules = dict(self.fx_model.named_modules())
        self.new_graph = copy.deepcopy(self.fx_model.graph)

    @staticmethod
    def _parent_name(target : str) -> tuple[str, str]:
        """
        Splits a qualname into parent path and last atom.
        For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
        """
        *parent, name = target.rsplit('.', 1)
        return parent[0] if parent else '', name

    def _match_pattern(self, node: fx.Node) -> bool:
        """Check if the node matches the pattern to fuse Conv2d and BatchNorm2d.
        It only returns true when the current node is Conv2d.
        """
        if len(node.args) == 0:
            return False
        if not isinstance(node, fx.Node) or not isinstance(node.target, str):
            return False
        if node.op != 'call_module':
            return False
        if node.target not in self.modules:
            return False
        if isinstance(self.modules[node.target], nn.Conv2d):
            next_node = list(node.users)[0]
            if next_node.op == 'call_module' and isinstance(self.modules[next_node.target], nn.BatchNorm2d):
                return True
            
        return False
    
    def _replace_node_module(self, node: fx.Node, new_module: nn.Module) -> None:
        """Replace the module in the node with a new module."""
        parent_name, name = self._parent_name(node.target)
        self.modules[node.target] = new_module
        setattr(self.modules[parent_name], name, new_module)
        
        self.modules[node.target] = new_module

    @staticmethod
    def _fuse(conv: nn.modules.Conv2d, bn: nn.modules.BatchNorm2d):
        """Fuse Conv2d and BatchNorm2d in-place.

        Eval mode only. 
        
        batch_norm = gamma * (x - mean) / sqrt(var + eps) + beta
        Mathematically, this is equivalent to performing the convolution with 
        bn_weight * (conv_weight * x + conv_bias - mean) / sqrt(var + eps) + beta

        scaling = bn.weight / sqrt(bn.running_var + bn.eps)

        """
        fused_conv = copy.deepcopy(conv)
        conv_w = fused_conv.weight
        conv_b = fused_conv.bias

        if conv_b is None:
            conv_b = torch.zeros_like(bn.running_mean)
        if bn.weight is None:
            bn.weight = torch.ones_like(bn.running_mean)
        if bn.bias is None:
            bn.bias = torch.zeros_like(bn.running_mean)
        
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        # reshape only works for conv2D, 
        # From out_channels to [out_channels, in_channels, kernel_h, kernel_w]
        fused_conv_w = conv_w * scale.reshape(-1, 1, 1, 1).to(dtype=conv.weight.dtype)
        fused_conv_b = ((conv_b - bn.running_mean)*scale + bn.bias).to(dtype=conv.weight.dtype)
        fused_conv.weight = torch.nn.Parameter(fused_conv_w, requires_grad=conv.weight.requires_grad)
        fused_conv.bias = torch.nn.Parameter(fused_conv_b, requires_grad=conv.bias.requires_grad)

        assert fused_conv.weight.shape == conv.weight.shape
        assert fused_conv.bias.shape == conv.bias.shape

        return fused_conv

    def transform(self) -> fx.GraphModule:
        for node in self.new_graph.nodes:
            # Found the current node is Conv2d
            if self._match_pattern(node):
                next_node = list(node.users)[0]
                conv = self.modules[node.target]
                bn = self.modules[next_node.target]
                fused_conv = self._fuse(conv, bn)
                # replace Conv2D with fused Conv2D
                self._replace_node_module(node, fused_conv)
                # Replace all uses of the BatchNorm2d output with the Conv2d output
                next_node.replace_all_uses_with(node)
                self.new_graph.erase_node(next_node)
        return fx.GraphModule(self.fx_model, self.new_graph)


def optimize_model():
    """
    This function demonstrates how to trace a model using torch.fx
    
    It outputs this graph:

    Original:

        opcode       name       target    args                   kwargs
    -----------  ---------  --------  ---------------------  --------
    placeholder  x          x         ()                     {}
    call_method  permute    permute   (x, 0, 2, 3, 1)        {}
    call_method  permute_1  permute   (permute, 0, 3, 1, 2)  {}
    call_module  conv1      conv1     (permute_1,)           {}
    call_module  bn1        bn1       (conv1,)               {}
    call_method  permute_2  permute   (bn1, 0, 2, 3, 1)      {}
    call_module  relu       relu      (permute_2,)           {}
    call_method  permute_3  permute   (relu, 0, 3, 1, 2)     {}
    call_module  conv2      conv2     (permute_3,)           {}
    call_module  bn2        bn2       (conv2,)               {}
    call_module  relu_1     relu      (bn2,)                 {}
    output       output     output    (relu_1,)              {}

    Self CPU time total: 197.853ms
    Self CUDA time total: 109.180ms


    Optimized:

        opcode       name    target    args       kwargs
    -----------  ------  --------  ---------  --------
    placeholder  x       x         ()         {}
    call_module  conv1   conv1     (x,)       {}
    call_module  relu    relu      (conv1,)   {}
    call_module  conv2   conv2     (relu,)    {}
    call_module  relu_1  relu      (conv2,)   {}
    output       output  output    (relu_1,)  {}

    Self CPU time total: 31.520ms
    Self CUDA time total: 60.264ms
    
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model= BadModel().to(device)
    model.eval()
    tracer = fx.symbolic_trace(model)
    input_tensor = torch.randn(64, 3, 256, 256).to(device)
    before_graph_output = inference(tracer, input_tensor)

    transformer = RemoveRedundantPermute(tracer)    
    tracer = transformer.transform()
    remove_redundant_graph_output = inference(tracer, input_tensor)

    # Fuse conv, bn using torch.fx
    transformer = FuseConv2dBatchNorm2d(tracer)    
    tracer = transformer.transform()
    after_graph_output = inference(tracer, input_tensor)

    
    assert torch.allclose(before_graph_output, remove_redundant_graph_output)
    assert torch.allclose(before_graph_output, after_graph_output, atol=1e-6)
  
    
    
def inference(model, input_tensor):
    model.graph.print_tabular()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # Reset memory stats
    before_memory = torch.cuda.memory_allocated()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("remove_permute_inference"):
            with torch.no_grad():
                graph_output = model(input_tensor)


    # Measure memory after forward pass
    after_memory = torch.cuda.memory_allocated()

    # Peak memory usage
    peak_memory = torch.cuda.max_memory_allocated()
    
    print(f"Memory Diff: {(after_memory-before_memory) // 1024} KB")
    print(f"Peak Memory: {peak_memory // 1024} KB")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    return graph_output

if __name__=="__main__":
    
    torch.manual_seed(42)
    optimize_model()
