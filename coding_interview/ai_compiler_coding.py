import torch
import torch.nn as nn
import torch.fx as fx
from torch.profiler import profile, record_function, ProfilerActivity

class BadModel(nn.Module):
    def __init__(self):
        super(BadModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)

    def forward(self, x):
        # input to be in NCHW format
        x = x.permute(0,2,3,1) # convert to NHWC
        x = self.conv1(x.permute(0,3,1,2)) # convert to NCHW
        x = torch.relu(x)

        x = x.permute(0,2,3,1) # convert to NHWC
        x = self.conv2(x.permute(0,3,1,2)) # convert to NHWC
        x = torch.relu(x)
        return x


class RemoveRedundantPermute(fx.Transformer):
    def call_method(self, target, args, kwargs):
        if target == 'permute':
            if args[1:] == (0, 2, 3, 1) or args[1:] == (0, 3, 1, 2):
                # I just return args[0] whcih is just the input tensor x
                return args[0]
        return super().call_method(target, args, kwargs)


class FuseConv2dRelu(fx.Transformer):
    def call_function(self, target, args, kwargs):
        if target == torch.relu:
            relu_input = args[0] # conv2d output 
            if isinstance(relu_input, fx.Proxy):
                if relu_input.node.op == 'call_module':
                    conv_module = self.submodules.get(relu_input.node.target)
                    if isinstance(conv_module, nn.Conv2d):
                        # Replace with my IMC fused relu in PyObject
                        return torch.nn.functional.relu_(relu_input)
        return super().call_function(target, args, kwargs)


def optimize_model():
    """
    This function demonstrates how to trace a model using torch.fx
    
    It outputs this graph:

    opcode         name       target                                                   args                     kwargs
    -------------  ---------  -------------------------------------------------------  -----------------------  --------
    placeholder    x          x                                                        ()                       {}
    call_method    permute    permute                                                  (x, 0, 2, 3, 1)          {}
    call_method    permute_1  permute                                                  (permute, 0, 3, 1, 2)    {}
    call_module    conv1      conv1                                                    (permute_1,)             {}
    call_function  relu       <built-in method relu of type object at 0x7f60d3e3a500>  (conv1,)                 {}
    call_method    permute_2  permute                                                  (relu, 0, 2, 3, 1)       {}
    call_method    permute_3  permute                                                  (permute_2, 0, 3, 1, 2)  {}
    call_module    conv2      conv2                                                    (permute_3,)             {}
    call_function  relu_1     <built-in method relu of type object at 0x7f60d3e3a500>  (conv2,)                 {}
    output         output     output                                                   (relu_1,)                {}
    
    After removing redundant permutes:

    opcode         name    target                                                   args       kwargs
    -------------  ------  -------------------------------------------------------  ---------  --------
    placeholder    x       x                                                        ()         {}
    call_module    conv1   conv1                                                    (x,)       {}
    call_function  relu    <built-in method relu of type object at 0x7f60d3e3a500>  (conv1,)   {}
    call_module    conv2   conv2                                                    (relu,)    {}
    call_function  relu_1  <built-in method relu of type object at 0x7f60d3e3a500>  (conv2,)   {}
    output         output  output                                                   (relu_1,)  {}


    After fuse:

    opcode         name     target                                                    args        kwargs
    -------------  -------  --------------------------------------------------------  ----------  --------
    placeholder    x        x                                                         ()          {}
    call_module    conv1    conv1                                                     (x,)        {}
    call_function  relu_    <built-in method relu_ of type object at 0x7f60d3e3a500>  (conv1,)    {}
    call_module    conv2    conv2                                                     (relu_,)    {}
    call_function  relu__1  <built-in method relu_ of type object at 0x7f60d3e3a500>  (conv2,)    {}
    output         output   output                                                    (relu__1,)  {}


        
    """
    model= BadModel().to('cuda')
    tracer = fx.symbolic_trace(model)
    input_tensor = torch.randn(1, 3, 32, 32).to('cuda')
    before_graph_output = inference(tracer, input_tensor)

    transformer = RemoveRedundantPermute(tracer)    
    model = transformer.transform()
    remove_redundant_graph_output = inference(model, input_tensor)

    tracer = fx.symbolic_trace(model)
    transformer = FuseConv2dRelu(tracer)
    model = transformer.transform()
    after_graph_output = inference(model, input_tensor)

    
    assert torch.allclose(before_graph_output, remove_redundant_graph_output)
    assert torch.allclose(before_graph_output, after_graph_output)

def inference(model, input_tensor):
    model.graph.print_tabular()
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
    
    print(f"Memory Before: {before_memory // 1024} KB")
    print(f"Memory After: {after_memory // 1024} KB")
    print(f"Peak Memory: {peak_memory // 1024} KB")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    return graph_output

if __name__=="__main__":
    
    torch.manual_seed(42)
    optimize_model()