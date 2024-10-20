
import copy
import torch
import torch.fx as fx
import torch.nn as nn
import click

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
    def _fuse(conv: nn.modules.Conv2d, bn: nn.modules.BatchNorm2d) -> nn.modules.Conv2d:
        """Fuse Conv2d and BatchNorm2d in-place.

        Eval mode only. 
        
        batch_norm = gamma * (x - mean) / sqrt(var + eps) + beta
        Mathematically, this is equivalent to performing the convolution with 
        bn_weight * (conv_weight * x + conv_bias - mean) / sqrt(var + eps) + beta

        scaling = bn.weight / sqrt(bn.running_var + bn.eps)

        Args:
            conv (nn.modules.Conv2d): Conv2d module
            bn (nn.modules.BatchNorm2d): BatchNorm2d module

        Returns:
            nn.modules.Conv2d: Fused Conv2d module
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
        """Fuse Conv2D and BatchNorm2d in the model.

        Returns:
            fx.GraphModule: fused graph mdoel.
        """
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


def analyze_layer(graph_module: fx.GraphModule) -> bool:
    """Analyze layer to determine if there is an optimization opportunity.

    Args:
        graph_module (fx.GraphModule): graph 

    Returns:
        bool: is optimized or not
    """
    is_optimized = True
    for node in graph_module.graph.nodes:
        if node.op == 'call_module' and 'conv2' in node.target:
            next_node = node.next
            if next_node and next_node.op == 'call_module' and 'bn2' in next_node.target:
                is_optimized = False
                click.echo(click.style(f"🔍 Found optimization opportunity: Conv2d {node.target} and BatchNorm2d {next_node.target}!", fg="yellow"))


    return is_optimized

def fuse_conv2_bn2(graph_module: fx.GraphModule) -> fx.GraphModule:
    """Optimize th graph by fusing conv2 and bn2.

    Args:
        graph_module (fx.GraphModule): _description_

    Returns:
        fx.GraphModule: _description_
    """

    print("Start to fuse Conv2d and BatchNorm2d...")

    transformer = FuseConv2dBatchNorm2d(graph_module)    
    optimized_graph_module = transformer.transform()

   
    click.echo(click.style("⚙️  Start to fuse Conv2d and BatchNorm2d...", fg="blue"))

    transformer = FuseConv2dBatchNorm2d(graph_module)    
    optimized_graph_module = transformer.transform()

    click.echo(click.style("🛠️  Before optimization:", fg="red", bold=True))
    print(graph_module.graph.print_tabular())

    click.echo(click.style("✅ After optimization:", fg="green", bold=True))
    print(optimized_graph_module.graph.print_tabular())
    

    
    return optimized_graph_module


                    