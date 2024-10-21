import torch
import torch.nn as nn
import torch.ao.quantization as quant
from torch.ao.quantization import MinMaxObserver
import click

class Custom4BitMinMaxObserver(MinMaxObserver):
    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False):
        super().__init__(dtype=dtype, qscheme=qscheme, reduce_range=reduce_range)

    def forward(self, x):
        if self.qscheme == torch.per_tensor_symmetric:
            min_val, max_val = -x.abs().max(), x.abs().max()
        else:
            min_val, max_val = x.min(), x.max()

        qmin = -2**2
        qmin = qmax = 2**2 - 1
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale

        # Store scale and zero_point
        self.scale = scale
        self.zero_point = zero_point

        return x

    def calculate_qparams(self):
        return torch.tensor([self.scale]), torch.tensor([self.zero_point])


def analyze_layerwise_quant(model: nn.Module) -> dict:
    """Analyze layerwise quantization stategies for the model.

    Args:
        model (nn.Module): model

    Returns:
        dict: A dictionary of quantization strategies for each layer
            that considers bit-width, quantization scheme, and calibration strategy.
            It's target hardware-aware.
    """
    quant_config = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            quant_config[name] = quant.QConfig(
                activation=quant.HistogramObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
                weight=quant.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
            )
            click.echo(click.style(f"🟢 Layer: {name}\n", fg="green", bold=True) +
                       click.style(f"   ➡️  Strategy:\n"
                                   f"   - Weight: int8 (per-channel quantization)\n"
                                   f"   - Activation: int8 (per-tensor quantization)\n"
                                   f"   - Rationale: Optimized for target hardware (Conv2D sensitive).\n"))
     
        
        elif isinstance(module, nn.Linear):
            quant_config[name] = quant.QConfig(\
                activation=quant.HistogramObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
                weight=Custom4BitMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
            )
            click.echo(click.style(f"🟢 Layer: {name}\n", fg="green", bold=True) +
                       click.style(f"   ➡️  Strategy:\n"
                                   f"   - Weight: customized int4\n"
                                   f"   - Activation: int8 (per-tensor quantization)\n"
                                   f"   - Rationale: Optimized for Linear layers with 4-bit weight.\n"))

        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            quant_config[name] = quant.QConfig(
                activation=quant.HistogramObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
                weight=None
            )
            click.echo(click.style(f"🟢 Layer: {name}\n", fg="green", bold=True) +
                       click.style(f"   ➡️  Strategy:\n"
                                   f"   - Activation: int8 (per-tensor quantization)\n"
                                   f"   - Rationale: Pooling layers don't require weight quantization.\n", fg="blue"))


        # Fallback strategy in case ReLU is not fused
        elif isinstance(module, nn.ReLU):
            quant_config[name] = quant.QConfig(\
                activation=quant.HistogramObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
                weight=None
            )
            click.echo(click.style(f"🟢 Layer: {name}\n", fg="green", bold=True) +
                       click.style(f"   ➡️  Strategy:\n"
                                   f"   - Activation: int8 (per-tensor quantization)\n"
                                   f"   - Rationale: Fallback strategy if ReLU is not fused.\n", fg="blue"))

        
        else: 
            quant_config[name] = quant.get_default_qconfig('fbgemm')
            click.echo(click.style(f"🟢 Layer: {name}\n", fg="green", bold=True) +
                       click.style(f"   ➡️  Strategy:\n"
                                   f"   - Default: Facebook General Matrix Multiplication (FBGEMM)\n"
                                   f"   - Rationale: Default low-precision strategy for x86.\n"))
