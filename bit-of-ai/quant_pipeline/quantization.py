from tqdm import tqdm
import torch
import torch.nn as nn
import torch.ao.quantization as quant
from torch.ao.quantization import MinMaxObserver
from torch.utils.data import DataLoader
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


def calculate_quantization_error(
        orig_model: nn.Module,
        quant_model: nn.Module,
        dataloader: DataLoader,
        threshold: float = 1e-6,
        device: str = 'cuda'
    )->torch.Tensor:
    """Calculate quantization errors.

    TODO ideally, we should calculate the error for each layer and return a dictionary of errors.

    Args:
        orig_model (nn.Module): original model
        quant_model (nn.Module): quantized model
        dataloader (DataLoader): data loader for calculation
        device (str, optional): device. Defaults to 'cuda'.
    
    Returns:
        torch.Tensor: quantization error
    """
    
    orig_model.eval()
    quant_model.eval()
    total_error = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="🔄 Evaluating Quantization Error"):
            imgs = imgs.to(device)
            orig_output = orig_model(imgs)
            quant_output = quant_model(imgs)

            # Ensure that the outputs have the same shape
            if orig_output.shape != quant_output.shape:
                print(f"Shape mismatch between original and quantized output!")
                click.echo(click.style(f'❌ Shape mismatch between original {orig_output.shape} "\
                                       "and quantized output {quant_output.shape}!', fg="red"))
                continue

            error = torch.mean(torch.abs(orig_output - quant_output))
            total_error += error.item()
            num_batches += 1

    avg_error = total_error / num_batches if num_batches > 0 else 0.0
    if avg_error > threshold:
        click.echo(click.style(f'❌ Average quantization Error over dataset: {avg_error:.9f}'))
    else:
        click.echo(click.style(f'✅ Average quantization Error over dataset: {avg_error:.9f}'))
    return torch.tensor(avg_error)