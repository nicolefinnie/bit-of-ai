from tqdm import tqdm
import torch
import torch.nn as nn
import torch.ao.quantization as quant
from torch.ao.quantization import MinMaxObserver
from torch.utils.data import DataLoader
import click
from dataloader import get_cifar_dataloader

class Custom4BitMinMaxObserver(MinMaxObserver):
    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False):
        super().__init__(dtype=dtype, qscheme=qscheme, reduce_range=reduce_range)

    def forward(self, x):
        if self.qscheme == torch.per_tensor_symmetric:
            min_val, max_val = -x.abs().max(), x.abs().max()
        else:
            min_val, max_val = x.min(), x.max()

        qmin = -2**2
        qmax = 2**2 - 1
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
     
        
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            quant_config[name] = quant.QConfig(\
                activation=quant.HistogramObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
                weight=Custom4BitMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
            )
            click.echo(click.style(f"🟢 Layer: {name}\n", fg="green", bold=True) +
                       click.style(f"   ➡️  Strategy:\n"
                                   f"   - Weight: customized int4\n"
                                   f"   - Activation: int8 (per-tensor quantization)\n"
                                   f"   - Rationale: Optimized for Linear layers with customized int4 weight.\n"))


        elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d)):
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


def post_quantize(model: nn.Module, device: str='cuda') -> nn.Module:
    """Quantize the model with data calibration.

    Args:
        graphed_model (nn.Module): graphed model
        device: where the model should be quantized

    Returns:
        nn.Module: quantized model
    """
    model.eval()
    #model_fused = quant.fuse_modules(model, [['conv2', 'bn', 'relu']])
    click.echo(click.style("⚙️ Analyzing quantization strategies layer by layer.", fg="blue", bold=True))
    
    model.qconfig = analyze_layerwise_quant(model)
    model_prepared   = quant.prepare(model, inplace=False)
    model_prepared.to(device)
    calibration_dataloader = get_cifar_dataloader(split='calibration')
    with torch.no_grad():
        for imgs, _ in tqdm(calibration_dataloader, desc="🔄 Calibrating"):
            imgs = imgs.to(device)
            model_prepared(imgs)

    model_quantized  = quant.convert(model_prepared, inplace=False)

    return model_quantized

def finetune_qat(model: nn.Module, dataloader: DataLoader, device: str = 'cuda', num_epochs: int = 5) -> nn.Module:
    """Finetune the model in quantization-aware training way.

    Args:
        model (nn.Module): model to finetune
        dataloader (DataLoader): calibration dataloader
        device (str, optional):  Defaults to 'cuda'.
        num_epochs (int, optional): Defaults to 5.
    
    Returns:
        nn.Module: QAT-quantized model
    """
    model.train()
    model.qconfig = analyze_layerwise_quant(model)
    model_prepared = quant.prepare_qat(model, inplace=False)
    model_prepared.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for imgs, labels in tqdm(dataloader, desc=f"🔄 Finetuning Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model_prepared(imgs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item() / len(dataloader)}")
    
    model_quantized = quant.convert(model_prepared, inplace=False)
    click.echo(click.style(f'🏁 Finetuning completed after {num_epochs} epochs!', fg="green"))
    return model_quantized


def calculate_quantization_error(
        orig_model: nn.Module,
        quant_model: nn.Module,
        dataloader: DataLoader,
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
    click.echo(click.style(f'📊 Average quantization error over dataset: {avg_error:.9f}'))
    return torch.tensor(avg_error)