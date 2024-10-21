import click
import torch
import torch.nn as nn
import torch.fx as fx
from torch.utils.data import DataLoader
from model_tracer import load_model, trace_model
from optimize_graph import analyze_layer_graph, fuse_conv2_bn2
from quantization import post_quantize, calculate_quantization_error, finetune_qat
from dataloader import get_cifar_dataloader

def trace(model: nn.Module, model_name: str, tracer_fn: callable, quiet: bool) -> fx.GraphModule:
    """Trace the model.

    Args:
        model (nn.Module): model
        model_name (str): model name
        tracer_fn (callable): tracer function
        quiet (bool): True if running in non-interactive mode

    Returns:
        fx.GraphModule: graphed model
    """
    if quiet:
        graphed_model = trace_model(model=model, tracer_fn=tracer_fn)
    else:
        trace_confirm = click.prompt(
            click.style(f"🧐 Should I trace the model '{model_name}' for you? (y/n)", fg="yellow"),
            type=str,
            default="y"
        )
        if trace_confirm.lower() == 'y':
            graphed_model = trace_model(model=model, tracer_fn=tracer_fn)
        else:
            click.echo(click.style("Skipping model tracing...", fg="yellow"))
            graphed_model = None
    return graphed_model

def analyze(graphed_model: fx.GraphModule, model_name: str, quiet: bool) -> bool:
    """Analyze the graphed model.

    Args:
        graphed_model (fx.GraphModule): graphed model
        model_name (str): model name
        quiet (bool): True if running in non-interactive mode

    Returns:
        bool: is optimized or not
    """
    is_optimized = True
    if graphed_model:
        if quiet:
            is_optimized = analyze_layer_graph(graphed_model)
        else:
            analyze_confirm = click.prompt(
                click.style(f"🔍 Should I analyze the model '{model_name}' for optimization? (y/n)", fg="yellow"),
                type=str,
                default="y"
            )
            if analyze_confirm.lower() == 'y':
                is_optimized = analyze_layer_graph(graphed_model)
            else:
                click.echo(click.style("Skipping model analysis...", fg="yellow"))
                is_optimized = False
            
            if is_optimized:
                click.echo(click.style("🎉 Model is already optimized on the graph level! Nothing to do.", fg="green"))
    return is_optimized

def optimize(is_optimized: bool, graphed_model: fx.GraphModule, model_name: str, quiet: bool) -> fx.GraphModule:
    """Optimize the graphed model on the graph level.

    Args:
        is_optimized (bool): True skip optimization
        graphed_model (fx.GraphModule): graphed model
        model_name (str): model name
        quiet (bool): True if running in non-interactive mode
    """
    optimzed_model = graphed_model
    if not is_optimized and graphed_model:
        if quiet:
            optimzed_model = fuse_conv2_bn2(graphed_model)
        else:
            optimize_confirm = click.prompt(
                click.style(f"⚙️  Should I optimize the model '{model_name}' for you? (y/n)", fg="yellow"),
                type=str,
                default="y"
            )
            if optimize_confirm.lower() == 'y':
                optimzed_model = fuse_conv2_bn2(graphed_model)
            else:
                click.echo(click.style("Skipping optimization...", fg="yellow"))

    return optimzed_model


def quantize(model: nn.Module, device: str='cuda', quiet: bool = True) -> nn.Module:
    """Quantize the model with data calibration.

    Args:
        graphed_model (nn.Module): graphed model
        device: where the model should be quantized
        quiet: True if running in non-interactive mode

    Returns:
        nn.Module: quantized model
    """

    if quiet:
        click.echo(click.style("✅ Calibrating the model, be patient...", fg="blue", bold=True))
    else:
        quantize_confirm = click.prompt(
                click.style(f"⚙️  Should I quantize the model for you? (y/n)", fg="yellow"),
                type=str,
                default="y"
            )
        if quantize_confirm.lower() != 'y':
            click.echo(click.style("Skipping quantization...Model is not quantized", fg="yellow"))
            return model
    return post_quantize(model=model, device=device, quiet=quiet)

def evaluate_quantization_strategy(
        orig_model: nn.Module,
        post_quantized_model: nn.Module,
        quant_error_threshold: float,
        quiet: bool = True,
        device:str = 'cuda'
    ) -> dict:
    """Evaluate quantization strategy.

    Args:
        model (nn.Module): original model
        post_quantized_model (nn.Module): post quantized model
        quant_error_threshold (float): accepted quantization error threshold
        quiet (bool, optional): True if running in non-interactive mode. Defaults to True.
        device (str, optional): Defaults to 'cuda'.

    Returns:
        dict: new quantization strategy config
    """
    error = calculate_quantization_error(orig_model, post_quantized_model, get_cifar_dataloader(split='validation'), device=device)
    dummy_strategy = {}
    if error <= quant_error_threshold:
        return dummy_strategy
    
    if not quiet:
        quantize_confirm = click.prompt(
                click.style(f"⚙️  Post quantization error is higher than the threshold {quant_error_threshold},"
                            "Should I perform QAT for you? (y/n)", fg="yellow"),
                type=str,
                default="y"
            )
        if quantize_confirm.lower() != 'y':
            click.echo(click.style("Skipping quantization...Model is not quantized", fg="yellow"))
            return dummy_strategy

    qat_quantized_model = finetune_qat(post_quantized_model, get_cifar_dataloader(split='calibration'), device=device, num_epochs=5)
    qat_error = calculate_quantization_error(orig_model, qat_quantized_model, get_cifar_dataloader(split='validation'), device=device)
    
    if qat_error < error:
        click.echo(click.style(f'🎉 QAT is more effective than post quantization with this dataset.'))
    else:
        click.echo(click.style(f'😞 QAT is not more effective than post quantization, change strategy'))

    return dummy_strategy
        
@click.command()
@click.argument('model_name', type=str)
@click.option('--device', type=str, default='cuda', help="The device to run the model on")
@click.option('--quiet', is_flag=True, default=False, help="Run this pipeline in non-interactive mode")
@click.option('--quant_error_threshold', type=float, default=1e-7, help="The threshold for quantization errors of the model")
def run_pipeline(model_name: str, device: str, quiet: bool, quant_error_threshold: float):

    # Step 1: Identify model, load it and trace it.
    model, tracer_fn = load_model(model_name=model_name, device=device)
    # Step 2: Trace the model
    graphed_model = trace(model=model, model_name=model_name, tracer_fn=tracer_fn, quiet=quiet)
    # Step 3: Analyze the model (only if tracing happened or quiet mode is enabled)
    is_optimized = analyze(graphed_model=graphed_model, model_name=model_name, quiet=quiet)
    # Step 4: Optimize the model (if it is not optimized yet)
    graphed_model = optimize(is_optimized=is_optimized, graphed_model=graphed_model, model_name=model_name, quiet=quiet)
    # Step 5: Quantize the model, we use the original model because torch.ao will fuse modules, fusion was only for demo
    quantized_model = post_quantize(model=graphed_model, device=device)
    # Step 6: See if we need to perform QAT and perform QAT if necessary
    evaluate_quantization_strategy(model, quantized_model, quant_error_threshold, quiet=quiet, device=device) 
    

if __name__=='__main__':
    torch.random.manual_seed(42)
    run_pipeline()

