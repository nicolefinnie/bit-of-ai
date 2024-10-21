import click
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.fx as fx
import torch.ao.quantization as quant
from model_tracer import load_model, trace_model
from optimize_graph import analyze_layer_graph, fuse_conv2_bn2
from quantization import analyze_layerwise_quant
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

    model.eval()
    #model_fused = quant.fuse_modules(model, [['conv2', 'bn', 'relu']])
    click.echo(click.style("⚙️ Analyzing quantization strategies layer by layer.", fg="blue", bold=True))
    
    model.qconfig = analyze_layerwise_quant(model)
    model_prepared   = quant.prepare(model, inplace=False)
    model_prepared.to(device)
    calibration_dataloader = get_cifar_dataloader()
    with torch.no_grad():
        for imgs, _ in tqdm(calibration_dataloader, desc="🔄 Calibrating"):
            imgs = imgs.to(device)
            model_prepared(imgs)

    model_quantized  = quant.convert(model_prepared, inplace=False)

    return model_quantized



@click.command()
@click.argument('model_name', type=str)
@click.option('--device', type=str, default='cuda', help="The device to run the model on")
@click.option('--quiet', is_flag=True, default=False, help="Run this pipeline in non-interactive mode")
def run_pipeline(model_name: str, device: str, quiet: bool):

    # Step 1: Identify model, load it and trace it.
    model, tracer_fn = load_model(model_name=model_name, device=device)
    # Step 2: Trace the model
    graphed_model = trace(model=model, model_name=model_name, tracer_fn=tracer_fn, quiet=quiet)
    # Step 3: Analyze the model (only if tracing happened or quiet mode is enabled)
    is_optimized = analyze(graphed_model=graphed_model, model_name=model_name, quiet=quiet)
    # Step 4: Optimize the model (if it is not optimized yet)
    graphed_model = optimize(is_optimized=is_optimized, graphed_model=graphed_model, model_name=model_name, quiet=quiet)

    #Step 5: Quantize the model, we use the original model because torch.ao will fuse modules, fusion was only for demo
    quantized_model = quantize(model=graphed_model, device=device)



if __name__=='__main__':
    run_pipeline()
