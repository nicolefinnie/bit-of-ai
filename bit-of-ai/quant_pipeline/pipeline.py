import click

import torch.nn as nn
import torch.fx as fx
from model_tracer import load_model, trace_model
from optimize_graph import analyze_layer, fuse_conv2_bn2



def trace_gui(model: nn.Module, model_name: str, tracer_fn: callable, quiet: bool) -> fx.GraphModule:
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

def analyze_gui(graphed_model: fx.GraphModule, model_name: str, quiet: bool) -> bool:
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
            is_optimized = analyze_layer(graphed_model)
        else:
            analyze_confirm = click.prompt(
                click.style(f"🔍 Should I analyze the model '{model_name}' for optimization? (y/n)", fg="yellow"),
                type=str,
                default="y"
            )
            if analyze_confirm.lower() == 'y':
                is_optimized = analyze_layer(graphed_model)
            else:
                click.echo(click.style("Skipping model analysis...", fg="yellow"))
                is_optimized = False
    return is_optimized

def optimize_gui(is_optimized: bool, graphed_model: fx.GraphModule, model_name: str, quiet: bool) -> None:
    """Optimize the graphed model on the graph level.

    Args:
        is_optimized (bool): True skip optimization
        graphed_model (fx.GraphModule): graphed model
        model_name (str): model name
        quiet (bool): True if running in non-interactive mode
    """
    if not is_optimized and graphed_model:
        if quiet:
            fuse_conv2_bn2(graphed_model)
        else:
            optimize_confirm = click.prompt(
                click.style(f"⚙️  Should I optimize the model '{model_name}' for you? (y/n)", fg="yellow"),
                type=str,
                default="y"
            )
            if optimize_confirm.lower() == 'y':
                fuse_conv2_bn2(graphed_model)
            else:
                click.echo(click.style("Skipping optimization...", fg="yellow"))

@click.command()
@click.argument('model_name', type=str)
@click.option('--device', type=str, default='cuda', help="The device to run the model on")
@click.option('--quiet', is_flag=True, default=False, help="Run this pipeline in non-interactive mode")
def run_pipeline(model_name: str, device: str, quiet: bool):

    # Step 1: Identify model, load it and trace it.
    model, tracer_fn = load_model(model_name=model_name, device=device)
    # Step 2: Trace the model
    graphed_model = trace_gui(model=model, model_name=model_name, tracer_fn=tracer_fn, quiet=quiet)
    # Step 3: Analyze the model (only if tracing happened or quiet mode is enabled)
    is_optimized = analyze_gui(graphed_model=graphed_model, model_name=model_name, quiet=quiet)
    # Step 4: Optimize the model (if it is not optimized yet)
    optimize_gui(is_optimized=is_optimized, graphed_model=graphed_model, model_name=model_name, quiet=quiet)



if __name__=='__main__':
    run_pipeline()
