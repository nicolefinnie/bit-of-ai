
"""Module to load and trace models"""
import torch.nn as nn
import torch.fx as fx
import torchvision.models as torch_models
from transformers import GPT2LMHeadModel, PreTrainedModel, AutoModel
from transformers.utils import fx as hf_fx
from simple_cnn import SimpleCNN


KNOWN_MODELS = {
    "gpt2": lambda: GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="/home/fin2rng/.cache/huggingface/hub"),
    "resnet18": lambda: torch_models.resnet18(pretrained=False),
    "simplecnn": lambda: SimpleCNN(),
}

# Tracer functions dictionary
TRACER_FUNCTIONS = {
    "gpt2": hf_fx.symbolic_trace,
    "resnet18": fx.symbolic_trace,
    "simplecnn": fx.symbolic_trace,
}


def load_model(model_name: str, device: str | None) -> tuple[PreTrainedModel, callable]:
    try:
        model = KNOWN_MODELS[model_name.lower()]()
        tracer_fn = TRACER_FUNCTIONS[model_name.lower()]
        
    except KeyError:
        model, tracer_fn = dynamic_model_loader(model_name)
    
    if device is not None:
        model.to(device)

    return model, tracer_fn
    
def dynamic_model_loader(
        model_name:str,
        cache_dir: str = "/home/fin2rng/.cache/huggingface/hub"
        )-> tuple[PreTrainedModel, callable]:
    """Dynamically load the model from hugging face model hub.

    Args:
        model_name (str): a valid hugging face model name

    Raises:
        ValueError: the model name cannot be loaded by hugging face AutoModel

    Returns:
        tuple[PreTrainedModel, callable]: a hugging face model and its tracer function
    """
    try:
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        return model, hf_fx.symbolic_trace
    except Exception:
        pass

    raise ValueError(f"Model {model_name} not supported, please choose from {KNOWN_MODELS.keys()}"
                     " or a valid huggingface model name")

def trace_model(model: nn.Module | PreTrainedModel, tracer_fn: callable) -> fx.GraphModule:
    """Trace static or dynamic model graph.

    Args:
        model (nn.Module | PreTrainedModel): a torch/HF model
        tracer_fn (callable): tracer function

    Returns:
        fx.GraphModule: the traced model
    """
    model.eval()
    traced_model = tracer_fn(model)
    return traced_model    
