import importlib

from diffusers import DiffusionPipeline


def parallelize_pipe(pipe: DiffusionPipeline, *args, **kwargs) -> None:
    assert isinstance(pipe, DiffusionPipeline)

    pipe_cls_name = pipe.__class__.__name__
    if pipe_cls_name.startswith("Flux"):
        adapter_name = "flux"
    elif pipe_cls_name.startswith("Mochi"):
        adapter_name = "mochi"
    else:
        raise ValueError(f"Unknown pipeline class name: {pipe_cls_name}")

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    parallelize_pipe_fn = getattr(adapter_module, "parallelize_pipe")
    parallelize_pipe_fn(pipe, *args, **kwargs)
