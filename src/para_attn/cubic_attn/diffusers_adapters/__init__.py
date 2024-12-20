import importlib

from diffusers import DiffusionPipeline


def cubify_transformer(transformer, *args, **kwargs):
    transformer_cls_name = transformer.__class__.__name__
    if False:
        pass
    elif transformer_cls_name.startswith("Mochi"):
        adapter_name = "mochi"
    elif transformer_cls_name.startswith("CogVideoX"):
        adapter_name = "cogvideox"
    # elif transformer_cls_name.startswith("HunyuanVideo"):
    #     adapter_name = "hunyuan_video"
    else:
        raise ValueError(f"Unknown transformer class name: {transformer_cls_name}")

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    cubify_transformer_fn = getattr(adapter_module, "cubify_transformer")
    return cubify_transformer_fn(transformer, *args, **kwargs)


def cubify_pipe(pipe: DiffusionPipeline, *args, **kwargs):
    assert isinstance(pipe, DiffusionPipeline)

    pipe_cls_name = pipe.__class__.__name__
    if False:
        pass
    elif pipe_cls_name.startswith("Mochi"):
        adapter_name = "mochi"
    elif pipe_cls_name.startswith("CogVideoX"):
        adapter_name = "cogvideox"
    # elif pipe_cls_name.startswith("HunyuanVideo"):
    #     adapter_name = "hunyuan_video"
    else:
        raise ValueError(f"Unknown pipeline class name: {pipe_cls_name}")

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    cubify_pipe_fn = getattr(adapter_module, "cubify_pipe")
    return cubify_pipe_fn(pipe, *args, **kwargs)
