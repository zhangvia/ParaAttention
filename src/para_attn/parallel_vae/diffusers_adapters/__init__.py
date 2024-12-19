import importlib


def parallelize_vae(vae, *args, **kwargs) -> None:
    vae_cls_name = vae.__class__.__name__
    if vae_cls_name == "AutoencoderKL":
        adapter_name = "autoencoder_kl"
    elif vae_cls_name == "AutoencoderKLHunyuanVideo":
        adapter_name = "autoencoder_kl_hunyuan_video"
    else:
        raise ValueError(f"Unknown vae class name: {vae_cls_name}")

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    parallelize_vae_fn = getattr(adapter_module, "parallelize_vae")
    return parallelize_vae_fn(vae, *args, **kwargs)
