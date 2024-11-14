# ParaAttention

Context parallel attention that works with torch.compile,
supporting both [**Ulysses Style**](https://arxiv.org/abs/2309.14509) and [**Ring Style**](https://arxiv.org/abs/2310.01889) parallelism.

This aims to provide:

- [x] An easy to use interface to speed up model inference with context parallel and `torch.compile`. Make `FLUX` and `Mochi` inference much faster losslessly.
- [x] A unified interface to run context parallel attention, as well as keeping the maximum performance while working with `torch.compile`
- [ ] The fastest accurate attention implemented in Triton, running 50% faster than the originial FA2 implementation on RTX 4090.

# Performance

| Model | GPU | Method | Wall Time (s) | Speedup |
| --- | --- | --- | --- | --- |
| FLUX.1-dev | A100-SXM4-80GB | Baseline | 13.843 | 1.00x |
| FLUX.1-dev | A100-SXM4-80GB | `torch.compile` | 9.997 | 1.38x |
| FLUX.1-dev | A100-SXM4-80GB x 2 | `para-attn (ulysses)` | 8.379 | 1.65x |
| FLUX.1-dev | A100-SXM4-80GB x 2 | `para-attn (ring)` | 8.307 | 1.66x |
| FLUX.1-dev | A100-SXM4-80GB x 2 | `para-attn (ulysses)` + `torch.compile` | 5.915 | 2.34x |
| FLUX.1-dev | A100-SXM4-80GB x 2 | `para-attn (ring)` + `torch.compile` | 5.775 | 2.39x |
| FLUX.1-dev | A100-SXM4-80GB x 4 | `para-attn (ulysses + ring)` + `torch.compile` | ? | ? |
| mochi-1-preview | A100-SXM4-80GB | Baseline | 196.534 | 1.00x |
| mochi-1-preview | A100-SXM4-80GB | `torch.compile` | 149.868 | 1.31x |
| mochi-1-preview | A100-SXM4-80GB x 2 | `para-attn (ulysses)` | 110.146 | 1.78x |
| mochi-1-preview | A100-SXM4-80GB x 2 | `para-attn (ring)` | 109.435 | 1.80x |
| mochi-1-preview | A100-SXM4-80GB x 2 | `para-attn (ulysses)` + `torch.compile` | 83.912 | 2.34x |
| mochi-1-preview | A100-SXM4-80GB x 2 | `para-attn (ring)` + `torch.compile` | 82.176 | 2.39x |
| mochi-1-preview | A100-SXM4-80GB x 4 | `para-attn (ulysses + ring)` + `torch.compile` | ? | ? |

# Installation

## Install from PyPI

```bash
pip3 install 'torch==2.5.0'
pip3 install para-attn
```

## Local Installation

```bash
git clone https://github.com/chengzeyi/ParaAttention.git
cd ParaAttention
git submodule update --init --recursive

pip3 install 'torch==2.5.0'
pip3 install 'setuptools>=64' 'setuptools_scm>=8'

# Pass --no-use-pep517 to speed up rebuild by using the legacy build system
# which doesn't use a one-time tmp directory for the build
pip3 install -e '.[dev]' --no-build-isolation
# Or:
# python3 setup.py develop

# Code formatting and linting
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

# Usage

## Run FLUX.1-dev with Parallel Inference

``` python
import torch 
import torch.distributed as dist
from diffusers import FluxPipeline

dist.init_process_group()

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to(f"cuda:{dist.get_rank()}")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

parallelize_pipe(
    pipe,
    mesh=init_context_parallel_mesh(
        pipe.device.type,
        max_ring_dim_size=2,
    ),
)

torch._inductor.config.reorder_for_compute_comm_overlap = True
pipe.transformer = torch.compile(
   pipe.transformer, mode="max-autotune-no-cudagraphs"
)

image = pipe(
    "A cat holding a sign that says hello world", num_inference_steps=28
).images[0]

if dist.get_rank() == 0:
    image.save("flux.png")

dist.destroy_process_group()
```

Save the above code to `test.py` and run it with `torchrun`:

```bash
torchrun --nproc_per_node=2 test.py
```

## Run Mochi with Parallel Inference

``` python
import torch
import torch.distributed as dist
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

dist.init_process_group()

pipe = MochiPipeline.from_pretrained(
    "genmo/mochi-1-preview", torch_dtype=torch.float16
).to(f"cuda:{dist.get_rank()}")

# Enable memory savings
# pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

parallelize_pipe(
    pipe,
    mesh=init_context_parallel_mesh(
        pipe.device.type,
        max_batch_dim_size=2,
        max_ring_dim_size=2,
    ),
)

torch._inductor.config.reorder_for_compute_comm_overlap = True
pipe.transformer = torch.compile(
   pipe.transformer, mode="max-autotune-no-cudagraphs"
)

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
frames = pipe(prompt, num_frames=84).frames[0]

if dist.get_rank() == 0:
    export_to_video(frames, "mochi.mp4", fps=30)

dist.destroy_process_group()
```

Save the above code to `test.py` and run it with `torchrun`:

```bash
torchrun --nproc_per_node=2 test.py
```

## Run Unified Attention (Hybird Ulysses Style and Ring Style) with `torch.compile`

``` python
import torch
import torch.distributed as dist
import torch.nn.functional as F
from para_attn import para_attn_interface

dist.init_process_group()
world_size = dist.get_world_size()
rank = dist.get_rank()

assert world_size <= torch.cuda.device_count()
if world_size % 2 == 0:
    mesh_shape = (world_size // 2, 2)
else:
    mesh_shape = (world_size, 1)

B, H, S_Q, S_KV, D = 2, 24, 4096, 4096, 64
dtype = torch.float16
device = "cuda"

torch._inductor.config.reorder_for_compute_comm_overlap = True

with torch.no_grad(), torch.cuda.device(rank):
    torch.manual_seed(0)

    query = torch.randn(B, H, S_Q, D, dtype=dtype, device=device)
    key = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
    value = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
    attn_mask = None
    dropout_p = 0.0
    is_causal = False

    query_slice = query.chunk(world_size, dim=-2)[rank]
    key_slice = key.chunk(world_size, dim=-2)[rank]
    value_slice = value.chunk(world_size, dim=-2)[rank]

    def func(*args, **kwargs):
        return F.scaled_dot_product_attention(*args, **kwargs)

    func = torch.compile(func)

    for _ in range(2):
        mesh = dist.init_device_mesh(device, mesh_shape, mesh_dim_names=("ulysses", "ring"))
        with para_attn_interface.UnifiedAttnMode(mesh):
            out_slice = func(
                query_slice,
                key_slice,
                value_slice,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    out_slice_ref = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    ).chunk(world_size, dim=-2)[rank]

    torch.testing.assert_close(out_slice, out_slice_ref, rtol=1e-5, atol=1e-3 * world_size)

dist.destroy_process_group()
```

Save the above code to `test.py` and run it with `torchrun`:

```bash
torchrun --nproc_per_node=2 test.py
```

# Run Tests

```bash
DISTRIBUTED_TESTS_DEFAULT_TIMEOUT=3000 pytest tests
```
