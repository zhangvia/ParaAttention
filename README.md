# ParaAttention

Context parallel attention that works with torch.compile

This aims to provide:

- [ ] The fastest accurate attention implemented in Triton, running 50% faster than the originial FA2 implementation on RTX 4090.
- [x] A unified interface to run context parallel attention, as well as keeping the maximum performance while working with `torch.compile`

# Installation

```bash
git clone https://github.com/chengzeyi/ParaAttention.git
cd ParaAttention
git submodule update --init --recursive

pip3 install 'torch==2.5.0'
pip3 install packaging wheel 'setuptools>=64' 'setuptools_scm>=8'

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

## Run Unified Attention (Ulysses Style and Ring Style) with `torch.compile`

``` python
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import init_device_mesh
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
        mesh = init_device_mesh(device, mesh_shape, mesh_dim_names=("ulysses", "ring"))
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
        query_slice,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )

    torch.testing.assert_close(out_slice, out_slice_ref, rtol=1e-5, atol=1e-3 * world_size)

dist.destroy_process_group()
```

Save the above code to `test.py` and run it with `torchrun`:

```bash
torchrun --nproc_per_node=2 test.py
```
