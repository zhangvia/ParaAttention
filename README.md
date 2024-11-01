# ParaAttention

Context parallel attention that works with torch.compile

This aims to include:

- [ ] The fastest accurate attention implemented in Triton, running 50% faster than the originial FA2 implementation on RTX 4090.
- [x] The ability to run context parallel attention in a unified interface, as well as keeping the maximum performance while working with `torch.compile`

# Usage

## Run RingAttention with `torch.compile`

``` python
import torch
import torch.distributed as dist
import torch.nn.functional as F
from para_attn import para_attn_interface

dist.init_process_group()
world_size = dist.get_world_size()
rank = dist.get_rank()

assert world_size <= torch.cuda.device_count()

B, H, S_Q, S_KV, D = 1, 24, 4096, 4096, 64
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

    query_slice = query.chunk(world_size, dim=-1)[rank]
    key_slice = key.chunk(world_size, dim=-1)[rank]
    value_slice = value.chunk(world_size, dim=-1)[rank]

    ring_attn_func = para_attn_interface.ring_attn_func
    ring_attn_func = torch.compile(ring_attn_func, fullgraph=True)

    out_slice = ring_attn_func(
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

    torch.testing.assert_close(out_slice, out_slice_ref)

dist.destroy_process_group()
```

Save the above code to `test.py` and run it with `torchrun`:

```bash
torchrun --nproc_per_node=2 test.py
```
