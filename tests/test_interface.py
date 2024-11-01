import pytest

import torch
import torch.nn.functional as F
from para_attn import para_attn_interface
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms


class RingAttnTest(DTensorTestBase):
    @property
    def world_size(self):
        return min(2, torch.cuda.device_count())

    @pytest.mark.skipif("not torch.cuda.is_available()")
    @with_comms
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("device", ["cuda"])
    @parametrize(
        "B,H,S_Q,S_KV,D",
        [
            [1, 24, 4096, 4096, 64],
            [1, 24, 4096, 1024, 64],
        ],
    )
    @parametrize("is_causal", [False, True])
    def test_ring_attn_func(self, dtype, device, B, H, S_Q, S_KV, D, is_causal):
        if is_causal and S_Q != S_KV:
            return

        with torch.no_grad(), torch.cuda.device(self.rank):
            torch.manual_seed(0)

            query = torch.randn(B, H, S_Q, D, dtype=dtype, device=device)
            key = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
            value = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
            attn_mask = None
            dropout_p = 0.0
            is_causal = is_causal

            query_slice = query.chunk(self.world_size, dim=-1)[self.rank]
            key_slice = key.chunk(self.world_size, dim=-1)[self.rank]
            value_slice = value.chunk(self.world_size, dim=-1)[self.rank]

            out_slice = para_attn_interface.ring_attn_func(
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


instantiate_parametrized_tests(RingAttnTest)
