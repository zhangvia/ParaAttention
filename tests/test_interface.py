import pytest

import torch
import torch.distributed as dist
import torch.nn.functional as F
from para_attn import para_attn_interface
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms


class ParallelAttnTest(DTensorTestBase):
    @property
    def world_size(self):
        device_count = torch.cuda.device_count()
        if device_count <= 4:
            return device_count
        elif device_count < 6:
            return 4
        elif device_count < 8:
            return 6
        else:
            return 8

    @property
    def attn_func(self):
        raise NotImplementedError

    def attn_mode(self, device):
        raise NotImplementedError

    def _test_attn_func(self, dtype, device, B, H, S_Q, S_KV, D, is_causal, compile):
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

            query_slice = query.chunk(self.world_size, dim=-2)[self.rank]
            key_slice = key.chunk(self.world_size, dim=-2)[self.rank]
            value_slice = value.chunk(self.world_size, dim=-2)[self.rank]

            func = self.attn_func
            if compile:
                func = torch.compile(func)

            for _ in range(2 if compile else 1):
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
            ).chunk(self.world_size, dim=-2)[self.rank]

            torch.testing.assert_close(out_slice, out_slice_ref, rtol=1e-5, atol=1e-3 * self.world_size)

    def _test_attn_mode(self, dtype, device, B, H, S_Q, S_KV, D, is_causal, compile):
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

            query_slice = query.chunk(self.world_size, dim=-2)[self.rank]
            key_slice = key.chunk(self.world_size, dim=-2)[self.rank]
            value_slice = value.chunk(self.world_size, dim=-2)[self.rank]

            def func(*args, **kwargs):
                return F.scaled_dot_product_attention(*args, **kwargs)

            if compile:
                func = torch.compile(func)

            for _ in range(2 if compile else 1):
                with self.attn_mode(device):
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
            ).chunk(self.world_size, dim=-2)[self.rank]

            torch.testing.assert_close(out_slice, out_slice_ref, rtol=1e-5, atol=1e-3 * self.world_size)


class RingAttnTest(ParallelAttnTest):
    @property
    def attn_func(self):
        return para_attn_interface.ring_attn_func

    def attn_mode(self, device):
        return para_attn_interface.RingAttnMode()

    @pytest.mark.skipif("not torch.cuda.is_available()")
    @with_comms
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("device", ["cuda"])
    @parametrize(
        "B,H,S_Q,S_KV,D",
        [
            [2, 24, 4096, 4096, 64],
            [2, 24, 4096, 1024, 64],
        ],
    )
    @parametrize("is_causal", [False, True])
    @parametrize("compile", [False, True])
    def test_attn_func(self, dtype, device, B, H, S_Q, S_KV, D, is_causal, compile):
        super()._test_attn_func(dtype, device, B, H, S_Q, S_KV, D, is_causal, compile)

    @pytest.mark.skipif("not torch.cuda.is_available()")
    @with_comms
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("device", ["cuda"])
    @parametrize(
        "B,H,S_Q,S_KV,D",
        [
            [2, 24, 4096, 4096, 64],
            [2, 24, 4096, 1024, 64],
        ],
    )
    @parametrize("is_causal", [False, True])
    @parametrize("compile", [False, True])
    def test_attn_mode(self, dtype, device, B, H, S_Q, S_KV, D, is_causal, compile):
        super()._test_attn_mode(dtype, device, B, H, S_Q, S_KV, D, is_causal, compile)


class UlyssesAttnTest(ParallelAttnTest):
    @property
    def attn_func(self):
        return para_attn_interface.ulysses_attn_func

    def attn_mode(self, device):
        return para_attn_interface.UlyssesAttnMode()

    @pytest.mark.skipif("not torch.cuda.is_available()")
    @with_comms
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("device", ["cuda"])
    @parametrize(
        "B,H,S_Q,S_KV,D",
        [
            [2, 24, 4096, 4096, 64],
            [2, 24, 4096, 1024, 64],
        ],
    )
    @parametrize("is_causal", [False, True])
    @parametrize("compile", [False, True])
    def test_attn_func(self, dtype, device, B, H, S_Q, S_KV, D, is_causal, compile):
        super()._test_attn_func(dtype, device, B, H, S_Q, S_KV, D, is_causal, compile)

    @pytest.mark.skipif("not torch.cuda.is_available()")
    @with_comms
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("device", ["cuda"])
    @parametrize(
        "B,H,S_Q,S_KV,D",
        [
            [2, 24, 4096, 4096, 64],
            [2, 24, 4096, 1024, 64],
        ],
    )
    @parametrize("is_causal", [False, True])
    @parametrize("compile", [False, True])
    def test_attn_mode(self, dtype, device, B, H, S_Q, S_KV, D, is_causal, compile):
        super()._test_attn_mode(dtype, device, B, H, S_Q, S_KV, D, is_causal, compile)


class UnifiedAttnTest(ParallelAttnTest):
    def attn_mode(self, device):
        world_size = self.world_size
        if world_size % 2 == 0:
            mesh_shape = (world_size // 2, 2)
        else:
            mesh_shape = (world_size, 1)
        mesh = dist.init_device_mesh(device, mesh_shape, mesh_dim_names=("ulysses", "ring"))
        return para_attn_interface.UnifiedAttnMode(mesh)

    @pytest.mark.skipif("not torch.cuda.is_available()")
    @with_comms
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("device", ["cuda"])
    @parametrize(
        "B,H,S_Q,S_KV,D",
        [
            [2, 24, 4096, 4096, 64],
            [2, 24, 4096, 1024, 64],
        ],
    )
    @parametrize("is_causal", [False, True])
    @parametrize("compile", [False, True])
    def test_attn_mode(self, dtype, device, B, H, S_Q, S_KV, D, is_causal, compile):
        super()._test_attn_mode(dtype, device, B, H, S_Q, S_KV, D, is_causal, compile)


# instantiate_parametrized_tests(RingAttnTest)
instantiate_parametrized_tests(UlyssesAttnTest)
instantiate_parametrized_tests(UnifiedAttnTest)

if __name__ == "__main__":
    run_tests()
