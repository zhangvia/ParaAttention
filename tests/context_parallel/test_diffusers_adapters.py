import time

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms


class DiffusionPipelineTest(DTensorTestBase):
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

    def mesh(self, device, use_ring):
        world_size = self.world_size
        if use_ring and world_size % 2 == 0:
            mesh_shape = (world_size // 2, 2)
        else:
            mesh_shape = (world_size, 1)
        mesh = dist.init_device_mesh(device, mesh_shape, mesh_dim_names=("ulysses", "ring"))
        return mesh

    def new_pipe(self, dtype, device, rank):
        raise NotImplementedError

    def call_pipe(self, pipe, *args, **kwargs):
        raise NotImplementedError

    def _test_benchmark_pipe(self, dtype, device, parallelize, compile, use_ring):
        torch.manual_seed(0)

        pipe = self.new_pipe(dtype, device)

        if parallelize:
            from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

            mesh = self.mesh(device, use_ring=use_ring)
            parallelize_pipe(pipe, mesh=mesh)

        if compile:
            if parallelize:
                torch._inductor.config.reorder_for_compute_comm_overlap = True
            # If cudagraphs is enabled and parallelize is True, the test will hang indefinitely
            # after the last iteration.
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

        for _ in range(2):
            begin = time.time()
            self.call_pipe(pipe)
            end = time.time()
            print(f"Time taken: {end - begin:.3f} seconds")


class FluxPipelineTest(DiffusionPipelineTest):
    def new_pipe(self, dtype, device):
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=dtype,
        ).to(f"{device}:{self.rank}")
        return pipe

    def call_pipe(self, pipe, *args, **kwargs):
        return pipe(
            "A cat holding a sign that says hello world",
            num_inference_steps=28,
        )

    @pytest.mark.skipif("not torch.cuda.is_available()")
    @with_comms
    @parametrize("dtype", [torch.bfloat16])
    @parametrize("device", ["cuda"])
    @parametrize(
        "parallelize,compile,use_ring",
        [
            [False, False, False],
            [False, True, False],
            [True, False, False],
            [True, False, True],
            [True, True, False],
            [True, True, True],
        ],
    )
    def test_benchmark_pipe(self, dtype, device, parallelize, compile, use_ring):
        super()._test_benchmark_pipe(dtype, device, parallelize, compile, use_ring)


class MochiPipelineTest(DiffusionPipelineTest):
    def new_pipe(self, dtype, device):
        from diffusers import MochiPipeline

        pipe = MochiPipeline.from_pretrained(
            "genmo/mochi-1-preview",
            torch_dtype=dtype,
        ).to(f"{device}:{self.rank}")
        pipe.enable_vae_tiling()
        return pipe

    def call_pipe(self, pipe, *args, **kwargs):
        return pipe(
            "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k.",
            num_frames=84,
        )

    @pytest.mark.skipif("not torch.cuda.is_available()")
    @with_comms
    @parametrize("dtype", [torch.bfloat16])
    @parametrize("device", ["cuda"])
    @parametrize(
        "parallelize,compile,use_ring",
        [
            [False, False, False],
            [False, True, False],
            [True, False, False],
            [True, False, True],
            [True, True, False],
            [True, True, True],
        ],
    )
    def test_benchmark_pipe(self, dtype, device, parallelize, compile, use_ring):
        super()._test_benchmark_pipe(dtype, device, parallelize, compile, use_ring)


instantiate_parametrized_tests(DiffusionPipelineTest)
instantiate_parametrized_tests(FluxPipelineTest)
instantiate_parametrized_tests(MochiPipelineTest)

if __name__ == "__main__":
    run_tests()
