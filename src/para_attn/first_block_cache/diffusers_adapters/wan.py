import functools
import unittest

import torch
from diffusers import DiffusionPipeline, HunyuanVideoTransformer3DModel

from para_attn.first_block_cache import utils


def apply_cache_on_transformer(
    transformer: HunyuanVideoTransformer3DModel,
):
    if getattr(transformer, "_is_cached", False):
        return transformer

    blocks = torch.nn.ModuleList(
        [
            utils.CachedTransformerBlocks(
                transformer.blocks,
                transformer=transformer,
                return_hidden_states_only=True,
            )
        ]
    )

    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        *args,
        **kwargs,
    ):
        with unittest.mock.patch.object(
            self,
            "blocks",
            blocks,
        ):
            return original_forward(
                *args,
                **kwargs,
            )

    transformer.forward = new_forward.__get__(transformer)

    transformer._is_cached = True

    return transformer


def apply_cache_on_pipe(
    pipe: DiffusionPipeline,
    *,
    shallow_patch: bool = False,
    residual_diff_threshold=0.05,
    **kwargs,
):
    if not getattr(pipe, "_is_cached", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            with utils.cache_context(utils.create_cache_context(residual_diff_threshold=residual_diff_threshold)):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

    if not shallow_patch:
        apply_cache_on_transformer(pipe.transformer, **kwargs)

    return pipe
