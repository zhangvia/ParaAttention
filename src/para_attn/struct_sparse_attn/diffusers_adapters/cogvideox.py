import functools

import torch
from diffusers import CogVideoXTransformer3DModel, DiffusionPipeline

from para_attn.para_attn_interface import StructSparseAttnMode


def sparsify_transformer(
    transformer: CogVideoXTransformer3DModel,
):
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        batch_size, num_frames, num_channels, height, width = hidden_states.shape
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            post_patch_num_frames = num_frames
        else:
            post_patch_num_frames = (num_frames + p_t - 1) // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        sparse_mask = torch.eye(post_patch_num_frames, dtype=torch.bool)
        sparse_mask[1:].logical_or_(torch.eye(post_patch_num_frames, dtype=torch.bool)[:-1])
        sparse_mask[:, 1:].logical_or_(torch.eye(post_patch_num_frames, dtype=torch.bool)[:, :-1])
        with StructSparseAttnMode(
            sparse_mask=sparse_mask,
            sparse_range_query=(
                encoder_hidden_states.shape[-2],
                encoder_hidden_states.shape[-2] + post_patch_num_frames * post_patch_height * post_patch_width,
            ),
        ):
            output = original_forward(
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )

        return output

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

    return transformer


def sparsify_pipe(
    pipe: DiffusionPipeline,
    *,
    shallow_patch: bool = False,
    **kwargs,
):
    if not shallow_patch:
        sparsify_transformer(pipe.transformer, **kwargs)

    return pipe
