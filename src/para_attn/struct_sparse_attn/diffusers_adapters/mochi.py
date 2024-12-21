import functools

import torch
from diffusers import DiffusionPipeline, MochiTransformer3DModel

from para_attn.para_attn_interface import StructSparseAttnMode


def sparsify_transformer(
    transformer: MochiTransformer3DModel,
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
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p = self.config.patch_size

        post_patch_num_frames = num_frames
        post_patch_height = height // p
        post_patch_width = width // p

        sparse_mask = torch.eye(post_patch_num_frames, dtype=torch.bool)
        sparse_mask[1:] = torch.eye(post_patch_num_frames, dtype=torch.bool)[:-1]
        sparse_mask[:, 1:] = torch.eye(post_patch_num_frames, dtype=torch.bool)[:, :-1]
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

    original_time_embed_forward = transformer.time_embed.forward

    @functools.wraps(transformer.time_embed.__class__.forward)
    def new_time_embed_forward(
        self,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        *args,
        **kwargs,
    ):
        with StructSparseAttnMode.disable():
            output = original_time_embed_forward(
                timestep, encoder_hidden_states, encoder_attention_mask, *args, **kwargs
            )
        return output

    new_time_embed_forward = new_time_embed_forward.__get__(transformer.time_embed)
    transformer.time_embed.forward = new_time_embed_forward

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
