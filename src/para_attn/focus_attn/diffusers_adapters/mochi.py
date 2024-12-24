import functools

import torch
from diffusers import DiffusionPipeline, MochiTransformer3DModel

from para_attn.para_attn_interface import FocusAttnMode


def apply_focus_attn_on_transformer(
    transformer: MochiTransformer3DModel,
    *,
    diagonal_width=5,
    left_width=1,
):
    assert diagonal_width % 2 == 1, "diagonal_width must be an odd number"

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

        focus_mask = torch.zeros(post_patch_num_frames, post_patch_num_frames, dtype=torch.bool)
        for i in range(post_patch_num_frames):
            focus_mask[
                i, max(0, i - diagonal_width // 2) : min(post_patch_num_frames, i + diagonal_width // 2 + 1)
            ] = True
        focus_mask[:, :left_width] = True
        with FocusAttnMode(
            focus_mask=focus_mask,
            focus_range_query=(
                0,
                post_patch_num_frames * post_patch_height * post_patch_width,
            ),
            focus_range_key_value=(
                0,
                post_patch_num_frames * post_patch_height * post_patch_width,
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
        with FocusAttnMode.disable():
            output = original_time_embed_forward(
                timestep, encoder_hidden_states, encoder_attention_mask, *args, **kwargs
            )
        return output

    new_time_embed_forward = new_time_embed_forward.__get__(transformer.time_embed)
    transformer.time_embed.forward = new_time_embed_forward

    return transformer


def apply_focus_attn_on_pipe(
    pipe: DiffusionPipeline,
    *,
    shallow_patch: bool = False,
    **kwargs,
):
    if not shallow_patch:
        apply_focus_attn_on_transformer(pipe.transformer, **kwargs)

    return pipe
