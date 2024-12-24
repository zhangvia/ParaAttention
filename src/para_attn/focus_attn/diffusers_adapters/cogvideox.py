import functools

import torch
from diffusers import CogVideoXTransformer3DModel, DiffusionPipeline

from para_attn.para_attn_interface import FocusAttnMode


def apply_focus_attn_on_transformer(
    transformer: CogVideoXTransformer3DModel,
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
        batch_size, num_frames, num_channels, height, width = hidden_states.shape
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            post_patch_num_frames = num_frames
        else:
            post_patch_num_frames = (num_frames + p_t - 1) // p_t
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
                encoder_hidden_states.shape[-2],
                encoder_hidden_states.shape[-2] + post_patch_num_frames * post_patch_height * post_patch_width,
            ),
            focus_range_key_value=(
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


def apply_focus_attn_on_pipe(
    pipe: DiffusionPipeline,
    *,
    shallow_patch: bool = False,
    **kwargs,
):
    if not shallow_patch:
        apply_focus_attn_on_transformer(pipe.transformer, **kwargs)

    return pipe
