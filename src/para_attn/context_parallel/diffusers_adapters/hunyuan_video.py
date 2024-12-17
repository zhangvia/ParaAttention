import functools
import itertools
from typing import List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, HunyuanVideoTransformer3DModel

import para_attn.primitives as DP
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.para_attn_interface import UnifiedAttnMode


def parallelize_transformer(transformer: HunyuanVideoTransformer3DModel, *, mesh=None):
    mesh = init_context_parallel_mesh(transformer.device.type, mesh=mesh)
    batch_mesh = mesh["batch"]
    seq_mesh = mesh["ring", "ulysses"]._flatten()

    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        *args,
        **kwargs,
    ):
        with UnifiedAttnMode(mesh):
            output = original_forward(*args, **kwargs)

        return output

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

    original_context_embedder_forward = transformer.context_embedder.forward

    @functools.wraps(transformer.context_embedder.__class__.forward)
    def new_context_embedder_forward(
        self,
        *args,
        **kwargs,
    ):
        with UnifiedAttnMode.disable():
            output = original_context_embedder_forward(*args, **kwargs)

        return output

    new_context_embedder_forward = new_context_embedder_forward.__get__(transformer.context_embedder)
    transformer.context_embedder.forward = new_context_embedder_forward

    for transformer_block in itertools.chain(transformer.transformer_blocks, transformer.single_transformer_blocks):

        def patch_transformer_block(transformer_block):
            original_transformer_block_forward = transformer_block.forward

            @functools.wraps(transformer_block.__class__.forward)
            def new_transformer_block_forward(
                self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                temb: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                *args,
                **kwargs,
            ):
                if attention_mask is not None:
                    world_size = DP.get_world_size(seq_mesh)
                    hidden_states_len = hidden_states.shape[-2]
                    encoder_hidden_states_len = encoder_hidden_states.shape[-2]
                    total_len = hidden_states_len + encoder_hidden_states_len
                    new_attention_mask = torch.empty_like(attention_mask)
                    for i in range(world_size):
                        new_attention_mask[..., i * total_len : i * total_len + hidden_states_len, :] = attention_mask[
                            ..., i * hidden_states_len : (i + 1) * hidden_states_len, :
                        ]
                        new_attention_mask[
                            ..., i * total_len + hidden_states_len : (i + 1) * total_len, :
                        ] = attention_mask[
                            ...,
                            world_size * hidden_states_len
                            + i * encoder_hidden_states_len : world_size * hidden_states_len
                            + (i + 1) * encoder_hidden_states_len,
                            :,
                        ]
                    attention_mask = new_attention_mask
                    new_attention_mask = torch.empty_like(attention_mask)
                    for i in range(world_size):
                        new_attention_mask[..., :, i * total_len : i * total_len + hidden_states_len] = attention_mask[
                            ..., :, i * hidden_states_len : (i + 1) * hidden_states_len
                        ]
                        new_attention_mask[
                            ..., :, i * total_len + hidden_states_len : (i + 1) * total_len
                        ] = attention_mask[
                            ...,
                            :,
                            world_size * hidden_states_len
                            + i * encoder_hidden_states_len : world_size * hidden_states_len
                            + (i + 1) * encoder_hidden_states_len,
                        ]
                    attention_mask = new_attention_mask

                if image_rotary_emb is not None:
                    temporal_size = hidden_states.shape[1]
                    freqs_cos, freqs_sin = image_rotary_emb

                    def get_rotary_emb_chunk(freqs):
                        dim_thw = freqs.shape[-1]
                        freqs = freqs.reshape(temporal_size, -1, dim_thw)
                        freqs = DP.get_assigned_chunk(freqs, dim=-2, group=seq_mesh)
                        freqs = freqs.reshape(-1, dim_thw)
                        return freqs

                    freqs_cos = get_rotary_emb_chunk(freqs_cos)
                    freqs_sin = get_rotary_emb_chunk(freqs_sin)
                    image_rotary_emb = (freqs_cos, freqs_sin)

                output = original_transformer_block_forward(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    *args,
                    **kwargs,
                )

                return output

            new_transformer_block_forward = new_transformer_block_forward.__get__(transformer_block)
            transformer_block.forward = new_transformer_block_forward

        patch_transformer_block(transformer_block)

    first_transformer_block = transformer.transformer_blocks[0]
    original_first_transformer_block_forward = first_transformer_block.forward

    @functools.wraps(transformer_block.__class__.forward)
    def new_first_transformer_block_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=-2, group=seq_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=0, group=batch_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=-2, group=seq_mesh)

        output = original_first_transformer_block_forward(
            hidden_states,
            encoder_hidden_states,
            *args,
            **kwargs,
        )

        return output

    new_first_transformer_block_forward = new_first_transformer_block_forward.__get__(first_transformer_block)
    first_transformer_block.forward = new_first_transformer_block_forward

    last_single_transformer_block = transformer.single_transformer_blocks[-1]
    original_last_single_transformer_block_forward = last_single_transformer_block.forward

    @functools.wraps(transformer_block.__class__.forward)
    def new_last_single_transformer_block_forward(
        self,
        *args,
        **kwargs,
    ):
        output = original_last_single_transformer_block_forward(
            *args,
            **kwargs,
        )

        hidden_states = output[0]
        hidden_states = DP.get_complete_tensor(hidden_states, dim=-2, group=seq_mesh)
        hidden_states = DP.get_complete_tensor(hidden_states, dim=0, group=batch_mesh)
        return (hidden_states, *output[1:])

    new_last_single_transformer_block_forward = new_last_single_transformer_block_forward.__get__(
        last_single_transformer_block
    )
    last_single_transformer_block.forward = new_last_single_transformer_block_forward

    return transformer


def parallelize_pipe(pipe: DiffusionPipeline, *, shallow_patch: bool = False, mesh=None):
    original_call = pipe.__class__.__call__

    if not getattr(original_call, "is_parallelized", False):

        @functools.wraps(original_call)
        def new_call(self, *args, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, **kwargs):
            if generator is None:
                seed = torch.seed()
                seed += torch.iinfo(torch.int64).min
                seed_t = torch.full([1], seed, dtype=torch.int64, device=self.device)
                seed_t = DP.get_complete_tensor(seed_t, dim=0)
                seed_t = DP.get_assigned_chunk(seed_t, dim=0, idx=0)
                seed = seed_t.item()
                seed -= torch.iinfo(torch.int64).min
                generator = torch.Generator(self.device).manual_seed(seed)
            return original_call(self, *args, generator=generator, **kwargs)

        new_call.is_parallelized = True

        pipe.__class__.__call__ = new_call

    if not shallow_patch:
        parallelize_transformer(pipe.transformer, mesh=mesh)

    return pipe
