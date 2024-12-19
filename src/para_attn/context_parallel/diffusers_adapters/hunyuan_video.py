import functools
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, HunyuanVideoTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput

import para_attn.primitives as DP
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.para_attn_interface import SparseKVAttnMode, UnifiedAttnMode


def parallelize_transformer(transformer: HunyuanVideoTransformer3DModel, *, mesh=None):
    mesh = init_context_parallel_mesh(transformer.device.type, mesh=mesh)
    batch_mesh = mesh["batch"]
    seq_mesh = mesh["ring", "ulysses"]._flatten()

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        # 3. Attention mask preparation
        latent_sequence_length = hidden_states.shape[1]
        latent_attention_mask = torch.ones(
            batch_size, 1, latent_sequence_length, device=hidden_states.device, dtype=torch.bool
        )  # [B, 1, N]
        attention_mask = torch.cat(
            [latent_attention_mask, encoder_attention_mask.unsqueeze(1).to(torch.bool)], dim=-1
        )  # [B, 1, N + M]

        with SparseKVAttnMode(), UnifiedAttnMode(mesh):
            # 4. Transformer blocks
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}

                for block in self.transformer_blocks:
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        attention_mask,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

                for block in self.single_transformer_blocks:
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        attention_mask,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

            else:
                for block in self.transformer_blocks:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                    )

                for block in self.single_transformer_blocks:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                    )

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

    """
    torch._dynamo hit config.cache_size_limit (8)
       function: 'new_transformer_block_forward'
    """
    transformer_transformer_blocks = transformer.transformer_blocks
    for i in range(len(transformer_transformer_blocks)):
        transformer_transformer_blocks[i].attn.processor = transformer_transformer_blocks[0].attn.processor

    single_transformer_blocks = transformer.single_transformer_blocks
    for i in range(len(single_transformer_blocks)):
        single_transformer_blocks[i].attn.processor = single_transformer_blocks[0].attn.processor

    for transformer_block in itertools.chain(transformer.transformer_blocks, transformer.single_transformer_blocks):

        def patch_transformer_block(transformer_block):
            original_transformer_block_forward = transformer_block.__class__.forward

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
                world_size = DP.get_world_size(seq_mesh)
                if attention_mask is not None:
                    hidden_states_len = hidden_states.shape[-2]
                    encoder_hidden_states_len = encoder_hidden_states.shape[-2]

                    attention_mask = attention_mask[:1, ..., :1, :]

                    new_attention_mask = []
                    for i in range(world_size):
                        new_attention_mask.append(
                            attention_mask[..., :, i * hidden_states_len : (i + 1) * hidden_states_len]
                        )
                        new_attention_mask.append(
                            attention_mask[
                                ...,
                                :,
                                world_size * hidden_states_len
                                + i * encoder_hidden_states_len : world_size * hidden_states_len
                                + (i + 1) * encoder_hidden_states_len,
                            ]
                        )
                    new_attention_mask = torch.cat(new_attention_mask, dim=-1)
                    attention_mask = new_attention_mask

                if image_rotary_emb is not None:
                    freqs_cos, freqs_sin = image_rotary_emb

                    def get_rotary_emb_chunk(freqs):
                        freqs = DP.get_assigned_chunk(freqs, dim=0, group=seq_mesh)
                        return freqs

                    freqs_cos = get_rotary_emb_chunk(freqs_cos)
                    freqs_sin = get_rotary_emb_chunk(freqs_sin)
                    image_rotary_emb = (freqs_cos, freqs_sin)

                output = original_transformer_block_forward(
                    self,
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
