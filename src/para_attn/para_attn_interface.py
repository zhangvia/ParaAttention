import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed.tensor.experimental._attention import _templated_ring_attention

para_attn_ops = torch.ops.para_attn


class RingAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mesh,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
    ):
        out, lse = _templated_ring_attention(
            mesh,
            para_attn_ops.attention_forward_with_lse,
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        raise NotImplementedError("Backward pass for RingAttnFunc is not implemented")


def ring_attn_func(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    mesh=None,
):
    if mesh is None:
        mesh = c10d._get_default_group()
    return RingAttnFunc.apply(
        mesh,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
    )
