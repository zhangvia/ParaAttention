import torch

para_attn_ops = torch.ops.para_attn


class RingAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
    ):
        pass

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
):
    return RingAttnFunc.apply(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
    )
