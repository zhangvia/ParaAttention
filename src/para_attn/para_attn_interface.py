import contextlib
import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

import para_attn
import para_attn.ops as para_attn_ops
import para_attn.primitives as DP

try:
    from torch.distributed.tensor.experimental._attention import _templated_ring_attention
except ImportError:
    _templated_ring_attention = None

if _templated_ring_attention is not None:
    import torch.distributed.tensor.experimental._attention as torch_ring_attention
else:
    import torch.distributed.tensor as torch_ring_attention

__all__ = [
    "UnifiedAttnMode",
    "RingAttnMode",
    "UlyssesAttnMode",
    "InBatchAttnMode",
    "SparseKVAttnMode",
    "StructSparseAttnMode",
    "ring_attn_func",
    "ulysses_attn_func",
    "in_batch_attn_func",
    "sparse_kv_attn_func",
    "struct_sparse_attn_func",
]


def _sdpa_all_to_all_single(x, mesh):
    if x.requires_grad:
        x = DP.all_to_all_single_autograd_sync(x, output_split_sizes=None, input_split_sizes=None, group=mesh)
    else:
        x = DP.all_to_all_single_sync(x, output_split_sizes=None, input_split_sizes=None, group=mesh)
    return x


def _sdpa_input_all_to_all(x, mesh):
    world_size = DP.get_world_size(mesh)
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert h % world_size == 0, "h must be divisible by world_size, got {} and {}".format(h, world_size)

    x = x.permute(1, 0, 2, 3).contiguous()
    x = _sdpa_all_to_all_single(x, mesh)
    x = x.reshape(world_size, h // world_size, b, -1, d).permute(2, 1, 0, 3, 4).reshape(b, h // world_size, -1, d)
    return x


def _sdpa_output_all_to_all(x, mesh):
    world_size = DP.get_world_size(mesh)
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert s % world_size == 0, "s must be divisible by world_size, got {} and {}".format(s, world_size)

    x = x.permute(2, 0, 1, 3).contiguous()
    x = _sdpa_all_to_all_single(x, mesh)
    x = x.reshape(world_size, s // world_size, b, -1, d).permute(2, 0, 3, 1, 4).reshape(b, -1, s // world_size, d)
    return x


def ulysses_attn_func(
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
    assert query.ndim == 4, "query must have 4 dimensions, got {}".format(query.ndim)
    assert key.ndim == 4, "key must have 4 dimensions, got {}".format(key.ndim)
    assert value.ndim == 4, "value must have 4 dimensions, got {}".format(value.ndim)

    if mesh is None:
        mesh = DP.get_group()

    query = _sdpa_input_all_to_all(query, mesh)
    key = _sdpa_input_all_to_all(key, mesh)
    value = _sdpa_input_all_to_all(value, mesh)

    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )

    out = _sdpa_output_all_to_all(out, mesh)
    return out


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
        mesh,
    ):
        assert _templated_ring_attention is not None, "RingAttnFunc requires a newer version of PyTorch"

        with unittest.mock.patch.object(
            torch_ring_attention,
            "_convert_to_f32",
            not para_attn.config.attention.allow_reduced_precision_compute,
            create=True,
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
    pg = DP.get_group(mesh)
    world_size = DP.get_world_size(pg)
    if world_size <= 1:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    assert attn_mask is None, "attn_mask is not supported in ring_attn_func when world_size > 1"

    return RingAttnFunc.apply(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        pg,
    )


def in_batch_attn_func(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
):
    assert query.ndim == 4, "query must have 4 dimensions, got {}".format(query.ndim)
    assert key.ndim == 4, "key must have 4 dimensions, got {}".format(key.ndim)
    assert value.ndim == 4, "value must have 4 dimensions, got {}".format(value.ndim)
    assert attn_mask is None, "attn_mask is not supported in in_batch_attn_func"
    assert not is_causal, "is_causal is not supported in in_batch_attn_func"

    b, h, s_q, d_qk = query.shape
    _, _, s_kv, d_v = value.shape

    query = query.permute(1, 0, 2, 3).reshape(1, h, b * s_q, -1)
    key = key.permute(1, 0, 2, 3).reshape(1, h, b * s_kv, -1)
    value = value.permute(1, 0, 2, 3).reshape(1, h, b * s_kv, -1)

    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )

    out = out.reshape(h, b, s_q, -1).permute(1, 0, 2, 3)
    return out


class SparseKVAttnFunc(torch.autograd.Function):
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
        out = para_attn_ops.attention_forward_sparse_kv(
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
        raise NotImplementedError("Backward pass for SparseKVAttnFunc is not implemented")


def sparse_kv_attn_func(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
):
    return SparseKVAttnFunc.apply(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
    )


class StructuredSparseAttnFunc(torch.autograd.Function):
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
        sparse_mask,
        sparse_range_query,
        sparse_range_key_value,
    ):
        if sparse_mask is None:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

        assert torch_ring_attention is not None, "CubicAttnFunc requires a newer version of PyTorch"

        assert query.ndim == 4, "query must have 4 dimensions, got {}".format(query.ndim)
        assert key.ndim == 4, "key must have 4 dimensions, got {}".format(key.ndim)
        assert value.ndim == 4, "value must have 4 dimensions, got {}".format(value.ndim)

        assert not is_causal, "is_causal is not supported in CubicAttnFunc"

        assert sparse_mask.ndim == 2, "sparse_mask must have 2 dimensions, got {}".format(sparse_mask.ndim)

        b, h, s_q, d_qk = query.shape
        _, _, s_kv, d_v = value.shape

        if sparse_range_key_value is None:
            sparse_range_key_value = sparse_range_query

        assert (
            0 <= sparse_range_query[0] <= sparse_range_query[1] <= s_q
        ), f"sparse_range_query must be in [0, {s_q}], got {sparse_range_query}"
        assert (
            0 <= sparse_range_key_value[0] <= sparse_range_key_value[1] <= s_kv
        ), f"sparse_range_key_value must be in [0, {s_kv}], got {sparse_range_key_value}"

        assert (sparse_range_query[1] - sparse_range_query[0]) % sparse_mask.shape[
            0
        ] == 0, f"sparse_mask must divide the query dimension, got {sparse_mask.shape[0]} and {sparse_range_query}"
        assert (sparse_range_key_value[1] - sparse_range_key_value[0]) % sparse_mask.shape[
            1
        ] == 0, (
            f"sparse_mask must divide the key/value dimension, got {sparse_mask.shape[1]} and {sparse_range_key_value}"
        )

        query_left, query_sparse, query_right = query.split(
            [
                sparse_range_query[0],
                sparse_range_query[1] - sparse_range_query[0],
                s_q - sparse_range_query[1],
            ],
            dim=2,
        )
        key_left, key_sparse, key_right = key.split(
            [
                sparse_range_key_value[0],
                sparse_range_key_value[1] - sparse_range_key_value[0],
                s_kv - sparse_range_key_value[1],
            ],
            dim=2,
        )
        value_left, value_sparse, value_right = value.split(
            [
                sparse_range_key_value[0],
                sparse_range_key_value[1] - sparse_range_key_value[0],
                s_kv - sparse_range_key_value[1],
            ],
            dim=2,
        )

        output = []

        if query_left.numel() > 0:
            output.append(
                F.scaled_dot_product_attention(
                    query_left,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
            )

        if query_sparse.numel() > 0:
            sdpa_merger = torch_ring_attention._SDPAMerger(
                not para_attn.config.attention.allow_reduced_precision_compute
            )

            if key_left.numel() > 0:
                sdpa_merger.step(
                    *para_attn_ops.attention_forward_with_lse(
                        query_sparse,
                        key_left,
                        value_left,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=scale,
                    )
                )

            if key_sparse.numel() > 0:
                sparse_output, sparse_lse = [], []
                for mask_row, query_chunk in zip(sparse_mask, query_sparse.chunk(sparse_mask.shape[0], dim=2)):
                    sub_sdpa_merger = torch_ring_attention._SDPAMerger(
                        not para_attn.config.attention.allow_reduced_precision_compute
                    )
                    for cond, key_chunk, value_chunk in zip(
                        mask_row,
                        key_sparse.chunk(sparse_mask.shape[1], dim=2),
                        value_sparse.chunk(sparse_mask.shape[1], dim=2),
                    ):
                        if cond:
                            sub_sdpa_merger.step(
                                *para_attn_ops.attention_forward_with_lse(
                                    query_chunk,
                                    key_chunk,
                                    value_chunk,
                                    attn_mask=attn_mask,
                                    dropout_p=dropout_p,
                                    is_causal=is_causal,
                                    scale=scale,
                                )
                            )
                    row_output, row_lse = sub_sdpa_merger.results()
                    sparse_output.append(row_output)
                    sparse_lse.append(row_lse)
                    del sub_sdpa_merger
                    del row_output
                    del row_lse
                sparse_output = torch.cat(sparse_output, dim=2)
                sparse_lse = torch.cat(sparse_lse, dim=2)
                sdpa_merger.step(sparse_output, sparse_lse)
                del sparse_output
                del sparse_lse

            if key_right.numel() > 0:
                sdpa_merger.step(
                    *para_attn_ops.attention_forward_with_lse(
                        query_sparse,
                        key_right,
                        value_right,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=scale,
                    )
                )

            output.append(sdpa_merger.results()[0])
            del sdpa_merger

        if query_right.numel() > 0:
            output.append(
                F.scaled_dot_product_attention(
                    query_right,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
            )

        return torch.cat(output, dim=2)

    @staticmethod
    def backward(ctx, dout, *args):
        raise NotImplementedError("Backward pass for StructuredSparseAttnFunc is not implemented")


def struct_sparse_attn_func(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    sparse_mask=None,
    sparse_range_query=None,
    sparse_range_key_value=None,
):
    return StructuredSparseAttnFunc.apply(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        sparse_mask,
        sparse_range_query,
        sparse_range_key_value,
    )


def _get_arg(args, kwargs, *field):
    if len(field) == 1:
        if isinstance(field, int):
            if field < len(args):
                return args[field]
            else:
                return None
        else:
            return kwargs.get(field[0])
    else:
        index, name = field
        if index < len(args):
            return args[index]
        else:
            return kwargs.get(name)


def _get_args(args, kwargs, *names):
    results = []
    for i, name in enumerate(names):
        results.append(_get_arg(args, kwargs, i, name))
    return results


class RingAttnMode(TorchFunctionMode):
    disabled = False

    @torch.compiler.disable()
    def __init__(self, mesh=None):
        super().__init__()
        self._mesh = mesh

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if RingAttnMode.disabled:
            return func(*args, **kwargs)

        if func is torch.nn.functional.scaled_dot_product_attention:
            return ring_attn_func(*args, **kwargs, mesh=self._mesh)

        return func(*args, **kwargs)

    @torch.compiler.disable()
    def __enter__(self):
        super().__enter__()

    @torch.compiler.disable()
    def __exit__(self, *args):
        super().__exit__(*args)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable()
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled


class UlyssesAttnMode(TorchFunctionMode):
    disabled = False

    @torch.compiler.disable()
    def __init__(self, mesh=None):
        super().__init__()
        self._mesh = mesh

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if UlyssesAttnMode.disabled:
            return func(*args, **kwargs)

        if func is torch.nn.functional.scaled_dot_product_attention:
            return ulysses_attn_func(*args, **kwargs, mesh=self._mesh)

        return func(*args, **kwargs)

    @torch.compiler.disable()
    def __enter__(self):
        super().__enter__()

    @torch.compiler.disable()
    def __exit__(self, *args):
        super().__exit__(*args)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable()
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled


class UnifiedAttnMode(TorchFunctionMode):
    disabled = False

    @torch.compiler.disable()
    def __init__(self, mesh=None):
        super().__init__()

        self._parallel_method = "ulysses"

        if mesh is None:
            self._ulysses_mesh = DP.get_default_group()
            self._ring_mesh = None
        else:
            if isinstance(mesh, dist.ProcessGroup):
                self._ulysses_mesh = mesh
                self._ring_mesh = None
            else:
                assert isinstance(mesh, dist.DeviceMesh), "mesh must be a ProcessGroup or DeviceMesh"

                if "ulysses" in mesh.mesh_dim_names:
                    self._ulysses_mesh = mesh["ulysses"]
                else:
                    self._ulysses_mesh = None
                if "ring" in mesh.mesh_dim_names:
                    self._ring_mesh = mesh["ring"]
                else:
                    self._ring_mesh = None

                assert (
                    self._ulysses_mesh is not None or self._ring_mesh is not None
                ), "mesh must have ulysses or ring dim"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if UnifiedAttnMode.disabled:
            return func(*args, **kwargs)

        if func is torch.nn.functional.scaled_dot_product_attention:
            parallel_method = self._parallel_method
            if parallel_method == "ulysses":
                with self._set_parallel_method("ring"), self:
                    if self._ulysses_mesh is None:
                        return func(*args, **kwargs)
                    return ulysses_attn_func(*args, **kwargs, mesh=self._ulysses_mesh)
            elif parallel_method == "ring":
                with self._set_parallel_method("none"), self:
                    if self._ring_mesh is None:
                        return func(*args, **kwargs)
                    return ring_attn_func(*args, **kwargs, mesh=self._ring_mesh)
            elif parallel_method == "none":
                if para_attn.config.attention.force_dispatch_to_custom_ops:
                    return para_attn_ops.attention_forward(*args, **kwargs)
                return func(*args, **kwargs)
            else:
                raise ValueError(f"Unknown parallel method: {parallel_method}")

        return func(*args, **kwargs)

    @torch.compiler.disable()
    def __enter__(self):
        super().__enter__()

    @torch.compiler.disable()
    def __exit__(self, *args):
        super().__exit__(*args)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable()
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled

    @contextlib.contextmanager
    def _set_parallel_method(self, method):
        old_parallel_method = self._parallel_method
        self._parallel_method = method
        try:
            yield
        finally:
            self._parallel_method = old_parallel_method


class InBatchAttnMode(TorchFunctionMode):
    disabled = False

    @torch.compiler.disable()
    def __init__(self):
        super().__init__()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if InBatchAttnMode.disabled:
            return func(*args, **kwargs)

        if func is torch.nn.functional.scaled_dot_product_attention:
            return in_batch_attn_func(*args, **kwargs)

        return func(*args, **kwargs)

    @torch.compiler.disable()
    def __enter__(self):
        super().__enter__()

    @torch.compiler.disable()
    def __exit__(self, *args):
        super().__exit__(*args)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable()
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled


class SparseKVAttnMode(TorchFunctionMode):
    disabled = False

    @torch.compiler.disable()
    def __init__(self):
        super().__init__()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if SparseKVAttnMode.disabled:
            return func(*args, **kwargs)

        if func is torch.nn.functional.scaled_dot_product_attention:
            return sparse_kv_attn_func(*args, **kwargs)

        return func(*args, **kwargs)

    @torch.compiler.disable()
    def __enter__(self):
        super().__enter__()

    @torch.compiler.disable()
    def __exit__(self, *args):
        super().__exit__(*args)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable()
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled


class StructSparseAttnMode(TorchFunctionMode):
    disabled = False

    @torch.compiler.disable()
    def __init__(self, *, sparse_mask=None, sparse_range_query=None, sparse_range_key_value=None):
        super().__init__()
        self._sparse_mask = sparse_mask
        self._sparse_range_query = sparse_range_query
        self._sparse_range_key_value = sparse_range_key_value

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if StructSparseAttnMode.disabled:
            return func(*args, **kwargs)

        if func is torch.nn.functional.scaled_dot_product_attention:
            return struct_sparse_attn_func(
                *args,
                **kwargs,
                sparse_mask=self._sparse_mask,
                sparse_range_query=self._sparse_range_query,
                sparse_range_key_value=self._sparse_range_key_value,
            )

        return func(*args, **kwargs)

    @torch.compiler.disable()
    def __enter__(self):
        super().__enter__()

    @torch.compiler.disable()
    def __exit__(self, *args):
        super().__exit__(*args)

    @classmethod
    @contextlib.contextmanager
    def disable(cls):
        old_disabled = cls._set_disabled(True)
        try:
            yield
        finally:
            cls._set_disabled(old_disabled)

    @classmethod
    @torch.compiler.disable()
    def _set_disabled(cls, value):
        old_disabled = cls.disabled
        cls.disabled = value
        return old_disabled
