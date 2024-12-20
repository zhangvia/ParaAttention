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
    "CubicAttnMode",
    "ring_attn_func",
    "ulysses_attn_func",
    "in_batch_attn_func",
    "sparse_kv_attn_func",
    "cubic_attn_func",
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


class CubicAttnFunc(torch.autograd.Function):
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
        grid,
        structure_range,
    ):
        if grid is None:
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

        assert len(grid) == 2, "grid must have 2 dimensions, got {}".format(len(grid))

        b, h, s_q, d_qk = query.shape
        _, _, s_kv, d_v = value.shape

        if structure_range is None:
            structure_range = (0, min(s_q, s_kv))

        assert structure_range[0] < structure_range[1] and structure_range[1] <= min(
            s_q, s_kv
        ), "structure_range must be valid, got {}, which is not in [0, {}]".format(structure_range, min(s_q, s_kv))

        query_left, query_ts, query_right = query.split(
            [
                structure_range[0],
                structure_range[1] - structure_range[0],
                s_q - structure_range[1],
            ],
            dim=-2,
        )
        key_left, key_ts, key_right = key.split(
            [
                structure_range[0],
                structure_range[1] - structure_range[0],
                s_kv - structure_range[1],
            ],
            dim=-2,
        )
        value_left, value_ts, value_right = value.split(
            [
                structure_range[0],
                structure_range[1] - structure_range[0],
                s_kv - structure_range[1],
            ],
            dim=-2,
        )

        if isinstance(grid, int):
            grid_t, grid_s = grid, None
        else:
            grid_t, grid_s = grid
        structure_s = structure_range[1] - structure_range[0]

        assert structure_s % grid_t == 0, "structure_s must be divisible by grid_t, got {} and {}".format(
            structure_s, grid_t
        )
        if grid_s is not None:
            assert (
                structure_s // grid_t % grid_s == 0
            ), "structure_s // grid_t must be divisible by grid_s, got {} and {}".format(structure_s // grid_t, grid_s)

        output = []

        if structure_range[0] > 0:
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

        if structure_range[0] < structure_range[1]:
            sdpa_merger = torch_ring_attention._SDPAMerger(
                not para_attn.config.attention.allow_reduced_precision_compute
            )

            if structure_range[0] > 0:
                sdpa_merger.step(
                    *para_attn_ops.attention_forward_with_lse(
                        query_ts,
                        key_left,
                        value_left,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=scale,
                    )
                )

            query_cubic = query_ts.unflatten(2, (grid_t, -1))
            key_cubic = key_ts.unflatten(2, (grid_t, -1))
            value_cubic = value_ts.unflatten(2, (grid_t, -1))
            stream_output, stream_lse = para_attn_ops.attention_forward_with_lse(
                query_cubic.flatten(1, 2),
                key_cubic.flatten(1, 2),
                value_cubic.flatten(1, 2),
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
            stream_output = stream_output.unflatten(1, (-1, grid_t))
            stream_lse = stream_lse.unflatten(1, (-1, grid_t))
            stream_output = stream_output.flatten(2, 3)
            stream_lse = stream_lse.flatten(2, 3)
            sdpa_merger.step(stream_output, stream_lse)
            del stream_output, stream_lse

            if grid_s is not None:
                structure_output, structure_lse = [], []
                for q, k, v in zip(
                    query_cubic.chunk(grid_s, dim=3),
                    key_cubic.chunk(grid_s, dim=3),
                    value_cubic.chunk(grid_s, dim=3),
                ):
                    o, lse = para_attn_ops.attention_forward_with_lse(
                        q.flatten(2, 3),
                        k.flatten(2, 3),
                        v.flatten(2, 3),
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=scale,
                    )
                    structure_output.append(o.unflatten(2, (grid_t, -1)))
                    structure_lse.append(lse.unflatten(2, (grid_t, -1)))
                    del o, lse

                structure_output = torch.cat(structure_output, dim=3)
                structure_lse = torch.cat(structure_lse, dim=3)
                structure_output = structure_output.flatten(2, 3)
                structure_lse = structure_lse.flatten(2, 3)
                sdpa_merger.step(structure_output, structure_lse)
                del structure_output, structure_lse

            del query_cubic, key_cubic, value_cubic

            if structure_range[1] < s_kv:
                sdpa_merger.step(
                    *para_attn_ops.attention_forward_with_lse(
                        query_ts,
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

        if structure_range[1] < s_q:
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

        return torch.cat(output, dim=-2)

    @staticmethod
    def backward(ctx, dout, *args):
        raise NotImplementedError("Backward pass for CubicAttnFunc is not implemented")


def cubic_attn_func(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    grid=None,
    structure_range=None,
):
    return CubicAttnFunc.apply(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        grid,
        structure_range,
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


class CubicAttnMode(TorchFunctionMode):
    disabled = False

    @torch.compiler.disable()
    def __init__(self, *, grid=None, structure_range=None):
        super().__init__()
        self._grid = grid
        self._structure_range = structure_range

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if CubicAttnMode.disabled:
            return func(*args, **kwargs)

        if func is torch.nn.functional.scaled_dot_product_attention:
            return cubic_attn_func(*args, **kwargs, grid=self._grid, structure_range=self._structure_range)

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
