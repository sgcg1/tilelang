"""Ascend MTE DMA copy primitives and their pad-value / ND->NZ helpers."""

from __future__ import annotations

from tvm import tirx


def _to_int32(v):
    return tirx.IntImm("int32", v) if isinstance(v, int) else v


_PAD_VALUE_SUPPORTED_DTYPES = frozenset(
    {
        "int8",
        "uint8",
        "int16",
        "uint16",
        "float16",
        "bfloat16",
        "int32",
        "uint32",
        "float32",
        "float8_e4m3fn",
        "float8_e5m2",
        "float4_e2m1fn",
    }
)

# FP8/FP4 没有可用的 C++ 标量类型，填充值需要在 codegen 阶段编码为物理字节，
# 因此这类 dtype 只接受编译期常量。
_ENCODED_PAD_DTYPES = frozenset({"float8_e4m3fn", "float8_e5m2", "float4_e2m1fn"})


def ascend_set_copy_pad_value(value, dtype=None) -> None:
    """Set the padding fill value for subsequent Ascend padded MTE copies.

    Generates: asc_set_copy_pad_val(value_bits);

    This call moves no data; it configures the stateful hardware pad value
    consumed by a following padded GM -> UB/L1 copy. Set it immediately
    before each padded copy rather than relying on a single set persisting
    across unrelated copies.

    Parameters
    ----------
    value : int, float, or PrimExpr
        Padding fill value.
    dtype : str, optional
        Copy element dtype (one of int8/uint8/int16/uint16/float16/bfloat16/
        int32/uint32/float32, plus float8_e4m3fn/float8_e5m2/float4_e2m1fn,
        which are encoded into one physical pad byte). Required when ``value``
        is a Python literal; for a typed PrimExpr it defaults to ``value.dtype``.
        FP8/FP4 values must be Python constants.

    Example
    -------
    >>> T.ascend_set_copy_pad_value(0.0, dtype="float32")
    """
    if dtype is None:
        inferred = getattr(value, "dtype", None)
        if isinstance(inferred, str):
            dtype = inferred
    if dtype is None:
        raise ValueError('ascend_set_copy_pad_value requires dtype for Python literals; use dtype="float32" or pass a typed PrimExpr.')
    if dtype not in _PAD_VALUE_SUPPORTED_DTYPES:
        raise ValueError(
            f"ascend_set_copy_pad_value does not support dtype {dtype!r}; supported dtypes are {sorted(_PAD_VALUE_SUPPORTED_DTYPES)}."
        )
    if dtype in _ENCODED_PAD_DTYPES and not isinstance(value, (int, float)):
        raise ValueError(
            f"ascend_set_copy_pad_value requires a Python int or float constant for dtype {dtype!r}; "
            "the FP8/FP4 pad value is encoded into a physical pad byte during code generation, so it cannot be a runtime expression."
        )
    value = tirx.const(value, dtype) if isinstance(value, (int, float)) else tirx.Cast(dtype, value)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_set_copy_pad_value"), value)


def ascend_nd2nz_post_copy(dst, src, rows, cols, *, src_pitch, dst_pitch, dtype: str) -> None:
    """Copy a prepacked NZ region from UB to L1 using explicit storage pitches.

    ``rows`` and ``cols`` describe this transfer. Pitches are the distance
    between C0-group starts in 32-byte blocks, independent of the region size.
    Both pointers must include the region origin and be 32-byte aligned.
    This operation does not initialize gaps or convert element types.
    """
    from tvm.tirx import BufferLoad
    from tilelang.language.builtin import access_ptr

    names = {"float16": "half", "bfloat16": "bfloat16_t", "float32": "float"}
    dtype = names.get(dtype, dtype)
    if dtype not in names.values():
        raise ValueError("NZ post-copy dtype must be float16, bfloat16, or float32")
    if isinstance(dst, BufferLoad):
        dst = access_ptr(dst, "w")
    if isinstance(src, BufferLoad):
        src = access_ptr(src, "r")
    args = [dst, src, _to_int32(rows), _to_int32(cols), _to_int32(src_pitch), _to_int32(dst_pitch), dtype]
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_nd2nz_post_copy"), *args)


__all__ = [
    "ascend_set_copy_pad_value",
    "ascend_nd2nz_post_copy",
]
