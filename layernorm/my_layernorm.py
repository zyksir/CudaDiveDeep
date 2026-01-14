import torch
from torch import Tensor
from typing import Optional
import triton
import triton.language as tl

def triton_autotune_configs():
    # Maximum threads per block is architecture-dependent in theory, but in reality all are 1024
    max_threads_per_block = 1024
    # Default to warp size 32 if not defined by device
    warp_size = getattr(
        torch.cuda.get_device_properties(torch.cuda.current_device()), "warp_size", 32
    )
    # Autotune for warp counts which are powers of 2 and do not exceed thread per block limit
    return [
        triton.Config({}, num_warps=warp_count)
        for warp_count in [1, 2, 4, 8, 16, 32]
        if warp_count * warp_size <= max_threads_per_block
    ]

@triton.autotune(
    configs=triton_autotune_configs(),
    key=["N", "IS_RMS_NORM", "HAS_BIAS", "HAS_RESIDUAL", "HAS_OUT_SCALE", "HAS_OUT_SHIFT"],
)
@triton.jit
def _layer_norm_fwd_1pass_1row_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Bias,  # pointer to the bias
    Residual,  # pointer to the residual
    Residual_Out,  # pointer to the residual output
    Out_Scale,  # pointer to the output scale
    Out_Shift,  # pointer to the output shift
    x_stride_b: tl.int64,  # how much to increase the pointer when moving by 1 block
    y_stride_b: tl.int64,  # how much to increase the pointer when moving by 1 block
    res_stride_b: tl.int64,  # how much to increase the pointer when moving by 1 block
    res_out_stride_b: tl.int64,  # how much to increase the pointer when moving by 1 block
    out_scale_stride_b: tl.int64,  # how much to increase the pointer when moving by 1 block
    out_shift_stride_b: tl.int64,  # how much to increase the pointer when moving by 1 block
    N: tl.int64,  # number of columns in X
    eps: tl.constexpr,  # epsilon to avoid division by zero
    BLOCK_SIZE_N: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_RESIDUAL_OUT: tl.constexpr,
    HAS_OUT_SCALE: tl.constexpr,
    HAS_OUT_SHIFT: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * x_stride_b
    Y += row * y_stride_b
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    if HAS_RESIDUAL:
        Residual += row * res_stride_b
        residual = tl.load(Residual + cols, mask=mask, other=0.).to(tl.float32)
        x += residual
        if HAS_RESIDUAL_OUT:
            Residual_Out += row * res_out_stride_b
            tl.store(Residual_Out + cols, x, mask=mask)
    if IS_RMS_NORM:
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        mean = tl.sum(x, axis=0) / N
        xbar = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd if IS_RMS_NORM else (x - mean) * rstd
    if HAS_BIAS:
        b = tl.load(Bias + cols, mask=mask, other=0.0).to(tl.float32)
        y = x_hat * w + b
    else:
        y = x_hat * w
    if HAS_OUT_SCALE:
        Out_Scale += row * out_scale_stride_b
        out_scale = tl.load(Out_Scale + cols, mask=mask, other=0.0).to(tl.float32)
        y *= out_scale
    if HAS_OUT_SHIFT:
        Out_Shift += row * out_shift_stride_b
        out_shift = tl.load(Out_Shift + cols, mask=mask, other=0.0).to(tl.float32)
        y += out_shift
    tl.store(Y + cols, y, mask=mask)

@triton.autotune(
    configs=triton_autotune_configs(),
    key=["N", "IS_RMS_NORM", "HAS_BIAS", "HAS_RESIDUAL", "HAS_OUT_SCALE", "HAS_OUT_SHIFT"],
)
@triton.jit
def _layer_norm_fwd_1pass_multirow_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Bias,  # pointer to the bias
    Residual,  # pointer to the residual
    Residual_Out,  # pointer to the residual output
    Out_Scale,  # pointer to the output scale
    Out_Shift,  # pointer to the output shift
    x_stride_b: tl.int64,  # how much to increase the pointer when moving by 1 block
    y_stride_b: tl.int64,  # how much to increase the pointer when moving by 1 block
    res_stride_b: tl.int64,  # how much to increase the pointer when moving by 1 block
    res_out_stride_b: tl.int64,  # how much to increase the pointer when moving by 1 block
    out_scale_stride_b: tl.int64,  # how much to increase the pointer when moving by 1 block
    out_shift_stride_b: tl.int64,  # how much to increase the pointer when moving by 1 block
    B: tl.int64,  # number of rows in X
    N: tl.int64,  # number of columns in X
    eps: tl.constexpr,  # epsilon to avoid division by zero
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_RESIDUAL_OUT: tl.constexpr,
    HAS_OUT_SCALE: tl.constexpr,
    HAS_OUT_SHIFT: tl.constexpr,
):
    row_offset = tl.program_id(0) * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    row_mask = row_offset < B
    col_offset = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offset < N
    mask = row_mask[:, None] & col_mask[None, :]
    X_ptrs = X + row_offset[:, None] * x_stride_b + tl.arange(0, BLOCK_SIZE_N)[None, :]
    Y_ptrs = Y + row_offset[:, None] * y_stride_b + tl.arange(0, BLOCK_SIZE_N)[None, :]
    x = tl.load(X_ptrs, mask=mask, other=0.).to(tl.float32)
    if HAS_RESIDUAL:
        Residual_ptrs = Residual + row_offset[:, None] * res_stride_b + tl.arange(0, BLOCK_SIZE_N)[None, :]
        residual = tl.load(Residual_ptrs, mask=mask, other=0.).to(tl.float32)
        x += residual
        if HAS_RESIDUAL_OUT:
            Residual_Out_ptrs = Residual_Out + row_offset[:, None] * res_out_stride_b + tl.arange(0, BLOCK_SIZE_N)[None, :]
            tl.store(Residual_Out_ptrs, x, mask=mask)
    if IS_RMS_NORM:
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
    else:
        mean = (tl.sum(x, axis=1) / N).reshape(BLOCK_SIZE_B, 1)
        xbar = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
    rstd = 1 / tl.sqrt(var + eps).reshape(BLOCK_SIZE_B, 1)
    x_hat = x * rstd if IS_RMS_NORM else (x - mean) * rstd
    w = tl.load(W + col_offset, mask=col_mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(Bias + col_offset, mask=col_mask, other=0.0).to(tl.float32)
        y = x_hat * w[None, :] + b[None, :]
    else:
        y = x_hat * w[None, :]
    if HAS_OUT_SCALE:
        Out_Scale_ptrs = Out_Scale + row_offset[:, None] * out_scale_stride_b + tl.arange(0, BLOCK_SIZE_N)[None, :]
        out_scale = tl.load(Out_Scale_ptrs, mask=mask, other=0.0).to(tl.float32)
        y *= out_scale
    if HAS_OUT_SHIFT:
        Out_Shift_ptrs = Out_Shift + row_offset[:, None] * out_shift_stride_b + tl.arange(0, BLOCK_SIZE_N)[None, :]
        out_shift = tl.load(Out_Shift_ptrs, mask=mask, other=0.0).to(tl.float32)
        y += out_shift
    tl.store(Y_ptrs, y, mask=mask)

def fused_layer_norm(
    x: Tensor,   # [B, N]
    weight: Tensor, # [N]
    eps: float,
    out: Optional[Tensor] = None, # [B, N]
    bias: Optional[Tensor] = None, # [N]
    residual: Optional[Tensor] = None, # [B, N]
    residual_out: Optional[Tensor] = None, # [B, N]
    out_scale: Optional[Tensor] = None, # [B, N]
    out_shift: Optional[Tensor] = None, # [B, N]
    is_rms_norm: bool = True,
):
    N = x.shape[-1]
    if out is None:
        out = torch.empty_like(x)
    assert x.is_contiguous(), "x must be contiguous"
    assert N * x.element_size() < 65536, f"dim size must be less than 64KB, got {N * x.element_size()} bytes"
    assert out.shape == x.shape, f"{out.shape=} is not equal to {x.shape=}"
    assert out_scale is None or out_scale.shape == x.shape, f"{out_scale.shape=} is not equal to {x.shape=}"
    assert out_shift is None or out_shift.shape == x.shape, f"{out_shift.shape=} is not equal to {x.shape=}"
    assert weight.shape == (N,), f"{weight.shape=} is not equal to {N=}"
    assert bias is None or bias.shape == (N,), f"{bias.shape=} is not equal to {N=}"
    assert residual is None or residual.shape == x.shape, f"{residual.shape=} is not equal to {x.shape=}"
    x_, out_ = x.view(-1, N), out.view(-1, N)
    residual_ = residual.view(-1, N) if residual is not None else None
    residual_out_ = residual_out.view(-1, N) if residual_out is not None else None
    out_scale_ = out_scale.view(-1, N) if out_scale is not None else None
    out_shift_ = out_shift.view(-1, N) if out_shift is not None else None
    B = x_.shape[0]
    MIN_BLOCK_SIZE = 1024
    if N < MIN_BLOCK_SIZE:
        BLOCK_SIZE_N = triton.next_power_of_2(N)
        BLOCK_SIZE_B = max(1, MIN_BLOCK_SIZE // BLOCK_SIZE_N)
        _layer_norm_fwd_1pass_multirow_kernel[(triton.cdiv(B, BLOCK_SIZE_B),)](
            x_, out_, weight, bias, residual_, residual_out_, out_scale_, out_shift_,
            x_.stride(0), out_.stride(0), 
            residual_.stride(0) if residual_ is not None else 0,
            residual_out_.stride(0) if residual_out_ is not None else 0,
            out_scale_.stride(0) if out_scale_ is not None else 0,
            out_shift_.stride(0) if out_shift_ is not None else 0,
            B,
            N,
            eps,
            BLOCK_SIZE_B,
            BLOCK_SIZE_N,
            is_rms_norm,
            bias is not None,
            residual is not None,
            residual_out is not None,
            out_scale is not None,
            out_shift is not None,
        )
    else:
        BLOCK_SIZE_N = triton.next_power_of_2(N)
        _layer_norm_fwd_1pass_1row_kernel[(B,)](
            x_, out_, weight, bias, residual_, residual_out_, out_scale_, out_shift_,
            x_.stride(0), out_.stride(0), 
            residual_.stride(0) if residual_ is not None else 0,
            residual_out_.stride(0) if residual_out_ is not None else 0,
            out_scale_.stride(0) if out_scale_ is not None else 0,
            out_shift_.stride(0) if out_shift_ is not None else 0,
            N,
            eps,
            BLOCK_SIZE_N,
            is_rms_norm,
            bias is not None,
            residual is not None,
            residual_out is not None,
            out_scale is not None,
            out_shift is not None,
        )
    if residual is not None:
        return out, residual_out
    return out