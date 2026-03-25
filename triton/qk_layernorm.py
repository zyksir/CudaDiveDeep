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
        for warp_count in [1, 2, 4, 8, 16]
        if warp_count * warp_size <= max_threads_per_block
    ]

@triton.autotune(
    configs=triton_autotune_configs(),
    key=["N", "H", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.jit
def _qk_layer_norm_fwd_1pass_1row_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Bias,  # pointer to the bias
    x_stride_b: tl.int64,
    y_stride_b: tl.int64,
    H: tl.constexpr, N: tl.constexpr,
    eps: tl.constexpr,  # epsilon to avoid division by zero
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    row = tl.program_id(0)
    X += row * x_stride_b
    Y += row * y_stride_b
    cols_h = tl.arange(0, BLOCK_SIZE_ROW)
    cols_n = tl.arange(0, BLOCK_SIZE_N)
    h_mask = cols_h < H
    # N is always 64 or 128 (power of 2), so BLOCK_SIZE_N == N — h_mask is the only real mask
    x_mask = h_mask[:, None]
    offsets = cols_h[:, None] * N + cols_n[None, :]
    x = tl.load(X + offsets, mask=x_mask, other=0.).to(tl.float32)
    if IS_RMS_NORM:
        rstd = tl.math.rsqrt(tl.sum(x * x, axis=1) / N + eps).reshape(BLOCK_SIZE_ROW, 1)
        y = x * rstd
    else:
        mean = (tl.sum(x, axis=1) / N).reshape(BLOCK_SIZE_ROW, 1)
        x_centered = x - mean
        rstd = tl.math.rsqrt(tl.sum(x_centered * x_centered, axis=1) / N + eps).reshape(BLOCK_SIZE_ROW, 1)
        y = x_centered * rstd
    if HAS_WEIGHT:
        w = tl.load(W + cols_n).to(tl.float32)
        y = y * w[None, :]
    if HAS_BIAS:
        b = tl.load(Bias + cols_n).to(tl.float32)
        y = y + b[None, :]
    tl.store(Y + offsets, y, mask=x_mask)

@triton.autotune(
    configs=triton_autotune_configs(),
    key=["N", "H", "IS_RMS_NORM"],
)
@triton.jit
def _fused_qk_norm_kernel(
    Q, K, Q_W, K_W, Q_Y, K_Y,
    q_stride_b: tl.int64,
    k_stride_b: tl.int64,
    q_out_stride_b: tl.int64,
    k_out_stride_b: tl.int64,
    H: tl.constexpr, N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    row = tl.program_id(0)
    Q += row * q_stride_b
    K += row * k_stride_b
    Q_Y += row * q_out_stride_b
    K_Y += row * k_out_stride_b
    cols_h = tl.arange(0, BLOCK_SIZE_ROW)
    cols_n = tl.arange(0, BLOCK_SIZE_N)
    h_mask = (cols_h < H)[:, None]
    offsets = cols_h[:, None] * N + cols_n[None, :]

    q_x = tl.load(Q + offsets, mask=h_mask, other=0.).to(tl.float32)
    if IS_RMS_NORM:
        q_rstd = tl.math.rsqrt(tl.sum(q_x * q_x, axis=1) / N + eps).reshape(BLOCK_SIZE_ROW, 1)
        q_y = q_x * q_rstd
    else:
        q_mean = (tl.sum(q_x, axis=1) / N).reshape(BLOCK_SIZE_ROW, 1)
        q_x_centered = q_x - q_mean
        q_rstd = tl.math.rsqrt(tl.sum(q_x_centered * q_x_centered, axis=1) / N + eps).reshape(BLOCK_SIZE_ROW, 1)
        q_y = q_x_centered * q_rstd
    qw = tl.load(Q_W + cols_n).to(tl.float32)
    tl.store(Q_Y + offsets, q_y * qw[None, :], mask=h_mask)

    k_x = tl.load(K + offsets, mask=h_mask, other=0.).to(tl.float32)
    if IS_RMS_NORM:
        k_rstd = tl.math.rsqrt(tl.sum(k_x * k_x, axis=1) / N + eps).reshape(BLOCK_SIZE_ROW, 1)
        k_y = k_x * k_rstd
    else:
        k_mean = (tl.sum(k_x, axis=1) / N).reshape(BLOCK_SIZE_ROW, 1)
        k_x_centered = k_x - k_mean
        k_rstd = tl.math.rsqrt(tl.sum(k_x_centered * k_x_centered, axis=1) / N + eps).reshape(BLOCK_SIZE_ROW, 1)
        k_y = k_x_centered * k_rstd
    kw = tl.load(K_W + cols_n).to(tl.float32)
    tl.store(K_Y + offsets, k_y * kw[None, :], mask=h_mask)


def my_fused_qk_layernorm(
    q: Tensor,   # [B, H, N]
    k: Tensor,   # [B, H, N]
    q_weight: Tensor,  # [N]
    k_weight: Tensor,  # [N]
    eps: float = 1e-6,
    is_rms_norm: bool = True,
    q_out: Optional[Tensor] = None, # [B, H, N]
    k_out: Optional[Tensor] = None, # [B, H, N]
):
    H, N = q.shape[-2:]
    q_ = q.view(-1, H, N)
    k_ = k.view(-1, H, N)
    if q_out is None:
        q_out = torch.empty_like(q)
    if k_out is None:
        k_out = torch.empty_like(k)
    q_out_ = q_out.view(-1, H, N)
    k_out_ = k_out.view(-1, H, N)
    assert q_out.shape == q.shape and k_out.shape == k.shape, f"{q_out.shape=} != {q_.shape=} or {k_out.shape=} != {k_.shape=}"
    assert N in [64, 128], f"qk layernorm is designed for head_dim in [64, 128], got {N=}"
    B = q_.shape[0]
    assert q_.stride(2) == 1 and k_.stride(2) == 1
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    BLOCK_SIZE_ROW = triton.next_power_of_2(H)
    _fused_qk_norm_kernel[(B,)](
        q_, k_, q_weight, k_weight, q_out_, k_out_,
        q_.stride(0), k_.stride(0), q_out_.stride(0), k_out_.stride(0),
        H, N, eps,
        BLOCK_SIZE_N, BLOCK_SIZE_ROW,
        is_rms_norm,
        # num_warps=4,
    )
    return q_out, k_out


"""
y = norm(x) * weight + bias
"""
def my_qk_layernorm(
    x: Tensor,   # [B, H, N]
    eps: float,
    out: Optional[Tensor] = None, # [B, H, N]
    weight: Optional[Tensor] = None, # [N]
    bias: Optional[Tensor] = None, # [N]
    is_rms_norm: bool = True,
):
    if out is None:
        out = torch.empty_like(x)
    assert out.shape == x.shape, f"{out.shape=} != {x.shape=}"
    H, N = x.shape[-2:]
    x_ = x.view(-1, H, N)
    out_ = out.view(-1, H, N)
    x_.view(-1, H * N) # x.should be contiguous on H and N
    assert x_.stride(2) == 1, f"x should be contiguous on N, got {x_.stride(2)=}"
    assert out_.stride(2) == 1, f"out should be contiguous on N, got {out_.stride(2)=}"
    assert N in [64, 128], f"qk layernorm is designed for head_dim in [64, 128], got {N=}"
    assert H * N * x.element_size() <= 65536, f"one row should be less than 64KB, got {H * N * x.element_size()=} bytes"

    assert weight is None or weight.shape == (N,), f"{weight.shape=} != {N=}"
    assert bias is None or bias.shape == (N,), f"{bias.shape=} != {N=}"
    B = x_.shape[0]
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    BLOCK_SIZE_ROW = triton.next_power_of_2(H)
    _qk_layer_norm_fwd_1pass_1row_kernel[(B,)](
        x_, out_, weight, bias,
        x_.stride(0),
        out_.stride(0),
        H,
        N,
        eps,
        BLOCK_SIZE_N,
        BLOCK_SIZE_ROW,
        is_rms_norm,
        weight is not None,
        bias is not None,
        # num_warps=4,
    )
    return out