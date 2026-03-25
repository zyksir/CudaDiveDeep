import torch
import triton
import triton.language as tl
from typing import Optional

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

# Y = X * (W2d + W1d) + B2d + B1d
@triton.autotune(
    configs=triton_autotune_configs(),
    key=["L", "H"],
)
@triton.jit
def _mult_and_add_kernel(
    X, # pointer to the input A with shape [L, H]
    X_Bias, # pointer to the input X_Bias with shape [H]
    W1d, # pointer to the input W1d with shape [H]
    W2d, # pointer to the input W2d with shape [L, H]
    B1d, # pointer to the input B1d with shape [H]
    B2d, # pointer to the input B2d with shape [L, H]
    Y, # pointer to the input Y with shape [L, H]
    L: tl.int64,
    H: tl.int64,
    x_stride_l: tl.int64,
    w2d_stride_l: tl.int64,
    b2d_stride_l: tl.int64,
    HAS_W1D: tl.constexpr,
    HAS_W2D: tl.constexpr,
    HAS_B1D: tl.constexpr,
    HAS_B2D: tl.constexpr,
    HAS_X_BIAS: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SAVE_FP32: tl.constexpr,
):
    compute_dtype = tl.float32
    pid = tl.program_id(0)
    row = pid * BLOCK_L + tl.arange(0, BLOCK_L)[:, None]
    mask_rows = row < L

    for col_offset in range(0, H, BLOCK_H):
        cols = col_offset + tl.arange(0, BLOCK_H)[None, :]
        mask_cols = cols < H
        x = tl.load(X + row * x_stride_l + cols, mask=mask_cols & mask_rows, other=0.)
        x_compute = x.to(compute_dtype)
        if HAS_X_BIAS:
            x_bias = tl.load(X_Bias + cols, mask=mask_cols, other=0.).to(compute_dtype)
            x_compute += x_bias
        if HAS_W2D and HAS_W1D:
            w2d = tl.load(W2d + row * w2d_stride_l + cols, mask=mask_cols & mask_rows, other=0.).to(compute_dtype)
            w1d = tl.load(W1d + cols, mask=mask_cols, other=0.).to(compute_dtype)
            w = w2d + w1d
        elif not HAS_W2D and HAS_W1D:
            w1d = tl.load(W1d + cols, mask=mask_cols, other=0.).to(compute_dtype)
            w = w1d
        elif HAS_W2D and not HAS_W1D:
            w2d = tl.load(W2d + row * w2d_stride_l + cols, mask=mask_cols & mask_rows, other=0.).to(compute_dtype)
            w = w2d
        else:
            w = 1.0
        if HAS_B2D and HAS_B1D:
            b2d = tl.load(B2d + row * b2d_stride_l + cols, mask=mask_cols & mask_rows, other=0.).to(compute_dtype)
            b1d = tl.load(B1d + cols, mask=mask_cols, other=0.).to(compute_dtype)
            b = b2d + b1d
        elif not HAS_B2D and HAS_B1D:
            b1d = tl.load(B1d + cols, mask=mask_cols, other=0.).to(compute_dtype)
            b = b1d
        elif HAS_B2D and not HAS_B1D:
            b2d = tl.load(B2d + row * b2d_stride_l + cols, mask=mask_cols & mask_rows, other=0.).to(compute_dtype)
            b = b2d
        else:
            b = 0.0
        y = x_compute * w + b
        if not SAVE_FP32:
            y = y.to(x.dtype)
        tl.store(Y + row * x_stride_l + cols, y, mask=mask_cols & mask_rows)

# Y = X * (W2d + W1d) + B2d + B1d
def _my_mult_add(x: torch.Tensor, w1d: Optional[torch.Tensor], w2d: Optional[torch.Tensor], b1d: Optional[torch.Tensor], b2d: Optional[torch.Tensor], x_bias: Optional[torch.Tensor] = None, SAVE_FP32: bool = False):
    if SAVE_FP32:
        y = torch.empty_like(x, dtype=torch.float32)
    else:
        y = torch.empty_like(x)
    x_, y_ = x.view(-1, x.shape[-1]), y.view(-1, x.shape[-1])
    w1d_, w2d_, b1d_, b2d_, x_bias_ = None, None, None, None, None
    if x_bias is not None:
        x_bias_ = x_bias.view(-1, x.shape[-1])
    if w1d is not None:
        w1d_ = w1d.view(-1, x.shape[-1])
    if w2d is not None:
        w2d_ = w2d.view(-1, x.shape[-1])
    if b1d is not None:
        b1d_ = b1d.view(-1, x.shape[-1])
    if b2d is not None:
        b2d_ = b2d.view(-1, x.shape[-1])
    L, H = x_.shape[0], x_.shape[1]
    BLOCK_L = 8
    BLOCK_H = min(triton.next_power_of_2(H), 4096 // BLOCK_L)
    x_stride_l = x_.stride(0)
    w2d_stride_l = w2d_.stride(0) if w2d_ is not None else 0
    b2d_stride_l = b2d_.stride(0) if b2d_ is not None else 0
    _mult_and_add_kernel[(triton.cdiv(L, BLOCK_L),)](
        x_, 
        x_bias_,
        w1d_, 
        w2d_, 
        b1d_, 
        b2d_, 
        y_, 
        L, 
        H, 
        x_stride_l, 
        w2d_stride_l, 
        b2d_stride_l,
        w1d is not None, 
        w2d is not None, 
        b1d is not None, 
        b2d is not None,
        x_bias is not None,
        BLOCK_L, 
        BLOCK_H, 
        SAVE_FP32
    )
    return y

def my_mult_add(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    x_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if w is not None:
        w = w.view(-1, x.shape[-1])
    if b is not None:
        b = b.view(-1, x.shape[-1])
    if w is None:
        w1d, w2d = None, None
    elif x.shape == w.shape:
        w1d, w2d = None, w
    elif x.shape[-1] == w.numel():
        w1d, w2d = w, None
    else:
        raise ValueError(f"Invalid shape: {x.shape} != {w.shape}")
    
    if b is None:
        b1d, b2d = None, None
    elif x.shape == b.shape:
        b1d, b2d = None, b
    elif x.shape[-1] == b.numel():
        b1d, b2d = b, None
    else:
        raise ValueError(f"Invalid shape: {x.shape} != {b.shape}")
    
    return _my_mult_add(x, w1d=w1d, w2d=w2d, b1d=b1d, b2d=b2d, SAVE_FP32=False).view(x_shape)

@torch.compile
def torch_compiled_mult_add(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    return x * scale + shift