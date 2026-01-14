import pytest
import torch
import torch.nn.functional as F

import triton
from sgl_kernel import rmsnorm as flashinfer_rmsnorm
from sglang.multimodal_gen.runtime.layers.triton_ops import rms_norm_fn as sglang_rmsnorm
from quack.rmsnorm import rmsnorm as quack_rmsnorm
from flash_attn.ops.triton.layer_norm import rms_norm_fn as flash_attn_rmsnorm
from my_layernorm import fused_layer_norm as my_layer_norm

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def torch_rms_norm(x, weight, eps, bias=None, residual=None, out_scale=None, out_shift=None):
    x_dtype = x.dtype
    x_fp32 = x.to(torch.float)
    weight_fp32 = weight.to(torch.float)
    if residual is not None:
        x_fp32 = x_fp32 + residual.to(torch.float)
    y_fp32 = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps) * weight_fp32
    if bias is not None:
        y_fp32 += bias.to(torch.float)
    if out_scale is not None:
        y_fp32 *= out_scale.to(torch.float)
    if out_shift is not None:
        y_fp32 += out_shift.to(torch.float)
    if residual is not None:
        return y_fp32.to(x_dtype), x_fp32.to(x_dtype)
    return y_fp32.to(x_dtype)

def torch_layer_norm(x, weight, eps, bias=None, residual=None, out_scale=None, out_shift=None):
    x_dtype = x.dtype
    x_fp32 = x.to(torch.float)
    weight_fp32 = weight.to(torch.float)
    if residual is not None:
        x_fp32 = x_fp32 + residual.to(torch.float)
    y_fp32 = F.layer_norm(
        x_fp32,
        (x.shape[-1],),
        weight_fp32,
        bias.float() if bias is not None else None,
        eps,
    )
    if out_scale is not None:
        y_fp32 *= out_scale.to(torch.float)
    if out_shift is not None:
        y_fp32 += out_shift.to(torch.float)
    if residual is not None:
        return y_fp32.to(x_dtype), x_fp32.to(x_dtype)
    return y_fp32.to(x_dtype)

@pytest.mark.parametrize("B", [64])
@pytest.mark.parametrize("H", [4096])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_rms_norm", [True, False])
@pytest.mark.parametrize("use_bias", [True])
@pytest.mark.parametrize("use_residual", [True])
@pytest.mark.parametrize("use_out_scale", [True])
@pytest.mark.parametrize("use_out_shift", [True])
@torch.no_grad()
def test_norm(B, H, eps, dtype, use_rms_norm, use_bias, use_residual, use_out_scale, use_out_shift):
    x = torch.randn(B, H, dtype=dtype).cuda()
    weight = torch.randn(H, dtype=dtype).cuda()
    bias = torch.randn(H, dtype=dtype).cuda() if use_bias else None
    residual = torch.randn(B, H, dtype=dtype).cuda() if use_residual else None
    out_scale = torch.randn(B, H, dtype=dtype).cuda() if use_out_scale else None
    out_shift = torch.randn(B, H, dtype=dtype).cuda() if use_out_shift else None
    if use_rms_norm:
        torch_y = torch_rms_norm(x, weight, eps, bias, residual, out_scale, out_shift)
    else:
        torch_y = torch_layer_norm(x, weight, eps, bias, residual, out_scale, out_shift)
    my_residual_out = torch.empty_like(residual) if use_residual else None
    my_y = my_layer_norm(x, weight, eps, out=None, bias=bias, residual=residual, residual_out=my_residual_out, out_scale=out_scale, out_shift=out_shift, is_rms_norm=use_rms_norm)
    if use_residual:
        my_y, my_residual_out = my_y
        torch_y, torch_residual_out = torch_y
    if dtype == torch.bfloat16:
        rtol_diff, atol_diff = 1e-2, 1e-2
    else:
        rtol_diff, atol_diff = 1e-5, 1e-5
    torch.testing.assert_close(torch_y, my_y, rtol=rtol_diff, atol=atol_diff)
    if use_residual:
        torch.testing.assert_close(torch_residual_out, my_residual_out, rtol=rtol_diff, atol=atol_diff)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128, 1024, 3840, 4096, 5120, 8192],  # different possible values for `x_name`
        line_arg=
        "provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            # "torch",
            "flash-attn",
            "my",
            "flashinfer",
            "quack",
        ],  # possible values for `line_arg`
        line_names=[
            # "torch",
            "flash-attn",
            "my",
            "flashinfer",
            "quack",
        ],  # label name for the lines
        styles=[
            # ("green", "-"),
            ("red", "--"),
            ("blue", "-."),
            ("purple", "-."),
            ("yellow", "-."),
        ],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name=
        "RMSNorm throughput",  # name for the plot, used also as a file name for saving the plot.
        args={
            "B": 12800,
            "dtype": torch.bfloat16,
        },  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_rms_norm(B, N, dtype, provider):
    """Benchmark RMSNorm throughput across different implementations."""
    device = torch.device("cuda")

    x = torch.randn(B, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    eps = 1e-5

    def torch_norm():
        return torch_rms_norm(x, weight, eps)
    
    def my_norm():
        return my_layer_norm(x, weight=weight, eps=eps, out=None)
    
    def flashinfer_norm():
        return flashinfer_rmsnorm(x, weight, eps)
    
    def flash_attn_norm():
        return flash_attn_rmsnorm(x, weight=weight, bias=None, eps=eps)
    
    def quack_norm():
        return quack_rmsnorm(x, weight, eps=eps)
    
    if provider == "my":
        ms = triton.testing.do_bench(my_norm)
    # elif provider == "torch":
    #     ms = triton.testing.do_bench(torch_norm)
    elif provider == "flashinfer":
        ms = triton.testing.do_bench(flashinfer_norm)
    elif provider == "flash-attn":
        ms = triton.testing.do_bench(flash_attn_norm)
    elif provider == "quack":
        ms = triton.testing.do_bench(quack_norm)
    else:
        raise ValueError(f"Unknown provider: *{provider}*")

    gb_s = (2 * B * N) * 4 / ms * 1e-6
    return gb_s

if __name__ == "__main__":
    pytest.main([__file__])
    benchmark_rms_norm.run(print_data=True)
    # B, H, eps, dtype = 1024, 1024, 1e-5, torch.bfloat16
    # use_bias, use_residual, use_out_scale, use_out_shift = False, True, False, False
    # x = torch.randn(B, H, dtype=dtype).cuda()
    # weight = torch.randn(H, dtype=dtype).cuda()
    # bias = torch.randn(H, dtype=dtype).cuda() if use_bias else None
    # residual = torch.randn(B, H, dtype=dtype).cuda() if use_residual else None
    # out_scale = torch.randn(B, H, dtype=dtype).cuda() if use_out_scale else None
    # out_shift = torch.randn(B, H, dtype=dtype).cuda() if use_out_shift else None
    # print(f"{residual=}, {x=}")
    # torch_y = torch_rms_norm(x, weight, eps, bias, residual, out_scale, out_shift)
    # my_y = torch.empty_like(x)
    # my_layer_norm(x, weight, eps, out=my_y, bias=bias, residual=residual, out_scale=out_scale, out_shift=out_shift, is_rms_norm=True)
    # torch.testing.assert_close(torch_y, my_y, rtol=1e-2, atol=1e-2)