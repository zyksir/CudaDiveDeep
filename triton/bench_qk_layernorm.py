import pytest
import torch
import torch.nn.functional as F

import triton
from qk_layernorm import my_qk_layernorm, my_fused_qk_layernorm
from layernorm import my_fused_layer_norm as my_layer_norm

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def torch_qk_rmsnorm(x, weight, eps, bias=None):
    x_dtype = x.dtype
    x_fp32 = x.contiguous().to(torch.float)
    weight_fp32 = weight.to(torch.float)
    y_fp32 = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps) * weight_fp32
    if bias is not None:
        y_fp32 += bias.to(torch.float)
    return y_fp32.to(x_dtype)

def torch_qk_layernorm(x, weight, eps, bias=None):
    x_dtype = x.dtype
    x_fp32 = x.to(torch.float)
    weight_fp32 = weight.to(torch.float)
    y_fp32 = F.layer_norm(
        x_fp32,
        (x.shape[-1],),
        weight_fp32,
        bias.float() if bias is not None else None,
        eps,
    )
    return y_fp32.to(x_dtype)

@pytest.mark.parametrize("B", [48])
@pytest.mark.parametrize("H", [48])
@pytest.mark.parametrize("N", [128])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_rms_norm", [True])
@torch.no_grad()
def test_norm(B, H, N, eps, dtype, use_rms_norm):
    # x_raw = torch.randn(B, 2*H, N, dtype=dtype).cuda()
    # x = torch.chunk(x_raw, 2, dim=1)[0]
    x_raw = torch.randn(B, H, N, dtype=dtype).cuda()
    weight = torch.randn(2*N, dtype=dtype).cuda()
    q, k = torch.chunk(x_raw, 2, dim=1)
    q_weight, k_weight = torch.chunk(weight, 2, dim=0)
    if use_rms_norm:
        torch_y_q = torch_qk_rmsnorm(q, q_weight, eps)
        torch_y_k = torch_qk_rmsnorm(k, k_weight, eps)
    else:
        torch_y_q = torch_qk_layernorm(q, q_weight, eps)
        torch_y_k = torch_qk_layernorm(k, k_weight, eps)
    my_y_q = my_qk_layernorm(q, eps=eps, weight=q_weight, is_rms_norm=use_rms_norm)
    my_y_k = my_qk_layernorm(k, eps=eps, weight=k_weight, is_rms_norm=use_rms_norm)

    my_fused_y_q, my_fused_y_k = my_fused_qk_layernorm(q, k, q_weight, k_weight, eps=eps, is_rms_norm=use_rms_norm)
    if dtype == torch.bfloat16:
        rtol_diff, atol_diff = 1e-2, 1e-2
    else:
        rtol_diff, atol_diff = 1e-5, 1e-5
    torch.testing.assert_close(torch_y_q, my_y_q, rtol=rtol_diff, atol=atol_diff)
    torch.testing.assert_close(torch_y_k, my_y_k, rtol=rtol_diff, atol=atol_diff)
    torch.testing.assert_close(torch_y_q, my_fused_y_q, rtol=rtol_diff, atol=atol_diff)
    torch.testing.assert_close(torch_y_k, my_fused_y_k, rtol=rtol_diff, atol=atol_diff)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B"],
        x_vals=[1024, 4096, 12800],  # different possible values for `x_name`
        line_arg=
        "provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "torch",
            "my_layer_norm",
            "my_qk_layernorm",
            "my_fused",
            "sglang",
        ],  # possible values for `line_arg`
        line_names=[
            "torch",
            "my_layer_norm",
            "my_qk_layernorm",
            "my_fused",
            "sglang",
        ],  # label name for the lines
        styles=[
            ("red", "--"),
            ("purple", "-"),
            ("blue", "-."),
            ("orange", "-"),
            ("green", "-."),
        ],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name=
        "QKLayerNorm throughput (non-continuous)",  # name for the plot, used also as a file name for saving the plot.
        args={
            "H": 24,
            "N": 128,
            "dtype": torch.bfloat16,
        },  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_qk_layernorm_noncontinuous(B, H, N, dtype, provider):
    """Benchmark RMSNorm throughput across different implementations."""
    device = torch.device("cuda")

    x_ = torch.randn(B, H*2, N, device=device, dtype=dtype)
    weight = torch.randn(2*N, device=device, dtype=dtype)
    q, k = torch.chunk(x_, 2, dim=1)
    q_weight, k_weight = torch.chunk(weight, 2, dim=0)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    eps = 1e-5

    def torch_norm():
        return torch_qk_rmsnorm(q, q_weight, eps), torch_qk_rmsnorm(k, k_weight, eps)
    torch_norm = torch.compile(torch_norm)

    def my_qk_norm():
        return my_qk_layernorm(q, eps=eps, weight=q_weight, out=q_out), my_qk_layernorm(k, eps=eps, weight=k_weight, out=k_out)

    def my_fused_norm():
        my_fused_qk_layernorm(q, k, q_weight, k_weight, eps=eps)
    
    def my_layer_norm_():
        return my_layer_norm(q.reshape(-1, N), eps=eps, weight=q_weight), my_layer_norm(k.reshape(-1, N), eps=eps, weight=k_weight)

    def sglang_norm():
        from sglang.jit_kernel.norm import fused_inplace_qknorm
        return fused_inplace_qknorm(
            q=q.view(B, -1, N),
            k=k.view(B, -1, N),
            q_weight=q_weight,
            k_weight=k_weight,
            eps=eps,
        )
    
    if provider == "my_qk_layernorm":
        ms = triton.testing.do_bench_cudagraph(my_qk_norm)
    elif provider == "my_fused":
        ms = triton.testing.do_bench_cudagraph(my_fused_norm)
    elif provider == "torch":
        ms = triton.testing.do_bench_cudagraph(torch_norm)
    elif provider == "sglang":
        ms = triton.testing.do_bench_cudagraph(sglang_norm)
    elif provider == "my_layer_norm":
        ms = triton.testing.do_bench_cudagraph(my_layer_norm_)
    else:
        raise ValueError(f"Unknown provider: *{provider}*")

    gb_s = (4 * B * H * N) * dtype.itemsize / ms * 1e-6
    return gb_s

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B"],
        x_vals=[1024, 4096, 12800],  # different possible values for `x_name`
        line_arg=
        "provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "torch",
            "my_layer_norm",
            "my_qk_layernorm",
            "my_fused",
            "sglang",
        ],  # possible values for `line_arg`
        line_names=[
            "torch",
            "my_layer_norm",
            "my_qk_layernorm",
            "my_fused",
            "sglang",
        ],  # label name for the lines
        styles=[
            ("red", "--"),
            ("purple", "-"),
            ("blue", "-."),
            ("orange", "-"),
            ("green", "-."),
        ],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name=
        "QKLayerNorm throughput (continuous)",  # name for the plot, used also as a file name for saving the plot.
        args={
            "H": 24,
            "N": 128,
            "dtype": torch.bfloat16,
        },  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_qk_layernorm_continuous(B, H, N, dtype, provider):
    """Benchmark RMSNorm throughput across different implementations."""
    device = torch.device("cuda")

    x_ = torch.randn(B, H*2, N, device=device, dtype=dtype)
    weight = torch.randn(2*N, device=device, dtype=dtype)
    q, k = torch.chunk(x_, 2, dim=1)
    q, k = q.contiguous(), k.contiguous()
    q_weight, k_weight = torch.chunk(weight, 2, dim=0)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    eps = 1e-5

    def torch_norm():
        return torch_qk_rmsnorm(q, q_weight, eps), torch_qk_rmsnorm(k, k_weight, eps)
    torch_norm = torch.compile(torch_norm)

    def my_qk_norm():
        return my_qk_layernorm(q, eps=eps, weight=q_weight, out=q_out), my_qk_layernorm(k, eps=eps, weight=k_weight, out=k_out)

    def my_fused_norm():
        my_fused_qk_layernorm(q, k, q_weight, k_weight, eps=eps)
    
    def my_layer_norm_():
        return my_layer_norm(q.reshape(-1, N), eps=eps, weight=q_weight), my_layer_norm(k.reshape(-1, N), eps=eps, weight=k_weight)

    def sglang_norm():
        from sglang.jit_kernel.norm import fused_inplace_qknorm
        return fused_inplace_qknorm(
            q=q.view(B, -1, N),
            k=k.view(B, -1, N),
            q_weight=q_weight,
            k_weight=k_weight,
            eps=eps,
        )
    
    if provider == "my_qk_layernorm":
        ms = triton.testing.do_bench_cudagraph(my_qk_norm)
    elif provider == "my_fused":
        ms = triton.testing.do_bench_cudagraph(my_fused_norm)
    elif provider == "torch":
        ms = triton.testing.do_bench_cudagraph(torch_norm)
    elif provider == "sglang":
        ms = triton.testing.do_bench_cudagraph(sglang_norm)
    elif provider == "my_layer_norm":
        ms = triton.testing.do_bench_cudagraph(my_layer_norm_)
    else:
        raise ValueError(f"Unknown provider: *{provider}*")

    gb_s = (4 * B * H * N) * dtype.itemsize / ms * 1e-6
    return gb_s

if __name__ == "__main__":
    pytest.main([__file__])
    benchmark_qk_layernorm_noncontinuous.run(print_data=True)
    benchmark_qk_layernorm_continuous.run(print_data=True)