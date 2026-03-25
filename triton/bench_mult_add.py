import pytest
import torch
import triton
from mult_add import my_mult_add, torch_compiled_mult_add
from sglang.jit_kernel.diffusion.triton.scale_shift import fuse_scale_shift_kernel as sglang_fuse_scale_shift_kernel

@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("S", [128])
@pytest.mark.parametrize("H", [5120])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_mult_and_add(B, S, H, dtype):
    x = torch.randn(B, S, H, dtype=dtype).cuda()
    w1d = torch.randn(H, dtype=dtype).cuda()
    w2d = torch.randn(B, S, H, dtype=dtype).cuda()
    b1d = torch.randn(H, dtype=dtype).cuda()
    b2d = torch.randn(B, S, H, dtype=dtype).cuda()
    from mult_add import _my_mult_add
    y_my_base = _my_mult_add(x, w1d, w2d, b1d, b2d, SAVE_FP32=False)
    y_torch = (x.float() * (w2d.float() + w1d.float()) + (b2d.float() + b1d.float())).type_as(x)
    torch.testing.assert_close(y_my_base, y_torch, rtol=1e-3, atol=1e-3)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],
        x_vals=[512, 4096, 6144, 8192],  # different possible values for `x_name`
        line_arg=
        "provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "torch",
            "b10",
            "sglang",
        ],  # possible values for `line_arg`
        line_names=[
            "Torch",
            "B10",
            "SGLang",
        ],  # label name for the lines
        styles=[
            ("green", "-"),
            ("red", "--"),
            ("blue", "-."),
        ],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name=
        "Mult and Add throughput",  # name for the plot, used also as a file name for saving the plot.
        args={
            "B": 1,
            "H": 4608,
            "dtype": torch.float32,
        },  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_mult_and_add(B, S, H, dtype, provider):
    """Benchmark Mult and Add throughput across different implementations."""
    device = torch.device("cuda")

    x = torch.randn(B, S, H, device=device, dtype=dtype)
    w2d = torch.randn(B, S, H, device=device, dtype=dtype).cuda()
    b2d = torch.randn(B, S, H, device=device, dtype=dtype).cuda()

    def test_my_mult_and_add():
        return my_mult_add(x, w2d, b2d)

    def test_torch_mult_and_add():
        return torch_compiled_mult_add(x, w2d, b2d)

    def test_sglang_mult_and_add():
        return sglang_fuse_scale_shift_kernel(x, w2d, b2d, scale_constant=0.0)

    if provider == "b10":
        ms = triton.testing.do_bench(test_my_mult_and_add)
    elif provider == "torch":
        ms = triton.testing.do_bench(test_torch_mult_and_add)
    elif provider == "sglang":
        ms = triton.testing.do_bench(test_sglang_mult_and_add)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    gb_s = (3 * B * S * H) * dtype.itemsize / ms * 1e-6
    return gb_s

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],
        x_vals=[512, 9450, 18900],  # different possible values for `x_name`
        line_arg=
        "provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "torch",
            "b10",
        ],  # possible values for `line_arg`
        line_names=[
            "Torch",
            "B10",
        ],  # label name for the lines
        styles=[
            ("green", "-"),
            ("red", "--"),
        ],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name=
        "Just Add throughput",  # name for the plot, used also as a file name for saving the plot.
        args={
            "B": 1,
            "H": 5120,
            "dtype": torch.float32,
        },  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_just_add(B, S, H, dtype, provider):
    """Benchmark Mult and Add throughput across different implementations."""
    device = torch.device("cuda")

    x = torch.randn(B, S, H, device=device, dtype=dtype)
    b2d = torch.randn(B, S, H, device=device, dtype=dtype).cuda()

    def test_my_mult_and_add():
        return my_mult_add(x, None, b2d)

    def test_torch_mult_and_add():
        return x + b2d

    if provider == "b10":
        ms = triton.testing.do_bench(test_my_mult_and_add)
    elif provider == "torch":
        ms = triton.testing.do_bench(test_torch_mult_and_add)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    gb_s = (3 * B * S * H) * dtype.itemsize / ms * 1e-6
    return gb_s



if __name__ == "__main__":
    pytest.main([__file__])
    benchmark_mult_and_add.run(print_data=True)
    benchmark_just_add.run(print_data=True)