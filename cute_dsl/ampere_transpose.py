"""
Lecture 1: Copy

Coverage:

- Example of Tile Copy
- Predication:
    - `elem_less`
"""
import torch
import pytest
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

import cuda.bindings.driver as cuda

import triton
import triton.language as tl

class Naive2DTranspose:
    def __init__(
        self,
        cta_tiler: Tuple[int, int] = (128, 128),
        num_threads: int = 256,
    ):
        self.cta_tiler = cta_tiler
        self.num_threads = num_threads
        self.BLK_M, self.BLK_N = self.cta_tiler
    
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        stream: cuda.CUstream,
        num_bits_per_copy: cutlass.Constexpr = 32,
    ):
        M = cute.size(mA.shape[0])
        N = cute.size(mA.shape[1])
        dtype = mA.element_type
        num_vectorized = num_bits_per_copy // dtype.width
        
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mA.element_type,
            num_bits_per_copy=num_bits_per_copy,
        )
        if cutlass.const_expr(utils.LayoutEnum.from_tensor(mA) == utils.LayoutEnum.COL_MAJOR):
            print("mA is col major")
            CPY_M_A = self.BLK_M // num_vectorized
            CPY_N_A = self.num_threads // CPY_M_A
            thr_layout_A = cute.make_ordered_layout((CPY_M_A, CPY_N_A), order=(0, 1))
            val_layout_A = cute.make_ordered_layout((num_vectorized, 1), order=(0, 1))

            CPY_N_B = self.BLK_N // num_vectorized
            CPY_M_B = self.num_threads // CPY_N_B
            thr_layout_B = cute.make_ordered_layout((CPY_N_B, CPY_M_B), order=(0, 1))
            val_layout_B = cute.make_ordered_layout((num_vectorized, 1), order=(0, 1))
        else:
            print("mA is row major")
            CPY_N_A = self.BLK_N // num_vectorized # 32
            CPY_M_A = self.num_threads // CPY_N_A # 8
            thr_layout_A = cute.make_ordered_layout((CPY_M_A, CPY_N_A), order=(1, 0))
            val_layout_A = cute.make_ordered_layout((1, num_vectorized), order=(1, 0))

            CPY_M_B = self.BLK_M // num_vectorized
            CPY_N_B = self.num_threads // CPY_M_B
            thr_layout_B = cute.make_ordered_layout((CPY_N_B, CPY_M_B), order=(1, 0))
            val_layout_B = cute.make_ordered_layout((1, num_vectorized), order=(1, 0))
        
        sA_layout = cute.make_layout((self.BLK_M, self.BLK_N))
        sB_layout = cute.make_layout((self.BLK_N, self.BLK_M))
        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sA_layout)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sB_layout)], 1024
            ]

        tiled_copy_A = cute.make_tiled_copy_tv(copy_atom, thr_layout_A, val_layout_A)
        tiled_copy_B = cute.make_tiled_copy_tv(copy_atom, thr_layout_B, val_layout_B)

        print("Tiled Copy: tiled_copy_A = {}".format(tiled_copy_A))
        print("tiled_copy_B = {}".format(tiled_copy_B))
        print("thr_layout_A = {}, thr_layout_B = {}".format(thr_layout_A, thr_layout_B))
        print("val_layout_A = {}, val_layout_B = {}".format(val_layout_A, val_layout_B))
        print("sA_layout = {}, sB_layout = {}".format(sA_layout, sB_layout))
        grid_dim = (cute.ceil_div(M, self.BLK_M), cute.ceil_div(N, self.BLK_N), 1)
        block_dim = (cute.size(thr_layout_A), 1, 1)
        self.kernel(mA, mB, tiled_copy_A, tiled_copy_B, sA_layout, sB_layout, SharedStorage).launch(
            grid=grid_dim,
            block=block_dim,
            stream=stream,
        )
    
    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        tiler_coord = (bidx, bidy)
        # (M, N) -> (BLK_M, BLK_N)
        gA = cute.local_tile(mA, self.cta_tiler, tiler_coord, proj=(1, 1))
        gB = cute.local_tile(mB, self.cta_tiler[::-1], tiler_coord[::-1], proj=(1, 1))
        print("gA = {}, gB = {}".format(gA, gB))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout) # (BLK_M, BLK_N)
        sB = storage.sB.get_tensor(sB_layout) # (BLK_N, BLK_M)
        print("sA = {}, sB = {}".format(sA, sB))

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tBgB = thr_copy_B.partition_D(gB)
        tBsB = thr_copy_B.partition_S(sB)
        cute.copy(tiled_copy_A, tAgA, tAsA)
        print("tAgA = {}, tBgB = {}".format(tAgA, tBgB))
        print("tAsA = {}, tBsB = {}".format(tAsA, tBsB))


        # # # sA_transpose_layout = cute.composition(
        # #     sA_layout,
        # #     cute.make_ordered_layout((self.BLK_M, self.BLK_N), order=(1, 0))
        # # )
        # # print("sA_transpose_layout = {}".format(sA_transpose_layout))
        # tA_local = cute.flat_divide(sA_layout, (
        #     cute.make_layout(8, stride=1), cute.make_layout(32, stride=1)
        #     )
        # )
        # tB_local = cute.flat_divide(sB_layout, (
        #     cute.make_layout(32, stride=1), cute.make_layout(8, stride=1)
        #     )
        # )
        # sA_flat = storage.sA.get_tensor(cute.make_layout(self.BLK_M * self.BLK_N, stride=1)) # (BLK_M, BLK_N)
        # sB_flat = storage.sB.get_tensor(cute.make_layout(self.BLK_N * self.BLK_M, stride=1)) # (BLK_N, BLK_M)
        # warp_idx = cute.arch.warp_idx()
        # lane_idx = cute.arch.lane_idx()
        # if bidx == 0 and bidy == 0:
        #     if tidx == 33 and lane_idx == 1 and warp_idx == 1:
        #         cute.printf("tidx = {}, warp_idx = {}, lane_idx = {}, tA_local = {}, tB_local = {}".format(tidx, warp_idx, lane_idx, tA_local, tB_local))
        # print("tA_local = {}, tB_local = {}".format(tA_local, tB_local))
        # for i in cutlass.range(cute.size(tA_local, mode=[2]), unroll_full=True):
        #     for j in cutlass.range(cute.size(tA_local, mode=[3]), unroll_full=True):
        #         sB_flat[tB_local((lane_idx, warp_idx, j, i))] = sA_flat[tA_local((warp_idx, lane_idx, i, j))]
        # cute.arch.barrier()

        for idx in cutlass.range(tidx, cute.size(sA), self.num_threads, unroll_full=True):
            i = idx // cute.size(sA, mode=[1])
            j = idx % cute.size(sA, mode=[1])
            sB[j, i] = sA[i, j]
            # sB_flat[sB_layout((j, i))] = sA_flat[sA_layout((i, j))]
        cute.arch.barrier()
        cute.copy(tiled_copy_B, tBsB, tBgB)

@triton.jit
def transpose_2d_kernel(
    in_ptr, out_ptr,          # pointers
    M, N,                     # sizes: in is [M, N], out is [N, M]
    stride_in_m, stride_in_n,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # program ids for 2D launch grid
    pid_m = tl.program_id(0)  # block index along M
    pid_n = tl.program_id(1)  # block index along N

    # row/col indices this program will handle
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # build 2D index grids
    # in: [M, N]
    in_idx = offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
    # out: [N, M] → note swapped m/n
    out_idx = offs_m[:, None] * stride_out_n + offs_n[None, :] * stride_out_m

    # mask for boundaries
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # load a BLOCK_M x BLOCK_N tile
    x = tl.load(in_ptr + in_idx, mask=mask, other=0.0)

    # store transposed tile
    tl.store(out_ptr + out_idx, x, mask=mask)


def transpose_2d_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    M, N = x.shape
    y = torch.empty((N, M), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 128

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    transpose_2d_kernel[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return y


compile_cache = {}

@pytest.mark.parametrize("M", [512, 4096])
@pytest.mark.parametrize("N", [1024, 2048])
@pytest.mark.parametrize("dtype", [torch.float32])
@torch.no_grad()
def test_permute_mnk(M, N, dtype):
    x = torch.randn(M, N, dtype=dtype).cuda()
    y_torch = x.permute(1, 0).contiguous()
    y_triton = transpose_2d_triton(x)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    y_cute = torch.empty(N, M, device=x.device, dtype=torch.float32)
    x_cute_tensor = from_dlpack(x, assumed_align=128)
    y_cute_tensor = from_dlpack(y_cute, assumed_align=128)
    # x_cute_tensor = x_cute_tensor.mark_layout_dynamic().mark_compact_shape_dynamic(mode=1, divisibility=N)
    # y_cute_tensor = y_cute_tensor.mark_layout_dynamic().mark_compact_shape_dynamic(mode=1, divisibility=M)
    naive_2d_transpose = Naive2DTranspose()
    compile_key = (M, N, "naive_2d_transpose")
    if compile_key not in compile_cache:
        compile_cache[compile_key] = cute.compile(naive_2d_transpose, x_cute_tensor, y_cute_tensor, current_stream)
    compile_cache[compile_key](x_cute_tensor, y_cute_tensor, current_stream)

    torch.testing.assert_close(y_triton, y_torch, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y_cute, y_torch, rtol=1e-5, atol=1e-5)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[512, 4096],  # different possible values for `x_name`
        line_arg=
        "provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "torch",
            "triton",
            "cute",
        ],  # possible values for `line_arg`
        line_names=[
            "Torch",
            "Triton",
            "CuTeDSL",
        ],  # label name for the lines
        styles=[
            ("green", "-"),
            ("red", "--"),
            ("blue", ":"),
        ],  # line styles
        plot_name="",
        ylabel="GB/s",  # label name for the y-axis
        args={
            "N": 1024,
            "dtype": torch.float32,
        },  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_2d_transpose(M, N, dtype, provider):
    """Benchmark permute_mnk throughput across different implementations."""
    x = torch.randn(M, N, dtype=dtype, device="cuda")

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    y_cute = torch.empty(N, M, device=x.device, dtype=torch.float32)
    x_cute_tensor = from_dlpack(x, assumed_align=128)
    y_cute_tensor = from_dlpack(y_cute, assumed_align=128)
    # x_cute_tensor = x_cute_tensor.mark_layout_dynamic().mark_compact_shape_dynamic(mode=1, divisibility=N)
    # y_cute_tensor = y_cute_tensor.mark_layout_dynamic().mark_compact_shape_dynamic(mode=1, divisibility=M)
    naive_2d_transpose = Naive2DTranspose()
    compile_key = (M, N, "naive_2d_transpose")
    if compile_key not in compile_cache:
        compile_cache[compile_key] = cute.compile(naive_2d_transpose, x_cute_tensor, y_cute_tensor, current_stream)

    def _triton_2d_transpose():
        return transpose_2d_triton(x)

    def torch_2d_transpose():
        return x.permute(1, 0).contiguous()
    
    def _cute_2d_transpose():
        return compile_cache[compile_key](x_cute_tensor, y_cute_tensor, current_stream)

    if provider == "torch":
        ms = triton.testing.do_bench(torch_2d_transpose)
    elif provider == "triton":
        ms = triton.testing.do_bench(_triton_2d_transpose)
    elif provider == "cute":
        ms = triton.testing.do_bench(_cute_2d_transpose)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    gb_s = (M * N * 2) * dtype.itemsize / ms * 1e-6
    return gb_s


if __name__ == "__main__":
    device = "cuda"
    M, N = 4096, 2048
    torch.random.manual_seed(42)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    A = torch.randn(M, N, device=device, dtype=torch.float32)
    # A = torch.empty(M, N, device=device, dtype=torch.float32).random_(-5, 5)
    B = torch.empty(N, M, device=device, dtype=torch.float32)
    A_cute_tensor = from_dlpack(A, assumed_align=128)
    B_cute_tensor = from_dlpack(B, assumed_align=128)
    A_cute_tensor = A_cute_tensor.mark_layout_dynamic()
    B_cute_tensor = B_cute_tensor.mark_layout_dynamic()
    naive_2d_transpose = Naive2DTranspose()
    compile_key = (M, N, "naive_2d_transpose")
    if compile_key not in compile_cache:
        compile_cache[compile_key] = cute.compile(naive_2d_transpose, A_cute_tensor, B_cute_tensor, current_stream)
    compile_cache[compile_key](A_cute_tensor, B_cute_tensor, current_stream)
    A_T = A.permute(1, 0).contiguous()
    # print("A_T = {}, B = {}".format(A_T, B))
    assert torch.equal(A_T, B)

    # pytest.main([__file__])
    # triton.testing.do_bench(benchmark_2d_transpose)
    benchmark_2d_transpose.run(print_data=True, show_plots=False)