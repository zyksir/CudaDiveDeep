"""
Lecture 1: Copy

Coverage:

- Example of Tile Copy
- Predication:
    - `elem_less`
"""
import torch
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

import cuda.bindings.driver as cuda

class NaiveCopy:
    def __init__(
        self,
        cta_tiler: Tuple[int, int] = (128, 128),
        num_threads: int = 256,
    ):
        self.cta_tiler = cta_tiler
        self.num_threads = num_threads
        self.BLK_M, self.BLK_N = self.cta_tiler
    
    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        tiled_copy: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        tiler_coord = (bidx, bidy)
        # (M, N) -> (BLK_M, BLK_N)
        gA = cute.local_tile(mA, self.cta_tiler, tiler_coord, proj=(1, 1))
        gB = cute.local_tile(mB, self.cta_tiler, tiler_coord, proj=(1, 1))
        mcA = cute.make_identity_tensor(mA.shape)
        cA = cute.local_tile(mcA, self.cta_tiler, tiler_coord, proj=(1, 1))
        print(f"{gA.type=} {gB.type=} {cA.type=}")

        thr_copy = tiled_copy.get_slice(tidx)
        tAgA = thr_copy.partition_S(gA) # ((atom_v, rest_v), CPY_M, CPY_N)
        tBgB = thr_copy.partition_D(gB) # ((atom_v, rest_v), CPY_M, CPY_N)
        print(f"{tAgA.type=} {tBgB.type=}")
        tArA = cute.make_fragment_like(tAgA)
        tAcA = thr_copy.partition_S(cA) # ((atom_v, rest_v), CPY_M, CPY_N)
        tApA = cute.make_fragment_like(
            cute.make_layout(
                (
                    tArA.shape[0][1],
                    cute.size(tArA, mode=[1]),
                    cute.size(tArA, mode=[2]),
                ),
                stride=(cute.size(tArA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        print(f"{tAcA.type=} {tApA.type=}")
        # let just assum N is divisible by BLK_N
        for rest_v in cutlass.range_constexpr(tApA.shape[0]):
            for cpy_m in cutlass.range_constexpr(tApA.shape[1]):
                for cpy_n in cutlass.range_constexpr(tApA.shape[2]):
                    coord_A = tAcA[(rest_v, 0), cpy_m, cpy_n]
                    tApA[rest_v, cpy_m, cpy_n] = cute.elem_less(coord_A, mA.shape)
                    # if tidx == 1 and bidx == 0 and bidy == 0:
                    #     thr_local_coord = (rest_v, cpy_m, cpy_n)
                    #     # thr_local_coord = (0,0,0), coord_A = (0,4), coord_A_1 = (0,5)
                    #     cute.printf("thr_local_coord = {}, coord_A = {}, coord_A_1 = {}", 
                    #         thr_local_coord, coord_A, tAcA[(rest_v+1, 0), cpy_m, cpy_n]
                    #     )
        cute.copy(tiled_copy, tAgA, tArA, pred=tApA)
        cute.copy(tiled_copy, tArA, tBgB, pred=tApA)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        stream: cuda.CUstream,
        num_bits_per_copy: cutlass.Constexpr = 128,
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
            CPY_M = self.BLK_M // num_vectorized
            CPY_N = self.num_threads // CPY_M
            thr_layout = cute.make_ordered_layout((CPY_M, CPY_N), order=(0, 1))
            val_layout = cute.make_ordered_layout((num_vectorized, 1), order=(0, 1))
        else:
            print("mA is row major")
            CPY_N = self.BLK_N // num_vectorized # 32
            CPY_M = self.num_threads // CPY_N # 8
            thr_layout = cute.make_ordered_layout((CPY_M, CPY_N), order=(1, 0))
            val_layout = cute.make_ordered_layout((1, num_vectorized), order=(1, 0))
        
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        print("Tiled Copy: ", tiled_copy)
        grid_dim = (cute.ceil_div(M, self.BLK_M), cute.ceil_div(N, self.BLK_N), 1)
        block_dim = (cute.size(thr_layout), 1, 1)
        self.kernel(mA, mB, tiled_copy).launch(
            grid=grid_dim,
            block=block_dim,
            stream=stream,
        )

if __name__ == "__main__":
    compile_cache = {}
    device = "cuda"
    M, N = 1024, 1024
    torch.random.manual_seed(42)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    A = torch.empty(M, N, device=device, dtype=torch.float32).random_(-5, 5).permute(1, 0)
    B = torch.empty(M, N, device=device, dtype=torch.float32).permute(1, 0)
    A_cute_tensor = from_dlpack(A, assumed_align=128)
    B_cute_tensor = from_dlpack(B, assumed_align=128)
    # A_cute_tensor = A_cute_tensor.mark_layout_dynamic().mark_compact_shape_dynamic(mode=1, divisibility=M)
    # B_cute_tensor = B_cute_tensor.mark_layout_dynamic().mark_compact_shape_dynamic(mode=1, divisibility=N)
    naive_copy = NaiveCopy()
    compile_key = (M, N, "naive")
    if compile_key not in compile_cache:
        compile_cache[compile_key] = cute.compile(naive_copy, A_cute_tensor, B_cute_tensor, current_stream)
    compile_cache[compile_key](A_cute_tensor, B_cute_tensor, current_stream)
    assert torch.equal(A, B)