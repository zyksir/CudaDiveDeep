import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from typing import Tuple
import cutlass.cute.testing as testing

class SGEMM_layout_demo:

    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (128, 128, 8),
    ):
        self.cta_tiler = cta_tiler
        self.BLK_M, self.BLK_N, self.BLK_K = self.cta_tiler
    
    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor, # ((M, K); (1, M))
        mB: cute.Tensor, # ((N, K); (1, N))
        mC: cute.Tensor, # ((M, N); (1, M))
        sA_layout: cute.Layout, # (BLK_M, BLK_K)
        sB_layout: cute.Layout, # (BLK_N, BLK_K)
        tA_layout: cute.Layout, # (THR_M, THR_K)
        tB_layout: cute.Layout, # (THR_N, THR_K)
        tC_layout: cute.Layout, # (THR_M, THR_N)
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        cta_coord = (bidx, bidy, None)
        gA = cute.local_tile(mA, self.cta_tiler, cta_coord, proj=(1, None, 1));  # (BLK_M,BLK_K,k)
        gB = cute.local_tile(mB, self.cta_tiler, cta_coord, proj=(None, 1, 1));  # (BLK_N,BLK_K,k)
        gC = cute.local_tile(mC, self.cta_tiler, cta_coord, proj=(1, 1, None));  # (BLK_M,BLK_N)
        
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16) # (BLK_M, BLK_K)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16) # (BLK_N, BLK_K)

        tAgA = cute.local_partition(gA, tA_layout, tidx) # (THR_M, THR_K, k)
        tBgB = cute.local_partition(gB, tB_layout, tidx) # (THR_N, THR_K, k) 
        tAsA = cute.local_partition(sA, tA_layout, tidx) # (THR_M, THR_K)
        tBsB = cute.local_partition(sB, tB_layout, tidx) # (THR_N, THR_K)

        tCsA = cute.local_partition(sA, tC_layout, tidx, proj=(1, None));   # (THR_M,BLK_K)
        tCsB = cute.local_partition(sB, tC_layout, tidx, proj=(None, 1));   # (THR_N,BLK_K)
        tCgC = cute.local_partition(gC, tC_layout, tidx, proj=(1, 1));   # (THR_M,THR_N)

        tCrC = cute.make_fragment_like(tCgC) # (THR_M, THR_N)
        tCrC.fill(0.0)
        print(f"mA: {mA.type}, mB: {mB.type}, mC: {mC.type}")
        print(f"sA_layout: {sA_layout}, sB_layout: {sB_layout}")
        print(f"tA_layout: {tA_layout}, tB_layout: {tB_layout}, tC_layout: {tC_layout}")
        print(f"gA: {gA.type}, gB: {gB.type}, gC: {gC.type}")
        print(f"sA: {sA.type}, sB: {sB.type}")
        print(f"tAgA: {tAgA.type}, tBgB: {tBgB.type}")
        print(f"tAsA: {tAsA.type}, tBsB: {tBsB.type}")
        print(f"tCsA: {tCsA.type}, tCsB: {tCsB.type}")
        print(f"tCgC: {tCgC.type}, tCrC: {tCrC.type}")

        num_k_blocks = cute.size(tAgA, mode=[2])
        for k_block_idx in cutlass.range(num_k_blocks):
            cute.autovec_copy(tAgA[None, None, k_block_idx], tAsA)
            cute.autovec_copy(tBgB[None, None, k_block_idx], tBsB)
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            # cute::gemm(tCsA, tCsB, tCrC)
            for k in cutlass.range(cute.size(tCsA, mode=[1]), unroll_full=True):
                for m in cutlass.range(cute.size(tCrC, mode=[0]), unroll_full=True):
                    for n in cutlass.range(cute.size(tCrC, mode=[1]), unroll_full=True):
                        tCrC[m, n] += tCsA[m,k] * tCsB[n,k]
            cute.arch.barrier()
        
        cute.autovec_copy(tCrC, tCgC)
            
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor, # ((M, K); (1, M))
        mB: cute.Tensor, # ((N, K); (1, N))
        mC: cute.Tensor, # ((M, N); (1, M))``
        stream: cuda.CUstream,
    ):
        gA = cute.local_tile(mA, self.cta_tiler, (cute.Int32(0), cute.Int32(0), cute.Int32(0)), proj=(1, None, 1))
        print(f"gA: {gA.type}")
        sA = cute.make_tensor(gA.iterator.align(16), cute.make_layout((self.BLK_M, self.BLK_K)))
        identity_sA = cute.make_identity_tensor(sA.shape)
        print(f"identity_sA: {identity_sA.type}, {identity_sA.layout}, {cute.size(identity_sA)}, {cute.cosize(identity_sA)}, {identity_sA[(0, 1)]}")
        composition = cute.composition(sA, cute.make_layout((16, 16)))
        print(f"composition: {composition.type}, {composition.layout}, {cute.size(composition)}, {cute.cosize(composition)}")
        zipped_divide = cute.zipped_divide(sA, cute.make_layout((2, 4)))
        print(f"zipped_divide: {zipped_divide.type}, {zipped_divide.layout}, {cute.size(zipped_divide)}, {cute.cosize(zipped_divide)}")
        logical_divide = cute.logical_divide(sA, cute.make_layout((2, 4)))
        print(f"logical_divide: {logical_divide.type}, {logical_divide.layout}, {cute.size(logical_divide)}, {cute.cosize(logical_divide)}")

        local_tile = cute.local_tile(sA, (16, 16), (cute.Int32(0), cute.Int32(0)))
        print(f"local_tile: {local_tile.type}, {local_tile.layout}, {cute.size(local_tile)}, {cute.cosize(local_tile)}")
        print(f"sA: {sA.type}, {sA.layout}, {cute.size(sA)}, {cute.cosize(sA)}")
        tCsA_debug = cute.local_partition(sA, cute.make_layout((16, 16)), cute.Int32(0))
        print(f"tCsA_debug: {tCsA_debug.type}, {tCsA_debug.layout}, {cute.size(tCsA_debug)}, {cute.cosize(tCsA_debug)}")
        tCsA = cute.local_partition(sA, cute.make_layout((16, 16)), cute.Int32(0), proj=(1, None))
        print(f"tCsA: {tCsA.type}, {tCsA.layout}, {cute.size(tCsA)}, {cute.cosize(tCsA, mode=[1])}")
        print("***************************************************")
        
        sA_layout = cute.make_layout((self.BLK_M, self.BLK_K))
        sB_layout = cute.make_layout((self.BLK_N, self.BLK_K))
        tA_layout = cute.make_layout((32, 8))
        tB_layout = cute.make_layout((32, 8))
        tC_layout = cute.make_layout((16, 16))
        print(f"a_dtype: {mA.element_type}, b_dtype: {mB.element_type}, c_dtype: {mC.element_type}")

        M = cute.size(mA.shape[0])
        N = cute.size(mB.shape[0])
        grid = (cute.ceil_div(M, self.BLK_M), cute.ceil_div(N, self.BLK_N), 1)
        block = (cute.size(tC_layout), 1, 1)

        self.kernel(
            mA, 
            mB,
            mC,
            sA_layout,
            sB_layout,
            tA_layout,
            tB_layout,
            tC_layout,
        ).launch(
            grid=grid, 
            block=block, 
            stream=stream
        )

class SGEMM_tiled_demo:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (128, 128, 8),
        num_threads: int = 256,
        num_stages: int = 3,
    ):
        self.cta_tiler = cta_tiler
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.BLK_M, self.BLK_N, self.BLK_K = self.cta_tiler
        
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor, # ((M, K); (1, M))
        mB: cute.Tensor, # ((N, K); (1, N))
        mC: cute.Tensor, # ((M, N); (1, M))
        stream: cuda.CUstream,
    ):
        M = cute.size(mC.shape[0])
        N = cute.size(mC.shape[1])
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)
        TC = cutlass.Float32

        if cutlass.const_expr(self.a_major_mode == utils.LayoutEnum.COL_MAJOR):
            print(f"[tiled sgemm] {mA.type=} {mA.layout.max_alignment=}")
            num_vectorized = 4 if (mA.layout.max_alignment % 16 == 0) else 1
            atom_async_copy_A = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mA.element_type,
                num_bits_per_copy=mA.element_type.width * num_vectorized,
            )
            CPY_M = self.BLK_M // num_vectorized
            tA = cute.make_layout((CPY_M, self.num_threads // CPY_M))
            vA = cute.make_layout((num_vectorized, 1))
        else:
            num_vectorized = 1
            atom_async_copy_A = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mA.element_type,
                num_bits_per_copy=mA.element_type.width * num_vectorized,
            )
            CPY_K = self.BLK_K // num_vectorized
            tA = cute.make_layout((self.num_threads // CPY_K, CPY_K), stride=(CPY_K, 1))
            vA = cute.make_layout((1, num_vectorized))

        if cutlass.const_expr(self.b_major_mode == utils.LayoutEnum.COL_MAJOR):
            print(f"[tiled sgemm] {mB.layout.max_alignment=}")
            num_vectorized = 4 if (mB.layout.max_alignment % 16 == 0) else 1
            atom_async_copy_B = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mB.element_type,
                num_bits_per_copy=mB.element_type.width * num_vectorized,
            )
            CPY_N = self.BLK_N // num_vectorized
            tB = cute.make_layout((CPY_N, self.num_threads // CPY_N))
            vB = cute.make_layout((num_vectorized, 1))
        else:
            num_vectorized = 1
            atom_async_copy_B = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mB.element_type,
                num_bits_per_copy=mB.element_type.width * num_vectorized,
            )
            CPY_K = self.BLK_K // num_vectorized
            tB = cute.make_layout((self.num_threads // CPY_K, CPY_K), stride=(CPY_K, 1))
            vB = cute.make_layout((1, num_vectorized))

        tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy_A, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(atom_async_copy_B, tB, vB)
        print(f"[tiled sgemm]tiled_copy_A: {tA.type=}, {vA.type=}, {tiled_copy_A=}")

        if cutlass.const_expr(self.c_major_mode == utils.LayoutEnum.COL_MAJOR):
            atoms_layout = cute.make_layout(
                (16, self.num_threads // 16, 1), stride=(1, 16, 0)
            )
        else:
            atoms_layout = cute.make_layout(
                (self.num_threads // 16, 16, 1), stride=(16, 1, 0)
            )
        op = cute.nvgpu.MmaUniversalOp(TC)
        permutation_tiler_M = cute.make_layout((atoms_layout.shape[0], 4), stride=(4, 1))
        permutation_tiler_N = cute.make_layout((atoms_layout.shape[1], 4), stride=(4, 1))
        tiled_mma = cute.make_tiled_mma(
            op,
            atoms_layout,
            permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),
        )

        grid_dim = (cute.ceil_div(M, self.BLK_M), cute.ceil_div(N, self.BLK_N), 1)
        block_dim = (cute.size(atoms_layout), 1, 1)
        sA_layout = cute.make_layout((self.BLK_M, self.BLK_K, self.num_stages))
        sB_layout = cute.make_layout((self.BLK_N, self.BLK_K, self.num_stages))
        assert cute.size(tiled_copy_A) == self.num_threads
        assert cute.size(tiled_copy_B) == self.num_threads
        assert cute.size(tiled_mma) == self.num_threads
        print(f"[call kernel] {self.a_major_mode=}, {self.b_major_mode=}, {self.c_major_mode=}")
        print(f"[call kernel] {mA.type=}, {sA_layout.type=}, {mB.type=}, {sB_layout.type=}, {mC.type=}")
        print(f"[call kernel] {tA.type=}, {vA.type=}, {tB.type=}, {vB.type=}")
        print(f"[call kernel] {grid_dim=}, block = {block_dim} ")
        self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
        ).launch(
            grid=grid_dim,
            block=block_dim,
            stream=stream,
        )
        
        return
    
    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor, # ((M, K); (1, M))
        mB: cute.Tensor, # ((N, K); (1, N))
        mC: cute.Tensor, # ((M, N); (1, M))
        sA_layout: cute.Layout, # (BLK_M, BLK_K)
        sB_layout: cute.Layout, # (BLK_N, BLK_K)
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):  
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        tiler_coord = (bidx, bidy, None)
        thr_mma = tiled_mma.get_slice(tidx)
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        
        # gA: (BLK_M, BLK_K, k), gB: (BLK_N, BLK_K, k), gC: (BLK_M, BLK_N)
        gA = cute.local_tile(mA, self.cta_tiler, tiler_coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, self.cta_tiler, tiler_coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.cta_tiler, tiler_coord, proj=(1, 1, None))
        print(f"[tiled sgemm]gA: {gA.type}, gB: {gB.type}, gC: {gC.type}")
        # Move the pointer of gA/gB in the `-k` direction, 
        # so that the first tile is irregular with [-residue_k, -1] in K dimension
        residue_k = mA.shape[1] - gA.shape[2] * cutlass.Int32(self.BLK_K)
        gA = cute.domain_offset((0, residue_k, 0), gA)
        gB = cute.domain_offset((0, residue_k, 0), gB)
        print(f"[tiled sgemm]after offset: gA: {gA.type}, gB: {gB.type}, residue_k: {residue_k}")

        # create coordinate for predication
        # cA: (BLK_M, BLK_K), cB: (BLK_N, BLK_K)
        mcA = cute.make_identity_tensor(mA.shape)
        mcB = cute.make_identity_tensor(mB.shape)
        cA = cute.local_tile(mcA, self.cta_tiler, tiler_coord, proj=(1, None, 1))
        cB = cute.local_tile(mcB, self.cta_tiler, tiler_coord, proj=(None, 1, 1))
        # 
        cA = cute.domain_offset((0, residue_k, 0), cA)
        cB = cute.domain_offset((0, residue_k, 0), cB)
        print(f"[tiled sgemm]cA: {cA.type}, cB: {cB.type}")
        
        # sA: (BLK_M, BLK_K, PIPE), sB: (BLK_N, BLK_K, PIPE)
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

        # tAgA: (CPY, CPY_M, CPY_K, K_TILE_COUNT), tBgB: (CPY, CPY_N, CPY_K, K_TILE_COUNT)
        # tAsA: (CPY, CPY_M, CPY_K, PIPE), tBsB: (CPY, CPY_N, CPY_K, PIPE)
        # where K_TILE_COUNT = K / BLK_K, PIPE = num_stages
        # CPY: (atom_v, rest_v)=(num_vectorized, 1)
        # CPY_M * CPY_K = CPY_N * CPY_K = num_threads
        # atom_v * rest_v * num_threads = BLK_M * BLK_K = BLK_N * BLK_K
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)
        print(f"[tiled sgemm]sA: {sA.type}, sB: {sB.type}")
        print(f"[tiled sgemm]tAgA: {tAgA.type}, tBgB: {tBgB.type}")
        print(f"[tiled sgemm]tAsA: {tAsA.type}, tBsB: {tBsB.type}")

        # tAcA: (CPY, CPY_M, CPY_K, K_TILE_COUNT), tBcB: (CPY, CPY_N, CPY_K, K_TILE_COUNT)
        # tApA: (rest_v, CPY_M, CPY_K)
        # tBpB: (rest_v, CPY_N, CPY_K)
        tAcA = thr_copy_A.partition_S(cA)
        tBcB = thr_copy_B.partition_S(cB)
        print(f"[tiled sgemm]tAcA: {tAcA.type}, tBcB: {tBcB.type}")
        tApA = cute.make_fragment(
            cute.make_layout(
                (
                    tAsA.shape[0][1],
                    cute.size(tAsA, mode=[1]),
                    cute.size(tAsA, mode=[2]),
                ),
                stride=(cute.size(tAsA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_fragment(
            cute.make_layout(
                (
                    tBsB.shape[0][1],
                    cute.size(tBsB, mode=[1]),
                    cute.size(tBsB, mode=[2]),
                ),
                stride=(cute.size(tBsB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tApA_residue_k = cute.make_fragment(
            cute.make_layout(
                (
                    tAsA.shape[0][1],
                    cute.size(tAsA, mode=[1]),
                    cute.size(tAsA, mode=[2]),
                ),
                stride=(
                    cute.size(tAsA, mode=[1]) * cute.size(tAsA, mode=[2]),
                    cute.size(tAsA, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )
        tBpB_residue_k = cute.make_fragment(
            cute.make_layout(
                (
                    tBsB.shape[0][1],
                    cute.size(tBsB, mode=[1]),
                    cute.size(tBsB, mode=[2]),
                ),
                stride=(
                    cute.size(tBsB, mode=[1]) * cute.size(tBsB, mode=[2]),
                    cute.size(tBsB, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )
        print(f"[tiled sgemm]tApA: {tApA.type}, tBpB: {tBpB.type}, tApA_residue_k: {tApA_residue_k.type}, tBpB_residue_k: {tBpB_residue_k.type}")
        for rest_v in range(tApA.shape[0]):
            for cpy_m in range(tApA.shape[1]):
                coord_A = tAcA[(0, rest_v), cpy_m, 0, 0]
                tApA[rest_v, cpy_m, 0] = cute.elem_less(coord_A[0], mA.shape[0])
        for rest_v in range(tBpB.shape[0]):
            for cpy_n in range(tBpB.shape[1]):
                coord_B = tBcB[(0, rest_v), cpy_n, 0, 0]
                tBpB[rest_v, cpy_n, 0] = cute.elem_less(coord_B[0], mB.shape[0])
        for rest_v in range(tApA_residue_k.shape[0]):
            for cpy_m in range(tApA_residue_k.shape[1]):
                for cpy_k in range(tApA_residue_k.shape[2]):
                    coord_A = tAcA[(0, rest_v), cpy_m, cpy_k, 0]
                    tApA_residue_k[rest_v, cpy_m, cpy_k] = cute.elem_less(
                        (coord_A[0], cutlass.Int32(-1)), (mA.shape[0], coord_A[1])
                    )
        for rest_v in range(tBpB_residue_k.shape[0]):
            for cpy_n in range(tBpB_residue_k.shape[1]):
                for cpy_k in range(tBpB_residue_k.shape[2]):
                    coord_B = tBcB[(0, rest_v), cpy_n, cpy_k, 0]
                    tBpB_residue_k[rest_v, cpy_n, cpy_k] = cute.elem_less(
                        (coord_B[0], cutlass.Int32(-1)), (mB.shape[0], coord_B[1])
                    )

        PIPE = cute.size(tAsA, mode=[3])
        K_TILE_COUNT = cute.size(tAgA, mode=[3])
        print(f"[tiled sgemm]PIPE: {PIPE}, K_TILE_COUNT: {K_TILE_COUNT}")
        gmem_pipe_read = cute.Int32(0)
        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, 0],
            pred=tApA_residue_k,
        )
        cute.copy(
            tiled_copy_B,
            tBgB[None, None, None, gmem_pipe_read],
            tBsB[None, None, None, 0],
            pred=tBpB_residue_k,
        )
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = (
            gmem_pipe_read + 1
            if gmem_pipe_read + 1 < K_TILE_COUNT
            else cutlass.Int32(0)
        )

        for k_tile_idx in range(1, PIPE - 1):
            if k_tile_idx < K_TILE_COUNT:
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, gmem_pipe_read],
                    tAsA[None, None, None, k_tile_idx],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, k_tile_idx],
                    pred=tBpB,
                )
            gmem_pipe_read = (
                gmem_pipe_read + 1
                if gmem_pipe_read + 1 < K_TILE_COUNT
                else cutlass.Int32(0)
            )
            cute.arch.cp_async_commit_group()
        
        if K_TILE_COUNT < PIPE:
            for rest_v in range(tApA.shape[0]):
                for cpy_m in range(tApA.shape[1]):
                    tApA[rest_v, cpy_m, 0] = cutlass.Boolean(0)
            for rest_v in range(tBpB.shape[0]):
                for cpy_n in range(tBpB.shape[1]):
                    tBpB[rest_v, cpy_n, 0] = cutlass.Boolean(0)
        
        # tCsA: (MMA, MMA_M, MMA_K, PIPE), tCsB: (MMA, MMA_N, MMA_K, PIPE)
        # tCgC: (MMA, MMA_M, MMA_N)
        # tCrA: (MMA, MMA_M, MMA_K, NUM_MMA), tCrB: (MMA, MMA_N, MMA_K, NUM_MMA)
        # tCrC: (MMA, MMA_M, MMA_N)
        # MMA: 1
        # MMA_M * TC_LAYOUT_M = BLK_M
        # MMA_N * TC_LAYOUT_N = BLK_N
        # MMA_K = BLK_K
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = thr_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)
        print(f"[tiled sgemm]tCrA: {tCrA.type}, tCrB: {tCrB.type}, tCrC: {tCrC.type}")
        print(f"[tiled sgemm]tCsA: {tCsA.type}, tCsB: {tCsB.type}, tCgC: {tCgC.type}")

        smem_pipe_read = cute.Int32(0)
        smem_pipe_write = cute.Int32(PIPE - 1)
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        tCsB_p = tCsB[None, None, None, smem_pipe_read]
        NUM_MMA = cute.size(tCrA, mode=[2])
        if NUM_MMA > 1:
            # barrier: wait gmem -> smem
            cute.arch.cp_async_wait_group(PIPE - 2)
            cute.arch.barrier()

            # smem -> regs
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])

        for _ in range(K_TILE_COUNT):
            for k_mma in range(NUM_MMA, unroll_full=True):
                if k_mma == NUM_MMA - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(PIPE - 2)
                    cute.arch.barrier()
                    
                # 1. Load A, B from shared memory to registers for k_block + 1
                k_mma_next = (k_mma + 1) % NUM_MMA  # static
                cute.autovec_copy(
                    tCsA_p[None, None, k_mma_next],
                    tCrA[None, None, k_mma_next],
                )
                cute.autovec_copy(
                    tCsB_p[None, None, k_mma_next],
                    tCrB[None, None, k_mma_next],
                )

                if k_mma == 0:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, gmem_pipe_read],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_mma],
                    tCrB[None, None, k_mma],
                    tCrC,
                )
                if k_mma == 0:
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, None, gmem_pipe_read],
                        tBsB[None, None, None, smem_pipe_write],
                        pred=tBpB,
                    )
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == PIPE:
                        smem_pipe_read = cute.Int32(0)
                    gmem_pipe_read = (
                        gmem_pipe_read + 1
                        if gmem_pipe_read + 1 < K_TILE_COUNT
                        else cute.Int32(1)
                    )

        # Epilogue        
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        cC = cute.make_identity_tensor(gC.shape)
        tCpC = thr_mma.partition_C(cC)
        predC = cute.make_fragment(tCrC.layout, cutlass.Boolean)
        residue_m = mC.shape[0] - cutlass.Int32(self.BLK_M) * bidx
        residue_n = mC.shape[1] - cutlass.Int32(self.BLK_N) * bidy
        for i in range(cute.size(tCrC.shape)):
            predC[i] = cute.elem_less(tCpC[i], (residue_m, residue_n))
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        cute.copy(atom, tCrC, tCgC, pred=predC)
        return


def test_sgemm():
    device = "cuda"
    M, N, K = 2560, 5120, 4096
    GFLOPS = 2 * M * N * K * 1e-9

    torch.random.manual_seed(42)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    def generate_tensors():
        A = torch.empty(K, M, device=device, dtype=torch.float32).random_(-5, 5).permute(1, 0)
        B = torch.empty(K, N, device=device, dtype=torch.float32).random_(-5, 5).permute(1, 0)
        C = torch.empty(M, N, device=device, dtype=torch.float32)

        A_tensor = from_dlpack(A, assumed_align=16) # .mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(mode=0, divisibility=M)
        B_tensor = from_dlpack(B, assumed_align=16).mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(mode=0, divisibility=N)
        C_tensor = from_dlpack(C, assumed_align=16).mark_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(mode=1, divisibility=N)
        return testing.JitArguments(
            A_tensor, B_tensor, C_tensor, current_stream
        )

    A = torch.empty(K, M, dtype=torch.float32).random_(-5, 5).permute(1, 0).cuda()
    B = torch.empty(K, N, dtype=torch.float32).random_(-5, 5).permute(1, 0).cuda()
    A_tensor = from_dlpack(A, assumed_align=16) # .mark_layout_dynamic().mark_compact_shape_dynamic(mode=0, divisibility=M)
    B_tensor = from_dlpack(B, assumed_align=16) # .mark_layout_dynamic().mark_compact_shape_dynamic(mode=0, divisibility=N)
    C_layout_demo = torch.empty(M, N, dtype=torch.float32).cuda()
    C_tiled_demo = torch.empty(M, N, dtype=torch.float32).cuda()
    C_ref = torch.matmul(A, B.t())
    print(f"C({C_layout_demo.shape}, {C_layout_demo.dtype}) = A({A.shape}, {A.dtype}) * B({B.shape}, {B.dtype})")

    C_layout_demo_tensor = from_dlpack(C_layout_demo, assumed_align=16).mark_layout_dynamic().mark_compact_shape_dynamic(mode=1, divisibility=N)
    C_tiled_demo_tensor = from_dlpack(C_tiled_demo, assumed_align=16) # .mark_layout_dynamic().mark_compact_shape_dynamic(mode=1, divisibility=N)
    A_tensor.element_type = cutlass.Float32
    B_tensor.element_type = cutlass.Float32
    C_layout_demo_tensor.element_type = cutlass.Float32
    C_tiled_demo_tensor.element_type = cutlass.Float32

    sgemm_layout_demo = SGEMM_layout_demo()
    compile_key = (M, N, K, 1)
    if compile_key not in test_sgemm.compile_cache:
        test_sgemm.compile_cache[compile_key] = cute.compile(sgemm_layout_demo, A_tensor, B_tensor, C_layout_demo_tensor, current_stream)
    test_sgemm.compile_cache[compile_key](A_tensor, B_tensor, C_layout_demo_tensor, current_stream)

    torch.cuda.synchronize()
    if torch.equal(C_layout_demo, C_ref):
        print("layout demo success")
    else:
        print(f"layout demo failed: {C_layout_demo=}, {C_layout_demo.shape=}\n{C_ref=}, {C_ref.shape=}")
    avg_time_us = testing.benchmark(
        test_sgemm.compile_cache[compile_key],
        workspace_generator=generate_tensors,
        workspace_count=1,
        stream=current_stream,
        warmup_iterations=20,
        iterations=100,
    )
    print(f"layout demo kernel execution time: {avg_time_us / 1e3:.4f} ms, {GFLOPS / avg_time_us * 1e3:.4f} GFLOPS")

    sgemm_tiled_demo = SGEMM_tiled_demo()
    compile_key = (M, N, K, 2)
    if compile_key not in test_sgemm.compile_cache:
        test_sgemm.compile_cache[compile_key] = cute.compile(sgemm_tiled_demo, A_tensor, B_tensor, C_tiled_demo_tensor, current_stream)
    test_sgemm.compile_cache[compile_key](A_tensor, B_tensor, C_tiled_demo_tensor, current_stream)
    torch.cuda.synchronize()
    
    if torch.equal(C_tiled_demo, C_ref):
        print("tiled demo success")
    else:
        print(f"tiled demo failed: {C_tiled_demo=}, {C_tiled_demo.shape=}\n{C_ref=}, {C_ref.shape=}")
    
    avg_time_ms = testing.benchmark(
        test_sgemm.compile_cache[compile_key],
        workspace_generator=generate_tensors,
        workspace_count=1,
        stream=current_stream,
        warmup_iterations=20,
        iterations=100,
    )
    print(f"tiled demo kernel execution time: {avg_time_ms:.4f} ms, {GFLOPS / avg_time_ms * 1e3:.4f} GFLOPS")

test_sgemm.compile_cache = {}

if __name__ == "__main__":
    test_sgemm()

