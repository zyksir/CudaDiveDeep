import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import cutlass.cute.testing as testing
from typing import Tuple, Type, Callable
# from types import SimpleNamespace
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
import cutlass.utils.hopper_helpers as sm90_utils_basic

class Ampere_MatVec:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (128, 1, 128),
        num_threads: int = 128,
        num_stages: int = 2,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    ):

        self.cta_tiler = cta_tiler
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.BLK_M, self.BLK_N, self.BLK_K = self.cta_tiler
        assert self.BLK_N == 1
        self.acc_dtype = acc_dtype
    
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor, # ((M, K); (K, 1))
        mB: cute.Tensor, # ((1, K): (K, 1))
        mC: cute.Tensor, # ((M, 1): (1, M))
        stream: cuda.CUstream,
    ):
        M, K = cute.size(mA.shape[0]), cute.size(mA.shape[1])
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        print("[ampere] mA: {}, mB: {}, mC: {}".format(mA.type, mB.type, mC.type))

        num_vectorized = 1
        atom_async_copy_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width * num_vectorized,
        )
        CPY_K = self.BLK_K // num_vectorized
        tA = cute.make_layout((self.num_threads // CPY_K, CPY_K), stride=(CPY_K, 1))
        vA = cute.make_layout((1, num_vectorized))

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
        print("[tiled matvec] tiled_copy_A: {}\ntiled_copy_B: {}".format(tiled_copy_A, tiled_copy_B))

        atoms_layout = cute.make_layout(
            (self.num_threads, 1, 1), stride=(1, 0, 0)
        )
        op = cute.nvgpu.MmaUniversalOp(self.acc_dtype)
        tiled_mma = cute.make_tiled_mma(op, atoms_layout)
        print("[tiled matvec] tiled_mma: {}".format(tiled_mma))
        
        grid_dim = (cute.ceil_div(M, self.BLK_M), 1, 1)
        block_dim = (self.num_threads, 1, 1)
        sA_layout = cute.make_layout((self.BLK_M, self.BLK_K, self.num_stages))
        sB_layout = cute.make_layout((self.BLK_N, self.BLK_K, self.num_stages))
        assert cute.size(tiled_copy_A) == self.num_threads
        assert cute.size(tiled_copy_B) == self.num_threads
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
    
    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor, # ((M, K); (1, M))
        mB: cute.Tensor, # (K: 1)
        mC: cute.Tensor, # (M: 1)
        sA_layout: cute.Layout, # (BLK_M, BLK_K, NUM_STAGES)
        sB_layout: cute.Layout, # (BLK_K)
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tiler_coord = (bidx, 0, None)

        gA = cute.local_tile(mA, self.cta_tiler, tiler_coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, self.cta_tiler, tiler_coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.cta_tiler, tiler_coord, proj=(1, 1, None))
        # gA: (BLK_M, BLK_K, NUM_K_TILES), gB: (1, BLK_K, NUM_K_TILES), gC: (BLK_M, 1)
        print("[ampere] gA: {}, gB: {}, gC: {}".format(gA.type, gB.type, gC.type))

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
        # sA: (BLK_M, BLK_K, NUM_STAGES), sB: (1, BLK_K, NUM_STAGES)
        print("[ampere] sA: {}, sB: {}".format(sA.type, sB.type))

        NUM_K_BLOCKS = cute.size(gA, mode=[2])
        NUM_G2S_STAGES = cute.size(sA, mode=[2])

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)
        # tAgA: (CPY, CPY_M, CPY_K, K_TILE_COUNT), tBgB: (CPY, CPY_N, CPY_K, K_TILE_COUNT)
        print("[ampere] tAgA: {}, tBgB: {}".format(tAgA.type, tBgB.type))
        # tAsA: (CPY, CPY_M, CPY_K, NUM_STAGES), tBsB: (CPY, CPY_N, CPY_K, NUM_STAGES)
        print("[ampere] tAsA: {}, tBsB: {}".format(tAsA.type, tBsB.type))

        gmem_pipe_read = cute.Int32(0)
        for k_block_idx in range(NUM_G2S_STAGES-1):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, gmem_pipe_read],
                tAsA[None, None, None, k_block_idx],
            )
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, gmem_pipe_read],
                tBsB[None, None, None, k_block_idx],
            )
            cute.arch.cp_async_commit_group()
            gmem_pipe_read = (
                gmem_pipe_read + 1
                if gmem_pipe_read + 1 < NUM_K_BLOCKS
                else cutlass.Int32(0)
            )
        
        smem_pipe_read = cute.Int32(0)
        smem_pipe_write = cute.Int32(NUM_G2S_STAGES - 1)

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrC = thr_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)
        # tCsA: (MMA_M, MMA_K, MMA, PIPE), tCsB: (MMA_N, MMA_K, MMA, PIPE), tCgC: (MMA_M, MMA_N, 1), tCrC: (MMA_M, MMA_N, 1)
        print("[ampere] tCsA: {}, tCsB: {}, tCgC: {}, tCrC: {}".format(tCsA.type, tCsB.type, tCgC.type, tCrC.type))

        # tCsA_p = tCsA[None, None, None, smem_pipe_read]
        # tCsB_p = tCsB[None, None, None, smem_pipe_read]
        # tCrA = thr_mma.make_fragment_A(tCsA_p)
        # tCrB = thr_mma.make_fragment_B(tCsB_p)
        # NUM_MMA = cute.size(tCrA, mode=[2])
        # if NUM_MMA > 1:
        #     cute.arch.cp_async_wait_group(NUM_G2S_STAGES - 2)
        #     cute.arch.barrier()
        #     cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
        #     cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])

        for _ in range(NUM_K_BLOCKS):
            tCsA_p = tCsA[None, None, None, smem_pipe_read]
            tCsB_p = tCsB[None, None, None, smem_pipe_read]
            cute.arch.cp_async_wait_group(NUM_G2S_STAGES - 2)
            cute.arch.barrier()
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, gmem_pipe_read],
                tAsA[None, None, None, smem_pipe_write],
            )
            for m in cutlass.range(cute.size(tCsA_p, mode=[0]), unroll_full=True):
                for k in cutlass.range(cute.size(tCsA_p, mode=[1]), unroll_full=True):
                    for mma_idx in cutlass.range(cute.size(tCsA_p, mode=[2]), unroll_full=True):
                        tCrC[0, m, 0] += tCsA_p[m, k, mma_idx] * tCsB_p[0, k, mma_idx]
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, gmem_pipe_read],
                tBsB[None, None, None, smem_pipe_write],
            )
            cute.arch.cp_async_commit_group()
            smem_pipe_write = smem_pipe_read
            smem_pipe_read = (
                smem_pipe_read + 1
                if smem_pipe_read + 1 < NUM_G2S_STAGES
                else cutlass.Int32(0)
            )
            gmem_pipe_read = (
                gmem_pipe_read + 1
                if gmem_pipe_read + 1 < NUM_K_BLOCKS
                else cutlass.Int32(0)
            )

            # for k_mma in range(NUM_MMA, unroll_full=True):
            #     if k_mma == NUM_MMA - 1:
            #         tCsA_p = tCsA[None, None, None, smem_pipe_read]
            #         tCsB_p = tCsB[None, None, None, smem_pipe_read]
            #         cute.arch.cp_async_wait_group(NUM_G2S_STAGES - 2)
            #         cute.arch.barrier()
                
            #     k_mma_next = (k_mma + 1) % NUM_MMA  # static
            #     cute.autovec_copy(
            #         tCsA_p[None, None, k_mma_next],
            #         tCrA[None, None, k_mma_next],
            #     )
            #     cute.autovec_copy(
            #         tCsB_p[None, None, k_mma_next],
            #         tCrB[None, None, k_mma_next],
            #     )
            #     if k_mma == 0:
            #         cute.copy(
            #             tiled_copy_A,
            #             tAgA[None, None, None, gmem_pipe_read],
            #             tAsA[None, None, None, smem_pipe_write],
            #         )
            #     cute.gemm(
            #         tiled_mma,
            #         tCrC,
            #         tCrA[None, None, k_mma],
            #         tCrB[None, None, k_mma],
            #         tCrC,
            #     )
            #     if k_mma == 0:
            #         cute.copy(
            #             tiled_copy_B,
            #             tBgB[None, None, None, gmem_pipe_read],
            #             tBsB[None, None, None, smem_pipe_write],
            #         )
            #         cute.arch.cp_async_commit_group()
            #         smem_pipe_write = smem_pipe_read
            #         smem_pipe_read = (
            #             smem_pipe_read + 1
            #             if smem_pipe_read + 1 < NUM_G2S_STAGES
            #             else cutlass.Int32(0)
            #         )
            #         gmem_pipe_read = (
            #             gmem_pipe_read + 1
            #             if gmem_pipe_read + 1 < NUM_K_BLOCKS
            #             else cutlass.Int32(0)
            #         )
        
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        cute.copy(atom, tCrC, tCgC)

# class Hopper_MatVec:
#     def __init__(
#         self,
#         cta_tiler: Tuple[int, int, int] = (128, 1, 128),
#         num_threads: int = 128,
#         num_stages: int = 1,
#         acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
#     ):
#         self.cta_tiler = cta_tiler
#         self.num_threads = num_threads
#         self.num_stages = num_stages
#         self.BLK_M, self.BLK_N, self.BLK_K = self.cta_tiler
#         assert self.BLK_N == 1
#         self.acc_dtype = acc_dtype
    
#     @cute.jit
#     def __call__(
#         self,
#         mA: cute.Tensor, # ((M, K); (K, 1))
#         mB: cute.Tensor, # ((1, K): (K, 1))
#         mC: cute.Tensor, # ((M, 1): (1, M))
#         stream: cuda.CUstream,
#     ):
#         print("[hopper] mA: {}, mB: {}, mC: {}".format(mA.type, mB.type, mC.type))
#         M, K = cute.size(mA.shape[0]), cute.size(mA.shape[1])
#         A_major_mode = cutlass.utils.LayoutEnum.ROW_MAJOR
#         dtype = mA.element_type
#         sA_layout_atom = warpgroup.make_smem_layout_atom(
#             sm90_utils_basic.get_smem_layout_atom(
#                 A_major_mode,
#                 mA.element_type,
#                 self.BLK_K,
#             ),
#             mA.element_type
#         )
#         sA_layout_staged = cute.tile_to_shape(
#             sA_layout_atom,
#             (self.BLK_M, self.BLK_K, self.num_stages),
#             (1, 0, 2),
#         )
#         # sB_layout_atom = warpgroup.make_smem_layout_atom(
#         #     sm90_utils_basic.get_smem_layout_atom(
#         #         A_major_mode,
#         #         mB.element_type,
#         #         self.BLK_N,
#         #     ),
#         #     mA.element_type
#         # )
#         # sB_layout_staged = cute.tile_to_shape(
#         #     sB_layout_atom,
#         #     (self.BLK_N, self.BLK_K, self.num_stages),
#         #     (1, 0, 2),
#         # )
#         sB_layout_staged = cute.make_ordered_layout((self.BLK_N, self.BLK_K, self.num_stages), order=(1, 0, 2))
#         print("[hopper] sA_layout_staged: {}, sB_layout_staged: {}".format(sA_layout_staged.type, sB_layout_staged.type))
#         sA_layout = cute.slice_(sA_layout_staged, (None, None, 0))
#         sB_layout = cute.slice_(sB_layout_staged, (None, None, 0))
#         print("[hopper] sA_layout: {}, sB_layout: {}".format(sA_layout.type, sB_layout.type))

#         op_g2s = cpasync.CopyBulkTensorTileG2SOp()
#         op_s2g = cpasync.CopyBulkTensorTileS2GOp()
#         tma_atom_load_A, tma_tensor_A = cpasync.make_tiled_tma_atom(op_g2s, mA, sA_layout, (self.BLK_M, self.BLK_K))
#         print("[hopper] tma_atom_load_A: {}, tma_tensor_A: {}".format(tma_atom_load_A, tma_tensor_A.type))
#         tma_atom_load_B, tma_tensor_B = cpasync.make_tiled_tma_atom(op_g2s, mB, sB_layout, (self.BLK_N, self.BLK_K))
#         print("[hopper] tma_atom_load_B: {}, tma_tensor_B: {}".format(tma_atom_load_B, tma_tensor_B.type))

#         @cute.struct
#         class SharedStorage:
#             mbar_ptr : cute.struct.MemRange[cutlass.Int64, 2]
#             sA : cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sA_layout)], 128]
#             sB : cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sB_layout)], 128]

#         atoms_layout = cute.make_layout(
#             (self.num_threads, 1, 1), stride=(1, 0, 0)
#         )
#         op = cute.nvgpu.MmaUniversalOp(self.acc_dtype)
#         tiled_mma = cute.make_tiled_mma(op, atoms_layout)
#         print("[hopper] tiled_mma: {}".format(tiled_mma))
        
#         grid_dim = (cute.ceil_div(M, self.BLK_M), 1, 1)
#         block_dim = (self.num_threads, 1, 1)
#         self.kernel(
#             tma_tensor_A,
#             tma_tensor_B,
#             mC,
#             sA_layout_staged,
#             sB_layout_staged,
#             SharedStorage,
#             tma_atom_load_A,
#             tma_atom_load_B,
#             tiled_mma,
#         ).launch(
#             grid=grid_dim,
#             block=block_dim,
#             stream=stream,
#             smem=SharedStorage.size_in_bytes(),
#         )
    
#     @cute.kernel
#     def kernel(
#         self,
#         mA: cute.Tensor, # ((M, K); (1, M))
#         mB: cute.Tensor, # (K: 1)
#         mC: cute.Tensor, # (M: 1)
#         sA_layout: cute.ComposedLayout, # (BLK_M, BLK_K, NUM_STAGES)
#         sB_layout: cute.Layout, # (BLK_N, BLK_K, NUM_STAGES)
#         SharedStorage: cutlass.Constexpr[Callable],
#         tma_atom_load_A: cute.CopyAtom,
#         tma_atom_load_B: cute.CopyAtom,
#         tiled_mma: cute.TiledMma,
#     ):
#         tidx = cute.arch.thread_idx()[0]
#         bidx = cute.arch.block_idx()[0]
#         tiler_coord = (bidx, 0, None)
#         warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

#         smem = cutlass.utils.SmemAllocator()
#         storage = smem.allocate(SharedStorage)
#         sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
#         sB = storage.sB.get_tensor(sB_layout)
#         print("[hopper] sA: {}, sB: {}".format(sA, sB))
#         mbar_ptr = storage.mbar_ptr.data_ptr()
#         if warp_idx == 0:
#             cpasync.prefetch_descriptor(tma_atom_load_A)
#             cpasync.prefetch_descriptor(tma_atom_load_B)

#         if warp_idx == 1:
#             cute.arch.mbarrier_init(mbar_ptr, 1)

#         gA = cute.local_tile(mA, self.cta_tiler, tiler_coord, proj=(1, None, 1))
#         gB = cute.local_tile(mB, self.cta_tiler, tiler_coord, proj=(None, 1, 1))
#         gC = cute.local_tile(mC, self.cta_tiler, tiler_coord, proj=(1, 1, None))
#         # gA: (BLK_M, BLK_K, NUM_K_TILES), gB: (1, BLK_K, NUM_K_TILES), gC: (BLK_M, 1)
#         print("[ampere] gA: {}, gB: {}, gC: {}".format(gA.type, gB.type, gC.type))

#         NUM_K_BLOCKS = cute.size(gA, mode=[2])
#         NUM_G2S_STAGES = cute.size(sA_layout, mode=[2])
#         print("[hopper] NUM_K_BLOCKS: {}, NUM_G2S_STAGES: {}".format(NUM_K_BLOCKS, NUM_G2S_STAGES))

#         tAsA, tAgA = cpasync.tma_partition(
#             tma_atom_load_A,
#             0, # cta_coord
#             cute.make_layout(1), # cta_layout
#             cute.group_modes(sA, 0, 2),
#             cute.group_modes(gA, 0, 2),
#         )
#         tBsB, tBgB = cpasync.tma_partition(
#             tma_atom_load_B,
#             0, # cta_coord
#             cute.make_layout(1), # cta_layout
#             cute.group_modes(sB, 0, 2),
#             cute.group_modes(gB, 0, 2),
#         )
#         print("[hopper] tAgA: {}, tBgB: {}".format(tAgA, tBgB))
#         print("[hopper] tAsA: {}, tBsB: {}".format(tAsA, tBsB))
#         tma_copy_A_bytes = cute.size_in_bytes(sA.element_type, cute.select(sA_layout, mode=[0, 1]))
#         tma_copy_B_bytes = cute.size_in_bytes(sB.element_type, cute.select(sB_layout, mode=[0, 1]))
#         print("[hopper] tma_copy_A_bytes: {}, tma_copy_B_bytes: {}".format(tma_copy_A_bytes, tma_copy_B_bytes))

#         gmem_pipe_read = cute.Int32(0)
#         thr_mma = tiled_mma.get_slice(tidx)
#         tCsA = thr_mma.partition_A(sA)
#         tCsB = thr_mma.partition_B(sB)
#         tCgC = thr_mma.partition_C(gC)
#         tCrC = thr_mma.make_fragment_C(tCgC)
#         tCrC.fill(0.0)
#         tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
#         # tCsA: (MMA_M, MMA_K, MMA, PIPE), tCsB: (MMA_N, MMA_K, MMA, PIPE), tCgC: (MMA_M, MMA_N, 1), tCrC: (MMA_M, MMA_N, 1)
#         print("[ampere] tCsA: {}, tCsB: {}, tCgC: {}, tCrC: {}".format(tCsA.type, tCsB.type, tCgC.type, tCrC.type))

#         phase = 0
#         for block_idx in range(NUM_K_BLOCKS):
#             if warp_idx == 0:
#                 if tidx == 0:
#                     cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tma_copy_A_bytes + tma_copy_B_bytes)
#                 cute.copy(tma_atom_load_A, tAgA[(None, gmem_pipe_read)], tAsA[(None, 0)], tma_bar_ptr=mbar_ptr)
#                 cute.copy(tma_atom_load_B, tBgB[(None, gmem_pipe_read)], tBsB[(None, 0)], tma_bar_ptr=mbar_ptr)

#             cute.arch.mbarrier_wait(mbar_ptr, phase=phase)
#             phase ^= 1
#             tCsA_p = tCsA[None, None, None, 0]
#             tCsB_p = tCsB[None, None, None, 0]
#             cute.autovec_copy(tCsA_p, tCrA)
#             for m in cutlass.range(cute.size(tCsA_p, mode=[0]), unroll_full=True):
#                 for k in cutlass.range(cute.size(tCsA_p, mode=[1]), unroll_full=True):
#                     for mma_idx in cutlass.range(cute.size(tCsA_p, mode=[2]), unroll_full=True):
#                         tCrC[0, m, 0] += tCrA[m, k, mma_idx] * tCsB_p[0, k, mma_idx]
#             gmem_pipe_read = (
#                 gmem_pipe_read + 1
#                 if gmem_pipe_read + 1 < NUM_K_BLOCKS
#                 else cutlass.Int32(0)
#             )
#         atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
#         cute.copy(atom, tCrC, tCgC)

class Hopper_MatVec:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (128, 1, 128),
        num_threads: int = 128,
        num_stages: int = 1,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    ):
        self.cta_tiler = cta_tiler
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.BLK_M, self.BLK_N, self.BLK_K = self.cta_tiler
        assert self.BLK_N == 1
        self.acc_dtype = acc_dtype
    
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor, # ((M, K); (K, 1))
        mB: cute.Tensor, # ((1, K): (K, 1))
        mC: cute.Tensor, # ((1, M): (M, 1))
        stream: cuda.CUstream,
    ):
        print("[hopper] mA: {}, mB: {}, mC: {}".format(mA.type, mB.type, mC.type))
        M, K = cute.size(mA.shape[0]), cute.size(mA.shape[1])
        A_major_mode = cutlass.utils.LayoutEnum.ROW_MAJOR
        dtype = mA.element_type
        sA_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                A_major_mode,
                mA.element_type,
                self.BLK_K,
            ),
            mA.element_type
        )
        sA_layout_staged = cute.tile_to_shape(
            sA_layout_atom,
            (self.BLK_M, self.BLK_K, self.num_stages),
            (1, 0, 2),
        )
        sB_layout_staged = cute.make_ordered_layout((self.BLK_N, self.BLK_K, self.num_stages), order=(1, 0, 2))
        sC_layout = cute.make_ordered_layout((self.BLK_N, self.BLK_M), order=(1, 0))
        print("[hopper] sA_layout_staged: {}, sB_layout_staged: {}, sC_layout: {}".format(sA_layout_staged.type, sB_layout_staged.type, sC_layout.type))
        sA_layout = cute.slice_(sA_layout_staged, (None, None, 0))
        sB_layout = cute.slice_(sB_layout_staged, (None, None, 0))
        print("[hopper] sA_layout: {}, sB_layout: {}, sC_layout: {}".format(sA_layout.type, sB_layout.type, sC_layout.type))

        op_g2s = cpasync.CopyBulkTensorTileG2SOp()
        op_s2g = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_load_A, tma_tensor_A = cpasync.make_tiled_tma_atom(op_g2s, mA, sA_layout, (self.BLK_M, self.BLK_K))
        print("[hopper] tma_atom_load_A: {}, tma_tensor_A: {}".format(tma_atom_load_A, tma_tensor_A.type))
        tma_atom_load_B, tma_tensor_B = cpasync.make_tiled_tma_atom(op_g2s, mB, sB_layout, (self.BLK_N, self.BLK_K))
        print("[hopper] tma_atom_load_B: {}, tma_tensor_B: {}".format(tma_atom_load_B, tma_tensor_B.type))
        # tma_atom_store_C, tma_tensor_C = cpasync.make_tiled_tma_atom(op_s2g, mC, sC_layout, (self.BLK_N, self.BLK_M))
        # print("[hopper] tma_atom_store_C: {}, tma_tensor_C: {}".format(tma_atom_store_C, tma_tensor_C.type))

        @cute.struct
        class SharedStorage:
            mbar_ptr : cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sA : cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sA_layout_staged)], 128]
            sB : cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sB_layout_staged)], 128]
            sC : cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sC_layout)], 128]

        atoms_layout = cute.make_layout((self.num_threads, 1, 1), stride=(1, 0, 0))
        op = cute.nvgpu.MmaUniversalOp(self.acc_dtype)
        tiled_mma = cute.make_tiled_mma(op, atoms_layout)
        print("[hopper] tiled_mma: {}".format(tiled_mma))
        
        grid_dim = (cute.ceil_div(M, self.BLK_M), 1, 1)
        block_dim = (self.num_threads, 1, 1)
        self.kernel(
            tma_tensor_A,
            tma_tensor_B,
            mC,
            # tma_tensor_C,
            sA_layout_staged,
            sB_layout_staged,
            sC_layout,
            SharedStorage,
            tma_atom_load_A,
            tma_atom_load_B,
            # tma_atom_store_C,
            tiled_mma,
        ).launch(
            grid=grid_dim,
            block=block_dim,
            stream=stream,
            smem=SharedStorage.size_in_bytes(),
        )
    
    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor, # ((M, K); (1, M))
        mB: cute.Tensor, # ((1, K): (K, 1))
        mC: cute.Tensor, # ((1, M): (M, 1))
        sA_layout: cute.ComposedLayout, # (BLK_M, BLK_K, NUM_STAGES)
        sB_layout: cute.Layout, # (BLK_N, BLK_K, NUM_STAGES)
        sC_layout: cute.Layout, # (BLK_M, BLK_N)
        SharedStorage: cutlass.Constexpr[Callable],
        tma_atom_load_A: cute.CopyAtom,
        tma_atom_load_B: cute.CopyAtom,
        # tma_atom_store_C: cute.CopyAtom,
        tiled_mma: cute.TiledMma,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        tiler_coord = (bidx, 0, None)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout)
        sC = storage.sC.get_tensor(sC_layout)
        print("[hopper] sA: {}, sB: {}, sC: {}".format(sA, sB, sC))
        mbar_ptr = storage.mbar_ptr.data_ptr()
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_load_A)
            cpasync.prefetch_descriptor(tma_atom_load_B)
            # cpasync.prefetch_descriptor(tma_atom_store_C)

        if warp_idx == 1:
            for i in cutlass.range_constexpr(self.num_stages):
                cute.arch.mbarrier_init(mbar_ptr + i, 1)

        gA = cute.local_tile(mA, self.cta_tiler, tiler_coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, self.cta_tiler, tiler_coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.cta_tiler, tiler_coord, proj=(1, 1, None))
        # gA: (BLK_M, BLK_K, NUM_K_TILES), gB: (1, BLK_K, NUM_K_TILES), gC: (BLK_M, BLK_N)
        print("[ampere] gA: {}, gB: {}, gC: {}".format(gA.type, gB.type, gC.type))

        NUM_K_BLOCKS = cute.size(gA, mode=[2])
        NUM_G2S_STAGES = cute.size(sA_layout, mode=[2])
        print("[hopper] NUM_K_BLOCKS: {}, NUM_G2S_STAGES: {}".format(NUM_K_BLOCKS, NUM_G2S_STAGES))

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_load_A,
            0, # cta_coord
            cute.make_layout(1), # cta_layout
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_load_B,
            0, # cta_coord
            cute.make_layout(1), # cta_layout
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )
        # tCsC_tma, tCgC_tma = cpasync.tma_partition(
        #     tma_atom_store_C,
        #     0, # cta_coord
        #     cute.make_layout(1), # cta_layout
        #     sC,
        #     gC,
        # )
        print("[hopper] tAgA: {}, tBgB: {}".format(tAgA, tBgB))
        print("[hopper] tAsA: {}, tBsB: {}".format(tAsA, tBsB))
        tma_copy_A_bytes = cute.size_in_bytes(sA.element_type, cute.select(sA_layout, mode=[0, 1]))
        tma_copy_B_bytes = cute.size_in_bytes(sB.element_type, cute.select(sB_layout, mode=[0, 1]))

        gmem_pipe_read = cute.Int32(0)
        smem_pipe_read = cute.Int32(0)
        smem_pipe_write = cute.Int32(0)
        for k_block_idx in range(NUM_G2S_STAGES-1):
            if warp_idx == 0:
                if tidx == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + smem_pipe_write, tma_copy_A_bytes + tma_copy_B_bytes)
                cute.copy(tma_atom_load_A, tAgA[(None, gmem_pipe_read)], tAsA[(None, smem_pipe_write)], tma_bar_ptr=mbar_ptr + smem_pipe_write)
                cute.copy(tma_atom_load_B, tBgB[(None, gmem_pipe_read)], tBsB[(None, smem_pipe_write)], tma_bar_ptr=mbar_ptr + smem_pipe_write)
            gmem_pipe_read = (
                gmem_pipe_read + 1
                if gmem_pipe_read + 1 < NUM_K_BLOCKS
                else cutlass.Int32(0)
            )
            smem_pipe_write = (
                smem_pipe_write + 1
                if smem_pipe_write + 1 < NUM_G2S_STAGES
                else cutlass.Int32(0)
            )

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrC = thr_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)
        tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
        # tCsA: (MMA_M, MMA_K, MMA, PIPE), tCsB: (MMA_N, MMA_K, MMA, PIPE), tCgC: (MMA_M, MMA_N, 1), tCrC: (MMA_M, MMA_N, 1)
        print("[ampere] tCsA: {}, tCsB: {}, tCgC: {}, tCrC: {}".format(tCsA.type, tCsB.type, tCgC.type, tCrC.type))

        phase = 0
        for _ in range(NUM_K_BLOCKS):
            if warp_idx == 0:
                if tidx == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + smem_pipe_write, tma_copy_A_bytes + tma_copy_B_bytes)
                cute.copy(tma_atom_load_A, tAgA[(None, gmem_pipe_read)], tAsA[(None, smem_pipe_write)], tma_bar_ptr=mbar_ptr + smem_pipe_write)
                cute.copy(tma_atom_load_B, tBgB[(None, gmem_pipe_read)], tBsB[(None, smem_pipe_write)], tma_bar_ptr=mbar_ptr + smem_pipe_write)

            cute.arch.mbarrier_wait(mbar_ptr + smem_pipe_read, phase=phase)
            phase ^= 1
            tCsA_p = tCsA[None, None, None, smem_pipe_read]
            tCsB_p = tCsB[None, None, None, smem_pipe_read]
            cute.autovec_copy(tCsA_p, tCrA)
            for m in cutlass.range(cute.size(tCsA_p, mode=[0]), unroll_full=True):
                for k in cutlass.range(cute.size(tCsA_p, mode=[1]), unroll_full=True):
                    for mma_idx in cutlass.range(cute.size(tCsA_p, mode=[2]), unroll_full=True):
                        tCrC[0, m, 0] += tCrA[m, k, mma_idx] * tCsB_p[0, k, mma_idx]
            smem_pipe_write = smem_pipe_read
            smem_pipe_read = (
                smem_pipe_read + 1
                if smem_pipe_read + 1 < NUM_G2S_STAGES
                else cutlass.Int32(0)
            )
            gmem_pipe_read = (
                gmem_pipe_read + 1
                if gmem_pipe_read + 1 < NUM_K_BLOCKS
                else cutlass.Int32(0)
            )
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        cute.copy(atom, tCrC, tCgC)
        # cute.autovec_copy(tCrC, tCsC)
        # if warp_idx == 0:
        #     cute.copy(tma_atom_store_C, tCsC_tma, tCgC_tma)

def test_matvec():
    device = "cuda"
    M, K = 2048, 4096
    GFLOPS = 2 * M * K * 1e-9

    torch.random.manual_seed(42)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    def generate_tensors():
        A = torch.empty(M, K, device=device, dtype=torch.float32).random_(-5, 5)
        B = torch.empty(1, K, device=device, dtype=torch.float32).random_(-5, 5)
        C = torch.empty(M, 1, device=device, dtype=torch.float32)

        A_tensor = from_dlpack(A, assumed_align=16).mark_layout_dynamic(leading_dim=1) # .mark_compact_shape_dynamic(mode=0, divisibility=M)
        B_tensor = from_dlpack(B, assumed_align=16).mark_layout_dynamic(leading_dim=1) # .mark_compact_shape_dynamic(mode=0, divisibility=N)
        C_tensor = from_dlpack(C, assumed_align=16).mark_layout_dynamic(leading_dim=1) # .mark_compact_shape_dynamic(mode=1, divisibility=N)
        return testing.JitArguments(
            A_tensor, B_tensor, C_tensor, current_stream
        )

    A = torch.empty(M, K, dtype=torch.float32).random_(-5, 5).cuda()
    B = torch.empty(1, K, dtype=torch.float32).random_(-5, 5).cuda()
    A_tensor = from_dlpack(A, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    A_tensor.element_type = cutlass.Float32
    B_tensor = from_dlpack(B, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    B_tensor.element_type = cutlass.Float32
    C_ampere_matvec = torch.empty(M, 1, dtype=torch.float32).cuda()
    C_ampere_matvec_tensor = from_dlpack(C_ampere_matvec, assumed_align=16).mark_layout_dynamic(leading_dim=1) # .mark_compact_shape_dynamic(mode=1, divisibility=N)
    C_ampere_matvec_tensor.element_type = cutlass.Float32

    C_ref = torch.mv(A, B.view(-1)).view(M, 1)

    sgemm_tiled_demo = Hopper_MatVec()
    compile_key = (M, K, "ampere_matvec")
    if compile_key not in test_matvec.compile_cache:
        test_matvec.compile_cache[compile_key] = cute.compile(sgemm_tiled_demo, A_tensor, B_tensor, C_ampere_matvec_tensor, current_stream)
    test_matvec.compile_cache[compile_key](A_tensor, B_tensor, C_ampere_matvec_tensor, current_stream)
    torch.cuda.synchronize()

    if torch.equal(C_ampere_matvec.view(M, 1), C_ref):
        print("ampere matvec success")
    else:
        print(f"ampere matvec failed: {C_ampere_matvec=}, {C_ampere_matvec.shape=}\n{C_ref=}, {C_ref.shape=}")
    
    from utils import run_benchmark
    tma_time_ms = run_benchmark(10, 10, test_matvec.compile_cache[compile_key], A_tensor, B_tensor, C_ampere_matvec_tensor, current_stream)
    print(f"ampere matvec kernel execution time: {tma_time_ms:.4f} ms, {GFLOPS / tma_time_ms * 1e3:.4f} GFLOPS")

test_matvec.compile_cache = {}

if __name__ == "__main__":
    test_matvec()

