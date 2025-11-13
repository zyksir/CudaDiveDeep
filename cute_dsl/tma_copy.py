"""
This is a tutorial code for TMA copy
In this tutorial, we assume that the input tensor A has shape (M, N, K)
- IdentityTMACopy is identical to A.copy_(B)
- TestTMACopy is identical to B = A.permute(1, 0).contiguous()
"""

from typing import Tuple

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync, warpgroup
import cuda.bindings.driver as cuda
from utils import run_benchmark

def make_smem_layout(dtype, tile_shape: Tuple[int, int, int], tile_order: Tuple[int, int, int] = (2, 0, 1)):
    print(f"[make_smem_layout]tile_shape: {tile_shape}, tile_order: {tile_order}")
    atom = warpgroup.make_smem_layout_atom(
        # swizzle: each K-major 128B addresses in SMEM will be re-arranged to avoid bank conflicts
        warpgroup.SmemLayoutAtomKind.K_SW128,
        dtype,
    )
    # Tile to CTA tile shape (MxN, K)
    return cute.tile_to_shape(atom, tile_shape, tile_order)

class IdentityTMACopy:
    """
    This is a tutorial code for TMA copy to do A.copy_(B).
    """
    def __init__(self, cta_tiler: Tuple[int, int] = (128, 128)):
        self.cta_tiler = cta_tiler
        self.blk_n, self.blk_k = self.cta_tiler
        self.blk_m = 1

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        stream: cuda.CUstream,
    ):  
        print(f"[IdentityTMACopy] {mA.type=}, {mB.type=}")
        self.dtype = mA.element_type
        assert mA.element_type == mB.element_type, f"mA and mB must have the same element type, got {mA.element_type=} and {mB.element_type=}"
        cta_tiler = (self.blk_n, self.blk_k)
        smem_layout = make_smem_layout(self.dtype, cta_tiler, tile_order=(0, 1)) # (BLK_N, BLK_K), K_MAJOR
        print(f"[IdentityTMACopy] {smem_layout.type=}, {self.cta_tiler=}")
        op_g2s = cpasync.CopyBulkTensorTileG2SOp()
        op_s2g = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_load, tma_tensor_A = cpasync.make_tiled_tma_atom(op_g2s, mA, smem_layout, cta_tiler)
        tma_atom_store, tma_tensor_B = cpasync.make_tiled_tma_atom(op_s2g, mB, smem_layout, cta_tiler)
        print(f"[IdentityTMACopy] {tma_atom_load=}, {tma_tensor_A.type=}")
        print(f"[IdentityTMACopy] {tma_atom_store=}, {tma_tensor_B.type=}")

        buffer_align_bytes = 128
        @cute.struct
        class SharedStorage:
            tile: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout)], buffer_align_bytes
            ]
            # mbarrier should be 8 bytes
            mbar: cute.struct.MemRange[cutlass.Int64, 1]
        self.shared_storage = SharedStorage
        self.tma_copy_bytes = cute.size_in_bytes(self.dtype, cute.select(smem_layout, mode=[0,1])) # BLK_N * BLK_K * dtype
        print(f"[IdentityTMACopy] {self.dtype=}, {self.tma_copy_bytes=}")

        MxN = cute.size(mA.shape[0])
        K = cute.size(mA.shape[1])
        grid = (cute.ceil_div(MxN, self.blk_n), cute.ceil_div(K, self.blk_k), 1)
        block = (128, 1, 1)
        print(f"[IdentityTMACopy] {grid=}, {block=}")
        self.kernel(
            tma_tensor_A,
            tma_tensor_B,
            tma_atom_load,
            tma_atom_store,
            smem_layout,
        ).launch(
            grid=grid,
            block=block,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream
        )

        
    @cute.kernel
    def kernel(
        self,
        tma_tensor_A: cute.Tensor, # logical view consumed by the TMA unit
        tma_tensor_B: cute.Tensor,
        tma_atom_load: cute.CopyAtom, # G->S atom
        tma_atom_store: cute.CopyAtom, # S->G atom
        smem_layout: cute.ComposedLayout,
    ):
        bidx, bidy, _ = cute.arch.block_idx()

        # Elect one warp to prefetch the TMA descriptor (recommended)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_load)
            cpasync.prefetch_descriptor(tma_atom_store)
            cute.arch.sync_threads()
        
        # Shared allocations
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mbar = storage.mbar.data_ptr()
        sTile = storage.tile.get_tensor(smem_layout.outer, swizzle=smem_layout.inner)
        print(f"[IdentityTMACopy] {sTile.type=}")

        # Init mbarrier once per CTA (elect one thread)
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar, cnt=1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # Compute the GMEM tile this CTA handles
        # Use the TMA tensors and local_tile to pick the tile box
        # gA_tma: (BLK_N, BLK_K): COL_MAJOR
        cta_tiler = (self.blk_n, self.blk_k)
        gA_tile = cute.local_tile(tma_tensor_A, tiler=cta_tiler, coord=(bidx, bidy))
        gB_tile = cute.local_tile(tma_tensor_B, tiler=cta_tiler, coord=(bidx, bidy))

        # if we assume N % BLK_N == 0, then
        #     idx_m = BLK_N * bidx // N
        #     idx_n = BLK_N * bidx % N
        M = 128
        N = 5120
        gB_debug_1 = cute.local_tile(tma_tensor_B, tiler=(M, self.blk_k), coord=(None, bidy)) # [M, BLK_K, N]
        idx_m = bidx // N
        idx_n = bidx % N
        gB_debug_2 = gB_debug_1[idx_m, None, None] # [BLK_K, N]
        gB_debug_3 = cute.local_tile(gB_debug_2, tiler=(self.blk_k, self.blk_n), coord=(None, idx_n % self.blk_n)) # [BLK_K]
        gB_debug = gB_debug_3[None, None, 0]
        print(f"[IdentityTMACopy] {gA_tile.type=}, {gB_tile.type=}")
        print(f"[IdentityTMACopy] {gB_debug.type=}, {gB_debug_1.type=}, {gB_debug_2.type=}, {gB_debug_3.type=}")

        # Group modes to comply with TMA partitioning contract
        sTile_g = cute.group_modes(sTile, 0, 2) # (BLK_N, BLK_K) -> ((BLK_N, BLK_K))
        gA_g = cute.group_modes(gA_tile, 0, 2)
        gB_g = cute.group_modes(gB_tile, 0, 2)
        gB_g_debug = cute.group_modes(gB_debug, 0, 2)
        print(f"[IdentityTMACopy] {sTile_g=}, {gA_g.type=}, {gB_g.type=}, {gB_g_debug.type=}")

        # Partition tensors to produce TMA-ready views
        sA_tma, gA_tma = cpasync.tma_partition(atom=tma_atom_load, cta_coord=0, cta_layout=cute.make_layout(1),
            smem_tensor=sTile_g, gmem_tensor=gA_g)
        sB_tma, gB_tma = cpasync.tma_partition(atom=tma_atom_store, cta_coord=0, cta_layout=cute.make_layout(1),
            smem_tensor=sTile_g, gmem_tensor=gB_g)
        sB_tma_debug, gB_tma_debug = cpasync.tma_partition(atom=tma_atom_store, cta_coord=0, cta_layout=cute.make_layout(1),
            smem_tensor=sTile_g, gmem_tensor=gB_g_debug)
        print(f"[IdentityTMACopy] {sA_tma.type=}, {gA_tma.type=}, {sB_tma.type=}, {gB_tma.type=}")

        if warp_idx == 0:
            # Producer: launch G2S copy (elect one thread)
            # elect_one: Elects one thread within a warp.
            # actually we only need one thread per CTA to arrive on the mbarrier
            with cute.arch.elect_one():
                # Pre assert that the mbarrier will be arrived when an expected specified number of transaction bytes done
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, self.tma_copy_bytes)
            cute.copy(tma_atom_load, gA_tma, sA_tma, tma_bar_ptr=mbar)

            # Consumer: wait for phase 0 load to finish, then store to GMEM B via S2G TMA
            cute.arch.mbarrier_wait(mbar, 0)
            # cute.copy(tma_atom_store, sB_tma, gB_tma)
            cute.copy(tma_atom_store, sB_tma_debug, gB_tma_debug)

def test_tma_copy():
    device = "cuda"
    torch.random.manual_seed(42)

    M, N, K = 128, 5120, 5120

    inp = torch.randn(M, N, K, device=device).to(torch.float8_e4m3fn)
    inp_uint8 = inp.view(dtype=torch.uint8)    
    print(f"Input tensor shape: {inp.shape}, dtype: {inp.dtype}")

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    print("-" * 100)

    id_tma_copy = IdentityTMACopy()
    id_compile_key = (M, N, K, 2)
    out2 = torch.zeros_like(inp)
    out2_uint8 = out2.view(dtype=torch.uint8)
    input_tensor = from_dlpack(inp_uint8.view(M*N, K), assumed_align=16)
    output_tensor = from_dlpack(out2_uint8.view(M*N, K), assumed_align=16)
    input_tensor.element_type = cutlass.Float8E4M3FN
    output_tensor.element_type = cutlass.Float8E4M3FN
    if id_compile_key not in test_tma_copy.compile_cache:
        test_tma_copy.compile_cache[id_compile_key] = cute.compile(id_tma_copy, input_tensor, output_tensor, stream)
    test_tma_copy.compile_cache[id_compile_key](input_tensor, output_tensor, stream)
    torch.cuda.synchronize()
    if torch.equal(inp, out2):
        print("[IdentityTMACopy] success")
    else:
        print(f"[IdentityTMACopy] failed, expected: {inp}, actual: {out2}")
    
    # from IPython import embed; embed()

    torch.cuda.synchronize()
    base_time = run_benchmark(10, 10, lambda: inp.reshape(M, N, K).permute(1, 0, 2).contiguous())
    print(f"Base time: {base_time} milliseconds")
    tma_time = run_benchmark(10, 10, test_tma_copy.compile_cache[id_compile_key], input_tensor, output_tensor, stream)
    print(f"ID TMA COPY time: {tma_time} milliseconds")

test_tma_copy.compile_cache = {}

if __name__ == "__main__":
    test_tma_copy()