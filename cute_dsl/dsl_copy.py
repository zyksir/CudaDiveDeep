import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.jit
def idx2crd(idx, shape):
    id = cute.make_identity_layout(shape)
    crd = id(idx)
    # cute.printf("{} -> {}", idx, crd)
    return crd

mblock_size = 64
nblock_size = 64
num_threads_per_block = 128 # 4 warps

@cute.kernel
def copy_kernel(
    mA : cute.Tensor,
    mB : cute.Tensor,
    tiled_copy : cute.TiledCopy,
):
    tidx = cute.arch.thread_idx()[0]
    bidx, bidy, _ = cute.arch.block_idx()

    tiler_mn = tiled_copy.tiler_mn
    subtile_mn = (tiler_mn[0].shape, tiler_mn[1].shape)

    tv_layout = tiled_copy.layout_tv_tiled

    print(mA)
    print(mB)
    # if tidx == 0 and bidx == 0 and bidy == 0:
    #     cute.printf("mA = {}", mA)
    #     cute.printf("mB = {}", mB)

    # gA = cute.local_tile(mA, (mblock_size, nblock_size), (None, None))
    # gA = gA[None, None, bidx, bidy]
    tile_shape = (mblock_size, nblock_size)
    gA = cute.local_tile(mA, tile_shape, (bidx, bidy))
    print(gA)

    print("tv_layout = ", tv_layout)
    gA_subtiled = cute.flat_divide(gA, subtile_mn)
    print("Flat divide by shape = ", gA_subtiled)
    for k in cutlass.range_constexpr(2, len(gA_subtiled.shape)):
        tv_layout = cute.append(tv_layout, gA_subtiled.layout[k])
    print("tv layout append: ", tv_layout)
    gA_tv = cute.composition(gA_subtiled, tv_layout)
    print("gA_tv = ", gA_tv)

    cA = cute.make_identity_tensor(tile_shape)

    thr_copy = tiled_copy.get_slice(tidx)

    # tAgA = thr_copy.partition_S(gA) # (V, M, N)
    tAgA = gA_tv[tidx, None, None, None]
    tArA = cute.make_fragment_like(tAgA) # (V, M, N)
    tAcA = thr_copy.partition_S(cA)

    print("tAgA = ", tAgA)
    print("tArA = ", tArA)
    print("tAcA = ", tAcA)

    # if tidx == 0 and bidx == 0 and bidy == 0:
    #     cute.printf("tAgA = {}", tAgA)
    #     cute.printf("tArA = {}", tArA)

    cute.copy(tiled_copy, tAgA, tArA)

    # if tidx == 0 and bidx == 0 and bidy == 0:
    #     cute.printf("tAgA = {}", tAgA)
    #     cute.printf("tArA = {}", tArA)

    if tidx == 0 and bidx == 0 and bidy == 0:
        for i in cutlass.range_constexpr(cute.size(tArA)):
            frg_coord = idx2crd(i, tArA.shape)
            cute.printf("flat coord = {}, natural coord = {}, mn_coord = {}, val = {}",
                        i, frg_coord, tAcA[i], tArA[i])
        
        for j in cutlass.range_constexpr(cute.size(tArA.shape[2])):
            for i in cutlass.range_constexpr(cute.size(tArA.shape[1])):
                for v in cutlass.range_constexpr(cute.size(tArA.shape[0])):
                    rank_coord = (v, i, j)
                    frg_coord = idx2crd(rank_coord, tArA.shape)
                    cute.printf("rank coord = {}, natural coord = {}, mn_coord = {}, val = {}",
                                rank_coord, frg_coord, tAcA[rank_coord], tArA[rank_coord])

    gB = cute.local_tile(mB, (mblock_size, nblock_size), (bidx, bidy))
    tBgB = thr_copy.partition_D(gB)

    cute.copy(tiled_copy, tArA, tBgB)

    # if tidx == 0 and bidx == 0 and bidy == 0:
    #     cute.printf("tBgB = {}", tBgB)
    #     cute.printf("tArA = {}", tArA)

@cute.jit
def copy_host(
    mA : cute.Tensor,
    mB : cute.Tensor,
):
    m, n = mA.shape

    num_mblocks = cute.ceil_div(m, mblock_size)
    num_nblocks = cute.ceil_div(n, nblock_size)

    num_bits_per_copy = 128
    num_vals_per_thread = num_bits_per_copy // mA.element_type.width # 8
    val_layout = cute.make_ordered_layout((1, num_vals_per_thread), order=(1,0))
    print("Value Layout: ", val_layout)
    
    num_threads_per_row = nblock_size // num_vals_per_thread
    num_threads_per_col = num_threads_per_block // num_threads_per_row
    thr_layout = cute.make_ordered_layout((num_threads_per_col, num_threads_per_row), order=(1,0))

    # Warp-interleaved thread layout
    # Warp0 Warp1
    # Warp2 Warp3
    # thr_layout = cute.make_ordered_layout(((4, num_threads_per_col // 4), (8, num_threads_per_row // 8)), order=((1,3),(0,2)))
    print("Thread Layout: ", thr_layout)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mA.element_type, num_bits_per_copy=num_bits_per_copy)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    print("Tiled Copy: ", tiled_copy)

    kernel = copy_kernel(mA, mB, tiled_copy)
    kernel.launch(grid=(num_mblocks, num_nblocks, 1),
                  block=(num_threads_per_block, 1, 1))
    
    
# M = 512
# N = 512
M = mblock_size
N = nblock_size
# a = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
a = torch.arange(M, device='cuda').unsqueeze(1) * N + torch.arange(N, device='cuda')
a = a.to(torch.int32)
b = torch.zeros(M, N, device="cuda", dtype=torch.int32)

a_ = from_dlpack(a, assumed_align=128)
b_ = from_dlpack(b, assumed_align=128)
# a_ = a_.mark_layout_dynamic()
# b_ = b_.mark_layout_dynamic()
# a_ = a_.mark_compact_shape_dynamic(mode=1, divisibility=8)
# b_ = b_.mark_compact_shape_dynamic(mode=1, divisibility=8)

copy_host_compiled = cute.compile(copy_host, a_, b_)

copy_host_compiled(a_, b_)

assert torch.equal(a, b)