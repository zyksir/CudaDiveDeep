#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../lib/macros.cuh" 

namespace warptiling {
template<const int BM, const int BN, const int BK, const int rowStrideA, const int rowStrideB>
__device__ inline __attribute__((always_inline)) void loadBlockFromGlobalMemoryIntoSharedMemory(
  float* A, float* B, float* As, float* Bs, float* ldg_A_reg, int write_stage_idx, const int N, const int K,
  const int innerRowA, const int innerRowB, const int innerColA, const int innerColB) {
  #pragma unroll
  for(int i = 0; i < BM; i+=rowStrideA) {
    int ldg_index = i / rowStrideA * 4;
      FLOAT4(ldg_A_reg[ldg_index]) = 
        FLOAT4(Val(A, innerRowA + i, innerColA, K ));
      Val3D(As, 0, innerColA, innerRowA + i, BK, BM) = ldg_A_reg[ldg_index];
      Val3D(As, 0, innerColA+1, innerRowA + i, BK, BM) = ldg_A_reg[ldg_index+1];
      Val3D(As, 0, innerColA+2, innerRowA + i, BK, BM) = ldg_A_reg[ldg_index+2];
      Val3D(As, 0, innerColA+3, innerRowA + i, BK, BM) = ldg_A_reg[ldg_index+3];
  }
  #pragma unroll
  for(int i = 0; i < BK; i+=rowStrideB) {
    FLOAT4(Val3D(Bs, 0, innerRowB + i, innerColB, BK, BN)) = 
      FLOAT4(Val(B, innerRowB + i, innerColB, N));
  }
}

template<const int BM, const int BN, const int BK, const int rowStrideA, const int rowStrideB>
__device__ inline __attribute__((always_inline)) void loadBlockFromGlobalMemoryIntoRegister(
  float* A, float* B, float* ldg_A_reg, float* ldg_B_reg, const int tile_idx, const int N, const int K, 
  const int innerRowA, const int innerRowB, const int innerColA, const int innerColB) {
  #pragma unroll
  for(int i = 0; i < BM; i+=rowStrideA) {
    int ldg_index = i / rowStrideA * 4;
    FLOAT4(ldg_A_reg[ldg_index]) = 
      FLOAT4(Val(A, innerRowA + i, innerColA + tile_idx, K));
  }
  #pragma unroll
  for(int i = 0; i < BK; i+=rowStrideB) {
    int ldg_index = i / rowStrideB * 4;
    FLOAT4(ldg_B_reg[ldg_index]) = 
      FLOAT4(Val(B, tile_idx + innerRowB + i, innerColB, N));
  }
}

template<const int BM, const int BN, const int BK, const int rowStrideA, const int rowStrideB>
__device__ inline __attribute__((always_inline)) void loadBlockFromRegisterIntoSharedMemory(
  float* ldg_A_reg, float* ldg_B_reg, float* As, float* Bs, const int write_stage_idx,
  const int innerRowA, const int innerRowB, const int innerColA, const int innerColB) {
  #pragma unroll
  for(int i = 0; i < BM; i+=rowStrideA) {
    int ldg_index = i / rowStrideA * 4;
    Val3D(As, write_stage_idx, innerColA, innerRowA + i, BK, BM) = ldg_A_reg[ldg_index];
    Val3D(As, write_stage_idx, innerColA+1, innerRowA + i, BK, BM) = ldg_A_reg[ldg_index+1];
    Val3D(As, write_stage_idx, innerColA+2, innerRowA + i, BK, BM) = ldg_A_reg[ldg_index+2];
    Val3D(As, write_stage_idx, innerColA+3, innerRowA + i, BK, BM) = ldg_A_reg[ldg_index+3];
  }
  #pragma unroll
  for(int i = 0; i < BK; i+=rowStrideB) {
    int ldg_index = i / rowStrideB * 4;
    FLOAT4(Val3D(Bs, write_stage_idx, innerRowB + i, innerColB, BK, BN)) = 
      FLOAT4(ldg_B_reg[ldg_index]);
  }
}

template<const int TM, const int TN, const int WMITER, const int WNITER>
__device__ inline __attribute__((always_inline)) void computeThreadTile(
  const float* regA, const float* regB, float* threadResults, const int dotIdx) {
  #pragma unroll
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    #pragma unroll
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // calculate per-tile results
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          Val4D(threadResults, wSubRowIdx, resIdxM, wSubColIdx, resIdxN, TM, WNITER, TN) +=
              Val3D(regA, dotIdx%2, wSubRowIdx, resIdxM, WMITER, TM) * 
              Val3D(regB, dotIdx%2, wSubColIdx, resIdxN, WNITER, TN);
        }
      }
    }
  }
}

template<const int BM, const int BN, const int BK, const int TM, const int TN, 
  const int WM, const int WN, const int WMITER, const int WNITER, 
  const int WSUBM=WM/WMITER, const int WSUBN=WN/WNITER>
__device__ inline __attribute__((always_inline)) void loadThreadTile(
  float* As, float* Bs, float* regA, float* regB, const int dotIdx, const int load_stage_idx, 
  const int threadRowInWarp, const int threadColInWarp, const int warpRow, const int warpCol) {
  #pragma unroll
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    #pragma unroll
    for (uint i = 0; i < TM; i += 4) {
      FLOAT4(Val3D(regA, dotIdx&1, wSubRowIdx, i, WMITER, TM)) = 
        FLOAT4(Val3D(As, load_stage_idx, dotIdx, 
          warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i, BK, BM));
    }
  }
  #pragma unroll
  for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
    #pragma unroll
    for (uint i = 0; i < TN; i += 4) {
      FLOAT4(Val3D(regB, dotIdx&1, wSubColIdx, i, WNITER, TN)) = 
        FLOAT4(Val3D(Bs, load_stage_idx, dotIdx, 
          warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i, BK, BN));
    }
  }
}

}

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling2(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[2*BK*BM];
  __shared__ float Bs[2*BK*BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4) * 4;
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4) * 4;
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);
  const int ldg_num_A = BM * BK / (NUM_THREADS * 4);
  const int ldg_num_B = BN * BK / (NUM_THREADS * 4);
  float ldg_A_reg[4*ldg_num_A];
  float ldg_B_reg[4*ldg_num_B];

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regA[2 * WMITER * TM] = {0.0};
  float regB[2 * WNITER * TN] = {0.0};

  // Step0. Preload
  // Stage0.1: Load Block from global memory to register and from register to shared memory
  warptiling::loadBlockFromGlobalMemoryIntoSharedMemory<BM, BN, BK, rowStrideA, rowStrideB>
    (A, B, As, Bs, ldg_A_reg, 0/*write_stage_idx*/, N, K, innerRowA, innerRowB, innerColA, innerColB);
  __syncthreads();
  // Stage0.2: Load WarpTile from shared memory to register
  warptiling::loadThreadTile<BM, BN, BK, TM, TN, WM, WN, WMITER, WNITER>
    (As, Bs, regA, regB, 0/*dotIdx*/, 0/*load_stage_idx*/, threadRowInWarp, threadColInWarp, warpRow, warpCol);
  int write_stage_idx = 1;
  int tile_idx = 0;
  do {
    tile_idx += BK;
    if (tile_idx < K) {
      warptiling::loadBlockFromGlobalMemoryIntoRegister<BM, BN, BK, rowStrideA, rowStrideB>
        (A, B, ldg_A_reg, ldg_B_reg, tile_idx, N, K, innerRowA, innerRowB, innerColA, innerColB);
    }
    int load_stage_idx = write_stage_idx ^ 1;
    #pragma unroll
    for (uint dotIdx = 0; dotIdx < BK-1; ++dotIdx) {
      // Load next tile from current shared memory block to register
      warptiling::loadThreadTile<BM, BN, BK, TM, TN, WM, WN, WMITER, WNITER>
        (As, Bs, regA, regB, dotIdx+1, load_stage_idx, threadRowInWarp, threadColInWarp, warpRow, warpCol);

      // Compute current tile
      warptiling::computeThreadTile<TM, TN, WMITER, WNITER>
        (regA, regB, threadResults, dotIdx);
    }

    if(tile_idx < K){
      // Load next block from register into shared memory
      warptiling::loadBlockFromRegisterIntoSharedMemory<BM, BN, BK, rowStrideA, rowStrideB>
        (ldg_A_reg, ldg_B_reg, As, Bs, write_stage_idx, innerRowA, innerRowB, innerColA, innerColB);
      __syncthreads();
      write_stage_idx ^= 1;
    }
    
    // Load first tile from shared memory into register
    warptiling::loadThreadTile<BM, BN, BK, TM, TN, WM, WN, WMITER, WNITER>
      (As, Bs, regA, regB, 0/*dotIdx*/, load_stage_idx^1, threadRowInWarp, threadColInWarp, warpRow, warpCol);

    // Compute the last tile of the current block
    warptiling::computeThreadTile<TM, TN, WMITER, WNITER>
        (regA, regB, threadResults, 1/*dotIdx*/);
  } while (tile_idx < K);

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}

void run_sgemm_warptiling2(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  constexpr int WARPSIZE = 32;
  const uint NUM_THREADS = 128;
  const uint BN = 128;
  const uint BM = 64;
  const uint BK = 16;
  const uint WN = 64;
  const uint WM = 32;
  const uint WNITER = 1;
  const uint TN = 4;
  const uint TM = 4;
  constexpr uint NUM_WARPS = NUM_THREADS / WARPSIZE;
  constexpr uint WMITER =
      (WM * WN) / (WARPSIZE * TM * TN * WNITER);

  // warptile in blocktile
  static_assert((BN % WN == 0) && (BM % WM == 0), "blockTile dim shoule be divisible by warpTile dim");
  static_assert((BN / WN) * (BM / WM) == NUM_WARPS, "BN*BM/(WN*WM) should be NUM_WARPS");

  // threads in warpsubtile
  static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0, 
    "WMITER should be integer");
  // warpsubtile in warptile
  static_assert((WM % WMITER == 0) and (WN % WNITER == 0), 
    "WM/WMITER means number of iterations of M each thread should go");

  static_assert((NUM_THREADS*4) % BK == 0 && (NUM_THREADS*4) % BN == 0, 
      "number of items to load to be divisible by blockTile col dim");
  static_assert(BN % (16 * TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(BM % (16 * TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((BM * BK) % (4 * NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*NUM_THREADS to vectorize loads");
  static_assert((BN * BK) % (4 * NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*NUM_THREADS to vectorize loads");
  
  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemmWarptiling2<BM, BN, BK, WM, WN, WNITER, TM,
                  TN, NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
