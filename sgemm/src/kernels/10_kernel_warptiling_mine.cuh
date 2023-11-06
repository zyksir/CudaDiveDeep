#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../lib/macros.cuh" 

// static const int WARPSIZE = 32; // warpSize is not constexpr
namespace wt_mine {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // float4 tmp;
    // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
    //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
    // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
    //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace wt


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
    sgemmWarptiling_mine(int M, int N, int K, float alpha, float *A, float *B,
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
  #pragma unroll
  for(int i = 0; i < BM; i+=rowStrideA) {
    int ldg_index = i / rowStrideA * 4;
      FLOAT4(ldg_A_reg[ldg_index]) = 
        FLOAT4(A[OFFSET(innerRowA + i, innerColA, K)]);
      Val3D(As, 0, innerColA,   innerRowA + i, BK, BM) = ldg_A_reg[ldg_index];
      Val3D(As, 0, innerColA+1, innerRowA + i, BK, BM) = ldg_A_reg[ldg_index+1];
      Val3D(As, 0, innerColA+2, innerRowA + i, BK, BM) = ldg_A_reg[ldg_index+2];
      Val3D(As, 0, innerColA+3, innerRowA + i, BK, BM) = ldg_A_reg[ldg_index+3];
  }
  #pragma unroll
  for(int i = 0; i < BK; i+=rowStrideB) {
    FLOAT4(Val3D(Bs, 0, innerRowB + i, innerColB, BK, BN)) = 
      FLOAT4(B[OFFSET(innerRowB + i, innerColB, N)]);
  }
  __syncthreads();
  // Stage0.2: Load WarpTile from shared memory to register
  #pragma unroll
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint i = 0; i < TM; ++i) {
      Val(regA, wSubRowIdx, i, TM) = 
          As[warpRow * WM + wSubRowIdx * WSUBM +
              threadRowInWarp * TM + i];
    }
  }
  #pragma unroll
  for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
    for (uint i = 0; i < TN; ++i) {
      Val(regB, wSubColIdx, i, TN) = 
          Bs[warpCol * WN + wSubColIdx * WSUBN +
              threadColInWarp * TN + i];
    }
  }
  int write_stage_idx = 1;
  int tile_idx = 0;
  do {
    tile_idx += BK;
    if (tile_idx < K) {
      #pragma unroll
      for(int i = 0; i < BM; i+=rowStrideA) {
        int ldg_index = i / rowStrideA * 4;
        FLOAT4(ldg_A_reg[ldg_index]) = 
          FLOAT4(A[OFFSET(innerRowA + i, innerColA + tile_idx, K)]);
      }
      #pragma unroll
      for(int i = 0; i < BK; i+=rowStrideB) {
        int ldg_index = i / rowStrideB * 4;
        FLOAT4(ldg_B_reg[ldg_index]) = 
          FLOAT4(B[OFFSET(tile_idx + innerRowB + i, innerColB, N)]);
      }
    }
    int load_stage_idx = write_stage_idx ^ 1;
    #pragma unroll
    for (uint dotIdx = 0; dotIdx < BK-1; ++dotIdx) {
      // Stage1.2.1: Load Next Tile from Shared Memory to Register
      // load from the current shared memory block
      #pragma unroll
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        #pragma unroll
        for (uint i = 0; i < TM; i += 4) {
          FLOAT4(Val3D(regA, (dotIdx+1)%2, wSubRowIdx, i, WMITER, TM)) = 
            FLOAT4(Val3D(As, load_stage_idx, dotIdx+1, 
              warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i, BK, BM));
        }
      }
      #pragma unroll
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        #pragma unroll
        for (uint i = 0; i < TN; i += 4) {
          FLOAT4(Val3D(regB, (dotIdx+1)%2, wSubColIdx, i, WNITER, TN)) = 
            FLOAT4(Val3D(Bs, load_stage_idx, dotIdx+1, 
              warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i, BK, BN));
        }
      }
      // Stage1.2.2: Compute current tile
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

    if(tile_idx < K){
      #pragma unroll
      for(int i = 0; i < BM; i+=rowStrideA) {
        int ldg_index = i / rowStrideA * 4;
        Val3D(As, write_stage_idx, innerColA,   innerRowA + i, BK, BM) = ldg_A_reg[ldg_index];
        Val3D(As, write_stage_idx, innerColA+1, innerRowA + i, BK, BM) = ldg_A_reg[ldg_index+1];
        Val3D(As, write_stage_idx, innerColA+2, innerRowA + i, BK, BM) = ldg_A_reg[ldg_index+2];
        Val3D(As, write_stage_idx, innerColA+3, innerRowA + i, BK, BM) = ldg_A_reg[ldg_index+3];
      }
      #pragma unroll
      for(int i = 0; i < BK; i+=rowStrideB) {
        int ldg_index = i / rowStrideB * 4;
        FLOAT4(Val3D(Bs, write_stage_idx, innerRowB + i, innerColB, BK, BN)) = FLOAT4(ldg_B_reg[ldg_index]);
      }
      __syncthreads();
      write_stage_idx ^= 1;
    }
    
    #pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      #pragma unroll
      for (uint i = 0; i < TM; i += 4) {
        FLOAT4(Val3D(regA, 0, wSubRowIdx, i, WMITER, TM)) = 
          FLOAT4(Val3D(As, load_stage_idx^1, 0, 
            warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i, BK, BM));
      }
    }
    #pragma unroll
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      #pragma unroll
      for (uint i = 0; i < TN; i += 4) {
        FLOAT4(Val3D(regB, 0, wSubColIdx, i, WNITER, TN)) = 
          FLOAT4(Val3D(Bs, load_stage_idx^1, 0, 
            warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i, BK, BN));
      }
    }
    // Stage1.5: Compute the last tile of the current block
    #pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      #pragma unroll
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-tile results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            Val4D(threadResults, wSubRowIdx, resIdxM, wSubColIdx, resIdxN, TM, WNITER, TN) +=
                Val3D(regA, 1, wSubRowIdx, resIdxM, WMITER, TM) * 
                Val3D(regB, 1, wSubColIdx, resIdxN, WNITER, TN);
          }
        }
      }
    }
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

void runSgemmWarptiling_mine(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  const int WARPSIZE = 32;
  // Settings for A100
  const uint K10_NUM_THREADS = 128;
  const uint K10_BN = 128;
  const uint K10_BM = 64;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 32;
  const uint K10_WNITER = 1;
  const uint K10_TN = 4;
  const uint K10_TM = 4;
  // Settings for A6000
  // const uint K10_NUM_THREADS = 128;
  // const uint K10_BN = 128;
  // const uint K10_BM = 128;
  // const uint K10_BK = 16;
  // const uint K10_WN = 64;
  // const uint K10_WM = 64;
  // const uint K10_WNITER = 4;
  // const uint K10_TN = 4;
  // const uint K10_TM = 8;
  dim3 blockDim(K10_NUM_THREADS);

  constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
  static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) ==
                0);
  constexpr uint K10_WMITER =
      (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
  // warpsubtile in warptile
  static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

  static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K10_BN % (16 * K10_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K10_BM % (16 * K10_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
  sgemmWarptiling_mine<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                  K10_TN, K10_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
