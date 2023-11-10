#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../lib/macros.cuh" 

namespace double_buffering {
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

template<const int TM, const int TN>
__device__ inline __attribute__((always_inline)) void computeThreadTile(
  const float* regA, const float* regB, float* threadResults, const int dotIdx) {
  #pragma unroll
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    #pragma unroll
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      Val(threadResults, resIdxM, resIdxN, TN) += 
        Val(regA, dotIdx%2, resIdxM, TM) * Val(regB, dotIdx%2, resIdxN, TN);
    }
  }
}

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__device__ inline __attribute__((always_inline)) void loadThreadTile(
  float* As, float* Bs, float* regA, float* regB,
  const int dotIdx, const int load_stage_idx, const int threadRow, const int threadCol) {
  #pragma unroll
  for (uint i = 0; i < TM; i+=4) {
    FLOAT4(Val(regA, dotIdx&1/*dotIdx%2*/, i, TM)) = 
      FLOAT4(Val3D(As, load_stage_idx, dotIdx, threadRow * TM + i, BK, BM));
  }
  #pragma unroll
  for (uint i = 0; i < TN; i+=4) {
    FLOAT4(Val(regB, dotIdx&1/*dotIdx%2*/, i, TN)) = 
      FLOAT4(Val3D(Bs, load_stage_idx, dotIdx, threadCol * TN + i, BK, BN));
  }
}

}

template <const int BM, const int BN, const int BK, const int TM, const int TN, 
    const int TX=BM/TM, const int TY=BN/TN, const int THREAD_NUM_PER_BLOCK=TX*TY>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemmVectorize_double_buffering2(int M, int N, int K, float alpha, float *A,
                      float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const uint tid = threadIdx.x;
  const int threadCol = tid % TY;
  const int threadRow = tid / TY;

  // allocate space for the current blocktile in smem
  __shared__ float As[2*BK*BM];
  __shared__ float Bs[2*BK*BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const int ldg_num_A = BM * BK / (THREAD_NUM_PER_BLOCK * 4);
  const int ldg_num_B = BN * BK / (THREAD_NUM_PER_BLOCK * 4);
  float ldg_A_reg[4*ldg_num_A];
  float ldg_B_reg[4*ldg_num_B];
  const int A_TILE_THREAD_PER_ROW = BK / 4;
  const int B_TILE_THREAD_PER_ROW = BN / 4;
  const int innerRowA = tid / A_TILE_THREAD_PER_ROW;
  const int innerRowB = tid / B_TILE_THREAD_PER_ROW;
  const int innerColA = tid % A_TILE_THREAD_PER_ROW * 4; 
  const int innerColB = tid % B_TILE_THREAD_PER_ROW * 4;
  const int rowStrideA = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
  const int rowStrideB = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM*TN] = {0.0};
  // register caches for As and Bs
  float regA[2*TM] = {0.0};
  float regB[2*TN] = {0.0};

  // Step0. Preload
  // Stage0.1: Load Block from global memory to register and from register to shared memory
  double_buffering::loadBlockFromGlobalMemoryIntoSharedMemory<BM, BN, BK, rowStrideA, rowStrideB>
    (A, B, As, Bs, ldg_A_reg, 0/*write_stage_idx*/, N, K, innerRowA, innerRowB, innerColA, innerColB);
  __syncthreads();
  // Stage0.2: Load Tile from shared memory to register
  double_buffering::loadThreadTile<BM, BN, BK, TM, TN>
    (As, Bs, regA, regB, 0/*dotIdx*/, 0/*load_stage_idx*/, threadRow, threadCol);
  int write_stage_idx = 1;
  int tile_idx = 0;
  do {
    tile_idx += BK;
    if (tile_idx < K) {
      double_buffering::loadBlockFromGlobalMemoryIntoRegister<BM, BN, BK, rowStrideA, rowStrideB>
        (A, B, ldg_A_reg, ldg_B_reg, tile_idx, N, K, innerRowA, innerRowB, innerColA, innerColB);
    }
    int load_stage_idx = write_stage_idx ^ 1;
    #pragma unroll
    for (uint dotIdx = 0; dotIdx < BK-1; ++dotIdx) {
      // Load next tile from current shared memory block to register
      double_buffering::loadThreadTile<BM, BN, BK, TM, TN>
        (As, Bs, regA, regB, dotIdx+1, load_stage_idx, threadRow, threadCol);

      // Compute current tile
      double_buffering::computeThreadTile<TM, TN>
        (regA, regB, threadResults, dotIdx);
    }

    if(tile_idx < K){
      // Load next block from register into shared memory
      double_buffering::loadBlockFromRegisterIntoSharedMemory<BM, BN, BK, rowStrideA, rowStrideB>
        (ldg_A_reg, ldg_B_reg, As, Bs, write_stage_idx, innerRowA, innerRowB, innerColA, innerColB);
      __syncthreads();
      write_stage_idx ^= 1;
    }

    // Load first tile from shared memory into register
    double_buffering::loadThreadTile<BM, BN, BK, TM, TN>
      (As, Bs, regA, regB, 0/*dotIdx*/, load_stage_idx^1, threadRow, threadCol);

    // Compute the last tile of the current block
    double_buffering::computeThreadTile<TM, TN>
        (regA, regB, threadResults, 1/*dotIdx*/);
  } while (tile_idx < K);
  // write out the results
  #pragma unroll
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    #pragma unroll
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      float4 tmp = FLOAT4(C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN]);
      // perform GEMM update in reg
      tmp.x = alpha * Val(threadResults, resIdxM, resIdxN, TN) + beta * tmp.x;
      tmp.y = alpha * Val(threadResults, resIdxM, resIdxN+1, TN) + beta * tmp.y;
      tmp.z = alpha * Val(threadResults, resIdxM, resIdxN+2, TN) + beta * tmp.z;
      tmp.w = alpha * Val(threadResults, resIdxM, resIdxN+3, TN) + beta * tmp.w;
      // write back
      FLOAT4(C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN]) = tmp;
    }
  }
}

void run_sgemm_double_buffering2(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    const uint NUM_THREADS = (BM * BN) / (TM * TN);
    
    static_assert((BM % TM == 0) && (BN % TN == 0), 
      "blockTile dim should be divisible by threadTile dim");
    static_assert((NUM_THREADS*4) % BK == 0 && (NUM_THREADS*4) % BN == 0, 
      "number of items to load to be divisible by blockTile col dim");
    static_assert((BM * BK) % (4 * NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*NUM_THREADS to vectorize loads");
    static_assert((BN * BK) % (4 * NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*NUM_THREADS to vectorize loads");

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BN*BM)/(TN*TM));
    sgemmVectorize_double_buffering2<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize_double_buffering2<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}
