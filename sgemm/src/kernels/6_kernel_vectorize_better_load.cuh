#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN, 
    const int TX=BM/TM, const int TY=BN/TN, const int THREAD_NUM_PER_BLOCK=TX*TY>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemmVectorize_better_load(int M, int N, int K, float alpha, float *A,
                      float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  const uint tid = threadIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const int ldg_num_A = BM * BK / (THREAD_NUM_PER_BLOCK * 4);
  // const int ldg_num_B = BN * BK / (THREAD_NUM_PER_BLOCK * 4);
  float ldg_A_reg[4*ldg_num_A];
  // const float lgd_B_reg[4*ldg_num_B];
  const int A_TILE_THREAD_PER_ROW = BK / 4;
  const int B_TILE_THREAD_PER_ROW = BN / 4;
  const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
  const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
  const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4; 
  const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;
  const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
  const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM][TN] = {0.0};
  // register caches for As and Bs
  float regA[TM] = {0.0};
  float regB[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // load a block of size BM x BK from A to As
    #pragma unroll
    for(int i = 0; i < BM; i+=A_TILE_ROW_STRIDE) {
      int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_A_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i, // row
            A_TILE_COL, // col
            K )]);
        As[A_TILE_COL][A_TILE_ROW_START + i]=ldg_A_reg[ldg_index];
        As[A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_A_reg[ldg_index+1];
        As[A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_A_reg[ldg_index+2];
        As[A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_A_reg[ldg_index+3];
    }

    #pragma unroll
    for(int i = 0; i < BK; i+=B_TILE_ROW_STRIDE) {
      FETCH_FLOAT4(Bs[B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                B_TILE_ROW_START + i, // row
                B_TILE_COL, // col
                N )]);
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    #pragma unroll
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      #pragma unroll
      for (uint i = 0; i < TM; i+=4) {
        FETCH_FLOAT4(regA[i]) = FETCH_FLOAT4(As[dotIdx][threadRow * TM + i]);
      }
      #pragma unroll
      for (uint i = 0; i < TN; i+=4) {
        FETCH_FLOAT4(regB[i]) = FETCH_FLOAT4(Bs[dotIdx][threadCol * TN + i]);
      }
      #pragma unroll
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        #pragma unroll
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM][resIdxN] +=
              regA[resIdxM] * regB[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      float4 tmp = FETCH_FLOAT4(C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN]);
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM][resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM][resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM][resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM][resIdxN + 3] + beta * tmp.w;
      // write back
      FETCH_FLOAT4(C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN]) = tmp;
    }
  }
}

void runSgemmVectorize_better_load(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize_better_load<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize_better_load<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}
