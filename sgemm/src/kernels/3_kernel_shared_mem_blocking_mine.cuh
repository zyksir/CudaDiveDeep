#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const uint threadsPerBlock=BM*BN>
__global__ void sgemm_shared_mem_block_mine(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // the output block that we want to compute in this threadblock
  const uint blockX = blockIdx.x;
  const uint blockY = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  // the inner row & col that we're accessing in this thread
  const uint threadY = threadIdx.x % BN;
  const uint threadX = threadIdx.x / BN;

  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint strideA = threadsPerBlock / BK;

  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  const uint strideB = threadsPerBlock / BN;

  // advance pointers to the starting positions
  A += blockX * BM * K;                     // row=cRow, col=0
  B += blockY * BN;                         // row=0, col=cCol
  C += blockX * BM * N + blockY * BN;       // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    for(int i = 0; i < BM; i += strideA) {
      As[innerRowA + i][innerColA] = A[(innerRowA + i) * K + innerColA];
    }
    for(int i = 0; i < BK; i += strideB) {
      Bs[innerRowB + i][innerColB] = B[(innerRowB + i) * N + innerColB];
    }

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BK;
    B += BK * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      tmp += As[threadX][dotIdx] * Bs[dotIdx][threadY];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadX * N + threadY] = alpha * tmp + beta * C[threadX * N + threadY];
}

void run_sgemm_shared_mem_block_mine(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C) {
  static const int BM = 32, BN = 32, BK = 32;
  dim3 dimGrid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
  dim3 dimBlock(BM*BN);
  cudaFuncSetAttribute(sgemm_shared_mem_block<32>,
                      cudaFuncAttributePreferredSharedMemoryCarveout,
                      cudaSharedmemCarveoutMaxShared);
  sgemm_shared_mem_block_mine<BM, BN, BK> 
    <<< dimGrid, dimBlock >>>(M, N, K, alpha, A, B, beta, C);
}