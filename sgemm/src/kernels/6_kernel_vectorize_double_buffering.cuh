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
    sgemmVectorize_double_buffering(int M, int N, int K, float alpha, float *A,
                      float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const uint tid = threadIdx.x;
  const int threadCol = tid % TY;
  const int threadRow = tid / TY;
  // const int threadCol = threadIdx.x;
  // const int threadRow = threadIdx.y;
  // const int tid = threadRow * TX + threadCol;

  // allocate space for the current blocktile in smem
  __shared__ float As[2][BK][BM];
  __shared__ float Bs[2][BK][BN];

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
  const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
  const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
  const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4; 
  const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;
  const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
  const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM][TN] = {0.0};
  // register caches for As and Bs
  float regA[2][TM] = {0.0};
  float regB[2][TN] = {0.0};
  int write_stage_idx = 0;
  int load_stage_idx = 0;

  // Step0. Preload
  // Stage0.1: Load Block from global memory to register and from register to shared memory
  #pragma unroll
  for(int i = 0; i < BM; i+=A_TILE_ROW_STRIDE) {
    int ldg_index = i / A_TILE_ROW_STRIDE * 4;
      FETCH_FLOAT4(ldg_A_reg[ldg_index]) = 
        FETCH_FLOAT4(A[OFFSET(A_TILE_ROW_START + i, A_TILE_COL, K )]);
      As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]=ldg_A_reg[ldg_index];
      As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_A_reg[ldg_index+1];
      As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_A_reg[ldg_index+2];
      As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_A_reg[ldg_index+3];
  }
  #pragma unroll
  for(int i = 0; i < BK; i+=B_TILE_ROW_STRIDE) {
    FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = 
      FETCH_FLOAT4(B[OFFSET(B_TILE_ROW_START + i, B_TILE_COL, N)]);
  }
  __syncthreads();
  // A += BK;     // move BK columns to right
  // B += BK * N; // move BK rows down
  // Stage0.2: Load Tile from shared memory to register
  for (uint i = 0; i < TM; i+=4) {
    FETCH_FLOAT4(regA[0][i]) = FETCH_FLOAT4(As[write_stage_idx][0][threadRow * TM + i]);
  }
  #pragma unroll
  for (uint i = 0; i < TN; i+=4) {
    FETCH_FLOAT4(regB[0][i]) = FETCH_FLOAT4(Bs[write_stage_idx][0][threadCol * TN + i]);
  }
  write_stage_idx ^= 1;

  // Step1. The Outer Loop: one load and one compute each iteration
  // in the beginning, load_stage_idx = 0, write_stage_idx = 1
  for (uint bkIdx = BK; bkIdx < K; bkIdx += BK) {

    // Step1.1: Load Next Block from Global Memory to Register
    #pragma unroll
    for(int i = 0; i < BM; i+=A_TILE_ROW_STRIDE) {
      int ldg_index = i / A_TILE_ROW_STRIDE * 4;
      FETCH_FLOAT4(ldg_A_reg[ldg_index]) = 
        FETCH_FLOAT4(A[OFFSET(A_TILE_ROW_START + i, A_TILE_COL + bkIdx, K)]);
    }
    #pragma unroll
    for(int i = 0; i < BK; i+=B_TILE_ROW_STRIDE) {
      int ldg_index = i / B_TILE_ROW_STRIDE * 4;
      FETCH_FLOAT4(ldg_B_reg[ldg_index]) = 
        FETCH_FLOAT4(B[OFFSET(B_TILE_ROW_START + i + bkIdx, B_TILE_COL, N)]);
    }
    // A += BK;     // move BK columns to right
    // B += BK * N; // move BK rows down
    // Stage1.1 End

    // Stage1.2: The inner Loop: one load and one compute 
    #pragma unroll
    for (uint dotIdx = 0; dotIdx < BK-1; ++dotIdx) {
      // Stage1.2.1: Load Next Tile from Shared Memory to Register
      // load from the current shared memory block
      #pragma unroll
      for (uint i = 0; i < TM; i+=4) {
        FETCH_FLOAT4(regA[(dotIdx+1)%2][i]) = FETCH_FLOAT4(As[load_stage_idx][dotIdx+1][threadRow * TM + i]);
      }
      #pragma unroll
      for (uint i = 0; i < TN; i+=4) {
        FETCH_FLOAT4(regB[(dotIdx+1)%2][i]) = FETCH_FLOAT4(Bs[load_stage_idx][dotIdx+1][threadCol * TN + i]);
      }
      // Stage1.2.2: Compute current tile
      #pragma unroll
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        #pragma unroll
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM][resIdxN] += regA[dotIdx%2][resIdxM] * regB[dotIdx%2][resIdxN];
        }
      }
    }
    // Stage1.3: Load Next Block from register into Shared Memory
    #pragma unroll
    for(int i = 0; i < BM; i+=A_TILE_ROW_STRIDE) {
      int ldg_index = i / A_TILE_ROW_STRIDE * 4;
      As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]=ldg_A_reg[ldg_index];
      As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_A_reg[ldg_index+1];
      As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_A_reg[ldg_index+2];
      As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_A_reg[ldg_index+3];
    }
    #pragma unroll
    for(int i = 0; i < BK; i+=B_TILE_ROW_STRIDE) {
      int ldg_index = i / B_TILE_ROW_STRIDE * 4;
      FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_B_reg[ldg_index]);
    }
    __syncthreads();
    // Stage1.4: Load Next Tile for register double buffering
    // this is the first tile of the next block, so we load from write_stage_idx
    for (uint i = 0; i < TM; i+=4) {
      FETCH_FLOAT4(regA[0][i]) = FETCH_FLOAT4(As[write_stage_idx][0][threadRow * TM + i]);
    }
    #pragma unroll
    for (uint i = 0; i < TN; i+=4) {
      FETCH_FLOAT4(regB[0][i]) = FETCH_FLOAT4(Bs[write_stage_idx][0][threadCol * TN + i]);
    }
    // Stage1.5: Compute the last tile of the current block
    #pragma unroll
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
      #pragma unroll
      for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
        threadResults[resIdxM][resIdxN] += regA[1][resIdxM] * regB[1][resIdxN];
      }
    }
    load_stage_idx = write_stage_idx;
    write_stage_idx ^= 1;
  }

  // Stage2.1: One More Computation
  #pragma unroll
  for (uint dotIdx = 0; dotIdx < BK-1; ++dotIdx) {
    // Stage2.2.1: load block from Shared Memory to Register
    // we should load from the previous shared memory block
    #pragma unroll
    for (uint i = 0; i < TM; i+=4) {
      FETCH_FLOAT4(regA[(dotIdx+1)%2][i]) = FETCH_FLOAT4(As[load_stage_idx][dotIdx+1][threadRow * TM + i]);
    }
    #pragma unroll
    for (uint i = 0; i < TN; i+=4) {
      FETCH_FLOAT4(regB[(dotIdx+1)%2][i]) = FETCH_FLOAT4(Bs[load_stage_idx][dotIdx+1][threadCol * TN + i]);
    }
    // Stage2.2.2: Compute current tile of the current block
    #pragma unroll
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
      #pragma unroll
      for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
        threadResults[resIdxM][resIdxN] += regA[dotIdx%2][resIdxM] * regB[dotIdx%2][resIdxN];
      }
    }
  }
  // Stage2.2.3: Compute the last tile of the current block
  #pragma unroll
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    #pragma unroll
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      threadResults[resIdxM][resIdxN] += regA[1][resIdxM] * regB[1][resIdxN];
    }
  }

  // write out the results
  #pragma unroll
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    #pragma unroll
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

void runSgemmVectorize_double_buffering(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BN*BM)/(TN*TM));
    sgemmVectorize_double_buffering<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize_double_buffering<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}
