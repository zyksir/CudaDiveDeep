#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <random>
#include <cassert>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublas_api.h>

#include "test.hpp"
#include "macros.cuh"
#include "kernels/gemm_seq.hpp"
#include "kernels/gemm_kernels.hpp"

int main() {
  const uint num_repeats = 1;
  CudaDeviceInfo();
  const int64_t float_calculation_num = 2*static_cast<uint64_t>(BatchSize)*Nnn*Nii;
  auto input_length = BatchSize*Nii;
  auto output_length = BatchSize*Nnn;
  auto weight_length = Nii * Nnn;
  auto gemm_test = Test<float, decltype(gemm_naive<BatchSize, Nii, Nnn, BatchSize, 16>)>
    (input_length, output_length, weight_length, float_calculation_num, "GEMM ", num_repeats);
  gemm_test.run_seq(gemm_seq<BatchSize, Nii, Nnn>);
  gemm_test.test_cuda(gemm_naive<BatchSize, Nii, Nnn, BatchSize, 64>, "CUDA NAIVE");
  gemm_test.test_cuda(gemm_coalescing<BatchSize, Nii, Nnn, BatchSize, 64>, "CUDA coalescing");
  // gemm_test.test_cuda(gemm_naive_shared<BatchSize, Nii, Nnn, BatchSize, 64, 512>, "CUDA NAIVE SHARED");
  gemm_test.test_cuda(gemm_shared<BatchSize, Nii, Nnn, BatchSize, 64, 64>, "CUDA SHARED");
  gemm_test.test_cuda(gemm_block_tiling<BatchSize, Nii, Nnn, BatchSize, 64, 64, 2, 2>, "CUDA TILING");
  gemm_test.test_cuda(gemm_vectorize<BatchSize, Nii, Nnn, BatchSize, 64, 64, 2, 2>, "CUDA VECTERIZE");
  auto cublas = [](const float* A, const float* B, float* C, dim3& grid, dim3& block) {
    cublasHandle_t err; cublasCreate(&err);
    cudaDeviceSynchronize();
    static const float alpha = 1.0, beta = 0.0;
    constexpr int M = BatchSize, N = Nnn, K = Nii;
    cublasSgemm(err, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
  };
  gemm_test.test_cuda(cublas, "CUBLAS");
}