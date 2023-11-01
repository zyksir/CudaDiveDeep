#include <vector>
#include <tuple>
#include <sstream> // for ostringstream
#include "lib/macros.cuh"
#include "kernels/sgemm_kun.cu"
#include "gemm_test.hpp"

using triple = std::tuple<size_t, size_t, size_t>;
int main() {
  const uint num_repeats = 40;
  std::vector<triple> problem_size = {
    triple(4096, 4096, 4096),
    triple(128, 4096, 1024)
  };
  CudaDeviceInfo();
  for(const auto& problem : problem_size) {
    size_t M, N, K;
    std::tie(M, N, K) = problem;
    std::ostringstream test_name;
    test_name << "SGEMM ";
    test_name << M << "x" << N << "x" << K << " ";
    auto sgemm_test = GemmTest(M, N, K, test_name.str(), num_repeats);
    sgemm_test.run_baseline();
    auto kun_cuda = [](float* gpu_A, float* gpu_B, float* gpu_C, size_t M, size_t N, size_t K) {
      static const int BLOCK_SIZE_M = 128;
      static const int BLOCK_SIZE_K = 8;
      static const int BLOCK_SIZE_N = 128;
      static const int THREAD_SIZE_X = 8;
      static const int THREAD_SIZE_Y = 8;
      static const bool ENABLE_DOUBLE_BUFFER = false;

      dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
      dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
      Sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> 
        <<< dimGrid, dimBlock >>>(gpu_A, gpu_B, gpu_C, M, N, K);
    };
    sgemm_test.run_cuda(kun_cuda, "kun");
  }
  return 0;
}