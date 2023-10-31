#include <vector>
#include <tuple>
#include <sstream> // for ostringstream
#include "lib/macros.cuh"
#include "gemm_test.hpp"

using triple = std::tuple<size_t, size_t, size_t>;
int main() {
  const uint num_repeats = 4;
  std::vector<triple> problem_size = {
    triple(4096, 4096, 4096),
    triple(16, 4096, 1024)
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
  }
}