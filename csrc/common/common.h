#pragma once

#include <iostream>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

template <typename T>
inline T *get_ptr(torch::Tensor &t) {
    return reinterpret_cast<T *>(t.data_ptr());
}
