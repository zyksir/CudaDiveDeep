""" usage: TORCH_LOGS="output_code" python reduce.py
torch sum take 245.106426 ms, got -18174.291016
compiled sum take 1397.798635 ms, got -18174.300781
my sum take 261.682916 ms, got -18174.294922
"""
import time
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = '''
#include <iostream>
#include <torch/extension.h>
#include <cuda.h>
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

#define MAX_BLOCK_SZ 1024
constexpr int blockSize = MAX_BLOCK_SZ;

__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void reduce7(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = ((i < n) ? g_idata[i] : 0) + ((i+blockDim.x < n) ? g_idata[i+blockDim.x] : 0); i+= gridSize;
    while (i < n) { sdata[tid] += ((i < n) ? g_idata[i] : 0) + ((i+blockDim.x < n) ? g_idata[i+blockDim.x] : 0); i += gridSize; }
    __syncthreads();
    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

float gpu_sum_reduce(float* d_in, unsigned int d_in_len, decltype(reduce7) kernel, 
    unsigned int coarse_factor, unsigned int unroll_factor=1) {
    // std::cout << d_in_len << "," << coarse_factor << "," << unroll_factor << std::endl;
	float total_sum = 0;

	// Set up number of threads and blocks
	// If input size is not power of two, the remainder will still need a whole block
	// Thus, number of blocks must be the least number of 2048-blocks greater than the input size
	unsigned int block_sz = MAX_BLOCK_SZ; // Halve the block size due to reduce3() and further 
											  //  optimizations from there
	// our block_sum_reduce()
	unsigned int max_elems_per_block = block_sz * coarse_factor; // due to binary tree nature of algorithm	
	unsigned int grid_sz = (d_in_len + max_elems_per_block - 1) / max_elems_per_block / unroll_factor;
    grid_sz = grid_sz == 0 ? 1 : grid_sz;
    // std::cout << max_elems_per_block << "," << grid_sz << std::endl;

	// Allocate memory for array of total sums produced by each block
	// Array length must be the same as number of blocks / grid size
	float* d_block_sums;
	checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(float) * grid_sz));
	checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(float) * grid_sz));

	// Sum data allocated for each block
	kernel<<<grid_sz, block_sz, sizeof(float) * MAX_BLOCK_SZ>>>(d_in, d_block_sums, d_in_len);

	// Sum each block's total sums (to get global total sum)
	// Use basic implementation if number of total sums is <= 2048
	// Else, recurse on this same function
	if (grid_sz <= max_elems_per_block) {
		float* d_total_sum;
		checkCudaErrors(cudaMalloc(&d_total_sum, sizeof(float)));
		checkCudaErrors(cudaMemset(d_total_sum, 0, sizeof(float)));
		kernel<<<1, block_sz, sizeof(float) * MAX_BLOCK_SZ>>>(d_block_sums, d_total_sum, grid_sz);
		checkCudaErrors(cudaMemcpy(&total_sum, d_total_sum, sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_total_sum));
	} else {
		float* d_in_block_sums;
		checkCudaErrors(cudaMalloc(&d_in_block_sums, sizeof(float) * grid_sz));
		checkCudaErrors(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(float) * grid_sz, cudaMemcpyDeviceToDevice));
		total_sum = gpu_sum_reduce(d_in_block_sums, grid_sz, kernel, coarse_factor);
		checkCudaErrors(cudaFree(d_in_block_sums));
	}

	checkCudaErrors(cudaFree(d_block_sums));
	return total_sum;
}

float reduce(torch::Tensor matrix) {
    const auto size = matrix.size(0);
    return gpu_sum_reduce(matrix.data_ptr<float>(), size, reduce7, 2, 16);
}
'''

cpp_source = "float reduce(torch::Tensor matrix);"

reduce_extension = load_inline(
    name='reduce_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['reduce'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory='./reduce_extension',
)

@torch.compile
def compiled_sum(matrix):
    result = torch.sum(matrix)
    return result


if __name__ == "__main__":
    size = int(1e4)
    input_ = torch.randn(size).cuda()
    
    s = time.perf_counter()
    torch_sum = torch.sum(input_)
    torch.cuda.synchronize()
    for _ in range(100):
        torch_sum = torch.sum(input_)
    torch.cuda.synchronize()
    e = time.perf_counter()
    print(f"torch sum take {(e - s)*1000:.6f} ms, got {torch_sum:.6f}")
    
    s = time.perf_counter()
    compiled_result = compiled_sum(input_)
    torch.cuda.synchronize()
    for _ in range(100):
        compiled_result = compiled_sum(input_)
    torch.cuda.synchronize()
    e = time.perf_counter()
    print(f"compiled sum take {(e - s)*1000:.6f} ms, got {compiled_result:.6f}")
    
    s = time.perf_counter()
    my_sum = reduce_extension.reduce(input_)
    torch.cuda.synchronize()
    for _ in range(0):
        my_sum = reduce_extension.reduce(input_)
    torch.cuda.synchronize()
    e = time.perf_counter()
    print(f"my sum take {(e - s)*1000:.6f} ms, got {my_sum:.6f}")