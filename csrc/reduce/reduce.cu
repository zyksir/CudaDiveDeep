#include <cstdlib>
#include <mutex>
#include <system_error>
#include <vector>
#include "common/common.h"

namespace cuda_extension {

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

__global__ void reduce7(const float *g_idata, float *g_odata, int n) {
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

void gpu_sum_reduce(const float* d_in, float* d_out, unsigned int d_in_len, decltype(reduce7) kernel, 
    unsigned int coarse_factor, unsigned int unroll_factor=1) {
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
		kernel<<<1, block_sz, sizeof(float) * MAX_BLOCK_SZ>>>(d_block_sums, d_out, grid_sz);
	} else {
		gpu_sum_reduce(d_block_sums, d_out, grid_sz, kernel, coarse_factor);
	}
	checkCudaErrors(cudaFree(d_block_sums));
}

at::Tensor reduce_sum(const at::Tensor& input_) {
	const auto matrix = input_.dim() == 1 ? input_ : input_.view(-1);
    const auto size = matrix.size(0);
	const at::ScalarType _st = matrix.scalar_type();
	at::Tensor output_ = at::empty(
      {1},
      at::TensorOptions().dtype(_st).device(torch::kCUDA).requires_grad(false));
    gpu_sum_reduce(
		matrix.data_ptr<float>(), 
		output_.data_ptr<float>(), 
		size, reduce7, 2, 16
	);
	return output_;
}

};