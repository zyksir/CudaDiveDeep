/* usage: 
nvcc -o reduce_nv reduce_nv.cu && ./reduce_nv 
*/
#include <iostream>
#include <cuda.h>
#include <ctime>
#include "cuda_runtime.h"

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
#define MAX_BLOCK_SZ 1024
constexpr int blockSize = MAX_BLOCK_SZ;

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

__global__
void block_sum_reduce(int* d_in,
    int* d_block_sums, 
	int d_in_len) {
	extern __shared__ unsigned int s_out[];

	unsigned int glbl_tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	
	// Zero out shared memory
	// Especially important when padding shmem for
	//  non-power of 2 sized input
	s_out[threadIdx.x] = 0;
	s_out[threadIdx.x + blockDim.x] = 0;

	__syncthreads();

	// Copy d_in to shared memory per block
	if (glbl_tid < d_in_len) {
		s_out[threadIdx.x] = d_in[glbl_tid];
		if (glbl_tid + blockDim.x < d_in_len)
			s_out[threadIdx.x + blockDim.x] = d_in[glbl_tid + blockDim.x];
	}
	__syncthreads();

	// Actually do the reduction
	for (unsigned int s = blockDim.x; s > 0; s >>= 1) {
		if (tid < s) {
			s_out[tid] += s_out[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		d_block_sums[blockIdx.x] = s_out[0];
}

__global__ void reduce0(int *g_idata, int *g_odata, int n) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce1(int *g_idata, int *g_odata, int n) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce2(int *g_idata, int *g_odata, int n) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce3(int *g_idata, int *g_odata, int n) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = ((i < n) ? g_idata[i] : 0) + ((i+blockDim.x < n) ? g_idata[i+blockDim.x] : 0);
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce4(int *g_idata, int *g_odata, int n) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = ((i < n) ? g_idata[i] : 0) + ((i+blockDim.x < n) ? g_idata[i+blockDim.x] : 0);
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// a warp has 32 threads
// this function reduce 5 synchthreads
__device__ void warpReduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce5(int *g_idata, int *g_odata, int n) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = ((i < n) ? g_idata[i] : 0) + ((i+blockDim.x < n) ? g_idata[i+blockDim.x] : 0);
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) warpReduce(sdata, tid);
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce6(int *g_idata, int *g_odata, int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + tid;
    sdata[tid] = ((i < n) ? g_idata[i] : 0) + ((i+blockDim.x < n) ? g_idata[i+blockDim.x] : 0);
    __syncthreads();
    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce7(int *g_idata, int *g_odata, int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockSize]; i+= gridSize;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

unsigned int gpu_sum_reduce(int* d_in, unsigned int d_in_len, decltype(reduce0) kernel, 
    unsigned int coarse_factor, unsigned int unroll_factor=1) {
	unsigned int total_sum = 0;

	// Set up number of threads and blocks
	// If input size is not power of two, the remainder will still need a whole block
	// Thus, number of blocks must be the least number of 2048-blocks greater than the input size
	unsigned int block_sz = MAX_BLOCK_SZ; // Halve the block size due to reduce3() and further 
											  //  optimizations from there
	// our block_sum_reduce()
	unsigned int max_elems_per_block = block_sz * coarse_factor; // due to binary tree nature of algorithm	
	unsigned int grid_sz = (d_in_len + max_elems_per_block - 1) / max_elems_per_block / unroll_factor;
    // std::cout << max_elems_per_block << "," << grid_sz << std::endl;

	// Allocate memory for array of total sums produced by each block
	// Array length must be the same as number of blocks / grid size
	int* d_block_sums;
	checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(int) * grid_sz));
	checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(int) * grid_sz));

	// Sum data allocated for each block
	kernel<<<grid_sz, block_sz, sizeof(int) * MAX_BLOCK_SZ>>>(d_in, d_block_sums, d_in_len);

	// Sum each block's total sums (to get global total sum)
	// Use basic implementation if number of total sums is <= 2048
	// Else, recurse on this same function
	if (grid_sz <= max_elems_per_block) {
		int* d_total_sum;
		checkCudaErrors(cudaMalloc(&d_total_sum, sizeof(int)));
		checkCudaErrors(cudaMemset(d_total_sum, 0, sizeof(int)));
		kernel<<<1, block_sz, sizeof(int) * MAX_BLOCK_SZ>>>(d_block_sums, d_total_sum, grid_sz);
		checkCudaErrors(cudaMemcpy(&total_sum, d_total_sum, sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_total_sum));
	} else {
		int* d_in_block_sums;
		checkCudaErrors(cudaMalloc(&d_in_block_sums, sizeof(int) * grid_sz));
		checkCudaErrors(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(int) * grid_sz, cudaMemcpyDeviceToDevice));
		total_sum = gpu_sum_reduce(d_in_block_sums, grid_sz, kernel, coarse_factor);
		checkCudaErrors(cudaFree(d_in_block_sums));
	}

	checkCudaErrors(cudaFree(d_block_sums));
	return total_sum;
}

void generate_input(int* input, unsigned int input_len) {
	for (unsigned int i = 0; i < input_len; ++i)
		input[i] = i;
}

unsigned int cpu_simple_sum(int* h_in, unsigned int h_in_len) {
	unsigned int total_sum = 0;

	for (unsigned int i = 0; i < h_in_len; ++i)
		total_sum = total_sum + h_in[i];

	return total_sum;
}

int main() {
	// Set up clock for timing comparisons
	std::clock_t start;
	double duration;

	for (int k = 27; k < 28; ++k) {
		unsigned int h_in_len = (1 << k);
		std::cout << "h_in_len: " << h_in_len << std::endl;
		int* h_in = new int[h_in_len];
		generate_input(h_in, h_in_len);

		// Set up device-side memory for input
		int* d_in;
		checkCudaErrors(cudaMalloc(&d_in, sizeof(int) * h_in_len));
		checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(int) * h_in_len, cudaMemcpyHostToDevice));

		// Do CPU sum for reference
		start = std::clock();
		int cpu_total_sum = cpu_simple_sum(h_in, h_in_len);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "CPU time: " << duration << " s"
                 << ", CPU result: " << cpu_total_sum                
                 << std::endl;

		int gpu_total_sum;
        // Do GPU scan
		start = std::clock();
		gpu_total_sum = gpu_sum_reduce(d_in, h_in_len, reduce0, 1);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "[reduce0]GPU time: " << duration << " s"
                << ", Match: " << (cpu_total_sum == gpu_total_sum)
                << ", Result: " << gpu_total_sum
                << std::endl;
        
        start = std::clock();
		gpu_total_sum = gpu_sum_reduce(d_in, h_in_len, reduce1, 1);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "[reduce1]GPU time: " << duration << " s"
                << ", Match: " << (cpu_total_sum == gpu_total_sum)
                << ", Result: " << gpu_total_sum
                << std::endl;
        
        start = std::clock();
		gpu_total_sum = gpu_sum_reduce(d_in, h_in_len, reduce2, 1);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "[reduce2]GPU time: " << duration << " s"
                << ", Match: " << (cpu_total_sum == gpu_total_sum)
                << ", Result: " << gpu_total_sum
                << std::endl; 
        
        start = std::clock();
		gpu_total_sum = gpu_sum_reduce(d_in, h_in_len, reduce3, 2);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "[reduce3]GPU time: " << duration << " s"
                << ", Match: " << (cpu_total_sum == gpu_total_sum)
                << ", Result: " << gpu_total_sum
                << std::endl;

        start = std::clock();
		gpu_total_sum = gpu_sum_reduce(d_in, h_in_len, reduce4, 2);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "[reduce4]GPU time: " << duration << " s"
                << ", Match: " << (cpu_total_sum == gpu_total_sum)
                << ", Result: " << gpu_total_sum
                << std::endl;
        
        start = std::clock();
		gpu_total_sum = gpu_sum_reduce(d_in, h_in_len, reduce5, 2);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "[reduce5]GPU time: " << duration << " s"
                << ", Match: " << (cpu_total_sum == gpu_total_sum)
                << ", Result: " << gpu_total_sum
                << std::endl;
        
        start = std::clock();
		gpu_total_sum = gpu_sum_reduce(d_in, h_in_len, reduce6, 2);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "[reduce6]GPU time: " << duration << " s"
                << ", Match: " << (cpu_total_sum == gpu_total_sum)
                << ", Result: " << gpu_total_sum
                << std::endl;
        
        for(int i = 1; i <= h_in_len / MAX_BLOCK_SZ / 2; i <<= 1) {
            start = std::clock();
            gpu_total_sum = gpu_sum_reduce(d_in, h_in_len, reduce7, 2, i);
            duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
            std::cout << "[reduce7_" << i << "]GPU time: " << duration << " s"
                    << ", Match: " << (cpu_total_sum == gpu_total_sum)
                    << ", Result: " << gpu_total_sum
                    << std::endl;
        }

		checkCudaErrors(cudaFree(d_in));
		delete[] h_in;

		std::cout << std::endl;
	}
}