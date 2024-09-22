/* usage: 
nvcc -o reduce_cuda_mode_demo reduce_cuda_mode_demo.cu && ./reduce_cuda_mode_demo 
*/
#include <iostream>
#include <cuda.h>

__global__ void SimpleSumReductionKernel(float* input, float* output) {
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();

    }
    if (threadIdx.x == 0) {
    *output = input[0];
    }
    input[i] = 1;
}

__global__ void FixDivergenceKernel(float* input, float* output) {
    unsigned int i = threadIdx.x; //threads start next to each other
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) { // furthest element is blockDim away
        if (threadIdx.x < stride) { // 
            input[i] += input[i + stride]; // each thread adds a distant element to its assigned position
        }
        __syncthreads();

    }
    if (threadIdx.x == 0) {
    *output = input[0];
    }
    input[i] = 1;
}

#define BLOCK_DIM 1024
__global__ void SharedMemoryReduction(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /=2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (threadIdx.x == 0) {
        *output = input_s[0];
    }
}

int main() {
    // Size of the input data
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
	cudaEventCreate(&stop);
    float elapsed_ms;
    const int warmup = 0;
    const int repeated = 1000;
    const int size = 2048;
    const int bytes = size * sizeof(float);
    std::cout << "for size of " << size << std::endl;

    // Allocate memory for input and output on host
    float* h_input = new float[size];
    float* h_output = new float;

    // Initialize input data on host
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f; // Example: Initialize all elements to 1
    }

    // Allocate memory for input and output on device
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Profile SimpleSumReductionKernel Kernel
    for(int i = 0; i < warmup; i++) {
        SimpleSumReductionKernel<<<1, size / 2>>>(d_input, d_output);
    }
    cudaEventRecord(start);
    for(int i = 0; i < repeated; i++) {
        SimpleSumReductionKernel<<<1, size / 2>>>(d_input, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[SimpleSumReductionKernel] Output:" << *h_output << ", Elasped time: " << elapsed_ms << "ms" << std::endl;
    // [SimpleSumReductionKernel] Output:2048, Elasped time: 8.3671ms

    // Profile FixDivergenceKernel Kernel
    for(int i = 0; i < warmup; i++) {
        FixDivergenceKernel<<<1, size / 2>>>(d_input, d_output);
    }
    cudaEventRecord(start);
    for(int i = 0; i < repeated; i++) {
        FixDivergenceKernel<<<1, size / 2>>>(d_input, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[FixDivergenceKernel] Output:" << *h_output << ", Elasped time: " << elapsed_ms << "ms" << std::endl;
    // [FixDivergenceKernel] Output:3.31853e+25, Elasped time: 4.88858ms

    // Profile SharedMemoryReduction Kernel
    for(int i = 0; i < warmup; i++) {
        SharedMemoryReduction<<<1, size / 2>>>(d_input, d_output);
    }
    cudaEventRecord(start);
    for(int i = 0; i < repeated; i++) {
        SharedMemoryReduction<<<1, size / 2>>>(d_input, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[SharedMemoryReduction] Output:" << *h_output << ", Elasped time: " << elapsed_ms << "ms" << std::endl;
    // [SharedMemoryReduction] Output:2048, Elasped time: 4.5015ms

    // Cleanup
    delete[] h_input;
    delete h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventDestroy(start);
	cudaEventDestroy(stop);

    return 0;
}
