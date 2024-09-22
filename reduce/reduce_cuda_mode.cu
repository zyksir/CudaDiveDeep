/* usage: 
nvcc -o reduce_cuda_mode reduce_cuda_mode.cu && ./reduce_cuda_mode 
ncu -o reduce_perf --kernel-id ::regex:"SharedMemoryReduction|CoarsenedReduction":1 ./reduce_cuda_mode
*/
#include <iostream>
#include <cuda.h>

#define BLOCK_DIM 1024
__global__ void SharedMemoryReduction(float* input, float* output, int n) {
    __shared__ float input_s[BLOCK_DIM]; 
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; // index within a block
    unsigned int t = threadIdx.x; // global index

    // Load elements into shared memory
    if (idx < n) {
        input_s[t] = input[idx];
    } else {
        input_s[t] = 0.0f;
    }
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride && idx + stride < n) {
            input_s[t] += input_s[t + stride];
        }
        __syncthreads();
    }

    // Reduction across blocks in global memory
    // needs to be atomic to avoid contention
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

#define COARSE_FACTOR 2
__global__ void CoarsenedReduction(float* input, float* output, int size) {
    __shared__ float input_s[BLOCK_DIM];

    unsigned int i = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;
    unsigned int t = threadIdx.x;
    float sum = 0.0f;

    // Reduce within a thread
    for (unsigned int tile = 0; tile < COARSE_FACTOR; ++tile) {
        unsigned int index = i + tile * blockDim.x;
        if (index < size) {
            sum += input[index];
        }
    }

    input_s[t] = sum;
    __syncthreads();
    
    //Reduce within a block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
        __syncthreads();
    }

    //Reduce over blocks
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

int main() {
    // Size of the input data
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
	cudaEventCreate(&stop);
    float elapsed_ms;
    const int repeated = 1000;
    const int size = 1000000;
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

    // Profile SharedMemoryReduction Kernel
    *h_output = 0;
    cudaMemcpy(d_output, h_output, sizeof(float), cudaMemcpyHostToDevice);
    int numBlocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
    SharedMemoryReduction<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, size);
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[SharedMemoryReduction] Output:" << *h_output;

    cudaEventRecord(start);
    for(int i = 0; i < repeated; i++) {
        SharedMemoryReduction<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << ", Elasped time: " << elapsed_ms << "ms" << std::endl;
    // [SharedMemoryReduction] Output:1e+06, Elasped time: 17.4511ms

    // Profile CoarsenedReduction Kernel
    *h_output = 0;
    cudaMemcpy(d_output, h_output, sizeof(float), cudaMemcpyHostToDevice);
    numBlocks = (size + BLOCK_DIM * COARSE_FACTOR - 1) / (BLOCK_DIM * COARSE_FACTOR);
    CoarsenedReduction<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, size);
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[CoarsenedReduction] Output:" << *h_output;

    cudaEventRecord(start);
    for(int i = 0; i < repeated; i++) {
        CoarsenedReduction<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << ", Elasped time: " << elapsed_ms << "ms" << std::endl;
    // [CoarsenedReduction] Output:1e+06, Elasped time: 10.7805ms

    // Cleanup
    delete[] h_input;
    delete h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventDestroy(start);
	cudaEventDestroy(stop);

    return 0;
}
