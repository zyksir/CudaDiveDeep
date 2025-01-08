/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "permute.h"

#include <torch/torch.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ATen/cuda/CUDAContext.h"

#include "cutlass/arch/memory.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"


using torch::Tensor;

namespace cuda_extension {

// map the sorted idx to origin idx.
static __global__ void moe_permute_topK_lisan_row_map(
    const int *sorted_row_id,
    int *row_id_map,
    const int num_rows,
    const int num_topK,
    const int num_out_tokens) {
    // Each block corresponds to one source token
    // row_id_map[num_topK][num_rows]
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = bid * blockDim.x + tid;

    if (idx >= num_rows * num_topK)
        return;

    int source_row = sorted_row_id[idx];
    int source_token_id = source_row / num_topK;
    int source_topK_id = source_row % num_topK;

    if (idx >= num_out_tokens) {
        row_id_map[source_topK_id * num_rows + source_token_id] = -1;
    } else {
        row_id_map[source_topK_id * num_rows + source_token_id] = idx;
    }
}


/// @brief This function was modified from GroupedGemm repository.
/// Original Author: Shiqing Fan.
/// Licience: Apache 2.0
/// Source: https://github.com/fanshiqing/grouped_gemm
// if T == TCompute, We DONOT Need a converter!
// This Optimization is useless according to the exp result
template <typename T,
          int kElementsPerAccess>
__global__ void moe_permute_topK_lisan_kernel(
    const T *input,
    T *output,
    const int *row_id_map,
    const int num_rows,
    const int num_topK,
    const int num_cols) {
    // using FragmentLoadStore = cutlass::Array<T, kElementsPerAccess>;
    // using FragmentCompute = cutlass::Array<T, kElementsPerAccess>;
    // cutlass::NumericArrayConverter<T, T, kElementsPerAccess> src_converter;
    // cutlass::NumericArrayConverter<T, T, kElementsPerAccess> dst_converter;
    const int source_token = blockIdx.x;
    const int tid = threadIdx.x;
    // FragmentLoadStore frag_load_store;
    float4 frag_load_store;
    const T *source_row_ptr = input + source_token * num_cols;
    for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess) {
        // cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
        //     frag_load_store, (source_row_ptr + i), true);
        frag_load_store = *(float4 *)(source_row_ptr + i);
        // FragmentCompute frag_sum;
        // frag_sum = src_converter(frag_load_store);
        int index = source_token;
        for (int k = 0; k < num_topK; k++) {
            int dest_row = row_id_map[index];
            index += num_rows;
            if (dest_row == -1)
                continue;
            T *dest_row_ptr = output + dest_row * num_cols;
            // frag_load_store = dst_converter(frag_sum);
            // *(float4 *)(dest_row_ptr + i) = *(float4 *)(frag_load_store.data());
            *(float4 *)(dest_row_ptr + i) = frag_load_store;
        }
    }
}


template <typename T, int kElementsPerAccess>
void moe_permute_topK_lisan_kernel_launcher(
    const T *input,
    T *output,
    const int *sorted_row_id,
    int *row_id_map,
    const int num_rows,
    const int num_topK,
    const int num_cols,
    const int num_out_tokens,
    cudaStream_t stream) {
    // permute_topK fwd
    int threads = 64;
    int blocks = (num_rows * num_topK + threads - 1) / threads;
    moe_permute_topK_lisan_row_map<<<blocks, threads, 0, stream>>>(
        sorted_row_id,
        row_id_map,
        num_rows,
        num_topK,
        num_out_tokens);

    blocks = num_rows;
    threads = std::min(num_cols / kElementsPerAccess, 1024);
    moe_permute_topK_lisan_kernel<T, kElementsPerAccess><<<blocks, threads, 0, stream>>>(
        input,
        output,
        row_id_map,
        num_rows,
        num_topK,
        num_cols);
}


/// @brief This function was modified from GroupedGemm repository.
/// Original Author: Shiqing Fan.
/// Licience: Apache 2.0
/// Source: https://github.com/fanshiqing/grouped_gemm
/// modify this function in the case of not using FP8
std::tuple<Tensor, Tensor, std::vector<Tensor>> moe_permute_lisan(
    Tensor              input,
    Tensor              indices,
    int64_t             num_out_tokens,
    std::vector<Tensor> workspace,
    int64_t             max_expanded_token_num) {
    const int num_tokens = input.size(0);
    const int num_cols = input.size(1);
    const int num_topK = indices.size(1);

    // allocate the temp variable in the first run
    if (workspace.empty()) {
        auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);

        Tensor sorted_indices = torch::empty(max_expanded_token_num, options);
        Tensor row_id = torch::range(0, max_expanded_token_num - 1, 1, options);
        Tensor sorted_row_id =
            torch::empty(max_expanded_token_num, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

        size_t temp_storage_bytes = 0;
        int *temp_ptr = nullptr;
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                        temp_ptr, temp_ptr,
                                        temp_ptr, temp_ptr, max_expanded_token_num);
        Tensor temp_storage =
            torch::empty(temp_storage_bytes, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

        workspace.push_back(sorted_indices);
        workspace.push_back(row_id);
        workspace.push_back(sorted_row_id);
        workspace.push_back(temp_storage);
    }

    int *indices_ptr = get_ptr<int>(indices);
    int *sorted_indices_ptr = get_ptr<int>(workspace[0]);
    int *row_id_ptr = get_ptr<int>(workspace[1]);
    int *sorted_row_id_ptr = get_ptr<int>(workspace[2]);

    void *d_temp_storage = get_ptr<void>(workspace[3]);
    size_t temp_storage_bytes = std::numeric_limits<size_t>::max();

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    indices_ptr, sorted_indices_ptr,
                                    row_id_ptr, sorted_row_id_ptr, num_tokens * num_topK);

    // activations type
    const at::ScalarType _st = input.scalar_type();

    // Output buffer alloc
    num_out_tokens = (num_out_tokens > 0) ? num_out_tokens : num_tokens * num_topK;
    Tensor permuted_output =
        torch::empty({num_out_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    Tensor row_id_map =
        torch::empty({num_tokens * num_topK}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st) {
    case at::ScalarType::Float: {
        using dType = float;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_lisan_kernel_launcher<dType, 4>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream);

        break;
    }
    case at::ScalarType::Half: {
        using dType = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_lisan_kernel_launcher<dType, 8>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream);

        break;
    }
    case at::ScalarType::BFloat16: {
        using dType = cutlass::bfloat16_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_lisan_kernel_launcher<dType, 8>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream);

        break;
    }
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return std::make_tuple(permuted_output, row_id_map, workspace);
}
}  // namespace grouped_gemm