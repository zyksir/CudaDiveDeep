/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
/** @brief This function was modified from GroupedGemm repository.
 *  Original Author: Shiqing Fan.
 *  Licience: Apache 2.0
 *  Source: https://github.com/fanshiqing/grouped_gemm
 *  */ 

#pragma once

#include "common/common.h"
#include <torch/extension.h>

using torch::Tensor;

namespace cuda_extension {

std::tuple<Tensor, Tensor, std::vector<Tensor>> moe_permute_topK_op(
    Tensor              input,
    Tensor              indices,
    int64_t             num_out_tokens,
    std::vector<Tensor> workspace,
    int64_t             max_expanded_token_num);

Tensor moe_recover_topK_op(
    Tensor input,
    Tensor row_id_map,
    Tensor prob_opt,
    int64_t num_tokens,
    int64_t num_topK);

}  // namespace grouped_gemm