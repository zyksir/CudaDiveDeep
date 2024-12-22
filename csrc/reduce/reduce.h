#pragma once
#include "common/common.h"

namespace cuda_extension {

at::Tensor reduce_sum(const at::Tensor& input_);

};