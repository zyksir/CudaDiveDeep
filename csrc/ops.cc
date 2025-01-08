#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <vector>

#include "reduce/reduce.h"
#include "permute/permute.h"
#include "permute/permute_lisan.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Example Code For Better Understanding of Cuda and Cutlass";
  m.def("reduce_sum",
        py::overload_cast<
          const at::Tensor&
        >(&cuda_extension::reduce_sum),
      py::arg("matrix"),
      "A Demo Reduce Sum Implementation");
  m.def(
      "_permute_fused_fan",
      py::overload_cast<
          at::Tensor,
          at::Tensor,
          int64_t,
          std::vector<at::Tensor>,
          int64_t>(&cuda_extension::moe_permute_topK_op),
      py::arg("input_data"),
      py::arg("indices"),
      py::arg("num_out_tokens"),
      py::arg("workspace"),
      py::arg("max_expanded_token_num"),
      "A kernel to sort and permute states");
  m.def(
      "_unpermute_fused_fan",
      py::overload_cast<at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t>(
          &cuda_extension::moe_recover_topK_op),
      py::arg("input_data"),
      py::arg("row_id_map"),
      py::arg("prob_opt"),
      py::arg("num_tokens"),
      py::arg("num_topk"),
      "A kernel to sort and permute states");
  m.def(
      "_permute_fused_lisan",
      py::overload_cast<
          at::Tensor,
          at::Tensor,
          int64_t,
          std::vector<at::Tensor>,
          int64_t>(&cuda_extension::moe_permute_lisan),
      py::arg("input_data"),
      py::arg("indices"),
      py::arg("num_out_tokens"),
      py::arg("workspace"),
      py::arg("max_expanded_token_num"),
      "A kernel to sort and permute states");
}
