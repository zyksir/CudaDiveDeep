#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <vector>

#include "reduce/reduce.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Example Code For Better Understanding of Cuda and Cutlass";
  m.def("reduce_sum",
        py::overload_cast<
          const at::Tensor&
        >(&cuda_extension::reduce_sum),
      py::arg("matrix"),
      "A Demo Reduce Sum Implementation");
}
