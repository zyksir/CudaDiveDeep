# Copyright (c) 2024 Aliyun Inc. All Rights Reserved.

r"""Building using setuptools."""

import os
import pathlib
import subprocess  # noqa: S404

import setuptools
from torch.utils import cpp_extension

library_name = "cuda_extension"
current_path = pathlib.Path(__file__).parent
cutlass_path = current_path / "csrc" / "cutlass"

cxx_src_files = []

nvcc_src_files = [
    "csrc/reduce/reduce.cu",
]

csrc_dir = str(current_path / "csrc")

include_dirs = [csrc_dir]

common_extra_compile_args = [
    "-g",
    "-O3",
    "-std=c++17",
    "-DNDEBUG",
]

torch_includes = cpp_extension.include_paths()

torch_extra_compile_args = ["-isystem", "/usr/local/cuda/include"] + [
    item for path in torch_includes for item in ["-isystem", path]
]

cxx_extra_compile_args = (
    [
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wformat",
        "-Wformat-security",
        "-Wno-sign-compare",
        "-Wno-unused-parameter",
        "-Wno-missing-field-initializers",
        "-fstack-protector",
    ]
    + common_extra_compile_args
    + torch_extra_compile_args
)

nvcc_extra_compile_args = (
    [
        "-isystem",
        str(cutlass_path / "include"),
        "-isystem",
        str(cutlass_path / "tools/util/include"),
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-Xptxas",
        "-Werror",
        "-Xcompiler",
        "-Wall,-Wextra,-Werror,-Wformat,-Wformat-security,"
        "-Wno-sign-compare,-Wno-unused-parameter,-Wno-missing-field-initializers,"
        f'-fstack-protector,-DNDEBUG',
    ]
    + common_extra_compile_args
    + torch_extra_compile_args
)

extra_link_args = [
    "-lcuda",
]

setuptools.setup(
    name=library_name,
    ext_modules=[
        cpp_extension.CUDAExtension(
            f"{library_name}._ops",
            ["csrc/ops.cc"] + cxx_src_files + nvcc_src_files,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": cxx_extra_compile_args,
                "nvcc": nvcc_extra_compile_args,
            },
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    package_data={"": ["*.json"]},
)
