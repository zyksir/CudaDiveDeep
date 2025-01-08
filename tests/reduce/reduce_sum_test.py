# Copyright (c) 2024 Aliyun Inc. All Rights Reserved.

r"""Unit test for Reduce Sum Kernel.

"""

import typing

import time
import torch
import pytest

from tests.util import use_deterministic_algorithms
from cuda_extension._ops import reduce_sum


def mock_tensors(tensor_size: int) -> torch.Tensor:
    """Mock tensors."""
    use_deterministic_algorithms(13)
    return torch.randn(tensor_size, dtype=torch.float, device="cuda")


@pytest.fixture(params=[1e3, 1e4, 1e5, 1e6])
def mock_input_tensor(request: typing.Any) -> torch.Tensor:
    """Mock tensors."""
    return mock_tensors(int(request.param))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_reduce_sum(mock_input_tensor: torch.Tensor) -> None:
    """Testing build top experts implementation."""
    output_pytorch = torch.sum(mock_input_tensor)
    output_my = reduce_sum(mock_input_tensor)
    torch.testing.assert_close(output_pytorch.view(-1), output_my.view(-1))
    
    s = time.perf_counter()
    torch.cuda.synchronize()
    for _ in range(100):
        _ = torch.sum(mock_input_tensor)
    torch.cuda.synchronize()
    e = time.perf_counter()
    print(f"torch sum take {(e - s)*1000:.6f} ms while size = {mock_input_tensor.numel()}")
    
    s = time.perf_counter()
    torch.cuda.synchronize()
    for _ in range(100):
        _ = reduce_sum(mock_input_tensor)
    torch.cuda.synchronize()
    e = time.perf_counter()
    print(f"reduce_sum take {(e - s)*1000:.6f} ms while size = {mock_input_tensor.numel()}")
    
    
