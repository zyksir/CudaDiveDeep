# Copyright (c) 2024 Aliyun Inc. All Rights Reserved.

r"""Unit test for Permute Kernel.
Permute is equal to argsort and index_select
"""

from typing import Any, Tuple

import time
import torch
import pytest

from tests.util import use_deterministic_algorithms
from tests.timer import Timer
from src.permute import permute_fan, permute_lisan, clear_workspace_for_test


@pytest.fixture(params=[(64, 4, 2048, 8192, 2048*4, torch.float)])
def mock_indices_and_scores(request: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mock tensors."""
    use_deterministic_algorithms(13)
    num_expert, topk, num_token, hidden_size, num_token_selected, dtype = request.param
    states = torch.randn(num_token, hidden_size, dtype=dtype, device="cuda")
    indices = torch.randint(0, num_expert, (num_token, topk), dtype=torch.int32, device="cuda")
    return states, indices, num_token_selected

def torch_permute(states: torch.Tensor, indices: torch.Tensor, num_out_token: int) -> torch.Tensor:
    permutated_indices = torch.argsort(indices.view(-1), stable=True)
    permutated_indices = permutated_indices[: num_out_token]
    permutated_folded_indices = permutated_indices // indices.size(1)
    permutated_states = states.index_select(0, permutated_folded_indices)
    return permutated_states, permutated_indices

def torch_unpermute(permuted_tokens, sorted_indices, num_in_token: int, num_topk: int = 1):
    """Unpermute the sorted tokens based on the indices.
    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The sorted indices tensor.
        merge_factor (int, optional): The merge factor. Defaults to 1.

    Returns:
        torch.Tensor: The unpermuted tensor.
    """
    unpermuted_tokens = torch.zeros(
        num_in_token*num_topk, 
        permuted_tokens.size(1), 
        dtype=permuted_tokens.dtype, 
        device="cuda"
    )
    unpermuted_tokens = unpermuted_tokens.index_copy(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, num_topk, permuted_tokens.size(-1))
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)
    return unpermuted_tokens

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_permute(mock_indices_and_scores: Tuple[torch.Tensor, torch.Tensor]) -> None:
    """Testing build top experts implementation."""
    clear_workspace_for_test()
    states, indices, num_token = mock_indices_and_scores
    torch.cuda.cudart().cudaProfilerStart()
    output_states_pytorch, _ = torch_permute(states, indices, num_token)
    output_states_fan, _ = permute_fan(states, indices, num_token)
    output_states_lisan, _ = permute_lisan(states, indices, num_token)
    torch.testing.assert_close(output_states_pytorch, output_states_fan)
    torch.testing.assert_close(output_states_pytorch, output_states_lisan)

    loop = 100
    timer = Timer()
    timer.start()
    for _ in range(loop):
        _ = torch_permute(states, indices, num_token)
    torch_timecost = timer.stop()
    print(f"permute sum take {torch_timecost:.6f} ms while size = {indices.shape}, {states.shape}, {num_token}")
    
    timer.start()
    for _ in range(loop):
        _ = permute_fan(states, indices, num_token)
    fan_timecost = timer.stop()
    print(f"permute_fan take {fan_timecost:.6f} ms while size = {indices.shape}, {states.shape}, {num_token}")
    
    timer.start()
    for _ in range(loop):
        _ = permute_lisan(states, indices, num_token)
    lisan_timecost = timer.stop()
    print(f"permute_lisan take {lisan_timecost:.6f} ms while size = {indices.shape}, {states.shape}, {num_token}")
    torch.cuda.cudart().cudaProfilerStop()