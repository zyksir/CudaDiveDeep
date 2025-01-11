# Copyright (c) 2024 Aliyun Inc. All Rights Reserved.

r"""Unit test for Reduce Sum Kernel.

"""

import typing

import time
import torch
import pytest

from tests.util import use_deterministic_algorithms
from src.rope import RotaryEmbeddingMegatron


@pytest.fixture(params=[(4096, 1, 40, 128)])
def mock_input_tensor(request: typing.Any) -> torch.Tensor:
    """Mock tensors."""
    use_deterministic_algorithms(13)
    seq_len, batch_size, num_heads, hidden_size = request.param
    return torch.randn(seq_len, batch_size, num_heads, hidden_size, dtype=torch.float32, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rope(mock_input_tensor: torch.Tensor) -> None:
    """Testing build top experts implementation."""
    seq_len, batch_size, num_heads, hidden_size = mock_input_tensor.shape
    rotary_pos_emb_megatron = RotaryEmbeddingMegatron(kv_channels = hidden_size)
    rotary_pos_emb = rotary_pos_emb_megatron(seq_len)
    rotary_pos_cos, rotary_pos_sin = rotary_pos_emb_megatron.get_cos_sin(
        seq_len
    )
    rotary_pos_cos = rotary_pos_cos.to(mock_input_tensor.dtype)
    rotary_pos_sin = rotary_pos_sin.to(mock_input_tensor.dtype)
    output_megatron = RotaryEmbeddingMegatron.apply_rotary_pos_emb_megatron(
        mock_input_tensor, rotary_pos_emb
    )
    
    output_apex = RotaryEmbeddingMegatron.apply_rotary_pos_emb_apex(
        mock_input_tensor, rotary_pos_emb
    )
    
    output_te = RotaryEmbeddingMegatron.apply_rotary_pos_emb_transformer_engine(
        mock_input_tensor, rotary_pos_emb
    )
    
    output_flash = RotaryEmbeddingMegatron.apply_rotary_pos_emb_flash(
        mock_input_tensor, rotary_pos_cos, rotary_pos_sin
    )

    torch.testing.assert_close(output_megatron, output_apex)
    torch.testing.assert_close(output_megatron, output_te)
    torch.testing.assert_close(output_megatron, output_flash)
    
    s = time.perf_counter()
    torch.cuda.synchronize()
    for _ in range(100):
        RotaryEmbeddingMegatron.apply_rotary_pos_emb_megatron(
            mock_input_tensor, rotary_pos_emb
        )
    torch.cuda.synchronize()
    e = time.perf_counter()
    print(f"RotaryEmbeddingMegatron take {(e - s)*1000:.6f} ms while size = {mock_input_tensor.numel()}")
    
    s = time.perf_counter()
    torch.cuda.synchronize()
    for _ in range(100):
        RotaryEmbeddingMegatron.apply_rotary_pos_emb_apex(
            mock_input_tensor, rotary_pos_emb
        )
    torch.cuda.synchronize()
    e = time.perf_counter()
    print(f"RotaryEmbeddingApex take {(e - s)*1000:.6f} ms while size = {mock_input_tensor.numel()}")
    
    s = time.perf_counter()
    torch.cuda.synchronize()
    for _ in range(100):
        RotaryEmbeddingMegatron.apply_rotary_pos_emb_transformer_engine(
            mock_input_tensor, rotary_pos_emb
        )
    torch.cuda.synchronize()
    e = time.perf_counter()
    print(f"RotaryEmbeddingTE take {(e - s)*1000:.6f} ms while size = {mock_input_tensor.numel()}")
    
    s = time.perf_counter()
    torch.cuda.synchronize()
    for _ in range(100):
        RotaryEmbeddingMegatron.apply_rotary_pos_emb_flash(
            mock_input_tensor, rotary_pos_cos, rotary_pos_sin
        )
    torch.cuda.synchronize()
    e = time.perf_counter()
    print(f"RotaryEmbeddingFlash take {(e - s)*1000:.6f} ms while size = {mock_input_tensor.numel()}")

    
    
