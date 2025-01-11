import torch
from cuda_extension._ops import _permute_fused_fan, _unpermute_fused_fan, _permute_fused_lisan

WORKSPACE_FAN = []
MAX_TOKEN_NUM_FAN = 0
def permute_fan(states: torch.Tensor, indices: torch.Tensor, num_out_tokens: int) -> torch.Tensor:
    global WORKSPACE_FAN, MAX_TOKEN_NUM_FAN
    assert indices.dim() == 2, f"expert indices to be 2-D tensor, got {indices.dim()}-D"
    assert states.dim() == 2, f"expert states to be 2-D tensor, got {states.dim()}-D"
    MAX_TOKEN_NUM_FAN = max(states.size(0), num_out_tokens) * indices.size(1)
    permuted_act, row_id_map, WORKSPACE_FAN = _permute_fused_fan(
        states,
        indices,
        num_out_tokens,
        WORKSPACE_FAN,
        MAX_TOKEN_NUM_FAN,
    )
    return permuted_act, row_id_map

def unpermute_fan(output_grad: torch.Tensor, row_id_map: torch.Tensor, num_in_tokens: int, num_topk: int) -> torch.Tensor:
    input_grad = _unpermute_fused_fan(output_grad, row_id_map, torch.Tensor([]), num_in_tokens, num_topk)
    return input_grad

WORKSPACE_LISAN = []
MAX_TOKEN_NUM_LISAN = 0
def permute_lisan(states: torch.Tensor, indices: torch.Tensor, num_out_tokens: int) -> torch.Tensor:
    global WORKSPACE_LISAN, MAX_TOKEN_NUM_LISAN
    assert indices.dim() == 2, f"expert indices to be 2-D tensor, got {indices.dim()}-D"
    assert states.dim() == 2, f"expert states to be 2-D tensor, got {states.dim()}-D"
    MAX_TOKEN_NUM_LISAN = max(states.size(0), num_out_tokens) * indices.size(1)
    permuted_act, row_id_map, WORKSPACE_LISAN = _permute_fused_lisan(
        states,
        indices,
        num_out_tokens,
        WORKSPACE_LISAN,
        MAX_TOKEN_NUM_LISAN,
    )
    return permuted_act, row_id_map

def clear_workspace_for_test():
    global WORKSPACE_LISAN, MAX_TOKEN_NUM_LISAN
    global WORKSPACE_FAN, MAX_TOKEN_NUM_FAN
    WORKSPACE_FAN, WORKSPACE_LISAN = [], []
    MAX_TOKEN_NUM_FAN, MAX_TOKEN_NUM_LISAN = 0, 0
