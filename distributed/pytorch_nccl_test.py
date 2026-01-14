"""
torchrun --nproc_per_node=4 pytorch_nccl_test.py
"""
import torch
import torch.distributed as dist
from typing import Callable

def benchmark_communication(comm_func: Callable, num_iterations: int = 100, num_warmup_iterations: int = 10) -> float:
    for _ in range(num_warmup_iterations):
        comm_func()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_iterations):
        comm_func()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_iterations # milliseconds

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    msg_size, dtype = int(4e9), torch.bfloat16
    x = torch.randn(msg_size, device=torch.cuda.current_device(), dtype=dtype)
    print(f"Rank {rank} of {world_size} initialized")

    all_reduce_time = benchmark_communication(lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM))
    bandwidth = msg_size * dtype.itemsize / all_reduce_time / 1e6 * 2 * (world_size - 1) / world_size  # GB/s
    print(f"Rank {rank} all_reduce time: {all_reduce_time:.2f} milliseconds, bandwidth: {bandwidth:.2f} GB/s")
    dist.destroy_process_group()
