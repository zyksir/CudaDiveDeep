import torch
import time

def run_benchmark(
    iterations: int,
    warmup_iterations: int,
    func,
    *args,
    **kwargs
):
    for _ in range(warmup_iterations):
        func(*args, **kwargs)

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(iterations):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return (end_time - start_time) / iterations * 1000  # milliseconds
