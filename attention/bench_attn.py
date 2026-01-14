import torch
from flash_attn.cute import flash_attn_varlen_func as flash_attn_varlen_func_cute
# flash_attn_varlen_func_sgl = None
from sgl_kernel.flash_attn import flash_attn_varlen_func as flash_attn_varlen_func_sgl
# flash_attn_varlen_func_cute = None

def bench_function(func, *args, **kwargs):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warm-up (important!)
    for _ in range(5):
        func(*args, **kwargs)

    start_event.record()
    for _ in range(100):
        func(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / 10

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    # profiler = torch.profiler.profile(record_shapes=True, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_stack=True)
    # profiler.start()
    for seq_lens in [512, 1024, 1024 * 2, 1024 * 4, 1024 * 8]:
        print(f"seq_lens: {seq_lens}")
        num_heads = 24
        kv_heads = 24
        head_dim = 128

        q = torch.randn(1, seq_lens, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(1, seq_lens, kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(1, seq_lens, kv_heads, head_dim, dtype=dtype, device=device)
        o = torch.empty(1, seq_lens, num_heads, head_dim, dtype=dtype, device=device)
        q_lens = torch.tensor([seq_lens] * 1, dtype=torch.int32).to(device=q.device)
        k_lens = torch.tensor([seq_lens] * 1, dtype=torch.int32).to(device=k.device)

        output_cute = None
        output_sgl = None
        if flash_attn_varlen_func_cute is not None:
            output_cute, _ = flash_attn_varlen_func_cute(
                q=q.flatten(0, 1),  # type: ignore[no-untyped-call]
                k=k.flatten(0, 1),
                v=v.flatten(0, 1),
                seqused_q=None,
                seqused_k=None,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                softmax_scale=1.0,
                causal=False,
            )
            print("\tcute time: ", bench_function(flash_attn_varlen_func_cute, q=q.flatten(0, 1), k=k.flatten(0, 1), v=v.flatten(0, 1), seqused_q=None, seqused_k=None, cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True), cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True), softmax_scale=1.0, causal=False))

        
        if flash_attn_varlen_func_sgl is not None:
            output_sgl = flash_attn_varlen_func_sgl(
                q=q,  # type: ignore[no-untyped-call]
                k=k,
                v=v,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=seq_lens,
                max_seqlen_k=seq_lens,
                softmax_scale=1.0,
                causal=False,
                return_softmax_lse=False,
                ver=3,
            )
            print("\tsgl time: ", bench_function(flash_attn_varlen_func_sgl, q=q, k=k, v=v, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=seq_lens, max_seqlen_k=seq_lens, softmax_scale=1.0, causal=False, return_softmax_lse=False, ver=3))
        if output_cute is not None and output_sgl is not None:
            print("\tmax diff: ", (output_cute - output_sgl).abs().max())
    # profiler.stop()
    # profiler.export_chrome_trace("trace.json.gz")