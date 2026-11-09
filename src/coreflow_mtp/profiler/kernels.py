"""AR paged attention measured with installed V1's actual FA2 entry point.

There is no artificial speculative expansion and num_splits=0 keeps native
automatic splitting. The default retains the original forward's input/setup
costs; an explicit prepared_kernel comparison can exclude these costs.
"""
import time

import numpy as np


class ARKernelProfiler:
    def __init__(self, geometry, device="cuda:0", warmup=5, repeat=10, blocks=100000,
                 timing_protocol="coreflow_forward"):
        import importlib.metadata
        import torch
        if importlib.metadata.version("vllm") != "0.18.1":
            raise ValueError("AR kernel adapter is pinned to installed vLLM 0.18.1")
        from vllm import _custom_ops as ops
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        self.torch, self.ops, self.attention = torch, ops, flash_attn_varlen_func
        self.g, self.device = geometry, device
        self.dtype = getattr(torch, geometry.dtype)
        self.warmup, self.repeat, self.blocks = warmup, repeat, blocks
        if timing_protocol not in {"coreflow_forward", "prepared_kernel"}:
            raise ValueError("invalid timing protocol")
        self.timing_protocol = timing_protocol
        if warmup < 0 or repeat < 1 or blocks < 2:
            raise ValueError("invalid timing/grid parameters")
        torch.cuda.set_device(device)
        shape = (blocks, geometry.block_size, geometry.num_kv_heads, geometry.head_size)
        self.k = torch.randn(shape, device=device, dtype=self.dtype)
        self.v = torch.randn_like(self.k)
        self.scale = torch.ones((), device=device, dtype=torch.float32)

    def measure(self, fn):
        torch = self.torch
        with torch.inference_mode():
            for _ in range(self.warmup):
                fn()
            samples = []
            for _ in range(self.repeat):
                torch.cuda.synchronize(self.device)
                begin = time.perf_counter()
                fn()
                torch.cuda.synchronize(self.device)
                samples.append(time.perf_counter() - begin)
        return {"seconds": float(np.median(samples)), "samples_seconds": samples}

    def attention_call(self, lengths, query_lengths=None):
        if self.timing_protocol == "coreflow_forward":
            return lambda: self._prepared_attention_call(lengths, query_lengths)()
        return self._prepared_attention_call(lengths, query_lengths)

    def _prepared_attention_call(self, lengths, query_lengths=None):
        torch, g = self.torch, self.g
        qlens = query_lengths or [1] * len(lengths)
        if len(lengths) != len(qlens) or any(q < 1 or q > c for c, q in zip(lengths, qlens)):
            raise ValueError("invalid causal query/history lengths")
        starts = np.cumsum([0] + list(qlens)).tolist()
        n = starts[-1]
        q = torch.randn((n, g.num_heads, g.head_size), device=self.device, dtype=self.dtype)
        table = torch.randint(0, self.blocks, (len(lengths), (max(lengths) + g.block_size - 1) // g.block_size),
                              device=self.device, dtype=torch.int32)
        seq = torch.tensor(lengths, device=self.device, dtype=torch.int32)
        cuq = torch.tensor(starts, device=self.device, dtype=torch.int32)
        decode = all(value == 1 for value in qlens)
        if decode:
            key = torch.randn((n, g.num_kv_heads, g.head_size), device=self.device, dtype=self.dtype)
            val = torch.randn_like(key)
            # Write the actual final token of each synthetic sequence's table.
            rows = torch.arange(len(lengths), device=self.device)
            slots = table[rows, (seq - 1) // g.block_size].long() * g.block_size + (seq - 1) % g.block_size
        def forward():
            if decode:
                self.ops.reshape_and_cache_flash(key, val, self.k, self.v, slots, "auto", self.scale, self.scale)
            self.attention(q=q, k=self.k, v=self.v, cu_seqlens_q=cuq,
                max_seqlen_q=max(qlens), seqused_k=seq, max_seqlen_k=max(lengths),
                softmax_scale=g.head_size ** -.5, causal=True, block_table=table,
                fa_version=2, num_splits=0)
        return forward

    def non_attention(self, batches):
        torch, g = self.torch, self.g
        device, dtype = self.device, self.dtype
        qsize, kvsize = g.num_heads * g.head_size, g.num_kv_heads * g.head_size
        weights = [torch.randn(shape, device=device, dtype=dtype) for shape in (
            (g.hidden_size, qsize + 2 * kvsize), (qsize, g.hidden_size),
            (g.hidden_size, 2 * g.intermediate_size), (g.intermediate_size, g.hidden_size))]
        norm_weight = torch.ones(g.hidden_size, device=device, dtype=dtype)
        def norm(x):
            return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(dtype) * norm_weight
        results = {}
        for b in batches:
            x = torch.randn((b, g.hidden_size), device=device, dtype=dtype)
            attn = torch.randn((b, qsize), device=device, dtype=dtype)
            cos = torch.randn((b, 1, g.head_size), device=device, dtype=dtype)
            sin = torch.randn_like(cos)
            def forward():
                qkv = norm(x) @ weights[0]
                q, k, _ = qkv.split([qsize, kvsize, kvsize], -1)
                for tensor, heads in ((q, g.num_heads), (k, g.num_kv_heads)):
                    tensor = tensor.view(b, heads, g.head_size)
                    a, z = tensor.chunk(2, -1)
                    rotated = torch.cat((-z, a), -1)
                    tensor * cos + rotated * sin
                hidden = norm(attn @ weights[1] + x)
                gate, up = (hidden @ weights[2]).chunk(2, -1)
                (torch.nn.functional.silu(gate) * up) @ weights[3]
            results[str(b)] = self.measure(forward)
        return results

    def swap(self, token_lengths):
        torch, g = self.torch, self.g
        result = []
        for tokens in token_lengths:
            count = tokens * g.kv_bytes_per_token
            gpu = torch.empty(count, dtype=torch.uint8, device=self.device)
            cpu = torch.empty(count, dtype=torch.uint8, device="cpu", pin_memory=True)
            gpu.zero_()
            out = self.measure(lambda: cpu.copy_(gpu, non_blocking=False))
            incoming = self.measure(lambda: gpu.copy_(cpu, non_blocking=False))
            result.append({"bytes": count, "tokens": tokens, "d2h": out, "h2d": incoming})
        return result
