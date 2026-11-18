# For the vLLM project

<!-- Post as: one issue with two sections, or split in two — the first is a patch, the second a question.
     Replace https://github.com/Astemiir/qwen38-27b-windows-vllm with the published repository URL first.
     Title: Triton backend rejects fp8_per_token_head on SM 8.x — the guard is one dtype predicate too wide
-->

Two items from running vLLM 0.33 with quantized KV caches on an RTX 3090 (`sm_86`) under Windows.
They are independent; the first comes with a patch, the second is a question.

## 1. `fp8_per_token_head` KV cache cannot start on SM 8.x

**File:** `vllm/v1/attention/backends/triton_attn.py:517` on `main` (525 in 0.33), `TritonAttentionImpl.__init__`.

```python
            if self.kv_cache_dtype.startswith("fp8") and not (
                current_platform.has_device_capability(89)
            ):
                suggested = (
                    "float16" if (cap is None or cap.to_int() < 80) else "bfloat16"
                )
                raise ValueError(
                    f"FP8 KV cache is not supported by the Triton attention backend "
```

With `--attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head` on a 3090 this raises at
startup, before any weight is loaded.

**Why the guard does not apply to this dtype.** The guard protects the native `fp8e4nv` path, where the
kernel performs the scaled dot product in fp8 and therefore needs SM89 tensor-core support.
`fp8_per_token_head` is not that: the cache is stored as `torch.uint8`, viewed as `float8_e4m3fn`, and
dequantized inside the Triton kernel with an explicit per-head, per-token scale — the same arithmetic
`int8_per_token_head` already uses, and which already runs on the same card through the same backend.
The condition is one dtype predicate too wide.

**Suggested change:**

```python
            if (
                self.kv_cache_dtype.startswith("fp8")
                and not self.kv_cache_dtype.endswith("per_token_head")
                and not (current_platform.has_device_capability(89))
            ):
```

**What it buys, stated honestly:** nothing in speed. With the guard widened the configuration starts,
answers a needle test correctly, and measures 1 185 / 76.8 tok/s against `int8_per_token_head`'s
1 172 / 76.0 at a 33 350-token fill — a tie. The native `fp8e4nv` path keeps its SM89 requirement. So this
is about a valid configuration being rejected, not about a faster one. If the maintainers prefer the
guard to stay as a hard rule for every fp8 dtype, closing this with "use int8_per_token_head or
FlashInfer" is a perfectly good answer; we would then just stop advertising the dtype.

## 2. Question: int4 KV with speculative decoding is pinned to the 2D path

**File:** `vllm/v1/attention/backends/triton_attn.py:164-175` on `main` (169-180 in 0.33).

```python
        max_query_len_3d = 1
        speculative_config = vllm_config.speculative_config
        # The INT4 launcher runs multi-query batches in 2D only.
        if (
            speculative_config is not None
            and kv_cache_spec.kv_quant_mode != KVQuantMode.INT4_PER_TOKEN_HEAD
        ):
```

The comment says the exclusion is deliberate, which is why this is a question and not a patch. What we
measure on a 3090 with MTP n3 and a 6.8 GB int4 cache, same model and same prompt:

| run                                                          | decode, tok/s |
| ------------------------------------------------------------ | ------------- |
| stock vLLM, `--max-num-seqs 2`                               | 28.0          |
| stock vLLM, `--max-num-seqs 8`                               | 29.3          |
| third-party int4 launcher (`vllm-quantpack`), `--max-num-seqs 2` | 23.6      |
| third-party int4 launcher (`vllm-quantpack`), `--max-num-seqs 8` | 85.5      |

On stock vLLM the batch size changes nothing. The dispatch in `v1/attention/ops/int4_per_token_head.py`
also requires `max_seqlen_q <= 1`, so with `1 + num_speculative_tokens` = 4 query rows per step the 2D
kernel runs at any `max_num_seqs`; the 3D branch that the scratch above sizes is never reached. The
23.6 → 85.5 pair is the third-party launcher, which overrides that dispatch and does split-KV over
multi-query batches — its scratch holds one row per sequence, so below `1 + num_speculative_tokens` it
declines and falls back. Two questions:

1. Is "multi-query batches in 2D only" a temporary limitation of the int4 launch path, or a permanent
   property? If permanent, a `logger.warning_once` — "int4 decode carries `1 + num_speculative_tokens`
   rows per sequence and stays on the 2D kernel" — would have saved us a day of measuring the wrong thing.
2. If permanent, is a split-KV path for int4 under speculation planned, or is the 2D kernel the intended
   answer and a third-party launcher the only way around it?

Background, if useful: `https://github.com/Astemiir/qwen38-27b-windows-vllm` — `PERFORMANCE.md` has the full table of KV formats on this card, and
`UPSTREAM-FIXES.md` §2 is the same text with the measurements attached. The int4/k8v4 kernels themselves
come from a third-party plugin (`vllm-quantpack`), not from vLLM.
