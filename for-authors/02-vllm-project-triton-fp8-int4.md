# For the vLLM project

<!-- Posted as vllm-project/vllm#68410. Item 1 was withdrawn there after the 9B result; this file is the
     final state, not the original text. Item 2 also carries a correction comment (the 23.6 → 85.5 pair
     belongs to the plugin's launcher, not to vLLM's 3D branch).
     Title as posted: Triton backend rejects fp8_per_token_head on SM 8.x — the guard is one dtype predicate too wide
-->

Two items from running vLLM 0.33 with quantized KV caches on an RTX 3090 (`sm_86`) under Windows.
The first is withdrawn, the second is still a question.

## 1. `fp8_per_token_head` KV cache cannot start on SM 8.x — withdrawn

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

**Why we thought the guard did not apply.** The guard protects the native `fp8e4nv` path, where the
kernel performs the scaled dot product in fp8 and therefore needs SM89 tensor-core support.
`fp8_per_token_head` is not that: the cache is stored as `torch.uint8`, viewed as `float8_e4m3fn`, and
dequantized inside the Triton kernel with an explicit per-head, per-token scale — the same arithmetic
`int8_per_token_head` already uses, and which already runs on the same card through the same backend.

**What we measured, and why this item is withdrawn.** We widened the condition with
`and not self.kv_cache_dtype.endswith("per_token_head")`. On a Qwen3.8-27B (head_dim 256, 4 KV heads) that
gives correct answers in three controls — MTP 3 with CUDA graphs, no speculation, `--enforce-eager` — with
the needle found at 33 350 tokens each time, at 1 185 / 76.8 against `int8_per_token_head`'s 1 172 / 76.0,
a tie with no speed to win. Then the maintainer of the Windows build ran the same widened check on a
Qwen3.5-9B GPTQ-4bit in eager mode: the engine starts, the text comes out as
`'Sunz $ initial … 执 moments …'`, and the needle misses.

Two shapes, two outcomes means the dequant is shape-dependent and not reliable. A `ValueError` at startup
is a better failure mode than quietly wrong tokens, so **we withdraw the patch and the suggestion**. If
you ever chase it, the useful data point is the shape difference: 27B with head_dim 256 and 4 KV heads is
fine, 9B GPTQ-4bit is not.

**Suggested change:** none.

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
