# Performance

All numbers are measured on one RTX 3090 (24 GB, `sm_86`) under Windows 11, Qwen3.8-27B GPTQ W4A16
with the MTP head, `num_speculative_tokens = 3`, `--max-num-batched-tokens 2048`.

**Prefill** is prompt tokens divided by time-to-first-token. **Decode** is
`(completion_tokens − 1)` divided by the span from the first to the last token, using the count the
server reports — not the number of chunks, which overstates decode by counting speculative drafts.

## How to read the table

Two rows can differ in three things at once: the KV format, the attention backend and whether our
patch is applied. Every row below is labelled with all three, because the interesting mistakes in this
stack come from reading a two-variable result as a one-variable effect.

## Main table — 33k fill (33 367 prompt tokens)

| KV format           | attention backend | patch              | max window | KV under that window | prefill, tok/s | decode, tok/s |
| ------------------- | ----------------- | ------------------ | ---------- | -------------------- | -------------- | ------------- |
| bf16                | Triton (stock)    | —                  | 65 536     | —                    | 1 038          | 85.4          |
| bf16                | FlashInfer        | —                  | 65 536     | —                    | **1 225**      | 82.8          |
| int8 per-token-head | Triton (stock)    | —                  | 131 072    | —                    | 828            | 76.0          |
| int8 per-token-head | Triton            | int8 prefill hook  | 131 072    | —                    | **1 172**      | 76.0          |
| fp8 per-token-head  | Triton            | hook + guard patch | 131 072    | 6.5 GB               | 1 185          | 76.8          |
| fp8 e4m3            | FlashInfer        | —                  | 131 072    | 6.5 GB               | 1 182          | 81.8          |
| int4 per-token-head | Triton (stock)    | —                  | 262 144    | —                    | 858.7          | 28.0          |
| int4 per-token-head | Triton            | plugin patch       | 262 144    | 6.8 GB               | **1 178**      | **85.5**      |
| k8v4 per-token-head | Triton            | plugin patch       | 262 144    | 8.8 GB               | 1 148          | 85.7          |

"KV under that window" is the `--kv-cache-memory-bytes` the run was started with; a dash means the run
did not pin it, and the engine sized the cache from what was left of VRAM.

Gains, each against its own baseline:

| what                            | prefill | decode         |
| ------------------------------- | ------- | -------------- |
| int4 stock → int4 patched       | +37.2 % | ×3.05 (+205 %) |
| int8 stock → int8 hook          | +41.5 % | 0 %            |
| int8 stock → fp8 per-token-head | +43.1 % | +1.1 %         |
| int8 stock → fp8 + FlashInfer   | +42.8 % | +7.6 %         |
| int8 hook → fp8 + FlashInfer    | +0.9 %  | +7.6 %         |

The last two lines are the same measurement against two different baselines. Against the *shipping*
stock kernel FlashInfer looks like a 42.8 % prefill win; once the stock int8 path is patched, that win
is 0.9 % — the prefill gain was never the backend, it was the missing fast kernel. What FlashInfer
does still buy is decode: +7.6 % over patched int8, and it is the only backend that can run fp8 KV at
all on `sm_86`.

## Deep fills

These are the numbers that decide usability, because a 262k window that is only fast at 32k is not a
262k window. Same probe, longer prompt.

| fill, tokens | int4 prefill / decode | k8v4 prefill / decode | fp8+FlashInfer prefill / decode | int8 hook prefill / decode |
| ------------ | --------------------- | --------------------- | ------------------------------- | -------------------------- |
| 33 367       | 1 178 / 85.5          | 1 148 / 85.7          | 1 182 / 81.8                    | 1 172 / 76.0               |
| 97 700       | **977.1 / 72.6**      | 918.9 / 70.9          | 912 / 68.4                      | 961.5 / 51.9               |
| 245 746      | 671.5 / 49.25         | 604.6 / 51.89         | — (window)                      | — (window)                 |

Reading it:

- **int4 wins prefill at every depth** and is the only format that holds 262k in 6.8 GB.
- **k8v4 overtakes decode only at the far end**: −0.2 % at 33k, −2.4 % at 98k, **+5.4 %** at 246k.
- **fp8 + FlashInfer loses to both at 98k** (−6.7 % prefill, −5.8 % decode versus int4). Its +31.8 %
  decode advantage over the int8 Triton path does not make it the fastest choice — it makes it the
  fastest *fp8* choice.
- decode falls with depth in every row. That is the memory bus, not a bug: every step re-reads the
  sequence's whole KV cache, so at 246k each token costs several gigabytes of reads however the kernel
  is written.

## int4 versus k8v4 — which to use

|                       | int4 per-token-head                                               | k8v4 per-token-head                                        |
| --------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------- |
| bits per KV element   | 4-bit codes, two per byte, one scale per head per token           | 8-bit codes, one per byte, one scale per group             |
| 262 144 tokens fit in | 6.8 GB                                                            | 8.8 GB                                                     |
| prefill               | faster everywhere (+2.6 % at 33k, +6.3 % at 98k, +11.1 % at 246k) | —                                                          |
| decode                | faster up to ~230k                                                | +5.4 % at 246k                                             |
| quantization error    | 16 levels per group                                               | 256 levels per group                                       |
| recommended for       | 262k windows, agent turns, long documents                         | the last 30k of a 262k window, or when you have spare VRAM |

Neither is lossless. int4 stores 16 levels with a scale per head per token, which was enough to return
the planted anchor correctly in every run we checked, but it is the coarser of the two. If your workload
is "read 200k tokens and quote exact figures", k8v4's extra 2 GB of VRAM buys 16× the levels. If your
workload is "fit 262k on a 24 GB card at all", int4 is the only one that fits with headroom.

## The other engine, for orientation

llama.cpp is the natural reference point on this machine, and we measured it with the same probe. It is
**not** a like-for-like comparison: different weights (GGUF Q4_K_M abliterated versus GPTQ W4A16), a
different KV format (q4_0), a different sampler path. Treat the table as orientation, not as a verdict.

| fill, tokens | vLLM int4 patched | llama.cpp b12413 |
| ------------ | ----------------- | ---------------- |
| 33 367       | 1 178 / 85.5      | 1 326 / 94.2     |
| 245 746      | **671.5 / 49.25** | 606.9 / 44.70    |

At 33k llama.cpp is ahead on both. At 246k vLLM is ahead on both for the first time (+10.6 % prefill,
+10.2 % decode) — which is the shape you expect when the KV cache is 6.8 GB compressed instead of a
q4_0 layout: the advantage grows with depth, and 33k does not show it.

## Accuracy checks

Speed without a correctness check is a number, not a result. The check used here: a unique sentence is
planted at the start of the prompt (`anchor: КРЕМНИЕВЫЙ-ЯКОРЬ-73`, `tile: 487291`), the same prompt ends
with a question about it, and the reply must return both values.

| format             | anchor at a 33 350-token fill |
| ------------------ | ----------------------------- |
| int4               | pass                          |
| k8v4               | pass                          |
| fp8 per-token-head | pass                          |
| int8               | pass                          |

Two limits of this check, stated plainly:

- The deeper runs (97.7k, 193.9k, 245.7k) were measured for speed only — the anchor was not repeated
  there. "Passes at 33k" is not evidence about 246k.
- A needle test proves the cache is not garbage. It does not measure degradation: a task that needs an
  exact 6-digit figure from deep in a 200k prompt is exactly where `int4`'s 16 levels can lose to
  `k8v4`'s 256, and this table would not show it.
