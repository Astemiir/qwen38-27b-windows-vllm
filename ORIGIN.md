# Why this exists, in plain words

We wanted faster answers from Qwen3.8-27B on one RTX 3090 under Windows. We ended up with a fast prompt
read and a 262k context instead. This page is the short honest version: what we did, what each step gave,
what did not work.

## The problem: the answer was slow

llama.cpp with this model gave us about 90 tok/s decode: 79–89 with MTP 3, 36 without. That is close to
what the card can do. Every new token re-reads the whole KV cache of the conversation, the weights are
15.4 GB, and the memory runs at 936 GB/s. The same engine cannot go faster.

So we needed an engine that produces more tokens per step. There are two ways: read fewer bytes per token
(narrower KV cache), or produce several tokens per read (speculative decoding: draft a few, verify them
in one pass).

## What we did, and what each step gave

| step | what it gave |
| ---- | ------------ |
| 1. Installed the native Windows vLLM 0.33 build (aivrar's kit). | An engine that already has speculative decoding, and a KV cache format that a plugin can register without patching the engine. The first int4 arm ran at 28 tok/s decode at a 262k window: the fast kernel was silently off, and fp8 KV was refused outright. |
| 2. Wanted fp8 KV, because the DFlash2 drafter card said that on Ampere it only works through FlashInfer. Ported FlashInfer to Windows: nvcc for host files, `.dll`, short paths, an MSVC shim, `if constexpr` instead of `#if MODE ==`. Prebuilt 29 kernels into a wheel. | fp8 KV works on the 3090: 131 072 window, 1 187 / 78.9 at a 33k fill, answer checked with a needle test. No nvcc or MSVC needed by a user. |
| 3. Measured the DFlash2 drafter against MTP. | DFlash2 lost: 71.6 against 78.9 tok/s at 32k, and at a 100k fill it fell to 424 prefill / 43 decode. We stayed on MTP. The reason we ported FlashInfer disappeared. |
| 4. Kept the port anyway. | At equal KV it gives +18 % prefill, and it is the only backend that can hold an fp8 cache on this card: 1 182 / 81.8 at 131k. |
| 5. Fixed the int4 plugin. Two gates, both silent: it wrote its generated kernel through the ANSI code page (the source has a `→` character, the write failed, the fast kernel was skipped), and its split-KV scratch was too small for 4 verify tokens. | 858 / 28.0 became 1 178 / 85.5 at a 33k fill, and 262 144 tokens of context fit in 6.8 GB of KV. |
| 6. Wrote an int8 prefill kernel: unpack the prefix, then hand it to FlashAttention. | 828 → 1 172 tok/s prefill. Decode did not change. Stock Triton now matches FlashInfer on prefill. |
| 7. Checked `fp8_per_token_head`: the guard blocked it, we widened the guard and measured. | It only ties int8 (1 185 / 76,8 against 1 172 / 76,0). Along the way we found that every per-token-head dtype answers the first request after a real start with noise in eager mode — a warm-up request or CUDA graphs fixes that, but the dtype still buys nothing, so the change was withdrawn. |

## The numbers: patched against stock

Stand: Qwen3.8-27B GPTQ W4A16 + MTP n3, one RTX 3090, vLLM 0.33, 500-token answer, thinking off. Decode is
counted from `completion_tokens` after the first token. Every arm passed a needle test.

| KV cache             | backend                  | max window | prefill 33k | decode | what changed                    |
| -------------------- | ------------------------ | ---------: | ----------: | -----: | ------------------------------- |
| `bfloat16`           | stock Triton             |     65 536 |       1 038 |   85.4 | nothing, this is a stock arm    |
| `bfloat16`           | FlashInfer               |     65 536 |       1 225 |   82.8 | +18.0 % prefill at equal KV     |
| `int8_per_token_head`| stock Triton             |    131 072 |         828 |   76.0 | nothing, this is the baseline   |
| `int8_per_token_head`| patched Triton           |    131 072 |       1 172 |   76.0 | +41.5 % prefill, decode same    |
| `fp8_per_token_head` | patched Triton           |    131 072 |       1 185 |   76.8 | +43.1 % prefill, decode same    |
| `fp8_e4m3`           | FlashInfer               |    131 072 |       1 182 |   81.8 | +42.8 % prefill, +7.6 % decode  |
| `int4_per_token_head`| stock Triton             |    262 144 |         859 |   28.0 | nothing, this is the baseline   |
| `int4_per_token_head`| patched Triton + plugin  |    262 144 |       1 178 |   85.5 | +37.2 % prefill, decode ×3.05   |
| `k8v4_per_token_head`| patched Triton + plugin  |    262 144 |       1 148 |   85.7 | 8-bit keys, 4-bit values        |

The same formats at deeper fills:

| fill  | int4      | k8v4      | fp8 + FlashInfer | patched int8 |
| ----- | --------- | --------- | ---------------- | ------------ |
| 33k   | 1 178 / 85.5 | 1 148 / 85.7 | 1 182 / 81.8 | 1 172 / 76.0 |
| 99k   | 977 / 72.6   | 919 / 70.9   | 912 / 68.4   | 962 / 51.9   |
| 246k  | 672 / 49.3   | 605 / 51.9   | window too small | window too small |

## What did not pay off

- **DFlash2.** It was the whole reason for the port and it is slower than MTP at every depth we tried.
- **`--max-num-seqs 8` on stock vLLM.** It does nothing there: 28.0 tok/s at 2 sequences, 29.3 at 8. The flag
  gates the plugin's launcher, not vLLM's own path.
- **`fp8_per_token_head`.** Ties int8 on our model, so there is nothing to win. Withdrawn. (Its first-request
  noise in eager mode turned out to be a property of all per-token-head dtypes, not of fp8 — a warm-up
  request or CUDA graphs removes it.)
- **The int8 split-KV knob.** Decode is limited by int8 arithmetic there, not by parallelism; more segments
  changed nothing.

## Conclusions

1. **The slow answer was an engine problem, not a model problem.** The same weights went from 28 to 85.5
   tok/s decode without touching the model.
2. **The int4 result belongs to the plugin's kernel, plus one flag.** Not to vLLM and not to the flag alone.
3. **Prefill stopped being a FlashInfer argument.** With our patch, stock Triton reads a prompt as fast
   (1 172 against 1 182). The often-quoted "+42.8 %" is measured against the unpatched stock path, and it
   shrinks to +0.9 % against patched int8.
4. **Decode follows the cache format, not the backend.** bf16 on stock Triton already gives 85.4; int4 with
   the plugin gives 85.5; int8 sits at 76.0 patched or not.
5. **FlashInfer still has exactly two jobs here:** fp8 KV, which no other backend can hold on this card, and
   +31.8 % decode at a 99k fill against patched int8, at the cost of −5.1 % prefill.
6. **For the longest context, int4 wins.** 262k in 6.8 GB against 8.8 GB for k8v4. k8v4 keeps more accurate
   keys and overtakes decode at the far end: 51.9 against 49.3 at 246k.

## If you start today

- **Fastest at 262k:** `int4_per_token_head` with the plugin and `--max-num-seqs 8`. 1 178 / 85.5.
- **Fewest moving parts:** `int8_per_token_head` with the int8 prefill patch. 1 172 / 76.0, no plugin, no
  FlashInfer.
- **You need fp8 KV:** the FlashInfer wheels in this repository. 1 182 / 81.8 at 131k.
- **Check the log before you measure.** Two of our three big losses were silent fallbacks with one line in
  a 300-line startup log.
- **Compare at equal fill depth.** The ranking changes at 246k; a 262k window measured at 33k tells you
  nothing about the far end.

## If we started again

If we had known on day one that we would stay on MTP, we would have started with int8 on stock vLLM and
probably would not have taken up FlashInfer at all. The port happened for DFlash2, and DFlash2 did not pay
off. Its value turned out different: fp8 KV and +18 % prefill. That value is real, it just is not the value
we were building.

## Patches in this repository

`FLASHINFER` (Windows wheels plus the build recipe), the `fp8` guard (one line in `triton_attn.py`), the
`int4`/`k8v4` plugin overlay, the `int8` prefill hook, and the `TRITON_ATTN` paths those hooks use. Each one
with file, line and replacement is in `UPSTREAM-FIXES.md`; the full measurement journal is `PERFORMANCE.md`.
