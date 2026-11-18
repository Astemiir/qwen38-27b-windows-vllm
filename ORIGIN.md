# How this started

A short account of how one RTX 3090, one Qwen3.8-27B model and a Linux-only kernel library ended up 
in the same sentence, and what survived of the original plan.

## 1. It began with a ceiling

llama.cpp on this machine reached about 94 tok/s decode with Qwen3.8-27B at a 33k fill, and about 45
tok/s at 246k. That curve is not a tuning problem: decode re-reads the sequence's entire KV cache on
every step, so the speed is set by memory bandwidth divided by bytes per element. At 246k that is
gigabytes of reads per token. Nothing in the sampler, the batch size or the thread count moves that
number.

There are only two ways through a bandwidth wall: read fewer bytes, or produce more tokens per read.
The first means a narrower KV format. The second means speculative decoding — draft several tokens,
verify them in one pass.

## 2. Why vLLM

llama.cpp's fork has both (MTP heads, DFlash2), and we measured them. What vLLM has that mattered more
here is a first-class KV cache dtype abstraction: a plugin can register a format, and the scheduler,
the block manager and the attention backend will use it without patching the engine. On a 24 GB card
where the weights take 15.4 GB, the difference between a 65k window and a 262k window *is* the KV
format.

So the target became: 4-bit weights, the narrowest KV that still answers correctly, speculative
decoding on top, in vLLM, on Windows.

## 3. The first wall was not the model

`fp8` KV is the obvious 1-byte choice, and it is the format the DFlash2 drafter wanted. On an 8.6 card
the stock Triton backend refuses it — `fp8e4nv` needs SM89+ for the native scaled dot product. The one
backend that can run fp8 KV on Ampere is FlashInfer, which dequantizes with explicit scales.

FlashInfer had no Windows wheel. Not "an old one" — none. The JIT generator is Linux by construction:
it calls `c++`, emits `.so`, passes GCC flags, and cannot cope with the `:` in `E:\`. The published
build recipe assumed a Linux box with a CUDA toolchain.

## 4. The port

It took five classes of change, all mechanical once understood: nvcc for host files as well as CUDA
files, `.dll` instead of `.so`, short paths because nvcc's `GetLongPathName` fails on deep spec
directories, a `fi_max`/`fi_min` shim for MSVC, and rewriting `#if MODE == …` blocks as `if constexpr`
because MSVC expands macro arguments before the preprocessor sees the directive — 743 blocks across 651
files.

Then we prebuilt the kernels: 29 modules for `sm_80`, packaged as a JIT-cache wheel, so that a user
never needs nvcc, MSVC or a first-start compile. `sm_80` SASS loads on `sm_86` and `sm_89` per the
runtime's own compatibility table, which is exactly the range this document supports.

First working run: `FLASHINFER` backend, `fp8_e4m3` KV, 131 072 tokens, answer verified with a needle
test.

## 5. The drafter did not pay off

FlashInfer was ported *for* DFlash2, and DFlash2 lost. At 32k it gave 71.4 tok/s against MTP's 78.9; at
100k it collapsed to 424 tok/s prefill and 43 decode against 912 and 68. MTP with 3 tokens won, and
still does. The port stayed useful for a different reason — fp8 KV, and +18 % prefill at equal KV — but
the original justification is dead and the document says so.

## 6. Then the plugin fought back

Two failures, both quiet, both costing a factor of three:

The plugin writes its generated CUDA kernel through `os.fdopen(fd, "w")`. Without `encoding`, that uses
the Windows ANSI code page. The kernel source contains an arrow character, U+2192, which no Western
code page can encode. The write throws, the exception is caught, and one warning in the middle of a
300-line startup log says the fast kernel was skipped. The server starts and serves at 28 tok/s decode
instead of 85.5.

After that was fixed, decode got *worse* before it got better: 23.6 tok/s. The 3D split-KV scratch is
sized `max_num_seqs × 1` rows, but a speculative step carries `1 + num_speculative_tokens`. With
`--max-num-seqs 2` the launcher quietly used the 2D path. `--max-num-seqs 8` fixed it without touching
code.

With both gates open, int4 went from 858.7/28.0 to 1 178/85.5 at a 33k fill — and 262 144 tokens of
context fit in 6.8 GB.

## 7. What is left

The int8 path had no fast kernel at all, so we wrote one: unpack in the prefill kernel and hand the
result to FlashAttention, 828 → 1 172 tok/s. A segment knob for split-KV looked promising and did
nothing — decode there is arithmetic-bound, not parallel-limited, at ~2 % of available bandwidth.

fp8 per-token-head turned out to be a dead end: the guard that blocked it was too wide, but removing it
only gets you a tie with int8. Honest result, kept in the numbers.

What remains is a working configuration: 262k context, 1 178 tok/s prefill, 85.5 tok/s decode, verified
answers, on one consumer card, under Windows, installed from wheels.

## 8. If you are starting today

Take the installation path in `INSTALL.md` and skip the history. The two things worth knowing before
you tune anything:

1. **Verify that the fast kernel is loaded before you measure.** Both of the big regressions here were
   fallbacks with a log line and no error. One grep of the startup log saves a day.
2. **Compare at equal fill depth.** A 262k window measured at a 33k fill tells you nothing about the
   far end; the ranking of KV formats changes there. The tables in `PERFORMANCE.md` are split by depth
   for that reason.
