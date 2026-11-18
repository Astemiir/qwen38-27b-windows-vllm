# Qwen3.8-27B on one RTX 3090 under native Windows: 262k context, 1 178 tok/s prefill

This repository is the missing pieces for running a Qwen3.8-27B model on a single 24 GB Ampere RTX3090 card 
under **native Windows** (no WSL, no Linux container) at a **262 144-token context window**, 
with speculative decoding: a Windows build of **FlashInfer 0.7.2.post1** with prebuilt `sm_80` kernels, 
and a small patch set for the vLLM Windows build and for the quantization plugin it ships with.

Two things are true at once, and it is worth being clear about both:

- **Upstream Windows support stops well short of this.** The vLLM Windows build works, but it has no
  FlashInfer, so `--kv-cache-dtype fp8` is unavailable and 131k does not fit at `bfloat16`.
- **The fast int4 path is in a third-party plugin that disables itself on Windows.** It writes a
  generated CUDA source file without an explicit encoding; on any Western Windows code page the write
  fails and the engine falls back to its slow stock kernel. The only trace is one warning buried in a
  300-line startup log — no error, no failed request, just three times slower decode.

We fixed the second one, built the first one, and measured the result. That is what is here.

## What you get

Single RTX 3090 (24 GB), Qwen3.8-27B 4-bit GPTQ with MTP speculative decoding, 500-token answers,
`vLLM 0.33`, prefilling a 33k prompt unless stated otherwise:

| KV cache format       | backend                |     context | prefill tok/s | decode tok/s |
| --------------------- | ---------------------- | ----------: | ------------: | -----------: |
| `int4_per_token_head` | patched Triton         | **262 144** |     **1 178** |     **85.5** |
| `k8v4_per_token_head` | patched Triton         |     262 144 |         1 148 |         85.7 |
| `fp8_e4m3`            | FlashInfer (this repo) |     131 072 |         1 182 |         81.8 |
| `int8_per_token_head` | patched Triton         |     131 072 |         1 172 |         76.0 |
| `int8_per_token_head` | stock Triton           |     131 072 |           828 |         76.0 |
| `int4_per_token_head` | stock Triton           |     262 144 |           859 |         28.0 |

The bottom line for a reader who wants one sentence: **int4 with the patched kernel gives 262k on a
24 GB card at 1 178 tok/s prefill and 85.5 tok/s decode — 37 % more prefill and 3× the decode of the
same card without the fixes.** If you only want 131k and no patching at all, the FlashInfer wheels in
this repo get you `fp8` at 1 182 / 81.8.

Longer prompts, same card and settings:

| KV cache format         | context | fill | prefill tok/s | decode tok/s |
| ----------------------- | ------: | ---: | ------------: | -----------: |
| `int4_per_token_head`   | 262 144 |  99k |           977 |         72.6 |
| `int4_per_token_head`   | 262 144 | 246k |           672 |         49.3 |
| `k8v4_per_token_head`   | 262 144 |  99k |           919 |         70.9 |
| `k8v4_per_token_head`   | 262 144 | 246k |           605 |         51.9 |
| `fp8_e4m3` (FlashInfer) | 131 072 |  99k |           912 |         68.4 |
| `int8_per_token_head`   | 131 072 |  99k |           962 |         51.9 |

## Which cards this works on

The FlashInfer kernels here are SASS for `sm_80`. CUDA runs `sm_80` binaries on `sm_86` and `sm_89`
(FlashInfer's own compatibility score for both is 1.0), so the wheels cover Ampere and Ada consumer
cards. They do **not** run on Hopper or Blackwell.

| compute capability | cards                                     | this repo                                                            |
| ------------------ | ----------------------------------------- | -------------------------------------------------------------------- |
| 8.0                | A100 40/80 GB                             | kernels run; vLLM Windows build is the limiting factor               |
| 8.6                | RTX 3090, 3090 Ti, 3080 Ti/3080/3070/3060 | **tested here** (3090 24 GB)                                         |
| 8.9                | RTX 4090/4080/4070, L40, RTX 6000 Ada     | kernels run; native fp8 is also available, so FlashInfer is optional |
| 9.0 and 10.x/12.x  | H100, B200, RTX 50xx                      | **not covered** — do not install these wheels                        |

VRAM: the recipe below targets a **single 24 GB card**. A 27B model in 4 bits is ~15.4 GB of weights,
which leaves ~7 GB for the KV cache — enough for 262 144 tokens at `int4` (6.8 GB) or 131 072 at
`fp8`/`int8` (6.5 GB). Cards with less than 24 GB cannot hold this model at these context lengths.

## Install in two steps

Everything that needs patching ships already patched, in one folder — `bundle/` in this repository and
the same files attached to the release. Only the vLLM kit and the weights come from elsewhere.

```bat
:: 1) the vLLM 0.33 Windows kit (aivrar's release, v0.33.0-win-cu130) - run its install.bat once

:: 2) this repo's bundle: four wheels + one guard fix, no downloads, no hand-editing
"<repo>\bundle\install.bat" "D:\vllm-windows-build-0.33.0-win-cu130"
```

Expected last line: `ok`. Then download a GPTQ W4A16 checkpoint with the MTP head and start the server
with the launch line from `INSTALL.md`.

Why the bundle instead of `pip install vllm-quantpack`: upstream 0.1.5 disables its own fast int4 kernel
on Windows (one warning in the log, nothing else), so the same server runs at 28 tok/s decode instead of
85. Our wheel is that package with the fixes applied, versioned `0.1.5+win.1`.

Want an assistant to do it? Hand it `AGENT-INSTALL.md` — the same procedure written as RUN / EXPECT /
IF NOT, including the checks that catch the silent failures.

## What is in this repository

| file                | what it is                                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `bundle/`           | **the whole solution in one place**: four wheels with the fixes baked in, `install.bat`, checks                    |
| `AGENT-INSTALL.md`  | the same install as a RUN / EXPECT / IF NOT runbook for an AI agent, with the failure table                        |
| `INSTALL.md`        | step-by-step install, launch lines per KV format, verification, troubleshooting                                    |
| `PERFORMANCE.md`    | all measurements, the measurement protocol, and how each number was obtained                                       |
| `STACK.md`          | who wrote which layer (Tencent, Jorge, Chris, flashinfer-ai, SystemPanic, us) and links                            |
| `UPSTREAM-FIXES.md` | the fixes each upstream author should apply, with exact file, line and replacement                                 |
| `for-authors/`      | the same fixes as five ready-to-post messages, one per addressee                                                   |
| `ORIGIN.md`         | why this exists — the story from "decode is too slow" to a 262k context                                            |
| `patches/`          | the plugin overlay and the vLLM one-line guard, both checked against clean wheel extractions                       |
| `tools/`            | FlashInfer build scripts (`make_wheel.py`, `build_jitcache.py`, the MSVC shim) — only needed to rebuild the wheels |

## Honest limits

| limit                                       | what it means                                                                                                               |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| This is a **community build**, not upstream | no support contract; version pins are exact (`FlashInfer 0.7.2.post1`)                                                      |
| `sm_80` kernels only                        | Hopper and Blackwell are not covered, and do not need to be                                                                 |
| `int8` KV + FlashInfer is impossible        | FlashInfer has no int8 KV path with per-token-head scales; `fp8` is the same 1 byte/element and works                       |
| Numbers are one card                        | RTX3090, one GPU, no tensor parallelism; another card will differ                                                           |
| 4-bit KV costs accuracy                     | `int4` keeps 16 levels per group, `k8v4` keeps 256; both dequantize to `bfloat16` before attention, so neither is lossless. |
|                                             | The anchor check passes for both at a 33k fill, deeper runs were speed-only, and a needle test is not a quality measurement |
## Credits

FlashInfer is [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer); the Windows port
is based on [SystemPanic/flashinfer-windows](https://github.com/SystemPanic/flashinfer-windows); the vLLM
Windows build is [aivrar/vllm-windows-build](https://github.com/aivrar/vllm-windows-build); the
quantization plugin and the quantized weights are
[jramirez-uab](https://github.com/jramirez-uab/vllm-quantpack) / JRamirez-UAB on Hugging Face. Details,
including what each of them still needs to change, are in `STACK.md` and `UPSTREAM-FIXES.md`.
