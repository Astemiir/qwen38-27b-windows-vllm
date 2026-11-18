# For the Windows build maintainer — `aivrar/vllm-windows-build`

<!-- Post as: a comment in the existing thread (#20) or a new issue if that one is closed.
     Replace https://github.com/Astemiir/qwen38-27b-windows-vllm with the published repository URL first.
     Title: Windows FlashInfer wheels now exist — plus a ready bundle, a patched plugin wheel and the
            two documentation lines that still say otherwise
-->

Follow-up to my comment in #20 (the one with the `PATCHES.md:415` replacement and the two
`troubleshooting.md` sections). Two things have changed since: the wheels are now packaged so that a user
needs nothing but `pip`, and we put the whole working set in one place.

## 1. What exists now

| artifact | what it is |
| --- | --- |
| `flashinfer_python` / `flashinfer_jit_cache` / `flashinfer_jit_cache_sm80`, `0.7.2.post1+win.1` | Windows build for CUDA 13.0 with 29 prebuilt `sm_80` modules; no nvcc, no MSVC, no JIT at first start; loads on 8.0/8.6/8.9 |
| `vllm_quantpack-0.1.5+win.1` | Jorge's plugin with four fixes (his fast int4 kernel disables itself on Western Windows code pages — one warning, 28 vs 85 tok/s decode) |
| `https://github.com/Astemiir/qwen38-27b-windows-vllm/bundle` | all four wheels plus `install.bat`, which installs them into a kit and verifies that a prebuilt kernel loads |

Measured on a 3090 with your kit, `fp8_e4m3` + `--attention-backend FLASHINFER`, 131 072 window: 1 182
tok/s prefill and 81.8 tok/s decode, versus 828 / 76.0 on the stock Triton int8 path. With `int4` KV from
the plugin: 262 144 tokens of context in 6.8 GB, 1 178 / 85.5 at a 33k fill.

Worth noting for your docs: **fp8 KV works without the plugin at all** — your kit plus our FlashInfer
wheels and `--kv-cache-dtype fp8_e4m3 --attention-backend FLASHINFER` is enough. The plugin is only needed
for int4/k8v4 and for the int8 prefill path.

## 2. The two documentation lines, again

`PATCHES.md`, section "What's NOT patched (known limitations)" — master:415 / `v0.33.0-win-cu130`:407:

```
- **FlashInfer** — No Windows wheel; the sampler defaults to Triton.
```

Replace with a line that keeps the true half (the sampler still defaults to Triton) and adds the wheel,
for example:

```
- **FlashInfer** — Prebuilt Windows wheels for 0.7.2.post1 exist (`sm_80`, CUDA 13.0, no build step) and
  give fp8 KV on Ampere: 1 182 / 81.8 tok/s at a 131k window against 828 / 76.0 on the stock int8 path.
  The sampler still defaults to Triton; install the wheels only if you want `--attention-backend
  FLASHINFER`. Blackwell FP4 limitations are unchanged.
```

`docs/troubleshooting.md` — after `### A .gguf model fails with UTF-8 or invalid JSON errors`
(master:164), two sections, both about failures that produce no error:

- **`int4_per_token_head` decodes at ~28 tok/s** — the plugin writes a generated CUDA source through
  `os.fdopen(fd, "w")`, which uses the ANSI code page; the source contains U+2192, which no Western code
  page can encode, so the fast kernel is skipped and one warning says `attention left stock`. Fix the
  encoding, then pass `--max-num-seqs 8` — with fewer sequences than `1 + num_speculative_tokens` the
  split-KV path is skipped as well. Both together: 28 → 85.5 tok/s.
- **`FP8 KV cache is not supported by the Triton attention backend … requires SM89+`** — the guard is one
  predicate too wide for `fp8_per_token_head` (that dtype dequantizes with per-token scales, it does not
  need native fp8 tensor cores). Either use `--attention-backend FLASHINFER`, or widen the condition with
  `and not kv_cache_dtype.endswith("per_token_head")`.

## 3. Optional, if you want the kit to carry it

`https://github.com/Astemiir/qwen38-27b-windows-vllm/bundle/patch_vllm.py` applies exactly that one guard change to `triton_attn.py` inside a kit,
idempotently, leaving a `.bak`. We are not asking you to take a dependency on us — the patch is three
lines and it fixes your own `fp8_per_token_head` dtype. If you would rather have it as a PR against
`v0.33.0-win-cu130`, say so and I will send one.

Same offer as before: if any of the doc text above is not how you want it phrased, tell me the wording and
I will rewrite it as a diff against the file.
