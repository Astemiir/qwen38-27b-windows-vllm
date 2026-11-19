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

Measured on your kit, RTX 3090, 33 367-token fill, 500-token answer, MTP n3 — five arms, so the port's
share is visible rather than implied:

| arm                                                                 | prefill, tok/s | decode, tok/s | window   |
| ------------------------------------------------------------------- | -------------- | ------------- | -------- |
| `int8_per_token_head`, Triton, stock                                | 828            | 76.0          | 131 072  |
| `int8_per_token_head`, Triton, + the plugin's int8 prefill hook      | **1 172**      | 76.0          | 131 072  |
| `fp8_e4m3`, FlashInfer (our wheels)                                 | 1 182          | **81.8**      | 131 072  |
| `int4_per_token_head`, Triton, plugin launcher                       | **1 178**      | **85.5**      | 262 144  |
| `bfloat16`, Triton, stock                                           | 1 038          | 85.4          | 65 536   |

Two things in that table are worth a line in your docs. **The prefill win is not the port**: with the
plugin's int8 prefill hook in place, stock Triton matches FlashInfer, 1 172 against 1 182. **Decode is a
cache-format story, not a backend one** — `bfloat16` on stock Triton already decodes at 85.4, int4 from
the plugin at 85.5, while the int8 path sits at 76.0 patched or not. What FlashInfer still owns on this
card is fp8 KV, which Triton refuses below SM89, and the +7.6 % decode over the *unpatched* int8 path.

Also worth noting: **fp8 KV works without the plugin at all** — your kit plus our FlashInfer wheels and
`--kv-cache-dtype fp8_e4m3 --attention-backend FLASHINFER` is enough. The plugin is needed for int4/k8v4
and for the int8 prefill path.

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
(master:164), one section, about a failure that produces no error:

- **`int4_per_token_head` with speculative decoding decodes at ~28 tok/s** — no error, correct answers,
  simply the 2D unpack kernel: the packed launcher only takes the 3D split-KV path when
  `max_seqlen_q <= 1`, so a step carrying `1 + num_speculative_tokens` rows stays on the slow path and no
  launch flag changes it (28.0 at `--max-num-seqs 2`, 29.3 at 8, same 33k fill). A faster multi-query
  split-KV launcher exists in the `vllm-quantpack` plugin, which is where this belongs rather than in the
  kit's docs — with the plugin's encoding fixed (U+2192 written through the ANSI code page) and
  `--max-num-seqs 8` the same stand gives 85.5.
- **`FP8 KV cache is not supported by the Triton attention backend … requires SM89+`** — the refusal is
  correct and we no longer ask for it to be lifted: widening the check starts the engine, but your 9B run
  returned garbled text while our 27B answered correctly, so the dequant is shape-dependent. The entry
  should send people to `--kv-cache-dtype int8_per_token_head`, or `fp8_e4m3` with
  `--attention-backend FLASHINFER`.

## 3. Optional, if you want the kit to carry it

`https://github.com/Astemiir/qwen38-27b-windows-vllm/bundle/patch_vllm.py` applies exactly that one guard change to `triton_attn.py` inside a kit,
idempotently, leaving a `.bak`. We are not asking you to take a dependency on us — the patch is three
lines and it fixes your own `fp8_per_token_head` dtype. If you would rather have it as a PR against
`v0.33.0-win-cu130`, say so and I will send one.

Same offer as before: if any of the doc text above is not how you want it phrased, tell me the wording and
I will rewrite it as a diff against the file.
