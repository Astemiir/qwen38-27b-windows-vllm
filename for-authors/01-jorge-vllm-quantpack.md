# For the plugin author — `jramirez-uab/vllm-quantpack`

<!-- Post as: new issue. Replace https://github.com/Astemiir/qwen38-27b-windows-vllm with the published repository URL first.
     Title: int4 fast kernel silently disabled on Windows: generated source is written without an
            explicit encoding, and U+2192 is not encodable in Western ANSI code pages
-->

Hi — thank you for `vllm_quantpack`. We run it on a single RTX 3090 under Windows 11 (vLLM 0.33, the
`v0.33.0-win-cu130` build, Qwen3.8-27B GPTQ W4A16 with the MTP head, `num_speculative_tokens = 3`) and
found one defect that costs a factor of three in decode with nothing but a single warning in the log.

## What happens

`int4_per_token_head` starts, serves correct answers, and decodes at **28.0 tok/s**. The only trace is
this line among ~300 at startup:

```
vllm-quantpack: int4_per_token_head attention left stock ('charmap' codec can't encode character '\u2192' in position 2341)
```

The stock kernel then runs the whole session: 858.7 tok/s prefill and 28.0 tok/s decode at a 33 350-token
fill. With the fast kernel actually loaded, the same stand gives **1 178 tok/s and 85.5 tok/s**.

## Cause

`vllm_quantpack/attention/int4_kv.py:181`:

```python
with os.fdopen(fd, "w") as fh:
```

The file being written is the generated CUDA source, and it contains an arrow in a comment:
`# 2 → INT4 nibble pair`. Without an explicit `encoding`, `os.fdopen` uses the process locale, which on
Windows is the ANSI code page. U+2192 is not representable in **any** Western Windows code page — we
tested cp1251, cp1252, cp1250, cp437, cp866 and cp874, all raise `UnicodeEncodeError`. Only cp936,
cp932, cp949 and UTF-8 encode it. So the write fails, the exception is caught by the `except` that logs
"left stock", and the fast kernel is never built.

The installed wheel was intact — its sha256 matched `RECORD` — so this is upstream, not a local edit.

## Fix

```python
with os.fdopen(fd, "w", encoding="utf-8") as fh:   # the kernel source carries non-ASCII comments
```

Two small follow-ups, both optional:

1. Put the cost into the warning — e.g. `int4_per_token_head fast kernel not loaded: int4 decode will run
   ~3x slower`. "attention left stock" reads like a note; three of us read past it before finding it.
2. `grep -rn "os.fdopen" vllm_quantpack/` — `turboquant_kv.py` shares the same `_load` helper, so any
   other generated source with non-ASCII content has the same exposure.

## How we are working around it

We ship a wheel `vllm_quantpack-0.1.5+win.1` (upstream 0.1.5, Apache-2.0, untouched except for four
changes listed in a `WINDOWS-OVERLAY.txt` inside the package), so Windows users get a working install
meanwhile. Besides the fix above it carries:

- `attention/int8_prefill.py` — a fast prefill path for `int8_per_token_head` and `fp8_per_token_head`
  (unpack long prefixes in Triton, hand them to FlashAttention): 828 → 1 172 tok/s, decode unchanged;
- `_unpack_segment_fp8` plus a branch in `unpack_segment`, so an fp8 cache (uint8 storage viewed as
  `float8_e4m3fn`, one code per channel) uses the same unpack path;
- two calls in `__init__.py` to register both.

Patch, files and the measurements: `https://github.com/Astemiir/qwen38-27b-windows-vllm` (`patches/quantpack/`, `PERFORMANCE.md`).

**We would rather point at your release than ship a fork.** If you publish 0.1.6 with the encoding fix,
that wheel disappears from our instructions. I am happy to open a PR containing only the one-line fix,
or the fix plus the int8 prefill module if you want it — your call.
