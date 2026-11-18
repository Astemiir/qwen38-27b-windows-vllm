# Fixes for the upstream projects

Four projects, four separate problems. Each one is written so it can be applied without reading the
rest of this repository: the file, the current line, the replacement, and how to check that it took.

Ready-to-post versions of these, one message per addressee, are in `for-authors/`. This file is the
audit view — it keeps the surrounding code and the cross-references; those are the posting view.

Everything below was found by running the stack on Windows 11 with an RTX 3090. Nothing here is a
request to change direction — each item is a case where the code assumes Linux, or assumes a batch
shape that only holds without speculative decoding.

| project                       | items                     | severity                                                          |
| ----------------------------- | ------------------------- | ----------------------------------------------------------------- |
| `jramirez-uab/vllm-quantpack` | 2                         | one silently disables the fast kernel, one silently halves decode |
| `vllm-project/vllm`           | 1                         | a valid KV format is rejected at startup                          |
| `aivrar/vllm-windows-build`   | 2 docs + 1 optional patch | documentation tells users the fast path does not exist            |
| `flashinfer-ai/flashinfer`    | 5                         | the JIT generator cannot build on Windows at all                  |

---

## 1. `vllm-quantpack` — the fast kernel disables itself on Windows

**File:** `vllm_quantpack/attention/int4_kv.py`, line 181.

```python
with os.fdopen(fd, "w") as fh:
```

**Replace with:**

```python
with os.fdopen(fd, "w", encoding="utf-8") as fh:
```

**Where it breaks.** The file being written is a generated CUDA source, and it contains an arrow in a
comment (`# 2 → INT4 nibble pair`). `os.fdopen(fd, "w")` without `encoding` uses the process locale,
which on Windows means the ANSI code page. U+2192 is not representable in **any** Western Windows code
page — cp1252, cp1251, cp1250, cp437, cp866 and cp874 all raise `UnicodeEncodeError`; only cp936,
cp932, cp949 and UTF-8 survive it. So on most Windows installs the write fails, the exception is
caught, and the log records:

```
vllm-quantpack: int4_per_token_head attention left stock ('charmap' codec can't encode character '\u2192' in position 2341)
```

at WARNING level, once, in the middle of a 300-line startup log. The server then starts and serves
correctly, using the stock kernel — 858.7 tok/s prefill and 28.0 tok/s decode instead of 1 178 and 85.5.
The warning is accurate and easy to miss; three of us read past it before finding it by grep.

**Two suggestions beyond the one-line fix:**

1. Carry the cost into the message. "attention left stock" reads like a note; something like
   `int4_per_token_head fast kernel not loaded: int4 decode will run ~3x slower (see the exception
   above)` makes it actionable, and repeats the reason in the line a user will actually grep for.
2. `grep -rn "os.fdopen" vllm_quantpack/` — `turboquant_kv.py` shares the same `_load` helper, so any
   other generated source with non-ASCII content has the same exposure.

**Verify:** start with `--kv-cache-dtype int4_per_token_head` and check the log for
`int4_per_token_head KV attention fixed` instead of `left stock`.

**How we ship it meanwhile:** the four changes are in `bundle/vllm_quantpack-0.1.5+win.1-py3-none-any.whl`
(upstream 0.1.5, Apache-2.0, with a `WINDOWS-OVERLAY.txt` inside listing every change), so Windows users
get a working install today. If you publish 0.1.6 with the same four edits, that wheel simply stops being
needed — we would rather point at yours.

## 2. `vllm-project/vllm` — int4 KV with speculation: measured, no patch proposed

**File:** `vllm/v1/attention/backends/triton_attn.py`, lines 169–180.

```python
        max_query_len_3d = 1
        speculative_config = vllm_config.speculative_config
        # The INT4 launcher runs multi-query batches in 2D only.
        if (
            speculative_config is not None
            and kv_cache_spec.kv_quant_mode != KVQuantMode.INT4_PER_TOKEN_HEAD
        ):
```

**What we measured.** With `num_speculative_tokens = 3` and `int4_per_token_head`, decode is 23.6 tok/s
at `--max-num-seqs 2` and 85.5 tok/s at `--max-num-seqs 8` — same model, same prompt, same 6.8 GB cache.
The comment above the condition says the exclusion is deliberate, which is why this item is written as a
question and not as a patch: the 2D path is clearly the slow one on Ampere, and something about a larger
`max_num_seqs` gets us off it, but we cannot tell from the code whether that is intended.

**Two questions for the maintainers:**

1. Is "multi-query batches in 2D only" a limitation of the int4 launch path that is expected to be lifted,
   or a permanent property? If permanent, the 3D branch could `logger.warning_once` that int4 decode
   carries `1 + num_speculative_tokens` rows per sequence and loses the split-KV path.
2. Is there an intended way to keep the split-KV path under speculation — a minimum batch, a scratch
   override, or a kernel flag — or is `--max-num-seqs ≥ 1 + num_speculative_tokens` the answer by design?

Until that is settled the workaround is a launch flag, documented in `INSTALL.md`: `--max-num-seqs 8`.
Worth knowing that it also costs nothing: at 8 sequences the 262k stand held 319 815 tokens of cache.

## 3. `vllm-project/vllm` — fp8 KV with per-token-head scales is rejected on `sm_86`

**File:** `vllm/v1/attention/backends/triton_attn.py`, line 525, inside the `TritonAttentionImpl`
constructor.

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

**Replace the first three lines with:**

```python
            if (
                self.kv_cache_dtype.startswith("fp8")
                and not self.kv_cache_dtype.endswith("per_token_head")
                and not (current_platform.has_device_capability(89))
            ):
```

**Where it breaks.** The guard protects the *native* `fp8e4nv` path, where the kernel does the scaled
dot product in fp8. A `fp8_per_token_head` cache is not that: it is stored as `torch.uint8`, viewed as
`float8_e4m3fn`, and dequantized with an explicit per-head, per-token scale inside the Triton kernel —
the same arithmetic that `int8_per_token_head` already uses and that already runs on `sm_86`. The guard
is one dtype predicate too wide, so a working configuration cannot start.

Once widened, `fp8_per_token_head` runs and produces correct output; it is not faster than int8
(1 185 versus 1 172 tok/s prefill, 76.8 versus 76.0 decode), but it starts.

**Verify:** `--attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head` reaches
"Application startup complete" and a needle test passes.

## 4. `aivrar/vllm-windows-build` — documentation

Two documents tell Windows users that the fast path is unavailable. Both statements were true when
written; neither is true now that prebuilt wheels exist.

### 4a. `PATCHES.md`

**Find** the note saying there is no FlashInfer wheel for Windows, or that FlashInfer must be built
from source with a Linux toolchain.

**Replace with:**

> **FlashInfer on Windows:** prebuilt wheels for `0.7.2.post1` are available, built for CUDA 13.0 and
> `sm_80` (loads on 8.0/8.6/8.9):
>
> ```
> pip install https://github.com/Astemiir/flashinfer-windows/releases/download/v0.7.2.post1-win.1/flashinfer_python-0.7.2.post1+win.1-py3-none-any.whl
> pip install https://github.com/Astemiir/flashinfer-windows/releases/download/v0.7.2.post1-win.1/flashinfer_jit_cache-0.7.2.post1+win.1-py3-none-any.whl
> pip install https://github.com/Astemiir/flashinfer-windows/releases/download/v0.7.2.post1-win.1/flashinfer_jit_cache_sm80-0.7.2.post1+win.1-cp39-abi3-win_amd64.whl
> ```
>
> Prebuilt kernels mean no nvcc, no MSVC and no JIT at first start. With these, `--attention-backend
> FLASHINFER --kv-cache-dtype fp8_e4m3` works on Ampere — including without `vllm-quantpack`, so fp8 KV
> no longer requires the plugin.

### 4b. `troubleshooting.md`

Add two sections, because both failures are silent and both cost hours:

> ### Decode is much slower than expected with `int4_per_token_head`
>
> Two independent gates keep this dtype on the stock kernel. First, the plugin writes a generated CUDA
> source through `os.fdopen(fd, "w")`, which uses the ANSI code page; the source contains U+2192, which
> no Western code page can encode. The write fails, the error is swallowed, and the log says
> `int4_per_token_head attention left stock`. Fix: pass `encoding="utf-8"` in
> `vllm_quantpack/attention/int4_kv.py:181`.
>
> Second, the 3D split-KV scratch is `max_num_seqs × 1` rows, while a speculative step needs
> `1 + num_speculative_tokens`. With `--max-num-seqs 2` the launcher silently uses the 2D path:
> 28 tok/s instead of 85. Pass `--max-num-seqs 8`.
>
> On an RTX 3090 with both fixed: 1 178 tok/s prefill and 85.5 tok/s decode at a 33k fill, 262 144
> tokens of context in 6.8 GB of KV.

> ### `FP8 KV cache is not supported by the Triton attention backend … requires SM89+`
>
> The `TRITON_ATTN` guard rejects every dtype starting with `fp8`. For `fp8_per_token_head` this is too
> wide — that format is stored as `uint8` and dequantized with per-token scales in the kernel, so it
> does not need native fp8 tensor cores. Either use `--attention-backend FLASHINFER` with
> `--kv-cache-dtype fp8_e4m3`, or widen the guard in
> `vllm/v1/attention/backends/triton_attn.py:525` with
> `and not kv_cache_dtype.endswith("per_token_head")`.

### 4c. Optional one-line patch

Item 3 above applies unchanged to the wheel shipped in the kit. It is included as
`patches/vllm-triton-fp8-per-token-head-guard.patch` so the kit can carry it without waiting for
upstream.

## 5. `flashinfer-ai/flashinfer` — Windows JIT

The JIT generator (`flashinfer/jit/`) is Linux-only by construction. Five blockers, in the order they
bite:

| #   | symptom                                                | cause                                                                                                                         | fix in our port                                                                                                                |
| --- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 1   | `FileNotFoundError: c++`                               | host compiler hardcoded as `c++` with GCC flags (`-fPIC`, `-shared`)                                                          | `cl.exe` for host objects, nvcc for `.cu`; `/LD` for linking                                                                   |
| 2   | `error: unrecognized file type` on `.cu`               | host compile step receives CUDA files                                                                                         | compile `.cu` with nvcc, and pass host files as `-x cu` in the same driver invocation                                          |
| 3   | `fatal error: cannot open input file` on `E:\…`        | nvcc and ninja treat `:` as a separator; `GetLongPathName` fails on long spec paths                                           | pass every path through a `_ShortPath` helper; export `_ninja_path` with `E$:` escaping                                        |
| 4   | `LNK2019: unresolved external fi_max / fi_min`         | macros defined for GCC device code only                                                                                       | shim header defining `fi_max`/`fi_min` over `max`/`min`; `/FORCE:UNRESOLVED` for genuinely unreachable template instantiations |
| 5   | `error C2059: syntax error: '=='` in hundreds of files | `#if MODE == …` appears *inside* a `DISPATCH_*` macro argument, which MSVC expands before the preprocessor sees the directive | rewrite those blocks as `if constexpr` — 743 blocks across 651 files, mechanical                                               |

Two further notes worth upstream attention:

- **The prebuilt cache lookup ignores `sm_86`.** In the `cu130` wheel index, JIT-cache artifacts exist
  for `sm_80` and Blackwell targets, but nothing for 8.6 or 8.9 — so even a successful build cannot be
  reused. We ship 29 `sm_80` modules and let the runtime's own compatibility table (`sm_80` SASS loads
  on `sm_86`/`sm_89`) do the work; upstream could publish an `sm_80` cache for Windows instead.
- **`spec.build()` returns `None`** even on success. The artifact is at `spec.jit_library_path`; anyone
  writing an installer will assume failure from the return value.

Our entire Windows patch set is in `E:\1` (build tree, 651 modified files) and the recipe is published
as `docs/build-windows-msvc.md` in the [FlashInfer fork](https://github.com/Astemiir/flashinfer-windows).
It is mechanical enough to be upstreamable; the offer to open a PR stands.

## 6. `SystemPanic/flashinfer-windows` — the base port

Our wheels build on the `flashinfer-windows-rel` branch at `v0.7.0.post1`. Everything in section 5 is
either already there or is a delta we can send as a patch. If this project moves forward, we would
rather contribute the five fixes back than maintain a second port.
