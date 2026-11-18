# Install

Four layers, in this order. Each step ends with a check — do not skip them, because two of the four
failures in this stack are silent.

Target: **one 24 GB card (RTX 3090 tested), Windows 10/11 x64, NVIDIA driver 580.88 or newer (R580
branch), ~60 GB free disk.** No CUDA toolkit is installed: the wheels carry their runtime, and the
driver is the only thing the stack needs from the system — 580.88 is the first Windows driver of the
R580 branch, which is the minimum for any CUDA 13.x application. We tested on 621.x.

| layer | what it is                                   | who ships it                                                                                          |
| ----- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 1     | vLLM 0.33 for Windows                        | [aivrar/vllm-windows-build](https://github.com/aivrar/vllm-windows-build)                             |
| 2     | FlashInfer 0.7.2.post1 with `sm_80` kernels  | [our FlashInfer fork](https://github.com/Astemiir/flashinfer-windows/releases/tag/v0.7.2.post1-win.1) |
| 3     | `vllm-quantpack` 0.1.5 + our Windows overlay | [jramirez-uab/vllm-quantpack](https://github.com/jramirez-uab/vllm-quantpack) + `patches/` here       |
| 4     | a model in GPTQ W4A16 with an MTP head       | JRamirez-UAB on Hugging Face                                                                          |

## What ends up installed, by version

These are the versions the numbers in `PERFORMANCE.md` were measured on, read out of the working
install (`python\python.exe -c "import torch, triton; …"`):

| component          | version                       | where it comes from                                            |
| ------------------ | ----------------------------- | -------------------------------------------------------------- |
| Python             | 3.13.14                       | portable, unpacked inside the vLLM kit by `install.bat`        |
| PyTorch            | 2.14.0+cu130                  | vLLM kit                                                       |
| Triton             | 3.8.0                         | vLLM kit                                                       |
| cuDNN              | 9.24                          | vLLM kit                                                       |
| CUDA runtime       | 13.0                          | carried by the `cu130` wheels, not installed by you            |
| CUDA toolkit       | **not needed**                | only if you rebuild kernels yourself (we build ours with 13.3) |
| NVIDIA driver      | 580.88 or newer (R580 branch) | system; tested on 621.x                                        |
| compute capability | 8.0 / 8.6 / 8.9               | prebuilt `sm_80` SASS loads on all three                       |

The order matters: driver R580+ → CUDA 13.0 runtime from `torch+cu130` → Triton 3.8 from the kit. A
driver below 580 fails at `import torch` with `CUDA driver version is insufficient for CUDA runtime
version`, before any weights are touched. A toolkit newer than the runtime is fine — a compiler may be
newer than the runtime it targets (minor version compatibility), which is why our `sm_80` kernels built
with nvcc 13.3 load under a 13.0 runtime.

---

## Step 1 — vLLM 0.33 for Windows

Download the `v0.33.0-win-cu130` release and run `install.bat`. It unpacks a portable Python 3.13.14
inside the kit directory and installs torch and the vLLM wheel.

Check:

```bat
python\python.exe -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda)"
```

Expected: `0.33.0`, `2.14.0+cu130`, `13.0`.

> If `pip` fails on the torch download with `SSLError: DECRYPTION_FAILED_OR_BAD_RECORD_MAC`, run
> `install.bat` again — the retry passes. It is a network hiccup, not a broken environment.

## The one-folder path (what most people should do)

Everything except the kit and the weights lives in `bundle/` in this repository, and the same eight files
are attached to **[release v0.1-win.1](https://github.com/Astemiir/qwen38-27b-windows-vllm/releases/tag/v0.1-win.1)**,
so there is nothing to fetch from four different places:

```
bundle\
  install.bat                                             one step, no arguments beyond the kit path
  flashinfer_python-0.7.2.post1+win.1-py3-none-any.whl
  flashinfer_jit_cache-0.7.2.post1+win.1-py3-none-any.whl
  flashinfer_jit_cache_sm80-0.7.2.post1+win.1-…whl
  vllm_quantpack-0.1.5+win.1-py3-none-any.whl             the plugin WITH the Windows fixes baked in
  patch_vllm.py                                           one guard fix inside the kit
  verify_flashinfer.py, smoke_test.py                     checks
```

```bat
:: 1. install the kit once (aivrar's release, v0.33.0-win-cu130)
:: 2. then:
"<repo>\bundle\install.bat" "D:\vllm-windows-build-0.33.0-win-cu130"
```

The script installs the four wheels, applies the guard fix with a `.bak` copy, and finishes with `ok`.
**The plugin wheel is the point**: the upstream `vllm-quantpack` 0.1.5 disables its own fast kernel on
Windows, so `pip install vllm-quantpack` gives a working-but-3×-slower server. Ours is the same code
with the four fixes applied, versioned `0.1.5+win.1` so it is easy to tell apart.

If you would rather have an assistant do the whole thing — install, start, verify, report — hand it
`AGENT-INSTALL.md`. It is the same procedure written as RUN / EXPECT / IF NOT for an AI agent.

The steps below are the manual path, for people who want to see every wheel go in.

---

## Step 2 — FlashInfer with prebuilt kernels

Use **the same** `python\python.exe` from step 1. Three wheels, python → shim → kernels:

```bat
pip install https://github.com/Astemiir/flashinfer-windows/releases/download/v0.7.2.post1-win.1/flashinfer_python-0.7.2.post1+win.1-py3-none-any.whl
pip install https://github.com/Astemiir/flashinfer-windows/releases/download/v0.7.2.post1-win.1/flashinfer_jit_cache-0.7.2.post1+win.1-py3-none-any.whl
pip install https://github.com/Astemiir/flashinfer-windows/releases/download/v0.7.2.post1-win.1/flashinfer_jit_cache_sm80-0.7.2.post1+win.1-cp39-abi3-win_amd64.whl
```

Check — with JIT disabled, a kernel must still load, which proves the prebuilt cache is in place:

```bat
set FLASHINFER_DISABLE_JIT=1
python\python.exe <repo>\bundle\verify_flashinfer.py
```

Expected output ends with `ok`:

```
flashinfer: 0.7.2.post1+win.1
from: …\site-packages\flashinfer\__init__.py
providers: 1
exports: 20
ok
```

If it stops with `no prebuilt kernel for this spec`, one of the three wheels is missing or the versions
do not match — all three must be exactly `0.7.2.post1+win.1`. If it printed `ok`, you will never need
nvcc or MSVC on this machine.

## Step 3 — the quantization plugin, then the Windows overlay

Install our wheel, which is upstream 0.1.5 with the overlay already applied:

```bat
pip install --no-deps --force-reinstall <repo>\bundle\vllm_quantpack-0.1.5+win.1-py3-none-any.whl
```

```bat
python\python.exe -c "import vllm_quantpack; print(vllm_quantpack.__version__)"
```

Expected: `0.1.5+win.1`. If you get `0.1.5`, the upstream wheel won — that is the build whose fast int4
kernel disables itself on Windows, so reinstall ours and never `pip install vllm-quantpack` afterwards.

<details>
<summary>Applying the same four edits by hand, to an upstream install</summary>

```bat
pip install vllm-quantpack==0.1.5
```

It is **four small edits**, two of them one-liners. Either use the patch file:

```bat
cd python\Lib\site-packages
git apply -p1 <path-to-this-repo>\patches\quantpack\quantpack-0.1.5-windows.patch
```

(GNU `patch -p1` reads the same file.) Or do it by hand:

| file (under `python\Lib\site-packages\vllm_quantpack`) | edit                                                                                                           |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `attention\int4_kv.py`, line 181                       | `os.fdopen(fd, "w")` → `os.fdopen(fd, "w", encoding="utf-8")`                                                  |
| `attention\int8_prefill.py`                            | copy the file from `patches\quantpack\attention\` in this repo                                                 |
| `attention\int4_prefill.py`                            | add the `_unpack_segment_fp8` kernel and its branch in `unpack_segment` (only needed for `fp8_per_token_head`) |
| `__init__.py`                                          | import `patch_int8_prefill, patch_int8_segments` and call them after `patch_int4_kv(logger)`                   |

</details>

**Why the first one matters.** The plugin writes a generated CUDA source file with
`os.fdopen(fd, "w")`, which uses the system ANSI code page. That source contains an arrow character
(`# 2 → INT4 nibble pair`), and U+2192 exists in **no** Western Windows code page — cp1252, cp1251,
cp1250, cp437, cp866 and cp874 all raise `charmap`. The exception is caught, and the log says
`int4_per_token_head attention left stock`. Nothing breaks; it just runs the slow stock kernel. Only
CJK code pages (cp936/932/949) and UTF-8 avoid it.

Check after starting the server (step 4): the log must contain **`KV attention fixed`**, not
`left stock`.

**Optional — only for `fp8_per_token_head` on the Triton backend.** Stock vLLM rejects that dtype on
cards below SM89, one predicate too wide. Apply the one-guard patch from the same directory:

```bat
git apply -p1 <path-to-this-repo>\patches\vllm\vllm-triton-fp8-per-token-head-guard.patch
```

It is not required for `int4`, `k8v4`, `int8` or `fp8_e4m3` through FlashInfer, and it buys no speed —
`fp8_per_token_head` measures a tie with int8. Skip it unless you specifically want that dtype.

## Step 4 — model and launch

Download a GPTQ W4A16 checkpoint with MTP head, for example
`JRamirez-UAB/Qwen3.8-27B-DSpine-vLLM-quants` (the `…-mtp-int8` variant). Launch lines:

**262k context, `int4` KV (the recommended default):**

```bat
set "PYTORCH_CUDA_ALLOC_CONF="
python\python.exe -m vllm.entrypoints.openai.api_server ^
  --model <path-to-model> --served-model-name qwen3.8-27b ^
  --language-model-only ^
  --attention-backend TRITON_ATTN ^
  --max-model-len 262144 ^
  --kv-cache-dtype int4_per_token_head ^
  --kv-cache-memory-bytes 6800000000 ^
  --max-num-batched-tokens 2048 ^
  --max-num-seqs 8 ^
  --enable-prefix-caching ^
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":3}" ^
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder ^
  --host 127.0.0.1 --port 8010
```

**131k context, `fp8` KV through FlashInfer (no plugin patching needed):**

```bat
set "PYTORCH_CUDA_ALLOC_CONF="
python\python.exe -m vllm.entrypoints.openai.api_server ^
  --model <path-to-model> --served-model-name qwen3.8-27b ^
  --attention-backend FLASHINFER ^
  --max-model-len 131072 ^
  --kv-cache-dtype fp8_e4m3 ^
  --kv-cache-memory-bytes 6500000000 ^
  --max-num-batched-tokens 2048 ^
  --max-num-seqs 2 ^
  --enable-prefix-caching ^
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":3}" ^
  --host 127.0.0.1 --port 8010
```

Two flags are not optional and are easy to get wrong:

- **`--max-num-seqs 8`** for `int4`. The 3D split-KV scratch is sized from `max_num_seqs` with one
  query row per sequence for this dtype; a speculative step carries `1 + num_speculative_tokens` rows.
  With `--max-num-seqs 2` the launcher silently falls back to the 2D path and decode drops from
  85.5 to 28.0 tok/s — with no error in the log.
- **`PYTORCH_CUDA_ALLOC_CONF` empty.** `expandable_segments` aborts startup on Windows with a GPU
  mapping OOM. If you inherited it from another setup, clear it.

## Verify

```bat
python\python.exe <repo>\bundle\smoke_test.py 8010
```

Expected:

```
content: ready
usage: {'prompt_tokens': 19, 'total_tokens': 21, 'completion_tokens': 2, …}
```

Then check the startup log for the lines that prove the fast paths are active:

| line in the log                                                              | proves                                                      |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `vllm-quantpack: int4_per_token_head KV attention fixed (… split-KV …)`      | the patched int4 kernel is loaded                           |
| `vllm-quantpack: int8/fp8 per-token-head long prefills run FlashAttention …` | the int8/fp8 prefill hook is loaded                         |
| `GPU KV cache size: 319,815 tokens`                                          | 262k fits, with the headroom the log then prints as `1.22x` |

The token count in the third line depends on your `--kv-cache-memory-bytes`; the ratio it prints must be
above 1.00× for the requested window to fit.

## Troubleshooting

| symptom                                                | cause                                                             | fix                                                                                                               |
| ------------------------------------------------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| decode is ~28 tok/s on `int4`                          | 2D fallback: scratch too small for the speculative batch          | `--max-num-seqs 8` (or ≥ `1 + num_speculative_tokens`)                                                            |
| log says `attention left stock`                        | the plugin could not write its generated kernel (code page)       | apply the `encoding="utf-8"` edit from step 3                                                                     |
| `native FP8 (fp8e4nv) requires SM89+`                  | stock Triton on an 8.x card                                       | use `--attention-backend FLASHINFER` (step 2), or the guard patch in `UPSTREAM-FIXES.md` for `fp8_per_token_head` |
| `kv_cache_dtype not supported` for `int8` + FLASHINFER | FlashInfer has no int8 KV path with per-token-head scales         | use `fp8_e4m3` — same 1 byte/element                                                                              |
| `GPU mapping OOM` at startup                           | `expandable_segments`                                             | clear `PYTORCH_CUDA_ALLOC_CONF`                                                                                   |
| `Engine core initialization failed` in ~5 s            | a JIT kernel failed to build (e.g. `topk` with a drafter)         | use MTP instead of DFlash2, or install the topk opt-out patch in `UPSTREAM-FIXES.md`                              |
| `--- Logging error --- charmap '\u2588'`               | the vLLM banner prints a block character into a non-UTF-8 console | cosmetic; redirect the log to a file or set `PYTHONUTF8=1`                                                        |
| first start takes 2–4 minutes                          | Triton compiles its kernels once                                  | warm up with one short request before measuring                                                                   |

## Rollback

| file                                        | restore                                                                 |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| `vllm_quantpack\attention\int4_kv.py`       | the `encoding="utf-8"` argument is the only change; remove it           |
| `vllm_quantpack\attention\int8_prefill.py`  | new file — delete it and its two calls in `__init__.py`                 |
| `vllm\v1\attention\backends\triton_attn.py` | only if you applied the guard patch; a `.bak` copy next to it is enough |

`pip install --force-reinstall vllm-quantpack==0.1.5` restores Jorge's package untouched, which also
undoes the overlay.
