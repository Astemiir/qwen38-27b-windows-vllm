# FlashInfer 0.7.5rc3 on Windows — what it takes

Verified on 2026-11-18: with these files and the environment below, `flashinfer-python==0.7.5rc3` builds
and runs kernels on Windows 11 / RTX 3090 / CUDA 13.3 / MSVC 14.44 — no upstream patch exists for this path.
Upstream's merged Windows work (#9415 cubins as COFF, #9462 arm64 CI driver, #9481 nvcc lookup) does not
touch the JIT generator that compiles kernels from source.

## Result

```
flashinfer: 0.7.5rc3
OK: import
rmsnorm out: (4, 128) torch.bfloat16        OK: rmsnorm JIT
prefill out: (8, 8, 128) torch.bfloat16     OK: prefill JIT (bf16, head_dim 128)
```

## The five things that block a stock install

| # | symptom | cause | fix |
| --- | --- | --- | --- |
| 1 | `ninja: error: build.ninja:71: expected build command name` | `build E:\...\norm.so: link …` — the drive colon is not escaped | `_ninja_path()` escapes `:` as `$:` and shortens names; artifacts become `m<md5[:10]>.dll` |
| 2 | `nvcc fatal : A single input file is required …` | `$cuda_home/bin/nvcc` with a space in `C:\Program Files\…` | point `FLASHINFER_NVCC` at a space-free nvcc (`C:/cuda133/bin/nvcc.exe`) and set `CUDA_HOME`/`CUDA_PATH` to the same tree |
| 3 | `fatal error C1189: MSVC/cl.exe with traditional preprocessor is used` | CCCL in 0.7.5 requires `/Zc:preprocessor` | `set NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor` |
| 4 | `error: identifier "ushort"/"uint" is undefined`, `__attribute__` unknown | Linux-only types and GCC attributes in the headers | force-include the shim: `set FLASHINFER_EXTRA_CUDAFLAGS=--pre-include flashinfer/flashinfer_win_shim.cuh` |
| 5 | access violation on the first kernel call | the DLL links without `tvm_ffi.lib`, so its imports are unresolved | `set FLASHINFER_WIN_LDFLAGS=<site-packages>/tvm_ffi/lib/tvm_ffi.lib`; the link rule then depends on `tvm_ffi.dll` |

One source fix rides along: `norm/fused_qk_rmsnorm_rope.cuh:469` — `tokenIdx % seq_len` is ambiguous for MSVC
(`int64 % IntFastDiv`), needs `int(tokenIdx) % seq_len`.

## Use

```bat
:: 1) install the package somewhere isolated
pip install --no-deps --target <dir> flashinfer-python==0.7.5rc3

:: 2) drop in the three patched modules and the shim (back up the originals first)
copy cpp_ext.py env.py core.py   <dir>\flashinfer\jit\
copy flashinfer_win_shim.cuh     <dir>\flashinfer\data\include\flashinfer\

:: 3) patch the sources MSVC cannot parse (constexpr, std::max → fi_max, the modulo cast)
set FI_SRC=<dir>
python patch_win_source.py

:: 4) build with this environment
set CUDA_HOME=C:\cuda133
set CUDA_PATH=C:\cuda133
set FLASHINFER_NVCC=C:/cuda133/bin/nvcc.exe
set NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor
set FLASHINFER_EXTRA_CUDAFLAGS=--pre-include flashinfer/flashinfer_win_shim.cuh
set FLASHINFER_WIN_LDFLAGS=<dir>/tvm_ffi/lib/tvm_ffi.lib
set FLASHINFER_WORKSPACE_BASE=<workdir>
call vcvars64.bat
```

`patch_win_source.py` lives in `tools/` at the repository root (`<repo>/tools/patch_win_source.py`); it
patches `csrc`/`include` of a tree given by `FI_SRC` and expects `<FI_SRC>/flashinfer/jit`,
`<FI_SRC>/include`, `<FI_SRC>/csrc`, which in an installed layout means junctions to
`flashinfer/data/include` and `flashinfer/data/csrc`.

Note on the interpreter: the portable Python inside the vLLM kit **ignores `PYTHONPATH`** (embedded build),
so an isolated install has to be put on `sys.path` from inside the script.

## What these files are

| file | change |
| --- | --- |
| `cpp_ext.py` | 0.7.5rc3 plus our Windows branches: `_ninja_path`, `_short_artifact`, nvcc for `.cu` and `.dll` artifacts, `nvcc_link` with `/FORCE:UNRESOLVED` and `FLASHINFER_WIN_LDFLAGS`, POSIX flag filter |
| `core.py` | `jit_library_name`/`jit_library_path` use the short `.dll` on Windows; `-Wno-switch-bool` dropped; `subninja` path escaped |
| `env.py` | workspace base from `FLASHINFER_WORKSPACE_BASE`, short hashed spec dirs |
| `flashinfer_win_shim.cuh` | `ushort`, `uint`, `uchar`, `__attribute__`, `__restrict__`, `M_PI`, `fi_max`/`fi_min` |

Measured timings on this machine: `norm` ≈ 20 s per object, `single_prefill` (bf16, head_dim 128) ≈ 6 objects
× ~40 s with `--threads=1`.

## Full AOT build

`aot.compile_and_package_modules` (config: FA2 head dims 128 and 256, bf16/fp16 + fp8_e4m3, no sliding
window, no soft cap) produced **33 modules** for `sm_80` in one run, and the resulting cache works as a
provider wheel: `providers: 1`, `aot_artifacts: 1`, `try_load() -> Module` with 20 exports under
`FLASHINFER_DISABLE_JIT=1`.

Four extra things the AOT path needs beyond the five above:

| item | symptom | fix |
| --- | --- | --- |
| `python3.lib` | `LNK1104: cannot open python3.lib` in the `spdlog` module | add `-L<kit>/python/libs` to `FLASHINFER_WIN_LDFLAGS` |
| fmt literal | `character value is out of range` in `fmt/bundled/format.h:3273` | that `U"\x9999999a…"` literal exceeds `char32_t` range for nvcc; replaced with a `switch` returning the same constants |
| MLA | `argument list for template … is missing` in `attention/mla.cuh` | not needed by the vLLM path; `FI_NO_MLA=1` skips the MLA specs (patch guard in `aot.py`) |
| cuDNN FMHA | `expression must be a pointer to a complete object type` in `cudnn_sdpa_kernel_launcher.cu` | same: `FI_NO_CUDNN=1` skips it |

Kernel-level timings of the built cache (RTX 3090, bf16, 8 heads × 128, no model weights — this measures
attention only):

| workload | time | tokens/s |
| --- | --- | --- |
| prefill, 33 000 tokens | 33.9 ms | 973 555 |
| prefill, 246 000 tokens | 1 679 ms | 146 495 |
| decode, 33k context (KV read) | 0.17 ms/token | 6 043 |
| decode, 246k context (KV read) | 1.14 ms/token | 874 |

Both decode points sit at ~800–880 GB/s of KV reads, i.e. at the card's memory bandwidth — the same ceiling
that sets vLLM's decode at 85 tok/s at 33k once the 15.4 GB of weights are added to each step.

The wheels built from that run: `dist/flashinfer_jit_cache_sm80-0.7.5rc3+win.1-cp39-abi3-win_amd64.whl`
(27.5 MB, 33 modules) and the 3.4 KB discovery shim `flashinfer_jit_cache-0.7.5rc3+win.1-py3-none-any.whl`.
Both are published as a **pre-release**:
<https://github.com/Astemiir/flashinfer-windows/releases/tag/v0.7.5rc3-win.1> — marked pre-release on purpose,
so `0.7.2.post1-win.1` stays the recommended build. Install both together: without the shim the provider is
invisible and `providers: 0`.

Engine comparison (vLLM 0.33, `fp8_e4m3` + `FLASHINFER`, 131 072 window, MTP n3): 33 366 tokens —
1 194.8 / 81.5 on 0.7.5rc3 versus 1 182 / 81.8 on 0.7.2.post1; 97 728 tokens — 910.8 / 66.8 versus
912.1 / 68.4. Equal within noise, as expected for a release line aimed at Blackwell and Rubin.

