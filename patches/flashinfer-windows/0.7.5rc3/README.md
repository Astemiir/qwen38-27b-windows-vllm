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
