# For FlashInfer upstream — `flashinfer-ai/flashinfer`

<!-- Post as: new issue. Replace https://github.com/Astemiir/qwen38-27b-windows-vllm with the published repository URL first.
     Title: Building the JIT cache on Windows: five blockers, and a mechanical patch for all of them
-->

We ported FlashInfer `0.7.2.post1` to Windows to get fp8 KV on Ampere (vLLM 0.33, RTX 3090, CUDA 13.0,
torch 2.14). It works: 29 `sm_80` modules build and load, and `--attention-backend FLASHINFER
--kv-cache-dtype fp8_e4m3` gives 1 182 tok/s prefill / 81.8 decode at a 131k window. The JIT generator
itself is Linux-only by construction, so the port is a patch set rather than a config change. Below are
the five blockers in the order they bite, with the fix we used for each. All of it is mechanical; we are
happy to send it as a PR.

| #   | symptom                                                    | cause                                                                                                                                         | fix in our port                                                                                                             |
| --- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 1   | `FileNotFoundError: c++`                                   | host compiler hardcoded as `c++`, with GCC flags (`-shared`, `-fPIC`) — see `flashinfer/jit/cpp_ext.py:303`                                   | `cl.exe` for host objects, nvcc for `.cu`, `/LD` to link                                                                    |
| 2   | `error: unrecognized file type` on `.cu`                   | the host compile step receives CUDA sources                                                                                                   | compile `.cu` with nvcc, and pass host files as `-x cu` in the same driver invocation                                       |
| 3   | `fatal error: cannot open input file …` on `E:\…`          | nvcc and ninja treat `:` as a separator, and `GetLongPathName` fails on long spec paths                                                       | route every path through a short-path helper; export `_ninja_path` with `E$:` escaping                                      |
| 4   | `LNK2019: unresolved external fi_max / fi_min`             | those macros are defined for GCC device code only                                                                                             | shim header mapping `fi_max`/`fi_min` to `max`/`min`; `/FORCE:UNRESOLVED` for genuinely unreachable template instantiations |
| 5   | `error C2059: syntax error: '=='` across hundreds of files | `#if MODE == …` appears **inside** a `DISPATCH_*` macro argument, and MSVC expands macro arguments before the preprocessor sees the directive | rewrite those blocks as `if constexpr` — 743 blocks across 651 files, scriptable                                            |

Two more things worth a look, independent of Windows:

1. **The artifact path is `.so`-only.** `flashinfer/jit/core.py:476` and `:484` hardcode
   `f"{name}.so"`, and `flashinfer/jit/env.py:206` looks for `{short}.so` when resolving a cached module.
   On Windows the artifact is a `.dll` loaded through `tvm_ffi`, so the lookup misses a cache that is
   actually present. A platform-dependent suffix in one place would cover it.
2. **`spec.build()` returns `None`** even when it succeeds — the artifact is at
   `spec.jit_library_path`. Anyone writing an installer will read that `None` as a failure. Returning the
   path, or documenting the return value, would remove the trap.

On the wheel side, if prebuilt caches are ever published for Windows: the `cu130` index has artifacts for
`sm_80` and Blackwell targets but nothing for 8.6/8.9, so even a successful local build cannot be reused
by someone with the same card. One `sm_80` cache covers all three (`sm_80` SASS loads on `sm_86`/`sm_89`
per the runtime's own compatibility table) — that is what we ship.

What we have: the build tree (651 modified files), two source patches (`windows_patch_format.patch`,
`windows_patch_stride.patch`), the shim header, and a script that produces a `jit_cache` wheel plus a
`flashinfer_python` wheel. Recipe: `docs/build-windows-msvc.md` in our fork, wheels and bundle at `https://github.com/Astemiir/qwen38-27b-windows-vllm`.

The port is a one-off on our side — we are not trying to maintain a Windows fork. If there is interest in
taking any of this upstream, tell me which part and I will prepare that piece as a PR rather than a patch
file.
