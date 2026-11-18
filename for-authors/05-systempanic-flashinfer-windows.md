# For the base port — `SystemPanic/flashinfer-windows`

<!-- Post as: a comment in issue #14 (the Windows JIT thread) or a new issue.
     Replace https://github.com/Astemiir/qwen38-27b-windows-vllm with the published repository URL first.
     Title: Delta on top of flashinfer-windows-rel for 0.7.2.post1 — five fixes, wheels published
-->

We took `flashinfer-windows-rel` from `v0.7.0.post1` to `0.7.2.post1` (the version vLLM 0.33 pins) and
got a full working port on an RTX 3090. Sharing the delta in case any of it belongs upstream of us:

| item            | what we changed                                                                    | why                                                                                              |
| --------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| host compile    | `cl.exe` for host objects, nvcc for `.cu` (including host files passed as `-x cu`) | `c++` and `-shared` do not exist here; one driver invocation keeps the objects consistent        |
| artifact suffix | `.dll` instead of `.so`, and the cached-module lookup taught the same suffix       | `core.py:476/484` and `env.py:206` look for `{name}.so` only                                     |
| paths           | short-path helper for every input, `_ninja_path` with `E$:` escaping               | nvcc and ninja split on `:`; `GetLongPathName` fails on deep spec directories                    |
| symbols         | shim for `fi_max`/`fi_min`, `/FORCE:UNRESOLVED` for unreachable instantiations     | MSVC has no GCC device builtins; some template variants are never instantiated                   |
| macros          | `#if MODE == …` inside `DISPATCH_*` arguments rewritten as `if constexpr`          | MSVC expands macro arguments before the preprocessor sees the directive — 743 blocks / 651 files |

Result: 29 `sm_80` modules build, load, and serve fp8 KV on Ampere — 1 182 tok/s prefill and 81.8 tok/s
decode at a 131 072-token window with vLLM 0.33, against 828 / 76.0 for the stock int8 path.

We published the wheels (`0.7.2.post1+win.1`: `flashinfer_python`, `flashinfer_jit_cache`,
`flashinfer_jit_cache_sm80`) so that nobody has to build anything, and the whole working setup — wheels,
installer, measurements — is at `https://github.com/Astemiir/qwen38-27b-windows-vllm`.

Two notes for anyone continuing this:

- The five items above are the difference between "cannot build" and "builds"; the recipe is
  `docs/build-windows-msvc.md` in our fork.
- We are not maintaining a Windows fork long-term: the port was a one-off to get fp8 KV on a 3090. If you
  want any of this merged into your branch, say which part and I will send it as a patch against
  `flashinfer-windows-rel` rather than a description.
