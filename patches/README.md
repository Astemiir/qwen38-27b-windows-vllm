# Patches

Most people do not need anything on this page: `bundle/install.bat` installs a plugin wheel that
already contains the quantpack patch. The files here are for auditing the change, for applying it to an
upstream install you already have, or for sending it on to the plugin's author.

Everything needed to make a stock install behave like the numbers in `PERFORMANCE.md`. Three files,
two one-line changes and one new module.

| patch                                             | target                      | changes                 | needed for                                              |
| ------------------------------------------------- | --------------------------- | ----------------------- | ------------------------------------------------------- |
| `quantpack/quantpack-0.1.5-windows.patch`         | `vllm_quantpack` 0.1.5      | 3 files, +52 / −2 lines | int4, k8v4, int8, fp8 per-token-head                    |
| `vllm/vllm-triton-fp8-per-token-head-guard.patch` | vLLM 0.33                   | 1 guard, +4 / −3 lines  | `fp8_per_token_head` on the Triton backend              |
| `quantpack/attention/int8_prefill.py`             | `vllm_quantpack/attention/` | new module, 8.8 KB      | included inside the first patch; here for hand installs |

Both patches were generated from the published wheels, not from a working tree, and both were checked
with `git apply --check` against a clean extraction.

## Apply

From the package directory, so that the `a/…` and `b/…` prefixes strip correctly:

```bat
cd python\Lib\site-packages
git apply -p1 <path-to-repo>\patches\quantpack\quantpack-0.1.5-windows.patch
git apply -p1 <path-to-repo>\patches\vllm\vllm-triton-fp8-per-token-head-guard.patch
```

Without git, GNU `patch -p1` reads the same files. Both patch programs handle the new file
(`int8_prefill.py`) and its registration in `__init__.py`.

## What is inside the quantpack patch

| file                        | change                                                        | why                                                                                                                                                                       |
| --------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `attention/int4_kv.py`      | `os.fdopen(fd, "w")` → `os.fdopen(fd, "w", encoding="utf-8")` | the generated kernel source contains U+2192, which no Western ANSI code page can encode; without `encoding` the write throws and the fast int4 kernel is silently skipped |
| `attention/int8_prefill.py` | new module                                                    | int8 and fp8 per-token-head caches had no fast prefill path; this unpacks long prefixes in a Triton kernel and hands them to FlashAttention: 828 → 1 172 tok/s            |
| `attention/int4_prefill.py` | `_unpack_segment_fp8` kernel + branch in `unpack_segment`     | the same unpack path can serve an fp8 cache: the storage is `uint8` viewed as `float8_e4m3fn`, so one code per channel and no bit tricks                                  |
| `__init__.py`               | import + two calls                                            | wires the hook into the plugin's `register()`                                                                                                                             |

The patch does **not** touch the int4 decode kernel, k8v4, EXL3, TurboQuant, or any model family code.

`int8_prefill.py` reads two environment variables, both optional:

| variable                           | meaning                                                      |
| ---------------------------------- | ------------------------------------------------------------ |
| `VLLM_QUANTPACK_INT8_FA_PREFILL=0` | disable the prefill hook entirely                            |
| `VLLM_QUANTPACK_INT8_SEGMENTS=N`   | split-KV segments for the int8 decode path (auto by default) |

The segment knob is kept because it is harmless, not because it helps: decode on these caches is
arithmetic-bound (≈800 MB per step at 246k, ≈2 % of the card's bandwidth), so changing the split does
not move it.

## What is inside the vLLM patch

`vllm/v1/attention/backends/triton_attn.py:525` rejects every `kv_cache_dtype` starting with `fp8` on
cards below SM89. For `fp8_per_token_head` that test is wrong: the cache is stored as `uint8`, viewed as
`float8_e4m3fn`, and dequantized with an explicit per-head, per-token scale inside the Triton kernel —
the same arithmetic `int8_per_token_head` uses on the same cards.

The patch narrows the guard with `and not kv_cache_dtype.endswith("per_token_head")`. It does not touch
the native `fp8e4nv` path, which still requires SM89.

Be aware of what this buys: nothing in speed. `fp8_per_token_head` measured 1 185 / 76.8 against int8's
1 172 / 76.0. The patch exists so a valid configuration can start, not because it is fast.

## FlashInfer

`flashinfer-windows/` holds the two source patches from the FlashInfer Windows port
(`windows_patch_format.patch`, `windows_patch_stride.patch`), and `../tools/` holds the build and wheel
scripts. They are not needed to use the prebuilt wheels — only to rebuild them. The full recipe is
`docs/build-windows-msvc.md` in the [FlashInfer fork](https://github.com/Astemiir/flashinfer-windows).

## Rollback

```bat
cd python\Lib\site-packages
git apply -R -p1 <path-to-repo>\patches\quantpack\quantpack-0.1.5-windows.patch
git apply -R -p1 <path-to-repo>\patches\vllm\vllm-triton-fp8-per-token-head-guard.patch
```

`pip install --force-reinstall vllm-quantpack==0.1.5` restores the plugin from the wheel without
touching vLLM; the reverse of the vLLM patch is the only thing that has to be undone by hand.
