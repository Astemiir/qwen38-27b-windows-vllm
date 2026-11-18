# The stack

Nothing here is one project. It is six independent efforts stacked on each other, each written for
Linux x86 by default, and the reason a 262k context works on a Windows desktop is that all six lines up.

```
        ┌──────────────────────────────────────────────────────────────┐
 model  │ Qwen3.8-27B                      Alibaba                     │
        │ GPTQ W4A16 + MTP head            JRamirez-UAB (Jorge)        │
        │ DSpine quantization method       Tencent                     │
        ├──────────────────────────────────────────────────────────────┤
 plugin │ vllm-quantpack 0.1.5             JRamirez-UAB + our overlay  │
        │ int4 / int8 / fp8 / k8v4 KV      E:\vLLM-0.33\pub\patches    │
        ├──────────────────────────────────────────────────────────────┤
 engine │ vLLM 0.33                        vLLM project                │
        │ Windows build 0.33.0-win-cu130   aivrar (Chris)              │
        ├──────────────────────────────────────────────────────────────┤
 kernels│ FlashInfer 0.7.2.post1           flashinfer-ai (upstream)    │
        │ Windows port + sm_80 wheels      SystemPanic, then us        │
        ├──────────────────────────────────────────────────────────────┤
 base   │ Triton 3.8, PyTorch 2.14.0+cu130, CUDA 13.0 runtime,         │
        │ cuDNN 9.24, NVIDIA driver 580.88+                            │
        └──────────────────────────────────────────────────────────────┘
```

## Layer by layer

| layer     | component                                             | author                                                           | where                                                                         | our role                                         |
| --------- | ----------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------ |
| model     | Qwen3.8-27B, 27B dense, hybrid attention              | Alibaba                                                          | [Qwen](https://huggingface.co/Qwen)                                           | user                                             |
| method    | DSpine post-training quantization                     | Tencent                                                          | [paper / repo](https://github.com/Tencent)                                    | user                                             |
| weights   | GPTQ W4A16 + MTP head, `…-mtp-int8`                   | [JRamirez-UAB](https://huggingface.co/JRamirez-UAB)              | Hugging Face                                                                  | user                                             |
| plugin    | `vllm_quantpack` 0.1.5, Apache-2.0                    | [jramirez-uab](https://github.com/jramirez-uab/vllm-quantpack)   | GitHub                                                                        | 4 fixes: 2 crash-class, 1 perf-class, 1 dead-end |
| engine    | vLLM 0.33                                             | [vllm-project](https://github.com/vllm-project/vllm)             | GitHub                                                                        | 1 guard widened                                  |
| build     | `v0.33.0-win-cu130`, portable Python 3.13.14          | [aivrar](https://github.com/aivrar/vllm-windows-build) (Chris)   | GitHub                                                                        | documentation fixes sent                         |
| kernels   | FlashInfer 0.7.2.post1                                | [flashinfer-ai](https://github.com/flashinfer-ai/flashinfer)     | GitHub                                                                        | Windows port                                     |
| port base | `flashinfer-windows-rel` branch                       | [SystemPanic](https://github.com/SystemPanic/flashinfer-windows) | GitHub                                                                        | predecessor of our port                          |
| wheels    | `0.7.2.post1+win.1`, three wheels, 29 `sm_80` modules | us                                                               | [Astemiir/flashinfer-windows](https://github.com/Astemiir/flashinfer-windows) | author                                           |

## What each layer contributes to the number

| layer               | contributes                                                                                                            | if this layer is missing or wrong                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| weights             | 15.4 GB of 4-bit weights that fit in 24 GB with room for KV                                                            | a 16-bit 27B needs 54 GB — not a 24 GB card                                                        |
| MTP head            | three draft tokens verified per step — on the same GGUF model llama.cpp decodes 79–89 tok/s with it against 36 without | decode falls by more than half; the int4 launcher also needs a batch ≥ 4 to keep its split-KV path |
| plugin              | the KV formats: int4 to 262k, k8v4, int8, fp8 per-token-head                                                           | stock vLLM has none of these dtypes                                                                |
| plugin patch (ours) | int4 prefill 858.7 → 1 178, decode 28.0 → 85.5                                                                         | int4 is slower than int8 and looks useless                                                         |
| vLLM Windows build  | everything runs at all on Windows                                                                                      | no native vLLM for Windows upstream                                                                |
| FlashInfer wheels   | fp8 KV on `sm_86`, +18 % prefill at equal KV, +7.6 % decode                                                            | fp8 KV impossible on 8.x cards; bf16 only                                                          |
| Triton              | the int8/int4/fp8 decode kernels                                                                                       | no quantized KV at all                                                                             |

## The two facts that shape this document

1. **Upstream FlashInfer has no Windows wheel.** The JIT generator is Linux-only — it invokes `c++`,
   emits `.so`, passes GCC flags, and chokes on the `:` in `E:\` paths. A port needs nvcc for `.cu`
   *and* for host files, `.dll` outputs, short paths, an unresolved-symbol policy, and a shim for
   `fi_max`/`fi_min`. Ours ships prebuilt so that no user ever needs nvcc.

2. **The plugin disables itself on Windows.** It generates a CUDA source file through
   `os.fdopen(fd, "w")`, which uses the ANSI code page; the file contains an arrow character that no
   Western code page can encode; the resulting `UnicodeEncodeError` is caught and logged as one WARNING
   among hundreds of startup lines — `int4_per_token_head attention left stock`. The server starts,
   serves, and runs the slow kernel. This is why "it works for the author" and "it is slow for me" are
   both true.

## What we did not do

- We did not fork vLLM. One guard line changed, documented in `UPSTREAM-FIXES.md`.
- We did not modify the weights. The plugin and the MTP head are used as shipped.
- We did not write the KV kernels. They are Jorge's; our changes are the prefill path, an encoding
  fix, a segment knob and a dtype branch.
- FlashInfer is ported, not rewritten: same sources, Windows toolchain, prebuilt `sm_80` cache.
