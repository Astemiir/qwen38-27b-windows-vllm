# Agent runbook — install and start Qwen3.8-27B on one RTX 3090 under Windows

You are an AI agent with shell access on the user's Windows machine. Follow this document literally.
It is written as RUN / EXPECT / IF NOT, because every step here has a silent failure mode and the
silent ones are the reason this document exists.

Do not upgrade, downgrade or "fix" any package version. Do not edit files except where told. If a step
does not produce its EXPECT line, take the IF NOT action and stop — do not improvise a workaround.

---

## 0. Collect four facts

Ask the user, or detect them:

| fact                                                                   | how                                                                     | default                |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------- |
| `KIT` — path to the vLLM Windows kit                                   | `dir /b C:\ D:\ E:\` and look for `vllm-windows-build-0.33.0-win-cu130` | none, must be known    |
| `MODEL` — path to a GPTQ W4A16 checkpoint with MTP head                | ask                                                                     | none                   |
| `PORT`                                                                 | ask                                                                     | `8010`                 |
| `BUNDLE` — path to the folder with `install.bat` and four `.whl` files | ask                                                                     | same repo as this file |

## 1. Preflight

RUN:

```bat
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
```

EXPECT: one line, `driver_version` **580.88 or higher**, `memory.total` about 24576 MiB, `memory.used`
under 1000 MiB.

IF NOT:

- driver below 580.88 → stop, tell the user to update the NVIDIA driver. Nothing else will work.
- `memory.used` above 2000 MiB → another model is holding the card. RUN (adjust the path):

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.ExecutablePath -like '<KIT>\*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

then re-check. Do not start a server on a partially occupied card.

RUN:

```bat
dir "<KIT>\python\python.exe"
```

EXPECT: the file exists.

IF NOT: the kit is not installed. Stop and tell the user to download it:

```
https://github.com/aivrar/vllm-windows-build/releases  ->  v0.33.0-win-cu130
```

and run its own `install.bat` once. Do not try to build vLLM from source.

## 2. Install the bundle

RUN:

```bat
"<BUNDLE>\install.bat" "<KIT>" nopause
```

EXPECT, in order:

```
torch 2.14.0+cu130 | cuda 13.0 | gpu NVIDIA GeForce RTX 3090
deps: tvm_ffi, numpy ok
Successfully installed flashinfer-jit-cache-0.7.2.post1+win.1 flashinfer-jit-cache-sm80-… flashinfer-python-0.7.2.post1+win.1
Successfully installed vllm-quantpack-0.1.5+win.1
patched: …\triton_attn.py            (or: already patched: …)
providers: 1
exports: 20
ok
```

IF NOT: the script stops with `[X] …` and names the broken step. Report that exact line to the user.
The usual causes: the kit was never installed (`torch is not importable`), or only some of the four
wheels were copied (`no prebuilt kernel for this spec`).

## 3. Confirm what got installed

RUN:

```bat
"<KIT>\python\python.exe" -c "import vllm, torch, triton, flashinfer, vllm_quantpack; print(vllm.__version__, torch.__version__, triton.__version__, flashinfer.__version__, vllm_quantpack.__version__)"
```

EXPECT:

```
0.33.0 2.14.0+cu130 3.8.0 0.7.2.post1+win.1 0.1.5+win.1
```

IF the last number is `0.1.5` without `+win.1`, the upstream wheel replaced ours: re-run step 2 and do
not `pip install vllm-quantpack` from PyPI afterwards.

## 4. Model

RUN:

```bat
dir "<MODEL>"
```

EXPECT: `config.json`, `tokenizer.json`, several `*.safetensors` shards, `*.safetensors.index.json`, and
a `.pth`/`mtp*` head file. Total size about 15 GB.

IF NOT: stop. The model must be a **GPTQ W4A16** checkpoint with an MTP head; a GGUF, an AWQ or a
plain bf16 checkpoint will not load here. Tell the user which folder is expected and why.

## 5. Start the server

RUN (one command; the `^` line continuations are for cmd):

```bat
cd /d "<KIT>"
set "PYTORCH_CUDA_ALLOC_CONF="
python\python.exe -m vllm.entrypoints.openai.api_server ^
  --model "<MODEL>" --served-model-name qwen3.8-27b ^
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
  --host 127.0.0.1 --port <PORT> > "<KIT>\log-agent-<PORT>.log" 2>&1
```

Run it as a background process; the first start takes 2–4 minutes while Triton compiles.

EXPECT, inside the log:

```
Application startup complete
GPU KV cache size: … tokens
vllm-quantpack: int4_per_token_head KV attention fixed (…
```

IF NOT:

| log contains                                     | meaning                                                        | action                                              |
| ------------------------------------------------ | -------------------------------------------------------------- | --------------------------------------------------- |
| `attention left stock`                           | the plugin's fast kernel did not load — wrong plugin installed | stop, re-run step 2, check step 3                   |
| `kv_cache_dtype not supported` with `FLASHINFER` | int8/KV dtype not in FlashInfer                                | you are on the wrong launch line; use the one above |
| `GPU mapping OOM`                                | `PYTORCH_CUDA_ALLOC_CONF` is set                               | clear the variable and restart                      |
| `Engine core initialization failed` within 5 s   | a kernel failed to build at warmup                             | report the traceback; do not retry blindly          |
| nothing after 8 minutes                          | first-start compile or a hang                                  | report the tail of the log                          |

## 6. Verify the endpoint

RUN:

```bat
"<KIT>\python\python.exe" "<BUNDLE>\smoke_test.py" <PORT>
```

EXPECT:

```
content: ready
usage: {'prompt_tokens': 19, …}
```

IF NOT: read the log tail. A server that answers `/health` but not `/v1/chat/completions` is usually
still loading weights or hit an OOM during warmup.

## 7. Report to the user

Give exactly this, filled in:

```
Installed:  vLLM 0.33.0, torch 2.14.0+cu130, Triton 3.8.0,
            FlashInfer 0.7.2.post1+win.1 (prebuilt sm_80 kernels),
            vllm-quantpack 0.1.5+win.1 (Windows fixes included)
Model:      <MODEL>, 262 144-token window, int4 KV in 6.8 GB
Server:     http://127.0.0.1:<PORT>/v1 , model name qwen3.8-27b
Verified:   smoke test answered "ready"; log shows "KV attention fixed"
Log:        <KIT>\log-agent-<PORT>.log
Flags that matter: --max-num-seqs 8, PYTORCH_CUDA_ALLOC_CONF empty
Expected speed on this card: ~1 178 tok/s prefill, ~85 tok/s decode at a 33k fill
```

## 8. Rules that must not be bent

1. **Never** `pip install --upgrade torch`, `vllm` or `flashinfer` to "fix" something. The versions in
   step 3 are the ones the numbers were measured on; anything newer is untested here.
2. **Never** install `vllm-quantpack` from PyPI after step 2. Upstream 0.1.5 silently disables its fast
   int4 kernel on Windows code pages, and the symptom is 28 tok/s decode with no error.
3. **Never** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`. It aborts startup on Windows.
4. **Never** lower `--max-num-seqs` below `1 + num_speculative_tokens` (4 here). 8 is tested.
5. Keep the `.bak-…` files that `patch_vllm.py` leaves. They are the only rollback.
6. If the user wants a different KV format, use the table below; do not invent flags.

## 9. Other configurations that are known to work

| goal                                | KV dtype              | backend       | window  | KV size | notes                                                  |
| ----------------------------------- | --------------------- | ------------- | ------- | ------- | ------------------------------------------------------ |
| 262k, fastest prefill (default)     | `int4_per_token_head` | `TRITON_ATTN` | 262 144 | 6.8 GB  | needs the plugin wheel from this bundle                |
| 262k, better decode at the very end | `k8v4_per_token_head` | `TRITON_ATTN` | 262 144 | 8.8 GB  | +5.4 % decode at 246k, −6 % prefill                    |
| 131k, fp8 without the plugin        | `fp8_e4m3`            | `FLASHINFER`  | 131 072 | 6.5 GB  | replace `--attention-backend` and dtype; keep the rest |
| 131k, int8                          | `int8_per_token_head` | `TRITON_ATTN` | 131 072 | 3.3 GB  | prefill fixed by the plugin wheel                      |

## 10. Stopping the server

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.ExecutablePath -like '<KIT>\*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

EXPECT after ~5 s: `0 MiB`. Killing only the `api_server` process leaves the EngineCore child holding
all of VRAM — the path filter above is the reliable form.
