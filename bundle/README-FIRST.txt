vLLM 0.33 + Qwen3.8-27B on one RTX 3090, Windows
=================================================

This folder is the whole solution. Nothing else needs to be downloaded from the internet except
the model weights.

What is here
------------
  install.bat                                               one-step installer
  flashinfer_python-0.7.2.post1+win.1-py3-none-any.whl      FlashInfer, Windows port
  flashinfer_jit_cache-0.7.2.post1+win.1-py3-none-any.whl   prebuilt-kernel namespace
  flashinfer_jit_cache_sm80-0.7.2.post1+win.1-*.whl         29 kernels for sm_80 (loads on 8.6, 8.9)
  vllm_quantpack-0.1.5+win.1-py3-none-any.whl               quantpack 0.1.5 with the Windows fixes baked in
  patch_vllm.py                                             applies one guard fix to vLLM in the kit
  verify_flashinfer.py                                      proves the prebuilt kernels load
  smoke_test.py                                             one request against a running server

Step 1 - the vLLM kit
---------------------
Download aivrar's vLLM 0.33 Windows kit and run ITS install.bat once:
  https://github.com/aivrar/vllm-windows-build/releases  ->  v0.33.0-win-cu130
It unpacks a portable Python 3.13 and installs torch. You end up with a folder like
  D:\vllm-windows-build-0.33.0-win-cu130\python\python.exe

Step 2 - this folder
--------------------
Run, from a normal command prompt:

  install.bat "D:\vllm-windows-build-0.33.0-win-cu130"

It installs the four wheels (no downloads - they are right here), patches one line in vLLM, and
verifies that a prebuilt kernel loads. Expected last line: ok

Step 3 - the model
------------------
A GPTQ W4A16 checkpoint with the MTP head, about 15 GB. Example used for all our numbers:
  https://huggingface.co/JRamirez-UAB  ->  a Qwen3.8-27B GPTQ W4A16 checkpoint, "...-mtp-int8"
Put it in any folder and pass that path as --model.

Step 4 - start and check
------------------------
The launch line is in INSTALL.md, step 4 of the repository this folder came with. After the server
says "Application startup complete", run:

  "D:\vllm-windows-build-0.33.0-win-cu130\python\python.exe" smoke_test.py 8010

Expected: content: ready

Two flags that silently cost 3x speed if forgotten:
  --max-num-seqs 8            (int4 KV; below 4 the split-KV decode path is skipped)
  set "PYTORCH_CUDA_ALLOC_CONF="   (must be empty on Windows)

Requirements: Windows 10/11 x64, NVIDIA driver 580.88 or newer, one 24 GB card
(RTX 3090 tested; sm_80 / 8.6 / 8.9 all work).

If you would rather have an assistant do it: hand it AGENT-INSTALL.md from the repository.
