"""Check that the three FlashInfer wheels are installed and that a prebuilt kernel loads.

Run it with the interpreter that serves vLLM:

    set FLASHINFER_DISABLE_JIT=1
    python\\python.exe <repo>\\tools\\verify_flashinfer.py

No nvcc, no MSVC, no source tree: with JIT disabled, a kernel can only load from the prebuilt cache.
"""
import os
import sys

os.environ.setdefault("FLASHINFER_DISABLE_JIT", "1")

import torch
import flashinfer
import flashinfer.jit.env as env
from flashinfer.jit.attention.modules import _gen_batch_prefill_module

print("flashinfer:", flashinfer.__version__)
print("from:", flashinfer.__file__)
print("providers:", len(env.FLASHINFER_AOT_PROVIDERS))

spec = _gen_batch_prefill_module(
    "fa2", torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16, torch.int32,
    256, 256, 0, False, False, False,
    paged_kv_stride_mode="equal", module_surface="full",
)
mod = spec.try_load()
if mod is None:
    sys.exit("no prebuilt kernel for this spec: the three wheels do not match (all must be 0.7.2.post1+win.1)")
print("exports:", len([n for n in dir(mod) if not n.startswith("_")]))
print("ok")
