"""Check the three wheels the way a user gets them: installed, no source tree.

Environment:
    FI_DIST_TEST  directory with the three installed wheels (default C:\\fi\\dist-test)
Run it with the same interpreter that will serve vLLM (it needs torch and tvm_ffi).
"""
import os
import pathlib
import sys

DIST = pathlib.Path(os.environ.get("FI_DIST_TEST", r"C:\fi\dist-test"))
sys.path.insert(0, str(DIST))
os.environ["FLASHINFER_DISABLE_JIT"] = "1"

import torch  # noqa: E402
import flashinfer  # noqa: E402
import flashinfer.jit.env as env  # noqa: E402
from flashinfer.jit.attention.modules import _gen_batch_prefill_module  # noqa: E402

print("flashinfer:", flashinfer.__version__)
print("from:", flashinfer.__file__)
assert str(DIST) in flashinfer.__file__, "the installed wheel was not imported"

print("providers:", len(env.FLASHINFER_AOT_PROVIDERS))
prov = env.FLASHINFER_AOT_PROVIDERS[0]
print(f"provider: {prov.provider_id} ver={prov.version} modules={len(prov.modules)}")

spec = _gen_batch_prefill_module(
    "fa2", torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16, torch.int32,
    256, 256, 0, False, False, False,
    paged_kv_stride_mode="equal", module_surface="full",
)
arts = spec.aot_artifacts
print(f"artifacts={len(arts)} path={len(str(arts[0].path))} chars")
print("   ", arts[0].path)
mod = spec.try_load()
print("try_load=", type(mod).__name__ if mod else "None")
if mod is not None:
    print("exports:", len([n for n in dir(mod) if not n.startswith("_")]))
