"""Build the AOT kernel set (Windows, Ampere) from a source tree.

Environment:
    FI_ROOT     build root, short path (default C:\\fi); `src`, `b`, `pkg` live there
See docs/build-windows-msvc.md for the full recipe.
"""
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(os.environ.get("FI_ROOT", r"C:\fi"))
PROJECT = ROOT / "src"
sys.path.insert(0, str(PROJECT))

import torch  # noqa: E402
from flashinfer import aot  # noqa: E402

OUT = ROOT / "pkg" / "flashinfer_jit_cache" / "providers" / "sm80" / "jit_cache"
BUILD = ROOT / "b" / "aot"
VERSION = os.environ.get("FI_VERSION", "0.7.2.post1+win.1")

CONFIG = {
    "fa2_head_dim": [(128, 128), (256, 256)],
    "fa3_head_dim": [],
    "f16_dtype": [torch.bfloat16, torch.float16],
    "f8_dtype": [torch.float8_e4m3fn],
    "use_sliding_window": [False],
    "use_logits_soft_cap": [False],
    "add_comm": False,
    "add_gemma": False,
    "add_oai_oss": False,
    "add_moe": False,
    "add_act": False,
    "add_misc": False,
    "add_xqa": False,
}

OUT.mkdir(parents=True, exist_ok=True)
BUILD.mkdir(parents=True, exist_ok=True)

print("arch list:", os.environ.get("FLASHINFER_CUDA_ARCH_LIST"))
print("source:", PROJECT)
print("out:", OUT)
print("build:", BUILD)
t0 = time.time()

aot.compile_and_package_modules(
    out_dir=OUT,
    build_dir=BUILD,
    project_root=PROJECT,
    config=CONFIG,
    verbose=True,
    skip_prebuilt=False,
)

# Module list comes from what was actually written: directories holding a .so.
mods = sorted(
    p.name for p in OUT.iterdir() if p.is_dir() and any(p.glob("*.so"))
)
manifest = {
    "schema_version": 1,
    "provider_id": "sm80",
    "distribution": "flashinfer-jit-cache-sm80",
    "version": VERSION,
    "cuda_architectures": ["sm80"],
    "modules": mods,
}
(OUT.parent / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
print("modules:", len(mods))
for m in mods:
    size = next((OUT / m).glob("*.so")).stat().st_size
    print(f"   {size / 1024**2:7.2f} MB  {m}")
