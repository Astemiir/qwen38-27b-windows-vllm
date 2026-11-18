"""Package the flashinfer-jit-cache shim (pure Python, same version as the kernels).

The shim is the discovery package: it reads the entry points in the group
`flashinfer.jit_cache.providers` and hands FlashInfer the provider manifests.
Without it, an installed kernel wheel is invisible.

Environment: FI_ROOT (default C:\\fi), FI_VERSION (default 0.7.2.post1+win.1).
"""
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(os.environ.get("FI_ROOT", r"C:\fi"))
WORK = ROOT / "shim-src"
PKG = WORK / "flashinfer_jit_cache"
SHIM_INIT = pathlib.Path(__file__).with_name("shim_init.py")
VERSION = os.environ.get("FI_VERSION", "0.7.2.post1+win.1")

if WORK.exists():
    shutil.rmtree(WORK)
PKG.mkdir(parents=True)

shutil.copy2(SHIM_INIT, PKG / "__init__.py")
(PKG / "_build_meta.py").write_text(
    f'__version__ = "{VERSION}"\n__git_version__ = "v{VERSION}"\n', encoding="utf-8"
)

(WORK / "setup.py").write_text(
    f'''from setuptools import setup

setup(
    name="flashinfer-jit-cache",
    version="{VERSION}",
    description="FlashInfer JIT cache provider discovery shim (Windows build)",
    license="Apache-2.0",
    url="https://github.com/flashinfer-ai/flashinfer",
    python_requires=">=3.9",
    packages=["flashinfer_jit_cache"],
    package_data={{"flashinfer_jit_cache": ["_build_meta.py"]}},
    zip_safe=False,
)
''',
    encoding="utf-8",
)

r = subprocess.run(
    [sys.executable, "setup.py", "bdist_wheel"],
    cwd=str(WORK),
    capture_output=True,
    text=True,
    errors="replace",
)
print("rc:", r.returncode)
print("\n".join(((r.stdout or "") + (r.stderr or "")).splitlines()[-6:]))

dst = ROOT / "dist-win"
dst.mkdir(exist_ok=True)
for w in sorted((WORK / "dist").glob("*.whl")):
    target = dst / w.name
    shutil.copy2(w, target)
    print(f"shim: {target.name}  {target.stat().st_size / 1024:.1f} KB")
