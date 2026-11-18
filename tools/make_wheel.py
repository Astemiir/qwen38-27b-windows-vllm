"""Package the compiled kernels into flashinfer_jit_cache_sm80-<version>-cp39-abi3-win_amd64.whl.

Expects build_jitcache.py to have produced pkg/flashinfer_jit_cache/providers/sm80/jit_cache.
Environment: FI_ROOT (default C:\\fi), FI_VERSION (default 0.7.2.post1+win.1).
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(os.environ.get("FI_ROOT", r"C:\fi"))
SRC_PKG = ROOT / "pkg" / "flashinfer_jit_cache" / "providers" / "sm80"
WHEEL_SRC = ROOT / "wheel-src"
PROVIDER = WHEEL_SRC / "flashinfer_jit_cache_provider"
PROVIDER_INIT = pathlib.Path(__file__).with_name("provider_init.py")
VERSION = os.environ.get("FI_VERSION", "0.7.2.post1+win.1")

if WHEEL_SRC.exists():
    shutil.rmtree(WHEEL_SRC)
PROVIDER.mkdir(parents=True)

shutil.copy2(PROVIDER_INIT, PROVIDER / "__init__.py")
(PROVIDER / "_build_meta.py").write_text(
    f'__version__ = "{VERSION}"\n__git_version__ = "v{VERSION}"\n',
    encoding="utf-8",
)

manifest = json.loads((SRC_PKG / "manifest.json").read_text(encoding="utf-8"))
manifest["version"] = VERSION
# Modules are counted from the tree: a kernel directory is one that holds a .so.
# After the short-name change these are s<md5[:12]> directories, and flashinfer
# accepts either spelling when it resolves them.
manifest["modules"] = sorted(
    p.name
    for p in (SRC_PKG / "jit_cache").iterdir()
    if p.is_dir() and any(p.glob("*.so"))
)
(PROVIDER / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("modules in manifest:", len(manifest["modules"]))

shutil.copytree(SRC_PKG / "jit_cache", PROVIDER / "jit_cache")

(WHEEL_SRC / "setup.py").write_text(
    f'''from setuptools import setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # older setuptools
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class bdist_wheel(_bdist_wheel):
    """The provider is not pure Python: tag it cp39-abi3 and platform-specific."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False
        self.py_limited_api = "cp39"

    def get_tag(self):
        _, _, platform_tag = super().get_tag()
        return "cp39", "abi3", platform_tag


PKG = "flashinfer_jit_cache.providers.sm80"

setup(
    name="flashinfer-jit-cache-sm80",
    version="{VERSION}",
    description="FlashInfer pre-compiled JIT cache provider for sm80 (Windows build)",
    license="Apache-2.0",
    url="https://github.com/flashinfer-ai/flashinfer",
    python_requires=">=3.9",
    packages=[PKG],
    package_dir={{PKG: "flashinfer_jit_cache_provider"}},
    package_data={{PKG: ["manifest.json", "jit_cache/**/*.so", "jit_cache/**/*.dll"]}},
    include_package_data=False,
    zip_safe=False,
    entry_points={{"flashinfer.jit_cache.providers": ["sm80=" + PKG + ":get_provider"]}},
    cmdclass={{"bdist_wheel": bdist_wheel}},
)
''',
    encoding="utf-8",
)

r = subprocess.run(
    [sys.executable, "setup.py", "bdist_wheel", "--py-limited-api", "cp39"],
    cwd=str(WHEEL_SRC),
    capture_output=True,
    text=True,
    errors="replace",
)
print("rc:", r.returncode)
print("\n".join(((r.stdout or "") + (r.stderr or "")).splitlines()[-8:]))

for w in sorted((WHEEL_SRC / "dist").glob("*.whl")):
    print(f"\nwheel: {w.name}\nsize: {w.stat().st_size / 1024**2:.1f} MB")
