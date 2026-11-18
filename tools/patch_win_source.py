"""Port FlashInfer 0.7.2.post1 to Windows/MSVC inside the source tree.

Unlike editing an installed package, this also patches aot.py (the jit-cache
packager) and the module repackaging that names artifacts the same way core.py does.

Environment:
    FI_SRC     FlashInfer source tree (default C:\\fi\\src)
    FI_SHIM    shim header to install; defaults to flashinfer_win_shim.cuh next to this file
"""
import hashlib
import os
import pathlib
import re

ROOT = pathlib.Path(os.environ.get("FI_SRC", r"C:\fi\src"))
JIT = ROOT / "flashinfer" / "jit"
INC = ROOT / "include"
CSRC = ROOT / "csrc"
SHIM_SRC = pathlib.Path(
    os.environ.get("FI_SHIM", str(pathlib.Path(__file__).with_name("flashinfer_win_shim.cuh")))
)

applied, missed = [], []


def sub(text, old, new, tag):
    if old in text:
        applied.append(tag)
        return text.replace(old, new, 1)
    missed.append(tag)
    return text


HELPERS = '''

def _ninja_path(p) -> str:
    """Path for build.ninja: forward slashes, drive colon escaped."""
    p = Path(p).resolve()
    if os.name == "nt":
        s = p.as_posix()
        if len(s) > 1 and s[1] == ":":
            s = s[0] + "$:" + s[2:]
        return s
    return str(p)


def _short_artifact(name: str) -> str:
    """Module library name: a short m<md5[:10]>.dll on Windows (the 260 limit)."""
    if os.name == "nt":
        return "m" + hashlib.md5(str(name).encode("utf-8")).hexdigest()[:10] + ".dll"
    return f"{name}.so"
'''

# ---------------------------------------------------------------- cpp_ext.py
cpp = JIT / "cpp_ext.py"
t = cpp.read_text(encoding="utf-8")

if "import hashlib" not in t:
    t = t.replace("import os\n", "import hashlib\nimport os\n", 1)

if "_ninja_path" not in t:
    t = t.rstrip() + "\n" + HELPERS

t = sub(
    t,
    '''        "rule compile",
        "  command = $cxx_launcher $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags",
        "  depfile = $out.d",
        "  deps = gcc",''',
    '''        "rule compile",
        (
            "  command = $nvcc -c $in -o $out $cflags"
            if os.name == "nt"
            else "  command = $cxx_launcher $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags"
        ),
        *([] if os.name == "nt" else ["  depfile = $out.d", "  deps = gcc"]),''',
    "compile rule",
)

t = sub(
    t,
    '''        "rule cuda_compile",
        "  command = $nvcc_launcher $nvcc --generate-dependencies-with-compile -MF $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
        "  depfile = $out.d",
        "  deps = gcc",''',
    '''        "rule cuda_compile",
        (
            "  command = $nvcc -c $in -o $out $cuda_cflags"
            if os.name == "nt"
            else "  command = $nvcc_launcher $nvcc --generate-dependencies-with-compile -MF $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags"
        ),
        *([] if os.name == "nt" else ["  depfile = $out.d", "  deps = gcc"]),''',
    "cuda_compile rule",
)

t = sub(
    t,
    "    if needs_device_linking:\n        lines.extend(",
    '    if needs_device_linking or os.name == "nt":\n        lines.extend(',
    "nvcc_link emitted",
)

t = sub(
    t,
    '''                "command = $nvcc -shared $cuda_arch_flags $in $ldflags -o $out",'''.replace(
        "command", "  command"
    ),
    '''                (
                    f"  command = $nvcc -shared $cuda_arch_flags $in -o $out {os.environ.get('FLASHINFER_WIN_LDFLAGS', '')}"
                    if os.name == "nt"
                    else "  command = $nvcc -shared $cuda_arch_flags $in $ldflags -o $out"
                ),''',
    "nvcc_link command",
)

t = sub(
    t,
    '        obj_name = f"{source.parent.name}_{source.stem}{object_suffix}"',
    '        obj_name = f"obj{len(objects)}{object_suffix}"',
    "short object names",
)

t = sub(
    t,
    "        obj = str((output_dir / obj_name).resolve())",
    "        obj = _ninja_path(output_dir / obj_name)",
    "object path",
)

t = sub(
    t,
    '        lines.append(f"build {obj}: {cmd} {source.resolve()}")',
    '        lines.append(f"build {obj}: {cmd} {_ninja_path(source)}")',
    "source path",
)

t = sub(
    t,
    '    link_rule = "nvcc_link" if needs_device_linking else "link"',
    '    link_rule = "nvcc_link" if (needs_device_linking or os.name == "nt") else "link"',
    "link rule choice",
)

t = sub(
    t,
    '''    output_so = str((output_dir / f"{name}.so").resolve())''',
    '    output_so = _ninja_path(output_dir / _short_artifact(name))',
    "artifact name",
)

cpp.write_text(t, encoding="utf-8")

# ------------------------------------------------------------------- core.py
core = JIT / "core.py"
c = core.read_text(encoding="utf-8")
c = sub(
    c,
    '        return jit_env.FLASHINFER_JIT_DIR / self.name / f"{self.name}.so"',
    '''        if os.name == "nt":
            import hashlib
            short = "m" + hashlib.md5(self.name.encode("utf-8")).hexdigest()[:10] + ".dll"
            return jit_env.FLASHINFER_JIT_DIR / self.name / short
        return jit_env.FLASHINFER_JIT_DIR / self.name / f"{self.name}.so"''',
    "core library path",
)
core.write_text(c, encoding="utf-8")

# -------------------------------------------------------------------- env.py
env = JIT / "env.py"
e = env.read_text(encoding="utf-8")
if "import hashlib" not in e:
    e = e.replace("import functools\n", "import functools\nimport hashlib\n", 1)

SHORT_PATH = '''

class _ShortPath(pathlib.Path):
    """A path whose over-long directory names are hashed on Windows.

    A JIT spec name reaches 185 characters and becomes the build directory name,
    so even a short root does not save you from the 260 limit. The segment is
    hashed inside __truediv__ so every writer (core, cpp_ext, aot) sees one path.
    """

    _MAX_SEGMENT = 32

    def __truediv__(self, key):
        if os.name != "nt":
            return super().__truediv__(key)
        name = os.fspath(key)
        if (
            isinstance(name, str)
            and len(name) > self._MAX_SEGMENT
            and "." not in name
            and "/" not in name
            and "\\\\" not in name
        ):
            name = "s" + hashlib.md5(
                f"{self.as_posix()}/{name}".encode("utf-8")
            ).hexdigest()[:12]
        return super().__truediv__(name)

    __div__ = __truediv__
'''

if "_ShortPath" not in e:
    e = sub(
        e,
        "def _get_workspace_dir_name() -> pathlib.Path:",
        SHORT_PATH.strip() + "\n\n\ndef _get_workspace_dir_name() -> pathlib.Path:",
        "_ShortPath class",
    )

e = sub(
    e,
    '''FLASHINFER_JIT_DIR: pathlib.Path = FLASHINFER_WORKSPACE_DIR / "cached_ops"''',
    '''FLASHINFER_JIT_DIR: pathlib.Path = _ShortPath(FLASHINFER_WORKSPACE_DIR) / "cached_ops"''',
    "env jit dir",
)
e = sub(
    e,
    '''FLASHINFER_GEN_SRC_DIR: pathlib.Path = FLASHINFER_WORKSPACE_DIR / "generated"''',
    '''FLASHINFER_GEN_SRC_DIR: pathlib.Path = (
    _ShortPath(FLASHINFER_WORKSPACE_DIR) / "generated"
)''',
    "env gen dir",
)
env.write_text(e, encoding="utf-8")

# -------------------------------------------------------------------- aot.py
aot = ROOT / "flashinfer" / "aot.py"
a = aot.read_text(encoding="utf-8")
a = sub(
    a,
    '''    jit_env.FLASHINFER_JIT_DIR = build_dir / "cached_ops"
    jit_env.FLASHINFER_GEN_SRC_DIR = build_dir / "generated"''',
    '''    jit_env.FLASHINFER_JIT_DIR = jit_env._ShortPath(build_dir) / "cached_ops"
    jit_env.FLASHINFER_GEN_SRC_DIR = jit_env._ShortPath(build_dir) / "generated"''',
    "aot build dirs short",
)
a = sub(
    a,
    '''        src = jit_env.FLASHINFER_JIT_DIR / jit_spec.name / f"{jit_spec.name}.so"''',
    '''        src = jit_env.FLASHINFER_JIT_DIR / jit_spec.name / jit_spec.jit_library_name''',
    "aot module source name",
)
aot.write_text(a, encoding="utf-8")

# core: expose the library name to aot.py as well
c = core.read_text(encoding="utf-8")
if "def jit_library_name" not in c:
    c = sub(
        c,
        "    @property\n    def jit_library_path(self) -> Path:",
        '''    @property
    def jit_library_name(self) -> str:
        if os.name == "nt":
            import hashlib
            return "m" + hashlib.md5(self.name.encode("utf-8")).hexdigest()[:10] + ".dll"
        return f"{self.name}.so"

    @property
    def jit_library_path(self) -> Path:''',
        "core library name property",
    )
    core.write_text(c, encoding="utf-8")

# ------------------------------------------------- headers: std::max/min
nmax = nmin = 0
for p in list(INC.rglob("*.cuh")) + list(INC.rglob("*.h")):
    src = p.read_text(encoding="utf-8", errors="replace")
    if "std::max(" not in src and "std::min(" not in src:
        continue
    nmax += src.count("std::max(")
    nmin += src.count("std::min(")
    p.write_text(
        src.replace("std::max(", "fi_max(").replace("std::min(", "fi_min("),
        encoding="utf-8",
    )

# -------------------------------------- csrc: #if X == A -> if constexpr
IF_RE = re.compile(r"^#if\s+(\w+)\s*==\s*(\w+)\s*$")
ELIF_RE = re.compile(r"^#elif\s+(\w+)\s*==\s*(\w+)\s*$")
files = blocks = 0
for p in list(CSRC.rglob("*.cuh")) + list(CSRC.rglob("*.cu")) + list(CSRC.rglob("*.h")):
    text = p.read_text(encoding="utf-8", errors="replace")
    if "#if " not in text:
        continue
    lines = text.splitlines()
    out, i, n = [], 0, 0
    while i < len(lines):
        m = IF_RE.match(lines[i].strip())
        if not m:
            out.append(lines[i])
            i += 1
            continue
        out.append(f"if constexpr ({m.group(1)} == {m.group(2)}) {{")
        i += 1
        closed = False
        while i < len(lines):
            s = lines[i].strip()
            e2 = ELIF_RE.match(s)
            if e2:
                out.append(f"}} else if constexpr ({e2.group(1)} == {e2.group(2)}) {{")
                i += 1
                continue
            if s == "#else":
                out.append("} else {")
                i += 1
                continue
            if s == "#endif":
                out.append("}")
                i += 1
                closed = True
                break
            if s.startswith("#error"):
                i += 1
                continue
            out.append(lines[i])
            i += 1
        if not closed:
            out.append("#endif")
        n += 1
    if n:
        p.write_text("\n".join(out) + "\n", encoding="utf-8")
        files += 1
        blocks += n

# ---------------------------------------------------------------- shim
shim_dst = INC / "flashinfer" / "flashinfer_win_shim.cuh"
shim = SHIM_SRC.read_text(encoding="utf-8")
if "fi_win" not in shim:
    shim += """
#include <type_traits>
namespace fi_win {
template <class A, class B>
__host__ __device__ constexpr typename std::common_type<A, B>::type fi_max(A a, B b) {
    using T = typename std::common_type<A, B>::type;
    return static_cast<T>(a) > static_cast<T>(b) ? static_cast<T>(a) : static_cast<T>(b);
}
template <class A, class B>
__host__ __device__ constexpr typename std::common_type<A, B>::type fi_min(A a, B b) {
    using T = typename std::common_type<A, B>::type;
    return static_cast<T>(a) < static_cast<T>(b) ? static_cast<T>(a) : static_cast<T>(b);
}
}
#define fi_max fi_win::fi_max
#define fi_min fi_win::fi_min
"""
shim_dst.write_text(shim, encoding="utf-8")

print("applied:", ", ".join(applied))
if missed:
    print("NOT FOUND:", ", ".join(missed))
print(f"fi_max/fi_min replaced: {nmax}/{nmin}")
print(f"if constexpr: {files} files, {blocks} blocks")
print("shim:", shim_dst)
