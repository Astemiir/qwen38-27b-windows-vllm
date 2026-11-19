r"""Apply the fp8 per-token-head guard fix to vLLM inside the Windows kit.

    python\python.exe patch_vllm.py <kit-root>

Not called by install.bat any more, and not recommended: the widened guard gave correct output on a 27B
but garbled text on a 9B GPTQ build, so the dtype is shape-dependent and the refusal is safer. Kept only
because our own fp8_per_token_head numbers were taken with it.

Idempotent: if the guard already mentions per_token_head, nothing is written. A .bak-<today> copy of
the file is left next to it, so the change can be undone by renaming.
"""
import datetime
import os
import sys

OLD = (
    '            if self.kv_cache_dtype.startswith("fp8") and not (\n'
    '                current_platform.has_device_capability(89)\n'
    '            ):\n'
)
NEW = (
    '            if (\n'
    '                self.kv_cache_dtype.startswith("fp8")\n'
    '                and not self.kv_cache_dtype.endswith("per_token_head")\n'
    '                and not (current_platform.has_device_capability(89))\n'
    '            ):\n'
)


def main():
    kit = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(kit, "python", "Lib", "site-packages", "vllm", "v1",
                          "attention", "backends", "triton_attn.py")
    if not os.path.exists(target):
        sys.exit("not found: " + target + "\npass the kit root, e.g. patch_vllm.py E:\\vLLM-0.33\\vllm-windows-build-0.33.0-win-cu130")

    with open(target, "r", encoding="utf-8") as fh:
        text = fh.read()

    if 'endswith("per_token_head")' in text:
        print("already patched:", target)
        return

    if OLD not in text:
        sys.exit("guard not found in the expected shape; vLLM version differs, apply by hand:\n" + target)

    stamp = datetime.date.today().isoformat()
    backup = target + ".bak-" + stamp + "-fp8pth"
    if not os.path.exists(backup):
        with open(backup, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)

    with open(target, "w", encoding="utf-8", newline="") as fh:
        fh.write(text.replace(OLD, NEW, 1))

    print("patched:", target)
    print("backup:", backup)


if __name__ == "__main__":
    main()
