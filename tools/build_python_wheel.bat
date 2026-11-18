@echo off
rem Build flashinfer_python-<base>+<suffix>-py3-none-any.whl from the patched tree.
rem Uses the vLLM portable interpreter: it already has torch and tvm_ffi.
setlocal

rem ---------------------------------------------------------------- CONFIG
set "FI_ROOT=C:\fi"
set "VLLM_PYTHON=C:\vllm-win\python\python.exe"
rem Local version suffix. The kernel wheel and the shim must carry the same
rem version string, otherwise FlashInfer refuses the provider.
set "FLASHINFER_LOCAL_VERSION=win.1"
rem -----------------------------------------------------------------------

set "BUILD_NVEP=0"
set "BUILD_NIXL_EP=0"
set "BUILD_NCCL_EP=0"
set "FLASHINFER_BUILD_NO_PIP=1"
cd /d "%FI_ROOT%\src"
"%VLLM_PYTHON%" -m pip wheel . --no-deps --no-build-isolation -w "%FI_ROOT%\dist-python"
echo rc=%ERRORLEVEL%
endlocal
