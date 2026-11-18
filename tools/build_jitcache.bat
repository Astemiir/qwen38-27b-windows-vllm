@echo off
rem Build the AOT kernel set for the sm80 / CUDA 13 / MSVC toolchain.
rem Edit the CONFIG block, then run this file. Nothing else in the recipe needs changing.
setlocal

rem ---------------------------------------------------------------- CONFIG
set "FI_ROOT=C:\fi"
set "VLLM_PYTHON=C:\vllm-win\python\python.exe"
set "CUDA_HOME=C:\cuda133"
set "TVM_FFI_LIB=C:\tvmffi\tvm_ffi.lib"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
rem architecture list: 8.0 covers A100 and RTX 3090 (sm80 SASS runs on sm86)
set "FLASHINFER_CUDA_ARCH_LIST=8.0"
rem -----------------------------------------------------------------------

call "%VCVARS%" >nul
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "FI_PY_LIBS=%FI_ROOT%\vllm-python-libs"
rem Keep TMP/TEMP short and local: nvcc writes tmpxft_* there and long paths bite.
set "TMP=%FI_ROOT%\tmp"
set "TEMP=%FI_ROOT%\tmp"
set "TMPDIR=%FI_ROOT:\=/%/tmp"
set "FLASHINFER_WORKSPACE_BASE=%FI_ROOT:\=/%/b"
set "FLASHINFER_EXTRA_CUDAFLAGS=--pre-include=%FI_ROOT:\=/%/src/include/flashinfer/flashinfer_win_shim.cuh -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING -Xcompiler=/utf-8 -Xcompiler=/wd4996"
rem Leave CFLAGS empty: bare /utf-8 /wd4996 are taken by nvcc as an extra input file.
set "FLASHINFER_EXTRA_CFLAGS="
set "FLASHINFER_WIN_LDFLAGS=-Xlinker /LIBPATH:%VLLM_PYTHON%\..\libs %TVM_FFI_LIB%"
set "FLASHINFER_DISABLE_VERSION_CHECK=1"

"%VLLM_PYTHON%" "%~dp0build_jitcache.py" > "%FI_ROOT%\build_jitcache.log" 2>&1
echo rc=%ERRORLEVEL% ^(log: %FI_ROOT%\build_jitcache.log^)
endlocal
