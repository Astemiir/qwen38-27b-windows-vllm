@echo off
setlocal
title vLLM 0.33 Windows bundle - install
set "HERE=%~dp0"
set "KIT=%~1"
if "%KIT%"=="" set "KIT=E:\vLLM-0.33\vllm-windows-build-0.33.0-win-cu130"
set "PY=%KIT%\python\python.exe"

echo.
echo === 0/5  looking for the vLLM kit ===
if not exist "%PY%" goto :nokit
echo     kit:  %KIT%

echo.
echo === 1/5  interpreter and torch ===
"%PY%" -c "import torch; print('    torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
if errorlevel 1 goto :notorch
"%PY%" -c "import tvm_ffi, numpy; print('    deps: tvm_ffi, numpy ok')"
if errorlevel 1 goto :nodeps

echo.
echo === 2/5  FlashInfer wheels ===
"%PY%" -m pip install --no-deps --force-reinstall "%HERE%flashinfer_python-0.7.2.post1+win.1-py3-none-any.whl" "%HERE%flashinfer_jit_cache-0.7.2.post1+win.1-py3-none-any.whl" "%HERE%flashinfer_jit_cache_sm80-0.7.2.post1+win.1-cp39-abi3-win_amd64.whl"
if errorlevel 1 goto :pipfail

echo.
echo === 3/5  patched quantization plugin ===
"%PY%" -m pip uninstall -y vllm-quantpack >nul 2>&1
"%PY%" -m pip install --no-deps --force-reinstall "%HERE%vllm_quantpack-0.1.5+win.1-py3-none-any.whl"
if errorlevel 1 goto :pipfail

echo.
echo === 4/5  vLLM fp8 guard ===
"%PY%" "%HERE%patch_vllm.py" "%KIT%"
if errorlevel 1 goto :patchfail

echo.
echo === 5/5  verification ===
set FLASHINFER_DISABLE_JIT=1
"%PY%" "%HERE%verify_flashinfer.py"
if errorlevel 1 goto :verifyfail
set FLASHINFER_DISABLE_JIT=

echo.
echo ============================================================
echo  DONE. Everything is installed into this kit:
echo    %KIT%
echo.
echo  Next: download a GPTQ W4A16 checkpoint with the MTP head,
echo  then start the server with the launch line from INSTALL.md
echo  (262k, int4) or AGENT-INSTALL.md (the same line, for agents).
echo.
echo  Required flags, easy to forget:
echo    --max-num-seqs 8          (or int4 decode falls to ~28 tok/s)
echo    PYTORCH_CUDA_ALLOC_CONF=  (must be EMPTY on Windows)
echo ============================================================
if not "%~2"=="nopause" pause
exit /b 0

:nokit
echo [X] No portable python at "%PY%"
echo     Pass the kit folder as the first argument:
echo       install.bat "D:\vllm-windows-build-0.33.0-win-cu130"
if not "%~2"=="nopause" pause
exit /b 1

:notorch
echo [X] torch is not importable in this kit - run the kit's own install.bat first.
if not "%~2"=="nopause" pause
exit /b 1

:nodeps
echo [X] python modules tvm_ffi / numpy are missing in this kit.
echo     Install the kit by its own install.bat first, then run this script again.
if not "%~2"=="nopause" pause
exit /b 1

:pipfail
echo [X] pip failed. Read the message above; nothing else was changed.
if not "%~2"=="nopause" pause
exit /b 1

:patchfail
echo [X] The fp8 guard patch did not apply. The bundle still works for int4/k8v4/int8;
echo     fp8_per_token_head stays unavailable until the guard is edited by hand.
if not "%~2"=="nopause" pause
exit /b 1

:verifyfail
echo [X] A prebuilt FlashInfer kernel did not load: the three wheels must all be 0.7.2.post1+win.1
if not "%~2"=="nopause" pause
exit /b 1
