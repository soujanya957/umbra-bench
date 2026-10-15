@echo off
rem Build nvdiffrast (CUDA rasterizer, torch extension) on Windows without a system CUDA toolkit.
rem
rem Needs: a conda env with CUDA-matching torch (here fh_l1, torch cu128), Visual Studio Build
rem Tools with the C++ workload, and a separate conda env holding nvcc + headers:
rem   conda create -n cuda128 -c nvidia/label/cuda-12.8.1 cuda-nvcc cuda-cudart-dev cuda-cccl ^
rem       cuda-nvrtc-dev libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev libcufft-dev
rem   git clone https://github.com/NVlabs/nvdiffrast %NVDR_SRC%
rem nvcc 12.8 does not know MSVC 14.50 (VS 2026), hence -allow-unsupported-compiler; the CCCL
rem headers (nv/target) live under include\targets\x64 in the conda layout.

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_HOME=%USERPROFILE%\miniconda3\envs\cuda128\Library"
set "PY=%USERPROFILE%\miniconda3\envs\fh_l1\python.exe"
set "NVDR_SRC=%USERPROFILE%\Documents\GitHub\_deps\nvdiffrast"
rem RTX 50xx = 12.0; RTX 40xx = 8.9; RTX 30xx = 8.6
set "TORCH_CUDA_ARCH_LIST=12.0"

call "%VCVARS%" >nul
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "NVCC_APPEND_FLAGS=-allow-unsupported-compiler -I%CUDA_HOME%\include\targets\x64"
set "LIB=%CUDA_HOME%\lib;%LIB%"
set "INCLUDE=%CUDA_HOME%\include\targets\x64;%INCLUDE%"
set DISTUTILS_USE_SDK=1
set MAX_JOBS=8
where nvcc || exit /b 1
where cl || exit /b 1
cd /d "%NVDR_SRC%" || exit /b 1
"%PY%" -m pip install ninja
"%PY%" -m pip install --no-build-isolation -v .
echo EXIT %ERRORLEVEL%
