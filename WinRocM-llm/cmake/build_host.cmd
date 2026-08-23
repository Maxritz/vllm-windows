@echo off
REM build_host.cmd — Phase 0.2+ host C++ build for WinRocM-llm (D012/D013).
REM Hand-rolled clang++ invocation avoids CMake's Windows-Clang platform module
REM (forces MSVC link semantics incompatible with GNU lld). ROCm bin on PATH so
REM clang resolves GNU-mode ld.lld instead of MSVC lld-link.
setlocal enabledelayedexpansion
REM D017 defaults OFF; explicit unset guards against inherited envs (e.g. PS session).
set "BUILD_ROCM_OPS_DLL="
cd /d G:\vllm-windows
set ROCM_HOME=G:\ROCM10RT-gfx1201
set CLANG=%ROCM_HOME%\lib\llvm\bin\clang++.exe
set MINGW_GCC=C:\Strawberry\c\lib\gcc\x86_64-w64-mingw32\13.2.0
set MINGW_LIB=C:\Strawberry\c\x86_64-w64-mingw32\lib
set MINGW_ROOT=C:\Strawberry\c\lib
set MSVC_LIB=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\lib\x64
REM ROCm 2.15 extracted libtorch headers + torch_hip/c10_hip import libs (param to env).
set TORCH_ROOT=G:\ROCM-versions\common\cp312\.extracted-torch\torch
set TORCH_INC=%TORCH_ROOT%\include
set TORCH_LIB=%TORCH_ROOT%\lib
set SRC=WinRocM-llm\src
set OBJ=WinRocM-llm\build-host\obj
REM Prepend ROCM bin so clang resolves GNU-mode ld.lld (not MSVC lld-link).
set "PATH=%ROCM_HOME%\lib\llvm\bin;%ROCM_HOME%\bin;%PATH%"
if exist WinRocM-llm\build-host\obj rmdir /s /q WinRocM-llm\build-host\obj
mkdir "%OBJ%" 2>nul

set SHIM=--target=x86_64-pc-windows-gnu -isystem %ROCM_HOME%\include -D__HIP_PLATFORM_AMD__=1 -D__NO_MATH_DEFINES=1 -D_MSC_VER=1900 -D_NATIVE_WCHAR_T_DEFINED=1 -D_WCHAR_T_DEFINED -fshort-wchar
set COMMON=%SHIM% -std=gnu++20 -O2 -DNDEBUG -I%SRC% -I%SRC%\hip_wrap -I%SRC%\engine\include -I. -DAMDT_SCOPED_RESOURCES_DYNAMIC=0

echo [1/6] engine.cpp
%CLANG% -c %COMMON% %SRC%\engine\engine.cpp -o "%OBJ%\engine.o" 2>&1
echo [2/6] logger.cpp
%CLANG% -c %COMMON% %SRC%\engine\logger.cpp -o "%OBJ%\logger.o" 2>&1
echo [3/6] gpu_windows.cpp
%CLANG% -c %COMMON% %SRC%\hip_wrap\hip_compat\gpu_windows.cpp -o "%OBJ%\gpu_windows.o" 2>&1
echo [4/6] gpu_buffer.cpp
%CLANG% -c %COMMON% %SRC%\engine\gpu_buffer.cpp -o "%OBJ%\gpu_buffer.o" 2>&1
echo [5/7] block_table.cpp
%CLANG% -c %COMMON% %SRC%\engine\block_table.cpp -o "%OBJ%\block_table.o" 2>&1
echo [6/8] model_runner.cpp
%CLANG% -c %COMMON% %SRC%\engine\model_runner.cpp -o "%OBJ%\model_runner.o" 2>&1
echo [7/8] main.cpp
%CLANG% -c %COMMON% %SRC%\cli\main.cpp -o "%OBJ%\main.o" 2>&1
if errorlevel 1 (echo COMPILE FAILED& exit /b 1)

REM === D017: rocm_ops.dll — MSVC pass (torch headers + device kernel link). ===
REM OFF by default (BUILD_ROCM_OPS_DLL=1 to enable).  The device kernel
REM builds via hipcc (D016 scalar-WMMA fallback for gfx1201); the torch
REM extension host TU (torch_bindings.cpp + paged_attention_bridge.cpp) is
REM compiled via cl.exe with the cuda_runtime.h shim (csrc/hip_wrap/) so
REM torch's <c10/cuda/*> resolve to stubs instead of HIP's GNU-attribute
REM headers that cl.exe cannot parse.  Set once the ROCm-on-Windows torch
REM toolchain is configured (see decisions D017/D018).
REM === D017: rocm_ops.dll — separate MSVC pass (see D017/D018). ===
if /i not "!BUILD_ROCM_OPS_DLL!"=="1" goto :skip_d017
echo [D017] BUILD_ROCM_OPS_DLL set — building rocm_ops.dll
echo [D017] locating vcvarsall...
for /f "usebackq tokens=*" %%V in (`"%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VC.Tools.x86.x64 -property installationPath`) do set "VSINST=%%V"
call ""!VSINST!\VC\Auxiliary\Build\vcvarsall.bat"" x64
set CL=/nologo /std:c++20 /O2 /EHsc /wd4819 /I"%TORCH_INC%" /IC:\vllm-windows /Icsrc /Isrc\engine\include /DUSE_ROCM /DC10_CUDA_NO_CMAKE_CONFIGURE_FILE /DC10_STATIC_DEFINE /I"%ROCM_HOME%\include" /D__HIP_PLATFORM_AMD__=1
set LINK=/nologo /DLL /OUT:"%OBJ%\rocm_ops.dll" /LIBPATH:"%TORCH_LIB%" /LIBPATH:"%ROCM_HOME%\lib"
echo [D017.1] device kernel (hipcc -c, gfx1201, D016 scalar WMMA fallback)
%ROCM_HOME%\bin\hipcc --offload-arch=gfx1201 --rocm-device-lib-path="%ROCM_HOME%\lib\llvm\amdgcn/bitcode" --target=x86_64-pc-windows-gnu -std=c++17 -D__NO_MATH_DEFINES=1 -DUSE_ROCM=1 -D__HIP_PLATFORM_AMD__=1 -D_MSC_VER=1900 -D_NATIVE_WCHAR_T_DEFINED=1 -D_WCHAR_T_DEFINED -fshort-wchar -c csrc\rocm\attention_gfx1201.cu -o "%OBJ%\attention_gfx1201.obj" 2>&1
if errorlevel 1 (echo DLL KERNEL COMPILE FAILED& exit /b 1)
echo [D017.2] torch bindings + C-ABI bridge (cl.exe /c)
cl %CL% /c csrc\rocm\torch_bindings.cpp /Fo"%OBJ%\" 2>&1
cl %CL% /c csrc\rocm\paged_attention_bridge.cpp /Fo"%OBJ%\" 2>&1
if errorlevel 1 (echo DLL HOST COMPILE FAILED& exit /b 1)
echo [D017.3] link rocm_ops.dll
link %LINK% "%OBJ%\attention_gfx1201.obj" "%OBJ%\torch_bindings.obj" "%OBJ%\paged_attention_bridge.obj" c10_hip.lib torch_hip.lib c10.lib torch_cpu.lib amdhip64.lib 2>&1
if errorlevel 1 (echo DLL LINK FAILED& exit /b 1)
echo rocm_ops.dll -^> "%OBJ%\rocm_ops.dll" (C-ABI paged_attention_rocm exported)
:skip_d017
if /i not "!BUILD_ROCM_OPS_DLL!"=="1" echo [D017] skipped (BUILD_ROCM_OPS_DLL unset); engine --attention throws runtime_error fail-closed (rocm_ops.dll not built; see D018)

echo Linking vllm_engine.exe (GNU ld.lld + MinGW runtime + amdhip64.lib)
%CLANG% --target=x86_64-pc-windows-gnu -fuse-ld=lld -nodefaultlibs ^
  -std=gnu++20 ^
  -L%MINGW_GCC% -L%MINGW_LIB% -L%MINGW_ROOT% -L%ROCM_HOME%\lib ^
  "%OBJ%\engine.o" "%OBJ%\logger.o" "%OBJ%\gpu_windows.o" "%OBJ%\gpu_buffer.o" "%OBJ%\block_table.o" "%OBJ%\model_runner.o" "%OBJ%\main.o" ^
  "%ROCM_HOME%\lib\amdhip64.lib" ^
  "%MINGW_LIB%\libwinpthread.a" "%MINGW_LIB%\libmingwex.a" "%MINGW_LIB%\libmsvcrt.a" ^
  "%MINGW_LIB%\libadvapi32.a" "%MINGW_LIB%\libuser32.a" "%MINGW_LIB%\libkernel32.a" ^
  "%MINGW_ROOT%\libstdc++.a" -lgcc_s -lgcc -lmingw32 ^
  -o WinRocM-llm\build-host\vllm_engine.exe 2>&1
if errorlevel 1 (echo LINK FAILED& exit /b 1)

echo === RUN ===
WinRocM-llm\build-host\vllm_engine.exe %* 2>&1

