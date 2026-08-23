@echo off
REM build_host.cmd — Phase 0.2+ host C++ build for WinRocM-llm (D012/D013).
REM Hand-rolled clang++ invocation avoids CMake's Windows-Clang platform module
REM (forces MSVC link semantics incompatible with GNU lld). ROCm bin on PATH so
REM clang resolves GNU-mode ld.lld instead of MSVC lld-link.
REM
REM Portable: all external paths resolved from env vars with autodetect fallback.
REM Override any of:  ROCM_HOME  TORCH_HOME  MINGW_ROOT  VS_LIB  PROJECT_ROOT
setlocal enabledelayedexpansion
REM [portability] machine-specific overrides live in gitignored local.env
if exist "%PROJECT_ROOT%\local.env" call "%PROJECT_ROOT%\local.env"
if exist "local.env" call local.env
REM D017 defaults OFF; preserve explicit BUILD_ROCM_OPS_DLL=1.
if not defined BUILD_ROCM_OPS_DLL set "BUILD_ROCM_OPS_DLL=0"
REM (value is preserved: empty string means OFF, not clobbered to "")REM (respect inherited value rather than blanking — parent shell set controls.)

REM --- portable path resolution (edit env vars, not this file) ---
if not defined PROJECT_ROOT  set "PROJECT_ROOT=%~dp0..\.."
pushd "%PROJECT_ROOT%" >nul 2>&1 && cd /d "%PROJECT_ROOT%" || (echo "set PROJECT_ROOT" & exit /b 1)

if not defined ROCM_HOME    set "ROCM_HOME=G:\ROCM10RT-gfx1201"
if not defined TORCH_HOME   set "TORCH_ROOT=E:\ROCM-versions\common\cp312\win_torch\torch"
if not defined TORCH_ROOT   set "TORCH_ROOT=%TORCH_HOME%\torch"
set TORCH_INC=%TORCH_ROOT%\include
set TORCH_LIB=%TORCH_ROOT%\lib
if not defined MINGW_GCC    set "MINGW_GCC=C:\Strawberry\c\lib\gcc\x86_64-w64-mingw32\13.2.0"
if not defined MINGW_LIB    set "MINGW_LIB=C:\Strawberry\c\x86_64-w64-mingw32\lib"
if not defined MINGW_ROOT   set "MINGW_ROOT=C:\Strawberry\c\lib"
if not defined VS_LIB       set "MSVC_LIB=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\lib\x64"
if defined   VS_LIB         set "MSVC_LIB=%VS_LIB%"

set CLANG=%ROCM_HOME%\lib\llvm\bin\clang++.exe
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
REM builds via hipcc (D016 scalar-WMMA fallback for gfx1201); the torch
REM extension host TU (torch_bindings.cpp + paged_attention_bridge.cpp) is
REM compiled via HIP-clang MSVC-triple with the cuda_runtime.h shim (csrc/hip_wrap/) so
REM torch's <c10/cuda/*> resolve to stubs instead of HIP's GNU-attribute
REM headers that cl.exe cannot parse.  Set once the ROCm-on-Windows torch
REM toolchain is configured (see decisions D017/D018).
REM === D017: rocm_ops.dll — separate MSVC pass (see D017/D018). ===
if /i not "!BUILD_ROCM_OPS_DLL: =!"=="1" goto :skip_d017
echo [D017] BUILD_ROCM_OPS_DLL=1 - building rocm_ops.dll (HIP-clang host TU, GNU-ldd link)
echo [D017] TORCH_INC=%TORCH_INC%  TORCH_LIB=%TORCH_LIB%  ROCM_HOME=%ROCM_HOME%
if not defined PY_INC      set "PY_INC=C:/Python314/Include"
set CL=%CLANG% -c -x c++ -std=c++20 -m64 -DNDEBUG -I%SRC% -Icsrc -Isrc\engine\include -I"%PROJECT_ROOT%" -DUSE_ROCM -DC10_CUDA_NO_CMAKE_CONFIGURE_FILE -DC10_STATIC_DEFINE -D__HIP_PLATFORM_AMD__=1 -fms-extensions -fms-compatibility -isystem "csrc/hip_wrap" -isystem "%TORCH_INC%" -isystem "%TORCH_INC%\torch\csrc\api\include" -isystem "%ROCM_HOME%\include" -isystem "%PY_INC%" -D__NO_MATH_DEFINES=1
REM D017.3: link with MSVC-triple clang++ object files.  clang++ link syntax, not link.exe.
set LINK=link /nologo /DLL /OUT:"%OBJ%\rocm_ops.dll" /LIBPATH:"%TORCH_LIB%" /LIBPATH:"%ROCM_HOME%\lib" /MACHINE:X64
echo [D017.1] device kernel (hipcc -c, gfx1201, D016 scalar WMMA fallback)
%ROCM_HOME%\bin\hipcc --offload-arch=gfx1201 --rocm-device-lib-path="%ROCM_HOME%\lib/llvm/amdgcn/bitcode" --target=x86_64-pc-windows-gnu -std=c++17 -D__NO_MATH_DEFINES=1 -DUSE_ROCM=1 -D__HIP_PLATFORM_AMD__=1 -D_MSC_VER=1900 -D_NATIVE_WCHAR_T_DEFINED=1 -D_WCHAR_T_DEFINED -fshort-wchar -c csrc\rocm\attention_gfx1201.cu -o "%OBJ%\attention_gfx1201.obj" 2>&1
if errorlevel 1 (echo DLL KERNEL COMPILE FAILED& exit /b 1)
REM [D017.2] host TU compiled via HIP-clang MSVC-triple (cl.exe avoided — see D013/D017)
REM NOTE: cl.exe/vcvarsall no longer needed since we link with clang++ GNU-mode lld (D012).
%CL% -c csrc\rocm\torch_bindings.cpp -o "%OBJ%\torch_bindings.obj" 2>&1
%CL% -c csrc\rocm\paged_attention_bridge.cpp -o "%OBJ%\paged_attention_bridge.obj" 2>&1
if errorlevel 1 (echo DLL HOST COMPILE FAILED& exit /b 1)
REM [D017.3] Link rocm_ops.dll with lld-link.exe (PE/COFF driver) directly —
REM clang++ MSVC-triple .obj + MSVC-format torch import libs (.lib).  NOT clang++ -fuse-ld=lld
REM (that routes clang++ itself to lld-link as an input).  Let lld-link report unresolved symbols.
set LLD_LINK=%ROCM_HOME%\lib\llvm\bin\lld-link.exe
if not exist "%LLD_LINK%" set "LLD_LINK=lld-link.exe"
echo [D017.3] %LLD_LINK% /DLL ...
"%LLD_LINK%" /nologo /DLL /OUT:"%OBJ%\rocm_ops.dll" /MACHINE:X64 ^
  /LIBPATH:"%TORCH_LIB%" /LIBPATH:"%ROCM_HOME%\lib" ^
  "%OBJ%\attention_gfx1201.obj" "%OBJ%\torch_bindings.obj" "%OBJ%\paged_attention_bridge.obj" ^
  c10_hip.lib torch_hip.lib c10.lib torch_cpu.lib amdhip64.lib ^
  "%MINGW_ROOT%\libstdc++.a" 2>&1
if errorlevel 1 (echo DLL LINK FAILED& exit /b 1)
echo rocm_ops.dll -^> "%OBJ%\rocm_ops.dll" (C-ABI paged_attention_rocm exported)
:skip_d017
if /i not "!BUILD_ROCM_OPS_DLL: =!"=="1" echo [D017] skipped (BUILD_ROCM_OPS_DLL not set to 1); engine --attention throws runtime_error fail-closed (rocm_ops.dll not built; see D018). Enable with: set BUILD_ROCM_OPS_DLL=1 && cmake\build_host.cmd (clang++ MSVC-triple host TU)

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

