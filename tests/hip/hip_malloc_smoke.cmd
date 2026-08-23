@echo off
REM hip_malloc_smoke.cmd — real test: hipMalloc/hipFree/hipMemcpy via ROCm HIP on RX 9070 XT.
REM Two-stage: hipcc -c (host+device) -> clang++ link (GNU ld.lld + MinGW + amdhip64.lib).
setlocal
cd /d G:\vllm-windows
set ROCM_HOME=G:\ROCM10RT-gfx1201
set LLVM_BIN=%ROCM_HOME%\lib\llvm\bin
set DEVICE_LIB=%ROCM_HOME%\lib\llvm\amdgcn/bitcode
set MINGW_GCC=C:\Strawberry\c\lib\gcc\x86_64-w64-mingw32\13.2.0
set MINGW_LIB=C:\Strawberry\c\x86_64-w64-mingw32\lib
set MINGW_ROOT=C:\Strawberry\c\lib
path=%LLVM_BIN%;%ROCM_HOME%\bin;%PATH%
if not exist tests\hip mkdir tests\hip
echo [1/2] hipcc compile (gfx1201)...
"%ROCM_HOME%\bin\hipcc" --offload-arch=gfx1201 --rocm-device-lib-path=%DEVICE_LIB% ^
  --target=x86_64-pc-windows-gnu -std=c++17 ^
  -D__HIP_PLATFORM_AMD__=1 -D__NO_MATH_DEFINES=1 -D_MSC_VER=1900 ^
  -D_NATIVE_WCHAR_T_DEFINED=1 -D_WCHAR_T_DEFINED -fshort-wchar ^
  -c -o tests\hip\hip_malloc_smoke.o tests\hip\hip_malloc_smoke.hip.cc
if errorlevel 1 (echo COMPILE FAIL& exit /b 1)
echo [2/2] link...
"%LLVM_BIN%\clang++.exe" --target=x86_64-pc-windows-gnu -fuse-ld=lld -nodefaultlibs ^
  -std=gnu++17 ^
  -L%MINGW_GCC% -L%MINGW_LIB% -L%MINGW_ROOT% -L%ROCM_HOME%\lib ^
  tests\hip\hip_malloc_smoke.o ^
  "%ROCM_HOME%\lib\amdhip64.lib" ^
  "%MINGW_LIB%\libwinpthread.a" "%MINGW_LIB%\libmingwex.a" "%MINGW_LIB%\libmsvcrt.a" ^
  "%MINGW_LIB%\libadvapi32.a" "%MINGW_LIB%\libuser32.a" "%MINGW_LIB%\libkernel32.a" ^
  "%MINGW_ROOT%\libstdc++.a" -lgcc_s -lgcc -lmingw32 ^
  -o tests\hip\hip_malloc_smoke.exe
if errorlevel 1 (echo LINK FAIL& exit /b 1)
echo === RUN ===
tests\hip\hip_malloc_smoke.exe
