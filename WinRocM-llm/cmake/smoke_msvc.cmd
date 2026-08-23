@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
echo __MSVC_OK__
cl.exe /nologo /? 2>nul | findstr /i "Microsoft.*Compiler"
echo __ROCM_HOME__=%ROCM_HOME%
set _HIPCC=G:\ROCM10RT-gfx1201\bin\hipcc
echo _HIPCC=%_HIPCC%
REM locate the amdgcn bitcode dir the way hipcc expects
for /f "delims=" %%i in ('"%_HIPCC%" --print-file-name=amdgcn/bitcode" 2^>nul') do echo _HIP_BITCODE=%%i
