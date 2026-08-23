# WinRocM-llm Toolchain (verified 2026-08-22)

## Compilers (use ROCm clang++, NOT MSVC cl.exe for C++ source)
- **C++ host compiler:** `G:\ROCM10RT-gfx1201\bin\clang++.exe`
- **HIP compiler:** `G:\ROCM10RT-gfx1201\bin\clang++.exe` (driven by `hipcc`)
- **MSVC cl.exe** is NOT used as a compiler — only Windows SDK rc.exe + MSVC linker via lld-link.

## Flags
```
offload-arch      = gfx1031        (compile target; RDNA2 for rocblas compat)
runtime HW        = gfx1201        (RDNA4; the preview ROCm on this box)
device lib path   = G:\ROCM10RT-gfx1201\lib\llvm\amdgcn\bitcode
rocm path         = G:\ROCM10RT-gfx1201
cxx standard      = c++17
frontend variant  = GNU            (critical: avoids MSVC /permissive- + builtin macros)
```

## Why MSVC cl.exe fails (the cmath conflict, root cause)
MSVC STL `<cmath>` emits `_CLANG_BUILTIN2(isgreater)` (a macro that expands to a
builtin declaration) at every include. HIP's `__clang_hip_cmath.h` then declares
`__DEVICE__ bool isgreater(...)` — a redeclaration under a different linkage → error:
"`__device__` function 'isgreater' cannot overload `__host__` `__device__` function 'isgreater'`".
MSVC STL has no `__NO_MATH_DEFINES` guard in v14.51, so `-D__NO_MATH_DEFINES` does nothing.

## Fix (verified working)
Use ROCm `clang++` as `CMAKE_CXX_COMPILER` with GNU frontend variant. The Windows
port's `rocm-windows-toolchain.cmake` + `strip-msvc-flags.cmake` enforce this.
With GNU frontend, clang uses its own headers (no MSVC builtin macros) → conflict gone.

## Verified test command (works)
```bat
set ROCM_HOME=G:\ROCM10RT-gfx1201
set DL=%ROCM_HOME%\lib\llvm\amdgcn\bitcode
%ROCM_HOME%\bin\hipcc --offload-arch=gfx1031 --rocm-device-lib-path="%DL%" -std=c++17 -o build\smoke_host.exe src\hip_wrap\kernel_smoke.hip.cc
```
