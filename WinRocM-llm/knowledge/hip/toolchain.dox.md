/**
@file hip/toolchain
@brief Windows ROCm toolchain configuration for WinRocM-llm C++ builds.
@details

## Status: verified 2026-08-22 — a real HIP kernel compiles, links, and runs on RX 9070 XT.

## Key constraints (D012)

Two independent Windows-specific breakage points block naive `hipcc`/`clang++` use:

1. **`sizeof(long)` data-model conflict (amd_warp_functions.h).**
   ROCm 7.16 headers `static_assert(sizeof(long) == 2*sizeof(int))` under `#ifndef _MSC_VER`
   (Linux LP64 path). Windows clang is **always LLP64** regardless of triple, so
   `sizeof(long)==4` — the assert fails when `_MSC_VER` is *not* defined. MSVC triple
   *does* define `_MSC_VER`, but...

2. **MSVC `<cmath>` `isgreater` macro conflict (isgreater forward decls).**
   Under MSVC triple, `<__clang_cuda_math_forward_declares.h>` declares
   `__DEVICE__ bool isgreater(double,double);` but MSVC injects `isgreater` as a
   *function-like macro* (via `<corecrt_math.h>` / `__NO_MATH_DEFINES` not honored for
   `__builtin_` forward decls), mangling the signature → `error: __device__ ...`.

## Resolution recipe (D012)

Compile with the **GNU target triple** so MSVC `<cmath>` macros are never injected,
but **define `_MSC_VER`** so `amd_warp_functions.h` takes its LLP64 `#else` branch:

```
hipcc  --offload-arch=gfx1201 \
       --rocm-device-lib-path=<ROCM>/lib/llvm/amdgcn/bitcode \
       --target=x86_64-pc-windows-gnu \
       -std=c++17 \
       -D__NO_MATH_DEFINES=1 -D_MSC_VER=1900 \
       -D_NATIVE_WCHAR_T_DEFINED=1 -D_WCHAR_T_DEFINED -fshort-wchar \
       -c -o smoke_host.o kernel_smoke.hip.cc
clang++ --target=x86_64-pc-windows-gnu -fuse-ld=lld -nodefaultlibs \
        -L<strawberry>/c/lib/gcc/x86_64-w64-mingw32/13.2.0 \
        -L<strawberry>/c/x86_64-w64-mingw32/lib -L<strawberry>/c/lib \
        -L<ROCM>/lib \
        smoke_host.o amdhip64.lib libmingwex.a libmsvcrt.a \
        libkernel32.a libadvapi32.a libuser32.a -lgcc_s -lgcc -lmingw32 \
        -o smoke_host.exe
```

**Link rationale:** GNU-triple objects carry MinGW-runtime expectations; MSVC
`link.exe`/`lld-link` ignore `-l`/`-L` MinGW flags. Use GNU `ld.lld` (invoked by
`clang++ -fuse-ld=lld`) against Strawberry MinGW 13.2 runtime objects. The host
`.lib` is `amdhip64.lib` (MSVC format) — lld accepts MSVC-format `.lib` natively.

## Verified run

```
G:\vllm-windows WinRocM-llm\cmake\smoke_hip.cmd
[1/2] hipcc device+host compile (gfx1201)...
__COMPILE_OK
[2/2] lld link...
__BUILD_OK __EXE_OK
saxpy y[0]=2 y[3]=8 PASS
__RUN_OK
```

## @toolchain_paths
- Compiler: `G:\ROCM10RT-gfx1201\lib\llvm\bin\clang++.exe` (HIP 7.16 / AMD clang 23.0.0git)
- Driver: `G:\ROCM10RT-gfx1201\bin\hipcc` (sets offload + device-lib flags)
- Runtime `.lib`: `G:\ROCM10RT-gfx1201\lib\amdhip64.lib`, `hipblas.lib`, `MIOpen.lib`
- Device lib (bitcode): `G:\ROCM10RT-gfx1201\lib\llvm\amdgcn/bitcode`
- MinGW runtime: `C:\Strawberry\c\lib\gcc\x86_64-w64-mingw32\13.2.0` + `C:\Strawberry\c\x86_64-w64-mingw32\lib`
- MSVC linker/SDK: NOT used for host link (GNU lld instead). Windows SDK ucrt at
  `C:\PROGRA~2\WI3CF2~1\10\Lib\100280~1.0\ucrt\x64` available if needed for pure-host C++.
- MSVC cl.exe: present (`VS18\Community\...\cl.exe`, 19.51.36256) but used only for
  CPU-only C++ host code, NEVER for HIP kernel files.
- AmD SMI: absent — GPU enumeration via `hipInfo`/`hipGetDeviceProperties` + Win32 WMI.

## @status done
@see knowledge/engine/core.dox.md
@see knowledge/decisions.log.md#D012
@see knowledge/gpu/windows_gpu.dox.md

@section host_exe Phase 0.2 host exe (builds + runs on RX 9070 XT)

The host engine exe (`vllm_engine.exe`) uses plain clang++ (no `hipcc`) with the
D012 shim flags via `cmake/build_host.cmd` (hand-rolled, avoiding CMake's
Windows-Clang platform module which forces MSVC link semantics). Validated run:
```
WinRocM-llm phase0.4 stub — argc=1
GPU 0: AMD Radeon RX 9070 XT (gfx1201) PCI=000e:0e:00 VRAM=16304 MB used=151 MB
```

@section host_link Link recipe
GNU `ld.lld.exe` (ROCm `lib/llvm/bin`, first on PATH) accepts both MinGW GNU `.a`
and MSVC COFF `.lib` in one link. Runtime libs in order:
1. `amdhip64.lib` (ROCm HIP runtime)
2. MinGW: `libwinpthread.a`, `libmingwex.a`, `libmsvcrt.a`, `libadvapi32.a`,
   `libuser32.a`, `libkernel32.a`
3. Strawberry `c/lib/libstdc++.a` (C++ runtime — provides `std::terminate`)
4. `-lgcc_s -lgcc -lmingw32` (GCC runtime helpers)

@toolchain_pitfalls
- Do NOT put Strawberry `c\bin` first on PATH during CMake configure (its `c++.exe`
  shadows ROCm clang++). Pass `-DCMAKE_CXX_COMPILER` to an absolute ROCm path, or
  set `CMAKE_CXX_COMPILER` as CACHE FORCE + `CMAKE_CXX_COMPILER_WORKS=TRUE` +
  `CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY`.
- `clang++` `--target=gnu` defaults to MSVC `lld-link`; force GNU-mode GNU lld via
  `-fuse-ld=lld` with `%ROCM_HOME%\lib\llvm\bin` first on PATH so `ld.lld.exe`
  wins over `lld-link.exe`.
- Do NOT mix MSVC `libcmt.lib`/`libvcruntime.lib` with MinGW `libmingw32.a` —
  duplicate symbols (e.g. `_IsNonwritableInCurrentImage` in pesect).
*/
