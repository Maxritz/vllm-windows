# WinRocM-llm — AGENTS

## Mission
Build a **pure C++/HIP-on-Windows** inference engine that replicates the behavior of vLLM's
Linux Python engine (`vllm/v1/engine/core.py` + `vllm/v1/core/*` + `vllm/v1/worker/*`),
using the existing C++ leaf kernels in `csrc/` and HIP-on-Windows for AMD GPU kernels.

## Constraints
- **Do not rewrite leaf math kernels.** Consume `csrc/libtorch_stable/*`, `csrc/cpu/*`,
  `csrc/rocm/*` as registered torch ops via libtorch.
- **Master control plane must be C++** and launch like a Windows CLI app (`vllm.exe`).
  No Python at runtime path.
- **HIP-on-Windows** via `hipcc` (ROCm) for all AMD GPU code paths.
- **No Vulkan.** No numpy. vLLM is torch/HIP-only.
- Knowledge tracked as Doxygen-style `.md` under `WinRocM-llm/knowledge/`.

## Available build toolchain (discovered 2026-08-22)

### GPU / ROCm
- **ROCm preview install:** `G:\ROCM10RT-gfx1201` (folder label misleading — actual HIP = 7.16.26323).
- **Compiler:** `G:\ROCM10RT-gfx1201\bin\hipcc` → `G:\ROCM10RT-gfx1201\lib\llvm\bin\clang.exe` (AMD clang 23.0.0git).
- **Device lib quirk:** needs `--rocm-device-lib-path=G:\ROCM10RT-gfx1201\lib` (or `-nogpulib`).
- **Runtime libs:** `amdhip64.lib`, `hipblas.lib`, hipblasLt, hipsolver, hipsparse, hipfft,
  hiprand, hipdnn_backend, **MIOpen.lib**, comgr — all present.
- **amdsmi:** absent — GPU enumeration via Win32 `Get-CimInstance`/WDDM only.
- **Target arch:** **compile = `gfx1031` (RDNA2)** for rocblas/hipBLAS kernel compatibility;
  **runtime HW = `gfx1201` (RDNA4)** — the preview stack runs the compiled kernels on this box. (D006)

### Host C++ toolchain
- **MSVC:** Visual Studio 2026 build 18.9.1 (Community), `cl.exe` 19.51.36256 for x64.
  - Path: `...\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe`
  - Source env: `"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64`
- **Strawberry Perl** at `C:\Strawberry` ships MinGW-w64 13.2 + CMake + **Ninja** in `C:\Strawberry\c\bin`
  (Perl, NOT a code fork). Used for Perl-driven sub-builds only (D005).
- Python: 3.12 venv at `.venv` (project-local).

## Workflow
1. Every file edited must update the matching `WinRocM-llm/knowledge/<component>.dox.md`.
2. Every decision is committed to `WinRocM-llm/knowledge/decisions.log.md` BEFORE code.
3. Track tasks in `WinRocM-llm/knowledge/TASKS.md`.
4. Run `WinRocM-llm\build\vllm_test.exe --self-check` after each phase.

## Directory layout
```
WinRocM-llm/
  AGENTS.md
  CLAUDE.md
  PHASES.md
  knowledge/     Doxygen-style component docs + decisions + tasks + actions log
  src/           C++ engine (engine, scheduler, model_runner, attention, serving, cli, hip_wrap)
  third_party/   vendored (asio/json/Numa shim)
  cmake/         build config
  build/
```

## DOX conventions
```
@file <component>
@brief one-line synopsis
@details full behavior
@requires <deps>
@windows_issue <Linux-ism that breaks on Windows>
@port_replaces <original vllm path>
@status <todo|partial|done>
@see <related .dox.md>
```
