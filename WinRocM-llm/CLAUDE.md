# WinRocM-llm — CLAUDE

> C++ HIP-on-Windows port of vLLM's Linux Python inference engine.
> Read `AGENTS.md` first.

## Environment snapshot (discovered 2026-08-22)

### GPU / ROCm stack
- **ROCm install:** `G:\ROCM10RT-gfx1201` (folder name is misleading internal label).
- **Actual HIP version:** `7.16.26323-af82d0abb4` (AMD clang 23.0.0git).
- **Target GPU arch:** `gfx1201` = AMD RDNA4 (e.g., Radeon RX 9xx / Strix Halo).
- **Compiler:** `G:\ROCM10RT-gfx1201\bin\hipcc` (wrapper) → calls `G:\ROCM10RT-gfx1201\lib\llvm\bin\clang.exe`.
- **Runtime libs (MSVC-format):** `amdhip64.lib`, `hipblas.lib`, `hipblasLt`, `hipsolver.lib`,
  `hipsparse.lib`, `hipfft.lib`, `hiprand.lib`, `hipdnn_backend.lib`, **MIOpen.lib**, `amd_comgr.lib`.
- **Device lib quirk:** needs `--rocm-device-lib-path=<ROCM_ROOT>/lib` or `-nogpulib`.
- **amdsmi:** NOT present in this ROCm RT install (no `amdsmi.dll`/`rocm-smi` found). GPU queries via Win32 `Get-CimInstance`/`WDDM` instead.

### Host build toolchain
- **Strawberry Perl** at `C:\Strawberry` (Perl, NOT a code fork). Ships MinGW-w64 13.2 + CMake + **Ninja**
  in `C:\Strawberry\c\bin`. `ninja` is on `PATH`.
- **C++ host compiler:** prefer MSVC (`cl.exe` via Visual Studio Build Tools) for the native engine;
  use HIP-Clang only for `.hip`/`.cu` device code paths. MinGW is fallback.
- Python: 3.12 venv at `.venv` (project-local).

## Setup commands (do first)
```bat
:: ROCm
set ROCM_HOME=G:\ROCM10RT-gfx1201
set PATH=%ROCM_HOME%\bin;%PATH%
:: Ninja + CMake (from Strawberry, already on PATH)
```

## Windows-breaking Linux-isms in upstream vllm/ (audit)

| Pattern | Count | Offenders | C++ fix |
|---|---|---|---|
| `/proc` paths | 9 | numa_utils, weight_utils, system_utils, torch_utils, platforms, cpu_attn, cpu_worker | Win32 API |
| `sys.platform`/`posix`/`os.name` | 13 | setup.py, platforms/*, cpu_worker, system_utils, pynvml, cpu_moe, network_utils, cpu_resource_utils | `#ifdef _WIN32` |
| `amdsmi` | 9 | platforms/rocm.py, collect_env, setup.py, aiter.py | `Get-CimInstance` WDDM query (no amdsmi here) |
| `numactl`/`libnuma` | 4 | numa_utils, config/parallel, cpu_worker | `GetNumaProcessorNodeEx` |
| `multiprocessing` fork/spawn | ~4 | engine/core.py, tensor_ipc | native `std::thread` + job objects |
| POSIX sockets / ZMQ | ~2 | engine/core.py, network_utils | Winsock / IOCP |

## Next actions
1. [ ] Confirm `cl.exe` (MSVC) available; else configure MinGW-only build.
2. [ ] Create `src/cli/main.cpp` + `cmake/CMakeLists.txt` (Phase 0 scaffold).
3. [ ] Port numa/proc utilities into `src/hip_wrap/`.
