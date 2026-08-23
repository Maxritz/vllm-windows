| ts | ACTION | who | target | desc | result |
|---|---|---|---|---|---|
| 2026-08-22T23:12 | CREATE | assistant | WinRocM-llm/ | scaffold folder tree (AGENTS/CLAUDE/knowledge/src/cmake/build) | ok |
| 2026-08-22T23:14 | AUDIT | assistant | vllm/ (3697 py) | scan Linux-isms: /proc (9), sys.platform/posix (13), amdsmi/ROCM (9), numactl (4), mp/zmq (4) | ok |
| 2026-08-22T23:16 | AUDIT | assistant | C:\Strawberry | discover Strawberry Perl = NOT a ROCm fork; ships MinGW-w64 13.2 + cmake + ninja | ok |
| 2026-08-22T23:17 | AUDIT | assistant | PATH | confirm `ninja` in PATH (Strawberry c\bin) | ok |
| 2026-08-22T23:18 | CONFIRM | assistant | vllm tree | 0 Vulkan files, 0 numpy imports, csrc/rocm-stubs absent in main tree | ok |
| 2026-08-22T23:19 | RECREATE | assistant | WinRocM-llm/ | recreate AGENTS.md / CLAUDE.md / PHASES.md / knowledge after accidental folder wipe | ok |
| 2026-08-22T23:20 | AUDIT | assistant | G:\ROCM10RT-gfx1201 | confirm hipcc = HIP 7.16.26323 (preview); full runtime + hipBLAS/MIOpen present; amdsmi absent; target gfx1201 | ok |
| 2026-08-22T23:22 | DECIDE | assistant | D006 | compile target = gfx1031 (RDNA2) for rocblas compat, runtime HW = gfx1201 (RDNA4) | accepted |
| 2026-08-22T23:23 | UPDATE | assistant | PHASES.md | add Phase 0.0: confirm MSVC cl.exe; configure gfx1031 compile arch via hipcc | todo |
| 2026-08-22T23:26 | CONFIRM | assistant | VS2026 / cl.exe | MSVC 19.51.36256 installed at VS18\Community; vcvarsall.bat x64 works | ok |
| 2026-08-22T23:27 | WRITE | assistant | knowledge/engine/core.dox.md | Doxygen DOCS for engine core: method mapping, windows issues, design | ok |
| 2026-08-22T23:28 | UPDATE | assistant | TASKS.md | mark P0.1 done; P0.2 in_progress with msvc+hipclang notes | ok |
| 2026-08-22T23:43 | DEBUG | assistant | kernel_smoke.hip.cc | MSVC triple: `<cmath>` isgreater macro clobbers `__device__` decls in `__clang_cuda_math_forward_declares.h` | blocker |
| 2026-08-22T23:44 | DEBUG | assistant | kernel_smoke.hip.cc | GNU triple fix cmath but trips `sizeof(long)==2*sizeof(int)` static_assert in `amd_warp_functions.h` | blocker |
| 2026-08-22T23:45 | RESOLVE | assistant | D012 | Two-shim fix: `--target=gnu` + `-D_MSC_VER=1900` + `-D__NO_MATH_DEFINES=1` + nativewchar defs. Compiles clean (only hipError_t nodiscard warnings). | accepted |
| 2026-08-22T23:52 | DEBUG | assistant | link stage | GNU-triple object fails lld-link (MSVC linker ignores `-L`/`-l` MinGW flags); GNU `ld.lld` needs MinGW CRT. MSVC `link.exe` needs COFF w/ MSVC runtime. | linker blocker |
| 2026-08-22T23:55 | RESOLVE | assistant | smoke_hip.cmd | Two-stage link: `hipcc -c` (GNU triple obj) → `clang++ -fuse-ld=lld -nodefaultlibs` + Strawberry MinGW libs (`libmingwex.a`, `libmsvcrt.a`, `libkernel32.a`, etc.) + `amdhip64.lib` from ROCm. | works |
| 2026-08-22T23:58 | RERUN | assistant | kernel_smoke.hip.cc | switched `--offload-arch` to `gfx1201` (matches on-device RX 9070 XT RDNA4, not gfx1100/gfx1031); added `hipSetDevice(0)` + `hipMalloc` error checks | ok |
| 2026-08-22T23:59 | VERIFY | assistant | smoke_hip.cmd | full run: `__COMPILE_OK` → `__BUILD_OK __EXE_OK` → `saxpy y[0]=2 y[3]=8 PASS` → `__RUN_OK` | **Phase 0.1 complete** |
| 2026-08-23T00:05 | ATTEMPT | assistant | CMake HIP-language | CMake 3.29 rejects `hipcc` as CMAKE_HIP_COMPILER; `Windows-Clang.cmake` injects MSVC flags assuming MSVC triple (conflicts D012 recipe) | incompatible |
| 2026-08-23T00:06 | RESOLVE | assistant | D013 | Don't enable CMake HIP language; drive HIP via `smoke_hip.cmd`; use CMake for host CXX only | accepted |
| 2026-08-23T00:08 | DEBUG | assistant | CMake CXX probe | `enable_language(RC)` fails; Windows-Clang.cmake needs `CMAKE_RC_COMPILER=llvm-rc` | fixed |
| 2026-08-23T00:11 | DEBUG | assistant | CMake CXX link | GNU-lld probe: MinGW runtime objects needed explicitly (`libgcc.a`, `libmingwex.a`, `libmsvcrt.a`, `libkernel32.a` etc.); MSVC `<cmath>`/`-D_MT` flags leak from Windows-Clang.cmake → stripped via `CMAKE_CXX_STANDARD_LIBRARIES_INIT` + toolchain flags | fixed |
| 2026-08-23T00:13 | FIX | assistant | logger.cpp | add `#include <cstdarg>` for `va_list`/`va_start` (MinGW doesn't implicit-include) | ok |
| 2026-08-23T00:14 | VERIFY | assistant | rocm-windows-toolchain.cmake | CMake configure OK (`Clang 23.0.0 GNU-like`); build produces `vllm_engine.exe` | Phase 0.2 complete |
| 2026-08-23T00:14 | RUN | assistant | vllm_engine.exe | prints `WinRocM-llm phase0.4 stub — argc=1` | ok |
| 2026-08-23T01:30 | REPLACE | assistant | P0.3 stub + main.cpp | wrote gpu_windows.h/.cpp (hipGetDeviceProperties+hipDeviceGetAttribute); wired enumerate_gpus() into main | compiled |
| 2026-08-23T02:10 | DEBUG | assistant | CMake host link | CMake Windows-Clang.cmake forced MSVC `lld-link` + `/MANIFEST` template, incompatible with GNU lld. Abandoned CMake CXX link; moved host build to hand-rolled build_host.cmd | root cause found |
| 2026-08-23T02:25 | DEBUG | assistant | linker symbol errors | `_commode`: MSVC libcmt vs MinGW libmingw32 collide (duplicate `__imp_`). `std::terminate`: needs C++ runtime. `pthread_mutex_init`: libstdc++ needs winpthread. | root cause found |
| 2026-08-23T02:35 | RESOLVE | assistant | build_host.cmd link recipe | dropped MSVC libcmt; added MinGW libwinpthread+libmingwex+libmsvcrt + Strawberry libstdc++.a + `-lgcc_s -lgcc -lmingw32`; `-fuse-ld=lld` w/ ROCm LLVM bin first on PATH forces GNU ld.lld | resolved |
| 2026-08-23T02:40 | VERIFY | assistant | build_host.cmd + vllm_engine.exe | builds via clang++ (GNU lld) on ROCM clang++ 23.0.0; runs on RX 9070 XT | P0.3+P0.4b complete |
| 2026-08-23T02:40 | RUN | assistant | vllm_engine.exe (gpu enum) | `GPU 0: AMD Radeon RX 9070 XT (gfx1201) PCI=000e:0e:00 VRAM=16304 MB used=151 MB` | ok |
| 2026-08-23T02:42 | UPDATE | assistant | knowledge/hip/toolchain.dox.md + D012 | record host_exe section + link recipe + 3 toolchain_pitfalls (PATH shadowing, lld-link vs ld.lld, libcmt/libmingw32 mix) | ok |
| 2026-08-23T03:05 | VERIFY | assistant | hip_malloc_smoke.hip.cc | hipMalloc/hipFree/hipMemcpy round-trip on RX 9070 XT passes (VRAM=16304MB) | P1 GPUBuffer foundation validated |
| 2026-08-23T03:12 | CREATE | assistant | src/engine/gpu_buffer.h/.cpp + block_table.h/.cpp | real RAII GPUBuffer + LRU BlockTable with prefix caching + multi-GPU placement | compile OK |
| 2026-08-23T03:15 | DEBUG | assistant | block_table.cpp | line 10 `std::hash<int>{}(...)` parse error under GNU target; switched to `std::hash<int> hasher;` local | fixed |
| 2026-08-23T03:18 | DEBUG | assistant | gpu_windows.cpp | ROCm 7.16 renamed `hipDeviceAttributeGcnArch`→`hipDeviceAttributeUnused4`; derive GCN arch from compute `major` (Navi12→gfx1201) | fixed |
| 2026-08-23T03:20 | VERIFY | assistant | build_host.cmd + vllm_engine.exe --self-check | builds via clang++/GNU ld.lld; self-check PASS: root prefix-shared, child prefix-shared, eviction alloc, GPUBuffer round-trip | Phase 1 complete |
| 2026-08-23T03:22 | UPDATE | assistant | knowledge/engine/core.dox.md + tasks + decisions | record P0+P1 done; add D014 (hand-rolled host build drops CMake CXX link) | ok |
| 2026-08-23T03:25 | DECIDE | assistant | D015 | ModelRunner interface: header-only structs/enums + stub .cpp, no torch coupling. prepare_inputs consumes real BlockTable. | accepted |
| 2026-08-23T03:25 | DECIDE | assistant | D016 | AttentionType enum kTorch/kROCm/kCPU mirrors vLLM AttentionBackendEnum dispatch (registry.py). | accepted |
| 2026-08-23T03:30 | CREATE | assistant | src/engine/include/model_runner.h | ModelRunner class: SequenceInputs, ModelRunnerOutput, AttentionType enum, prepare_inputs/execute/gather_outputs signatures. | ok |
| 2026-08-23T03:30 | CREATE | assistant | src/engine/model_runner.cpp | prepare_inputs allocates root+child blocks via BlockTable; execute/gather_outputs throw std::logic_error. | ok |
| 2026-08-23T03:32 | UPDATE | assistant | src/cli/main.cpp | wire --dry-run path: prepare_inputs + execute exception catch + gather_outputs exception catch. | ok |
| 2026-08-23T03:35 | UPDATE | assistant | cmake/build_host.cmd + CMakeLists.txt | add model_runner.cpp to build; bump std to gnu++20 (std::span). | ok |
| 2026-08-23T03:40 | BUILD+RUN | assistant | build_host.cmd --dry-run | build via clang++/GNU ld.lld on RX 9070 XT; prepare_inputs ok (blocks=67108864), execute threw, gather_outputs threw, dry-run PASS. | Phase 2 complete
| 2026-08-23T04:20 | RESEARCH | assistant | D016 kernel | upstream attention.cu `__GFX12__` path calls `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` — ROCm 7.16 clang 23 does NOT define this intrinsic for gfx1201 (IntrinsicsAMDGPU.td has gfx9/gfx11 WMMA only). | confirmed missing
| 2026-08-23T04:25 | CREATE | assistant | csrc/rocm/attention_gfx1201.cu | device-only kernel TU with `__GFX12__`/`__GFX12_HAS_WMMA__` arch detection + scalar fp16/bf16→fp32 fmaf fallback replacing the unavailable `_w32_gfx12` builtin. | compiles rc=0 for gfx1201 (hipcc, -D_MSC_VER)
| 2026-08-23T04:45 | CREATE | assistant | src/engine/include/paged_attention_bridge.h + csrc/rocm/paged_attention_bridge.cpp | C-ABI `extern "C" paged_attention_rocm(...)` trampoline: wraps raw device ptrs into torch tensors in the DLL, no torch headers cross into the GNU-ldd engine. | C-ABI surface locked (24 params incl shape)
| 2026-08-23T05:00 | CREATE | assistant | src/engine/include/model_runner.h/.cpp refactor | real `execute` loads rocm_ops.dll via LoadLibrary, calls C-ABI; prepare_inputs host-stages block-table+seq_lens via hipMemcpyHtoD. | compiles rc=0, self-check PASS
| 2026-08-23T05:15 | DEBUG | assistant | model_runner.cpp AV | raw host deref of hipMalloc'd block_tables/seq_lens ptrs SEGV'd on Windows (MSVC triple allows it; GNU mingw triple faults). Fixed via std::vector staging + hipMemcpyHtoD. | fixed
| 2026-08-23T05:30 | ATTEMPT | assistant | D017 rocm_ops.dll MSVC pass | cl.exe: torch headers route to HIP `cuda_runtime.h` → GNU-attr `amd_hip_vector_types.h` → cl.exe parser rejects (D013). hipcc host: torch 2.15 needs C++23 + generated `cuda_cmake_macros.h` + real CUDA rt headers; shim cascade unwinnable. | blocked (D018)
| 2026-08-23T05:45 | DECIDE | assistant | D017/D018 | Engine dispatch wiring stays (fail-closed runtime_error if DLL absent); rocm_ops.dll build deferred until ROCm-on-Windows torch toolchain configured. build_host.cmd guards D017 behind BUILD_ROCM_OPS_DLL=1. | accepted (fail-closed)
| 2026-08-23T06:00 | BUILD+RUN | assistant | build_host.cmd + --self-check + --attention | self-check PASS (KV round-trip, prefix-cache, eviction); --attention runs full prepare_inputs dispatch path then fail-closed at rocm_ops.dll LoadLibrary (expected, no stub). | P2.5 wiring verified |
