| id | phase | title | status | owner |
|---|---|---|---|---|
| P0 | 0 | Scaffold C++ toolchain + DOX docs | done | - |
| P0.1 | 0 | Verify hipcc (HIP 7.16) compiles+links+runs a real kernel on RX 9070 XT (gfx1201) | done | - |
| P0.1b | 0 | Freeze validated toolchain recipe in cmake/smoke_hip.cmd + knowledge/hip/toolchain.dox.md | done | - |
| P0.2 | 0 | CMake toolchain (rocm-windows-toolchain.cmake) for host CXX (GNU-triple + shim) | done | - |
| P0.2b | 0 | Host engine stub build via CMake+Ninja+GNU-lld | done | - |
| P0.3 | 0 | Write Windows GPU enumeration shim (hipGetDeviceProperties + hipDeviceGetAttribute replacing amdsmi) | done | - |
| P0.4 | 0 | Bootstrap vllm_engine.exe skeleton (host C++ only; GPU ops via prebuilt wheel) | done | - |
| P0.4b | 0 | Wire GPU smoke result into vllm_engine.cpp (live GPU enum on startup) | done | - |
| P1 | 1 | C++ scheduler + KV-cache block manager (LRU + prefix-caching + GPUBuffer RAII) | done | - |
| P1.1 | 1 | Phase 1 hardening: real KV tensor writes/reads via hipMemcpyHtoD/DtoH + BlockConfig | done | - |
| P2 | 2 | C++ model_runner (forward pass, reuse csrc) | partial | - |
| P2.1 | 2 | ModelRunner class: structs (SequenceInputs, ModelRunnerOutput), AttentionType enum, prepare_inputs/execute/gather_outputs signatures | done | - |
| P2.2 | 2 | prepare_inputs: real BlockTable allocation (allocate + allocate_child per chunk) | done | - |
| P2.3 | 2 | execute/gather_outputs stubs throw std::logic_error (torch wiring deferred to P2.5) | done | - |
| P2.5 | 2 | Real attention dispatch: ModelRunner::execute calls paged_attention via the C-ABI bridge; prepare_inputs stages block-table/seq-lens via hipMemcpyHtoD. | partial | - |
| P2.5.1 | 2 | gfx1201 device kernel compiles (D016 scalar WMMA fallback; csrc/rocm/attention_gfx1201.cu device-only rc=0). | done | - |
| P2.5.2 | 2 | rocm_ops.dll MSVC pass (torch ext + C-ABI bridge). | blocked (D018) | - |
| P2.4 | 2 | Wire --attention into main.cpp; build + run on RX 9070 XT (self-check PASS; attention fail-closed at DLL gate). | done | - |
| P2.6 | 2 | Approach A: single clang++ MSVC-triple TU (HIP 7.16 / clang 23.0.0git) compiles paged-attention device kernel + C-ABI bridge + torch-bindings host. | in-progress | - |
| P2.7 | 2 | Verify bridge+bindings compile under `clang++ --target=x86_64-pc-windows-msvc -std=c++20` + C10_CUDA_NO_CMAKE_CONFIGURE_FILE + C10_STATIC_DEFINE + cuda stub. | todo | - |
| P3 | 3 | C++ attention backend dispatch (ROCm/CPU) | todo | - |
| P4 | 4 | C++ quantization dispatch (Marlin/W8A8/FP8) | todo | - |
| P5 | 5 | C++ distributed (TP/DP via quickreduce) | todo | - |
| P6 | 6 | C++ KV offload tiering + disk backend | todo | - |
| P7 | 7 | C++ spec-decode (Eagle/Medusa/NGram) | todo | - |
| P8 | 8 | C++ structured output + streaming | todo | - |
| P9 | 9 | C++ OpenAI-compatible server + model loader | todo | - |
| P10 | 10 | MLA + advanced attention on ROCm Windows | todo | - |
| P11 | 11 | Full parity benchmark vs upstream | todo | - |
