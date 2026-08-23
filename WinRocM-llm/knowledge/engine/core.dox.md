/**
@file engine/core
@brief C++ reimplementation of vLLM's block manager + KV-cache + scheduler core.

## Method mapping (Linux → WinRocM-llm)
| vllM (linux)                         | WinRocM-llm                          | status |
|---|---|---|
| vllm/v1/core/block_manager.py        | src/engine/block_table.h/.cpp        | done |
| vllm/v1/core/block_table.py          | (folded into BlockTable)             | done |
| vllm/v1/worker/cache_engine.py       | src/engine/gpu_buffer.h/.cpp         | done |
| vllm/v1/core/kv_cache_utils.py       | (block_size * num_layers * 2 * num_heads * head_size) | done |
| vllm/v1/core/scheduler.py            | src/engine/scheduler.cpp (P2)        | todo |
| vllm/v1/worker/gpu/model_runner.py   | src/engine/include/model_runner.h +  | partial |
|                                      | src/engine/model_runner.cpp          |        |
| vllm/v1/outputs.py (ModelRunnerOutput)| ModelRunnerOutput struct            | partial |
| vllm/v1/attention/backends/registry.py | AttentionType enum (kTorch/kROCm/kCPU) | partial |

## Block layout (matches vLLM KV-cache tensor shape)
- Each block stores `[num_layers][2, block_size, num_heads, head_size]` floats.
- `num_layers` transformer layers (default 32 for Llama-3-70B).
- KV pair = 2 (key + value), each `[block_size, num_heads, head_size]`.
- Block size: `num_layers * 2 * block_size * num_heads * head_size * sizeof(float)`.
- Example: 32 layers, block_size=16, 32 heads, head_size=128 → ~33.5MB/block.

## BlockConfig
- `BlockConfig {int block_size, num_layers, num_heads, head_size}` set at BlockTable construction.
- Stored on BlockTable; blocks are allocated with `config.block_bytes()`.

## API surface (Phase 1 hardening)
- `BlockTable::write_block(int block_id, const float* kv_pair, ...)` — `hipMemcpyHtoD` full block KV.
- `BlockTable::read_block(int block_id, float* out, ...)` — `hipMemcpyDtoH` for verification.
- `Block` struct stores GPUBuffer sized to `config.block_bytes()`.

## Windows issues
- @windows_issue No `amdsmi` on Windows → GPUBuffer uses hipMalloc directly (D009).
- @windows_issue ROCm 7.16 preview removed gcnArchName from hipDeviceProp_t → derive from compute_major (D012).

## Design notes
- Prefix caching: blocks keyed by (parent_id, block_hash) enable reuse across sequences with shared prefixes.
- LRU eviction: least-recently-used blocks freed first; referenced blocks (ref_count > 0) are pinned.
- Multi-GPU placement: future (D011) — currently single-GPU with device_hint support.

## Phase 2.5 — ModelRunner real dispatch (C-ABI bridge, D017)
- `execute` loads `rocm_ops.dll` at runtime via `LoadLibrary`/`GetProcAddress`
  and calls the `extern "C" paged_attention_rocm` trampoline (header:
  `src/engine/include/paged_attention_bridge.h`).
- `prepare_inputs` stages the block-table map + seq-lens on the host (std::vector)
  and copies them to device via `hipMemcpyHtoD` (raw host deref of a
  hipMalloc'd pointer SEGV faults on Windows — D013 lesson).
- Scratch tensors (query/out/exp_sums/...) live in `GPUBuffer` slabs
  allocated on the BlockTable's primary device; `GPUBuffer::reserve` grew
  them on demand.  New accessors: `BlockTable::block_device_ptr` (per-block
  KV storage pointer for the paged indirection), `BlockTable::num_blocks_total`,
  `BlockTable::gpu_device_id`.
- **Fail-closed**: `execute` throws `std::runtime_error` if `rocm_ops.dll` or
  the `paged_attention_rocm` export is missing — no silent CPU fallback, no
  stubbed kernel.  `--self-check` (P1 KV round-trip) still PASS on RX 9070 XT.
- `rocm_ops.dll` is built by a separate MSVC pass (`BUILD_ROCM_OPS_DLL=1` in
  `cmake/build_host.cmd`): hipcc device kernel (D016) + cl.exe torch-extension
  host TU.  **BLOCKED on D018** (cl.exe cannot compile torch headers that
  route through HIP's GNU-attribute headers; hipcc host pass needs C++23 +
  generated torch macros).  Engine wiring is complete and await this DLL.

## Phase 2 — ModelRunner (interface + prepare_inputs)
- `ModelRunner` constructed with a `BlockTable&`, `AttentionType`, head/kv-head/head_size.
- `prepare_inputs` consumes the real BlockTable: allocates root blocks via `allocate` for
  fresh sequences, then `allocate_child` per block-size token chunk to build the KV-cache chain.
  Mirrors vLLM's `BlockTables.append_block_ids` (overwrite=True for new reqs).
- `--attention` self-check path in `main.cpp` exercises `prepare_inputs` + the
  real dispatch path; fail-closed throw at the DLL gate proves wiring.
- @status partial (Phase 2.5: dispatch wired, rocm_ops.dll deferred to D018)

@status partial (Phase 2.5: dispatch wired via C-ABI; rocm_ops.dll deferred to D018)
@see knowledge/gpu/windows_gpu.dox.md
@port_replaces vllm/v1/core/block_manager.py, vllm/v1/core/kv_cache_utils.py
*/
