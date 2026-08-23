# Phased Plan: C++/HIP-on-Windows vLLM Engine

**Goal:** Build a C++ (with HIP-on-Windows) vLLM inference engine that replicates the
Linux Python vLLM behavior. The Linux `vllm/` package is ~3697 files of Python orchestration
(engine, scheduler, KV-cache manager, model runner, quantization dispatch, serving API).
This plan splits the work into phases, each delivering a vertical slice that runs end-to-end.

**Key boundary:** vLLM already ships C++ leaf kernels in `csrc/libtorch_stable/` (quantization,
MoE, attention, CPU ops, ROCm gems). The Python layer is the *missing* part we rebuild in C++.
Torch ops registered via `csrc/ops.h` / `torch_bindings.cpp` remain the kernel ABI; we do not
rewrite those kernels — we reimplement the **Python orchestration** in C++ and call into the
existing C++ kernels / TorchScript / HIP kernels via libtorch on Windows.

---

## Phase 0 — Bootstrapping the Windows C++ toolchain

**Deliverable:** A C++ vLLM library that compiles on Windows with HIP/ROCm and links against
the existing `csrc/` kernels.

| # | Task | Detail |
|---|---|---|
| 0.1 | CMake on Windows (MSVC) | Adapt `CMakeLists.txt` / `cmake/utils.cmake` for MSVC + Ninja; confirm `csrc/libtorch_stable` + `csrc/cpu` + `csrc/rocm` compile. |
| 0.2 | HIP-on-Windows | Add `rocm-windows-toolchain.cmake` support: `hipcc` path detection, `HIP_PATH` env, Windows `.dll` output naming. |
| 0.3 | Torch libtorch import | Use precompiled `torch` Windows wheel (CUDA or ROCm build) via `setup.py` `VLLM_USE_PRECOMPILED=1` path; register custom ops into `torch.ops._C`. |
| 0.4 | Build artifact | `vllm.dll` exposing `PyInit__vllm` (Python extension) **or** a pure C++ shared lib `vllm_engine.dll` + small C-API shim. Chose C++-only if no Python interop desired. |

---

## Phase 1 — Core scheduling & KV-cache manager (C++)

**Deliverable:** A standalone C++ scheduler that accepts requests and manages the paginated KV cache, mirroring `vllm/v1/engine/core.py` + `vllm/v1/core/kv_cache_manager.py`.

| # | Task | Maps to (Python) |
|---|---|---|
| 1.1 | Request + Sequence data structures | `vllm/v1/engine/llm_engine.py`, `vllm/v1/core/request_queue.py`, `SchedulerOutput` |
| 1.2 | TokenBlockPool / GPU memory allocator | `vllm/v1/core/block_pool.py`, `vllm/worker/block_table.py` |
| 1.3 | Paged KV-cache manager (block allocator, eviction, preemption) | `vllm/v1/core/kv_cache_manager.py`, `single_type_kv_cache_manager.py` |
| 1.4 | Scheduler (prefill/decode, priority, chunk prefill) | `vllm/v1/core/sched/scheduler.py` |
| 1.5 | ZMQ control channel (engine <-> worker) | `vllm/v1/engine/core.py` `TensorIpcReceiver`, `make_zmq_socket` |
| 1.6 | Self-test: feed 100 dummy token requests, verify block allocation/free. | |

**Exit criterion:** C++ binary `vllm_sched_test` passes block-allocation round-trip.

---

## Phase 2 — Model runner (C++ forward pass)

**Deliverable:** C++ model executor that loads HF weights, runs RoPE + attention + feedforward + sampling for a single Llama-style model, calling the existing C++ Torch kernels.

| # | Task | Maps to (Python) |
|---|---|---|
| 2.1 | Model config + weight loader (npz/torch save) | `vllm/config/`, `model_loader/` |
| 2.2 | Transformer layer forward (QKV, RoPE, silu_and_mul, rotary_embedding) | `model_executor/layers/`, `ops.h` kernels (reuse as Torch ops) |
| 2.3 | Sampling head (logits -> sampler) | `vllm/v1/sample/sampler.py` |
| 2.4 | Tensor parallelism (all-reduce via `csrc/quickreduce`) | `vllm/distributed/` |
| 2.5 | CUDA/HIP graph capture (decode) | `cudagraph_utils.py`, `csrc/custom_quickreduce.cu` |
| 2.6 | Self-test: run `meta-llama/Llama-2-7b` 1-token decode, check logits shape. | |

**Exit criterion:** `vllm_model_test` produces correct log-probs within 1e-3 vs Python reference.

---

## Phase 3 — Attention backends (HIP/CUDA kernels)

**Deliverable:** C++ attention dispatch that swaps between CPU, ROCm (AITER), and CUDA (FlashInfer/FlashAttention) backends, mirroring the 48-file backend registry.

| # | Task | Maps to |
|---|---|---|
| 3.1 | Attention backend registry + selector | `vllm/v1/attention/backends/registry.py`, `ml/a/prefill/selector.py` |
| 3.2 | ROCm AITER unified attention (`rocm_aiter_unified_attn`) | `csrc/rocm/*` kernels |
| 3.3 | FlashInfer / FlashAttention prefill+decode | `vllm/v1/attention/ops/*triton*` / flashinfer |
| 3.4 | CPU native attention (`cpu_attn`) | `csrc/cpu/cpu_attn*` |
| 3.5 | MLA (Multi-Head Latent Attention) dispatch | `vllm/v1/attention/backends/mla/*` |
| 3.6 | Self-test: run 1024-batch decode with/without MLA. | |

**Exit criterion:** throughput matches Python Triton backend within 5%.

---

## Phase 4 — Quantization (Marlin / W8A8 / FP8 / MXFP4)

**Deliverable:** C++ quantization layer dispatch wired to the existing `csrc/libtorch_stable/quantization/` kernels.

| # | Task | Maps to |
|---|---|---|
| 4.1 | Quant scheme registry (RTN, Marlin, Machete, W8A8, FP8, NVFP4, MXFP4) | `model_executor/layers/quantization/` |
| 4.2 | Weight-only + activation quant dequant | `vllm_quantize_*` ops in `csrc/` |
| 43 | GPTQ/AWQ dequant | `auto_awq.py`, `auto_gptq.py` |
| 4.4 | TorchAO / Quark / INC backends (Windows-relevant) | `torchao.py`, `quark*.py`, `inc_*` |
| 4.5 | Self-test: load `TheBloke/Llama-2-7B-GPTQ` 4-bit, compare logits. | |

**Exit criterion:** quant models run; no Python quant dispatch used.

---

## Phase 5 — Distributed (tensor / pipeline / data / sequence parallel)

**Deliverable:** C++ distributed runtime replicating `vllm/distributed/` for multi-GPU Windows.

| # | Task | Maps to |
|---|---|---|
| 5.1 | NCCL/ROCm-clr all-reduce/all-gather (via `csrc/quickreduce`) | `vllm/distributed/parallel_state.py` |
| 5.2 | Tensor + pipeline + sequence parallel comms | `vllm/distributed/model_parallel` |
| 5.3 | Pipeline micro-batching + interleaving | `vllm/distributed/pipeline_parallel` |
| 5.4 | Self-test: 2x GPU tensor-parallel, verify identical logits. | |

**Exit criterion:** 2-GPU TP run matches single-GPU output.

---

## Phase 6 — KV Offload / Paged attention paging

**Deliverable:** C++ tiering manager — GPU↔CPU↔disk, mirroring `vllm/v1/kv_offload` + `simple_kv_offload`.

| # | Task | Maps to |
|---|---|---|
| 6.1 | Block swap policy (CPU/GPU/disk tiering) | `vllm/v1/kv_offload/config.py`, `policies/factory.py` |
| 6.2 | Disk backend (async read/write) | `simple_kv_offload/disk_backend.py` |
| 6.3 | Fault-tolerance sentinel | `vllm/v1/fault_tolerance/` |
| 6.4 | Self-test: 4× GPU memory pressure, verify swap-in/out correctness. | |

**Exit criterion:** runs under simulated 8 GB GPU budget without OOM.

---

## Phase 7 — Spec-decoding (Eagle / Medusa / NGram)

**Deliverable:** C++ draft-model + rejection-sampling pipeline.

| # | Task | Maps to |
|---|---|---|
| 7.1 | Draft model loader + proposer | `vllm/v1/spec_decode/draft_model.py`, `eagle.py` |
| 7.2 | Hidden-state extractor | `extract_hidden_states.py` |
| 7.3 | Rejection / adapted sampling | `rejection_sampler.py` |
| 7.4 | Self-test: 2x speedup on `meta-llama/Llama-2-7b` + Eagle-7B. | |

**Exit criterion:** achieves ≥1.8× decode speedup vs baseline.

---

## Phase 8 — Structured output + streaming sampler

**Deliverable:** C++ constrained-generation (regex/JSON-schema) + streaming token emission.

| # | Task | Maps to |
|---|---|---|
| 8.1 | Structured output (regex/JSON) | `vllm/v1/structured_output/` |
| 8.2 | Thinking-budget / stopping-criteria | `sample/thinking_budget.py`, `states.py` |
| 8.3 | Streaming token emitter over ZMQ | `vllm/v1/engine/core.py` output loop |
| 8.4 | Self-test: emit JSON `{"name": "x", "age": 5}` guaranteed valid. | |

**Exit criterion:** 100/100 valid JSON outputs under stress.

---

## Phase 9 — Serving API + multi-model loader

**Deliverable:** C++ OpenAI-compatible server replacing `vllm/entrypoints/openai`.

| # | Task | Maps to |
|---|---|---|
| 9.1 | HTTP server + REST routes (chat/completion/embeddings) | `entrypoints/openai/*` |
| 9.2 | LoRA adapter loading + switching | `vllm/lora/` |
| 9.3 | Multimodal processor (vision/audio/video) | `vllm/multimodal/` |
| 9.4 | Tokenizer port (use tokenizer-rs crate via `rust/` bindings, or C++ SentencePiece) | `rust/src/tokenizer`, `vllm/tokenizers` |
| 9.5 | Self-test: `curl` OpenAI chat endpoint, 10 concurrent. | |

**Exit criterion:** drop-in `curl localhost:8080/v1/chat/completions` works.

---

## Phase 10 — MLA + advanced attention (flash-infer, triton-MLA on ROCm)

**Deliverable:** Full MLA stack in C++ on Windows ROCm, mirroring the 48-file backend set.

| # | Task | Maps to |
|---|---|---|
| 10.1 | Cutlass-MLA + FlashAttention-MLA C++ dispatch | `ml/a/cutlass_mla.py`, `flashattn_mla.py` |
| 10.2 | Sparse MLA (sliding-window attention) | `ml/a/flashattn_mla_sparse.py` |
| 10.3 | ROCm AITER sparse MLA | `ml/a/rocm_aiter_mla_sparse.py` |
| 10.4 | Self-test: DeepSeek-V3 128K-context decode. | |

**Exit criterion:** 128K-context run matches Linux vLLM output token-for-token.

---

## Phase 11 — Full parity benchmark + validation

**Deliverable:** C++ engine passes the full vLLM correctness & benchmark suite.

| # | Task | Maps to |
|---|---|---|
| 11.1 | Port `tests/core/` scheduler tests to C++ | `tests/core/test_scheduler.py` |
| 11.2 | Port `tests/kernels/` correctness tests | `tests/kernels/` |
| 11.3 | Run `vllm bench serve` against C++ server | `benchmarks/benchmark_serving.py` |
| 11.4 | GSM8K / MMLU eval comparison vs Linux | `tests/evals/` |
| 11.5 | Self-test: ≥99% token-match vs Python reference on 50 models. | |

**Exit criterion:** C++ `vllm bench serve --model Llama-2-7b` throughput ≥ linux-python; correctness ≥99%.

---

## Execution order & dependencies

```
0 (toolchain) → 1 (scheduler, standalone) → 2 (model runner, needs 0.3 op table)
→ 3 (attention) → 4 (quantization) → 5 (distributed, needs 3 + csrc/quickreduce)
→ 6 (kv offload) → 7 (spec-decode, needs 2 + 3) → 8 (structured output)
→ 9 (serving API, needs 1-8) → 10 (MLA) → 11 (full parity)
```

Each phase's deliverable is a compiled C++ binary + unit test; Phase 9 delivers the production server.

## Non-goals (explicit exclusions)

- Rewriting `csrc/libtorch_stable/*` kernels — we consume them.
- Triton JIT — replaced by direct C++ kernel calls / FlashInfer C++ API.
- Python fallback at runtime — the engine is pure C++/HIP.
- Vulkan — out of scope (no Vulkan in the repo).
