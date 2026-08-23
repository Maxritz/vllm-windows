# vLLM Feature Inventory Report — upstream `main` @ `da329cc30`

## Source: pure vLLM (`vllm-project/vllm`), no Windows port modifications.

---

## 1. Repository composition

| Category | Files | Notes |
|---|---|---|
| Python `.py` | 3697 | `vllm/`, `tests/`, `benchmarks/`, `examples/`, `scripts/` |
| C++/CUDA/ROCm (`cuh cu cpp h hpp cc`) | 273 | all under `csrc/` + 1 in `vllm/` |
| Rust `.rs` | 247 | `rust/` crates (text backend, server, tokenizer) |
| YAML configs | 285 | `.buildkite/`, `.github/`, `benchmarks/`, `docs/` |
| Markdown `.md` | 281 | docs, READMEs |
| Shell `.sh` | 130 | build, benchmark, docker, CI |
| TOML | 19 | `pyproject.toml`, toolchain, requirements |
| **Total tracked** | **~5884** | |

---

## 2. Supported backends / platforms

| Backend | Platform file | Status |
|---|---|---|
| CUDA | `vllm/platforms/cuda.py` | primary |
| ROCm | `vllm/platforms/rocm.py` | AMD GPUs |
| XPU | `vllm/platforms/xpu.py` | Intel GPUs |
| CPU | `vllm/platforms/cpu.py` | CPU inference |
| TPU | `vllm/platforms/tpu.py` | TPU |
| Zen CPU | `vllm/platforms/zen_cpu.py` | AMD CPU tuned |
| Interface | `vllm/platforms/interface.py` | abstraction |

---

## 3. Python feature surface (top-level `vllm/` package, 1964 files)

| Feature area | Files | Contents |
|---|---|---|
| **Engine / V1** | 320 | `vllm/v1/` — engine, worker, attention, sample, spec_decode, kv_offload, structured_output, fault_tolerance, metrics, executor, pool |
| **Model executor (layers)** | 430+ | fused_moe, quantization, mamba, rotary_embedding, attention, pooler, hpc, fusion, minimax_rms_norm |
| **Model implementations** | 311 models | full model zoo (transformers-backed modeling layer) |
| **Transformers utils** | 114 | tokenizer + weight loading + processors |
| **Entrypoints** | 180 | OpenAI-compatible API server, CLI, ray, speech-to-text, scale-out, MCP |
| **Distributed** | 123 | tensor/pipeline/DP/SP, 3FS, mooncake, nixl, moriio, offloading connectors |
| **Input types** | 4 | request dataclasses |
| **IR / config / scalar_type** | 53 | config objects, scalar types, IR builders |
| **Logging / tracing / usage** | 24 | pydantic logs, OpenTelemetry, anonymous usage |
| **Utils** | 37 | cache, network, filesystem, system, serialization |
| **kernels** | 20 | python kernel wrappers (triton/flashinfer dispatch) |
| **reasoning** | 0 (empty dir marker) | reserved |
| **ray** | 2 | lazy utils, env setup |
| **parser** | 27 | deepseek_v3.2/v4, gemma4, kimi_k2/k3, mistral, nemotron, etc. |
| **tool_parsers** | 45 | tool-calling renderers per-model |
| **renderers** | 18 | output formatting (harmony, default, deepseek_v3/v4) |
| **tokenizers** | 11 | HF/tiktoken/tekken byte-level + incremental decode |
| **multimodal** | 24 | media, processing, video decoders |
| **lora** | 42 | layers, ops, punica wrapper |
| **plugins** | 7 | extensible plugin system |
| **device_allocator / cute_utils / third_party** | 27 | helpers |
| **vllm_flash_attn** | 2 | vendored FA3 shim |

---

## 4. V1 engine architecture (`vllm/v1/`, 320 files)

| Sub-system | Files | Features |
|---|---|---|
| **engine** | 14 | core loop, scheduler, utils, `core.py` |
| **worker** | 113 | GPU/CPU worker, model_states, spec_decode, sample, ec_connector |
| **attention** | 90 | backends + MLA (Multi-Head Latent Attention) suite |
| **kv_offload** | 46 | tiered CPU offloading, disk backend, policies |
| **simple_kv_offload** | 7 | simple disk offload |
| **spec_decode** | 19 | Eagle / Medusa / NGram / DFlash / Step3p5 / Gemma4 / Multi-module-MTP |
| **sample** | 15 | sampler, logits, thinking_budget, trace_replay |
| **executor** | 9 | Ray, multiproc, unproc executors |
| **structured_output** | 8 | constrained generation (regex/JSON schema) |
| **metrics** | 8 | Prometheus metrics |
| **pool / fault_tolerance** | 7 | worker pools, sentinels |

---

## 5. Attention backends (`vllm/v1/attention/backends/`, 48 files)

| Backend | Target | Notes |
|---|---|---|
| `triton_attn` | Triton | default GPU |
| `flash_attn` | CUDA | flash-attention v2/v3 |
| `flashinfer` | CUDA | FlashInfer |
| `flashinfer_mla` | CUDA | MLA via FlashInfer |
| `flashattn_mla` | CUDA | MLA via FlashAttention |
| `triton_mla` | Triton | MLA via Triton |
| `cutlass_mla` | CUTLASS | C++ cutlass MLA |
| `flex_attention` | CUDA | PyTorch flex |
| `linear_attn` | CPU | linear attention |
| `mamba1/2_attn` | CUDA | Mamba state-space |
| `short_conv_attn` | CUDA | short convolution (Mamba/Maia) |
| `cpu_attn` | CPU | native CPU attention |
| `amx_mla` | CPU | AMX-accelerated MLA |
| `cpu_native` | CPU | native CPU MLA |
| `rocm_attn` | ROCm | ROCm MIAPI |
| `rocm_aiter_fa` | ROCm | AMD AITER unified attn |
| `turboquant_attn` | CUDA | Inkling turboquant |
| `diffkv` | CUDA | differential KV cache |
| `gdn_attn` | CUDA | GDN sparse |
| `hpc_attn` | CUDA | HPC attention |
| `sparse_swa` | — | sliding-window sparse (MLA) |

---

## 6. C++ kernel surface (`csrc/`, 273 files)

| Area | Files | Kernels | Python owner |
|---|---|---|---|
| **libtorch_stable/quantization** | ~100 | Marlin, Machete, W8A8, FP8/QDQ, per-tok-group Quant, cutlass dispatches (sm70/80/89/90/100/120), scaled_mm, qdq | `vllm/model_executor/layers/quantization/*` |
| **libtorch_stable/moe** | 22 | fused_moe, marlin_moe_wna16, dynamic_4bit_int_moe, wna16, cutlass/aiter moe | `vllm/model_executor/layers/fused_moe.py` |
| **libtorch_stable/attention** | 7 | cutlass attention | `vllm/v1/attention/backends/` |
| **libtorch_stable/mamba** | 3 | mamba scan | `vllm/model_executor/layers/mamba` |
| **libtorch_stable/gdn** | 2 | GDN kernels | attention/gdn backend |
| **libtorch_stable/kimi_k3** | 4 | Kimi K3 custom kernels | `vllm/model_executor/models/kimi_k3.py` |
| **libtorch_stable/core** | 1 | scalar_type, registration | vllm core dispatch |
| **cpu** | 59 | AMX/NEON/RVV/VSX/VXE/SVE CPU: cpu_attn, cpu_fused_moe, cpu_tanhf, cpu_types | `vllm/v1/attention/backends/cpu_attn.py`, cpu moe |
| **rocm** | 8 | q_gemm, moe_q_gemm, qdq, skinny_gemms, attention | `vllm/v1/attention/backends/rocm*` |
| **cutlass_extensions** | 5 | cutlass helpers, scaled_mm | quantization utils |
| **quantization** | 4 | fp8/nvidia vs amd quant utils | quantization layers |
| **quickreduce** | 3 | custom_allreduce, custom_quickreduce | `vllm/distributed/` allreduce |
| **moe** (root) | 1 | moe registration glue | libtorch_stable/moe |
| **root files** | ~10 | ops.h, torch_bindings.cpp, cuda_utils.h, cumem_allocator | `vllm/_custom_ops.py` op table |

---

## 7. Quantization backends (`vllm/model_executor/layers/quantization/`, 80+ files)

| Scheme | Files | Notes |
|---|---|---|
| Marlin | 4 (marlin_utils*) | FP4/FP8 weight-only |
| Compressed Tensors | 15+ | w4a4-mxfp4, w4a4-nvfp4, w8a8-fp8, w8a8-int8, wNa16, rdna3, flydsl |
| TorchAO | 2 | torchao integration |
| INC (Intel) | 14+ | inc_w4a8, inc_mxfp4, inc_mxfp8, inc_xpu schemes |
| Quark (AMD) | 8+ | quark_nvfp4, quark_ocp_mx, quark_w4a8, quark_w8a8 |
| GPTQ / AWQ | 4 | auto_awq, auto_gptq, awq_triton |
| FBGEMM | 1 | fp8_utils |
| KV cache quant | 1 | kv_cache.py |
| Machete / QUTLASS | 3 | machete_utils, qutlass |
| FlashInfer QMoE | 2 | flashinfer_fp4_moe, flashinfer_mxint4_moe |
| Humming / B12X | 3 | humming, b12x_moe |
| Custom | 4 | experts_int8, input_quant_fp8, fp_quant, modelopt |

---

## 8. Spec-Decoding family (`vllm/v1/spec_decode/`, 19 files)

| Speculator | Method |
|---|---|
| Eagle | vision+language feature distillation |
| Medusa | auxiliary LM heads |
| NGram | n-gram proposal (CPU+GPU) |
| DFlash | DFLASH2 multi-module-MTP |
| Step3p5 | 3.5-step pipeline |
| Gemma4 | gemma-4 style |
| Multi-module-MTP | deep explicit MTP |
| Suffix decoding | token-suffix proposals |

---

## 9. Distributed KV-Transfer connectors (`vllm/distributed/kv_transfer/kv_connector/v1/`, 5 backends)

| Connector | Backend |
|---|---|
| HF3FS | Heterogeneous-3FS |
| LCache (lmcache_integration) | LMCache |
| Mooncake | mooncake-store |
| Moriio | moriio-connector |
| NIXL | nixl-based (PCP/DCP) |
| Simple offloading | disk/cpu tiering |

---

## 10. Model zoo scope

- **311 model implementation files** under `vllm/model_executor/models/` covering:
  - Major families: Llama 1–4, Qwen 2/3 (incl. MoE, ASR, VL), GPTQ/AWQ quant, Mamba 1/2/Mk2, DeepSeek V2/V3/MTP/OCR/Eagle, Mixtral, MixFormer, Aria, Airtable, Bagel, CogVLM, Chameleon, CLIP, ColPali/ColQwen, Mistral, Command, Gemma, Phi, Falcon, MPT, Bloom, Bert, and many more.
- **Multimodal**: vision encoders, video decoders, audio (whisper/flux), embedded media processors.

---

## 11. Entrypoints & serving (`vllm/entrypoints/`, 180 files)

| Sub-area | Features |
|---|---|
| `openai` | full OpenAI-compatible REST API server (chat/completion/embeddings/tools/pool/scale_out) |
| `cli` | `vllm bench`, `vllm serve`, `vllm run`, `vllm.install`, `vllm.convert` |
| `speech_to_text` | ASR transcription |
| `scale_out/scale_out_router` | multi-instance routing |
| `mcp` | Model-Context-Protocol server |
| `pooling` | embedding / reward / classification pooling endpoints |

---

## 12. Benchmarking & tooling

| Area | Files | Notes |
|---|---|---|
| `benchmarks/kernels/` | 58 | kernel-level (triton, cutlass, deepgemm, helion, mamba, moe, vit) |
| `benchmarks/attention_benchmarks/` | 18 | attention perf |
| `benchmarks/fused_kernels/` | 3 | fused op benchmarks |
| `benchmarks/cutlass_benchmarks/` | 3 | cutlass |
| `benchmarks/auto_tune/` | 3 | autotuning scaffolding |
| `benchmarks/multi_turn/` | 7 | conversational perf |
| top-level bench scripts | ~20 | throughput, latency, prefix caching, block pool, hash, pin memory, topk, structured output |

---

## 13. What is NOT present (confirms scope boundaries)

| Feature | Present? |
|---|---|
| **Vulkan** | **No** (0 files: no `.v/.vert/.frag/.comp/.spv/.glsl/.hlsl`) |
| **numpy direct usage** | **No** (vLLM is pure `torch`-based; no numpy dependency) |
| **JAX/TF/ONNX** backprop backends | No |
| **Java/C#/Go bindings** | No (Python, C++, Rust, CLI only) |
| **Non-GUI web UI** | No (API server only; examples link out to Gradio etc.) |

---

## 14. Summary

This is a complete upstream vLLM main tree. The **Python layer** (`vllm/`, 1964 files + 3697 total) provides all orchestration, modeling, serving, distributed KV-transfer, quantization-policy dispatch, and the V1 engine. The **C++ layer** (`csrc/`, 273 files) provides leaf math kernels — primarily quantization (Marlin/Machete/W8A8/FP8), MoE, CPU attention, and ROCm gems — each owned by a Python feature module in `vllm/`. The **Rust layer** (`rust/`, 247 files) provides the text backend and tokenizer crates. Backends span CUDA, ROCm, XPU, CPU (with AMX/NEON/RVV/SVE), and TPU.
