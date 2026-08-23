#pragma once
/**
@file engine/model_runner
@brief C++ ModelRunner interface mirroring vLLM's GPUModelRunner.
@details
NO torch coupling in this header — the engine compiles against standard
library + BlockTable only.  The forward pass (`execute`) crosses into the
ROCm attention kernels via the C-ABI bridge in `paged_attention_bridge.h`
(`extern "C" paged_attention_rocm`), which at runtime `LoadLibrary`s
`rocm_ops.dll` (built by the MSVC pass in build_host.cmd — torch headers
live there, NOT in this GNU-ldd TU, per D014).
  `prepare_inputs` -> `GPUModelRunner.prepare_inputs`  (builds InputBatch + block tables)
  `execute`         -> `GPUModelRunner.execute_model`   (forward pass)
  `gather_outputs`  -> `GPUModelRunner.sample` + output assembly

`prepare_inputs` consumes the real BlockTable to allocate root blocks (fresh
sequences) and child blocks (continuation / prefix-cache sharing) via
`allocate` + `allocate_child`, exactly as vLLM's KV-cache manager hands
block_ids to `BlockTables.append_block_ids`.

@requires vllm::engine::BlockTable, vllm::engine::GPUBuffer (block_table.h, gpu_buffer.h)
@windows_issue No amdsmi; GPUBuffer uses hipMalloc directly (D009/D012).
@port_replaces vllm/v1/worker/gpu/model_runner.py (GPUModelRunner class)
@status partial (Phase 2.5: prepare_inputs + execute via C-ABI bridge)
@see knowledge/engine/core.dox.md
@see src/engine/include/paged_attention_bridge.h
*/
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "block_table.h"
#include "gpu_buffer.h"

namespace vllm::engine {

struct SequenceInputs {
    int seq_id;
    std::vector<int> token_ids;
};

struct ModelRunnerOutput {
    std::vector<float> hidden_states;       ///< [num_tokens, hidden_size]
    std::vector<float> logits;              ///< [num_logits, vocab_size]
    std::vector<int> top_token_ids;         ///< [num_tokens] greedy decode
    std::vector<std::vector<int>> top_token_ids_logprobs;
};

enum class AttentionType {
    kTorch,
    kROCm,
    kCPU,
};

struct PreparedBatch {
    void* query = nullptr;
    void* out   = nullptr;
    void* exp_sums = nullptr;
    void* max_logits = nullptr;
    void* tmp_out = nullptr;
    void* key_cache   = nullptr;
    void* value_cache = nullptr;
    void* block_tables = nullptr;
    void* seq_lens = nullptr;
    void* query_start_loc = nullptr;
    void* alibi_slopes = nullptr;
    void* k_scale = nullptr;
    void* v_scale = nullptr;
    void* fp8_out_scale = nullptr;
    int64_t num_seqs = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_size = 0;
    int64_t block_size = 16;
    int64_t max_num_blocks_per_seq = 0;
    int64_t max_num_partition = 1;
    int64_t num_blocks = 0;
    int64_t num_tokens = 0;
    double scale = 1.0;
    const char* kv_cache_dtype = "auto";
    const char* mfma_type = "fp16";
};

class ModelRunner {
public:
    ModelRunner(BlockTable& block_table, AttentionType attn,
                int64_t num_heads, int64_t num_kv_heads, int64_t head_size,
                double attn_scale = 1.0);

    void prepare_inputs(std::span<const SequenceInputs> sequences);

    ModelRunnerOutput execute(std::span<const SequenceInputs> sequences);

    void gather_outputs(ModelRunnerOutput& out,
                        std::span<const SequenceInputs> sequences);

    AttentionType attention_type() const noexcept { return attn_; }
    BlockTable& block_table() const noexcept { return block_table_; }
    const PreparedBatch& prepared() const noexcept { return prepared_; }

private:
    BlockTable& block_table_;
    AttentionType attn_;
    int64_t num_heads_;
    int64_t num_kv_heads_;
    int64_t head_size_;
    double attn_scale_;
    PreparedBatch prepared_;
    GPUBuffer query_slab_;
    GPUBuffer out_slab_;
    GPUBuffer exp_sums_slab_;
    GPUBuffer max_logits_slab_;
    GPUBuffer tmp_out_slab_;
    GPUBuffer block_tables_slab_;
    GPUBuffer seq_lens_slab_;
};

} // namespace vllm::engine
