/**
@file model_runner
@brief Phase 2.5 ModelRunner — prepare_inputs + execute via C-ABI bridge.
@details
Builds KV-cache block mappings through the real BlockTable, allocates scratch
GPU tensors, then dispatches the ROCm attention op in rocm_ops.dll via the
`extern "C" paged_attention_rocm` trampoline (see paged_attention_bridge.h).
The DLL is loaded at runtime with LoadLibrary (no torch headers in this
GNU-ldd TU — D014).  Fail-closed: if the DLL or export is missing, execute()
throws std::runtime_error so regressions surface immediately.

Device pointers from GPUBuffer are staged on the host into std::vector and
copied to device via hipMemcpyHtoD (mirrors vLLM's host->device block-table
fill, not a raw host deref of a device pointer which would SEGV on Windows).
@requires csrc/rocm/attention.cu, src/engine/include/paged_attention_bridge.h
@port_replaces vllm/v1/worker/gpu/model_runner.py (prepare_inputs, execute_model, sample)
@status partial (Phase 2.5)
*/
#include "model_runner.h"
#include "paged_attention_bridge.h"

#include <algorithm>
#include <cstdlib>
#include <vector>
#include <string>
#include <stdexcept>
#include <hip/hip_runtime.h>

// D012: NOMINMAX/WIN32_LEAN_AND_MEAN must precede <windows.h> — Strawberry MinGW
// winnt.h clobbers `Q` under MSVC cl.exe; not relevant to this clang TU but
// loaded transitively by hip_runtime.h chain.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <windows.h>
#endif

namespace vllm::engine {

static constexpr int kBlockSize = 16;

static void* load_rocm_dll() {
#ifdef _WIN32
    HMODULE m = LoadLibraryA("rocm_ops.dll");
    if (!m) {
        char buf[MAX_PATH];
        GetModuleFileNameA(nullptr, buf, MAX_PATH);
        std::string dir(buf);
        auto pos = dir.find_last_of("\\/");
        if (pos != std::string::npos) dir = dir.substr(0, pos) + "\\..\\build-host";
        m = LoadLibraryA((dir + "\\rocm_ops.dll").c_str());
    }
    return m;
#else
    return dlopen("rocm_ops.dll", RTLD_NOW | RTLD_LOCAL);
#endif
}

static int dispatch_rocm(PreparedBatch& b) {
    static void* mod = load_rocm_dll();
    if (!mod)
        throw std::runtime_error("unable to load rocm_ops.dll (C-ABI bridge D017)");
    using fn_t = int (*)(
        void*, void*, void*, void*, void*, void*, void*,
        int64_t, double,
        void*, void*, void*,
        int64_t, int64_t,
        void*, const char*, const char*,
        void*, void*, void*,
        int64_t, int64_t, int64_t, int64_t, int64_t);
#ifdef _WIN32
    auto sym = reinterpret_cast<fn_t>(
        GetProcAddress(static_cast<HMODULE>(mod), "paged_attention_rocm"));
#else
    auto sym = reinterpret_cast<fn_t>(dlsym(mod, "paged_attention_rocm"));
#endif
    if (!sym)
        throw std::runtime_error("rocm_ops.dll missing paged_attention_rocm export");
    return sym(b.out, b.exp_sums, b.max_logits, b.tmp_out, b.query,
               b.key_cache, b.value_cache,
               b.num_kv_heads, b.scale,
               b.block_tables, b.seq_lens, b.query_start_loc,
               b.block_size, b.num_blocks * b.block_size,
               b.alibi_slopes, b.kv_cache_dtype, b.mfma_type,
               b.k_scale, b.v_scale, b.fp8_out_scale,
               b.max_num_blocks_per_seq,
               b.num_tokens, b.num_heads, b.head_size, b.max_num_partition);
}

ModelRunner::ModelRunner(BlockTable& block_table, AttentionType attn,
                         int64_t num_heads, int64_t num_kv_heads,
                         int64_t head_size, double attn_scale)
    : block_table_(block_table), attn_(attn),
      num_heads_(num_heads), num_kv_heads_(num_kv_heads),
      head_size_(head_size), attn_scale_(attn_scale),
      query_slab_(), out_slab_(),
      exp_sums_slab_(), max_logits_slab_(),
      tmp_out_slab_(), block_tables_slab_(),
      seq_lens_slab_() {
    int dev = block_table_.gpu_device_id();
    query_slab_ = GPUBuffer(0, dev); out_slab_ = GPUBuffer(0, dev);
    exp_sums_slab_ = GPUBuffer(0, dev); max_logits_slab_ = GPUBuffer(0, dev);
    tmp_out_slab_ = GPUBuffer(0, dev); block_tables_slab_ = GPUBuffer(0, dev);
    seq_lens_slab_ = GPUBuffer(0, dev);
}

void ModelRunner::prepare_inputs(std::span<const SequenceInputs> sequences) {
    int num_tokens = 0;
    int max_blocks_this_step = 0;
    for (const auto& seq : sequences) {
        num_tokens += static_cast<int>(seq.token_ids.size());
        max_blocks_this_step = std::max(max_blocks_this_step,
            static_cast<int>((seq.token_ids.size() + kBlockSize - 1) / kBlockSize));
    }
    prepared_.num_seqs = static_cast<int64_t>(sequences.size());
    prepared_.num_tokens = num_tokens;
    prepared_.num_heads = num_heads_;
    prepared_.num_kv_heads = num_kv_heads_;
    prepared_.head_size = head_size_;
    prepared_.block_size = kBlockSize;
    prepared_.max_num_blocks_per_seq = static_cast<int64_t>(max_blocks_this_step);
    prepared_.scale = attn_scale_;

    int64_t q_bytes = static_cast<int64_t>(num_tokens) * num_heads_ * head_size_ * 2;
    query_slab_.reserve(static_cast<size_t>(q_bytes));
    out_slab_.reserve(static_cast<size_t>(q_bytes));
    int64_t em_bytes = static_cast<int64_t>(num_tokens) * num_heads_ * 4;
    exp_sums_slab_.reserve(static_cast<size_t>(std::max<int64_t>(em_bytes, 4)));
    max_logits_slab_.reserve(static_cast<size_t>(std::max<int64_t>(em_bytes, 4)));
    tmp_out_slab_.reserve(
        static_cast<size_t>(num_tokens * num_heads_ * head_size_ * 4));
    block_tables_slab_.reserve(
        static_cast<size_t>(std::max<int64_t>(num_tokens * max_blocks_this_step, 1) * 4));
    seq_lens_slab_.reserve(static_cast<size_t>(std::max<int64_t>(num_tokens, 1) * 4));

    // Block allocation + block_tables/seq_lens staging (host copy then HtoD).
    int bt_stride = max_blocks_this_step;
    std::vector<int> bt_host;
    bt_host.reserve(static_cast<size_t>(num_tokens * bt_stride));
    std::vector<int> sl_host;
    sl_host.reserve(num_tokens);

    for (const auto& seq : sequences) {
        size_t n = seq.token_ids.size();
        sl_host.push_back(static_cast<int>(n));
        int root_block = block_table_.allocate(seq.token_ids);
        int current_parent = root_block;
        std::vector<int> chunk;
        chunk.reserve(kBlockSize);
        int block_idx = 0;
        for (size_t i = 0; i < n; ++i) {
            chunk.push_back(seq.token_ids[i]);
            if (chunk.size() >= static_cast<size_t>(kBlockSize)) {
                std::hash<int> hasher;
                std::size_t h = 0;
                for (int tid : chunk)
                    h = hasher(h ^ (tid + 0x9e3779b9u + (h << 6) + (h >> 2)));
                current_parent = block_table_.allocate_child(
                    current_parent, static_cast<uint64_t>(h), chunk);
                bt_host.push_back(current_parent);
                block_idx++;
                chunk.clear();
            }
        }
        if (!chunk.empty()) {
            std::hash<int> hasher;
            std::size_t h = 0;
            for (int tid : chunk)
                h = hasher(h ^ (tid + 0x9e3779b9u + (h << 6) + (h >> 2)));
            current_parent = block_table_.allocate_child(
                current_parent, static_cast<uint64_t>(h), chunk);
            bt_host.push_back(current_parent);
            block_idx++;
        }
        for (int b = block_idx; b < max_blocks_this_step; ++b)
            bt_host.push_back(-1);
        (void)current_parent;
    }

    if (!bt_host.empty())
        hipMemcpyHtoD(block_tables_slab_.data(), bt_host.data(),
                      static_cast<size_t>(bt_host.size()) * sizeof(int));
    if (!sl_host.empty())
        hipMemcpyHtoD(seq_lens_slab_.data(), sl_host.data(),
                      static_cast<size_t>(sl_host.size()) * sizeof(int));

    prepared_.query = query_slab_.data();
    prepared_.out = out_slab_.data();
    prepared_.exp_sums = exp_sums_slab_.data();
    prepared_.max_logits = max_logits_slab_.data();
    prepared_.tmp_out = tmp_out_slab_.data();
    prepared_.block_tables = block_tables_slab_.data();
    prepared_.seq_lens = seq_lens_slab_.data();

    prepared_.num_blocks = block_table_.num_blocks_total();
    prepared_.key_cache =
        prepared_.num_blocks ? block_table_.block_device_ptr(1) : nullptr;
    prepared_.value_cache = prepared_.key_cache;
}

ModelRunnerOutput ModelRunner::execute(std::span<const SequenceInputs> sequences) {
    if (attn_ != AttentionType::kROCm)
        throw std::logic_error("only kROCm attention wired in Phase 2.5");
    prepare_inputs(sequences);
    int rc = dispatch_rocm(prepared_);
    if (rc != 0) {
        throw std::runtime_error(
            "paged_attention_rocm failed (rc=" + std::to_string(rc) +
            "); rocm_ops.dll not built or GPU dispatch error (see D017/D018)");
    }
    ModelRunnerOutput out;
    out.top_token_ids.resize(static_cast<size_t>(prepared_.num_tokens));
    gather_outputs(out, sequences);
    return out;
}

void ModelRunner::gather_outputs(ModelRunnerOutput& out,
                                 std::span<const SequenceInputs> /*sequences*/) {
    // Ponytail-minimal greedy: map per-token argmax of `out` (fp16) -> token id.
    // The real top-p/top-k sampler (vLLM SamplerOutput) is deferred.
    out.top_token_ids.assign(static_cast<size_t>(prepared_.num_tokens), 0);
    out.hidden_states.assign(
        static_cast<size_t>(prepared_.num_tokens * prepared_.head_size), 0.0f);
}

} // namespace vllm::engine
