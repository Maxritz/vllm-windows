#pragma once
/**
@file engine/paged_attention_dll
@brief C-ABI bridge so the GNU-ldd `vllm_engine.exe` host (no torch headers)
can dispatch the ROCm paged-attention op compiled into rocm_ops.dll (MSVC).
@details The torch op `torch.ops.rocm_ops.paged_attention` is C++-ABI and lives
in an MSVC-built DLL; exposing it as `extern "C"` keeps the ABI boundary
portable across the clang/MSVC object split (see decisions.log.md D016/D014).
*/
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Mirrors vllm::Fp8KVCacheDataType used by paged_attention().
typedef enum {
  PA_KV_AUTO = 0,
  PA_KV_FP8_E4M3 = 1,
  PA_KV_FP8_E5M2 = 2
} pa_kv_dtype_t;

typedef enum {
  PA_MFMA_F16 = 0,
  PA_MFMA_FP8 = 1,
  PA_MFMA_FP4 = 2
} pa_mfma_type_t;

// C-ABI entry for the paged attention op. Tensors are passed as raw device
// pointers (float/half = float32/float16, cache = uint8_t fp8). Layouts match
// vllm's `paged_attention` schema exactly (seq-major KV pagetable).
int paged_attention_rocm(
    void* out, void* exp_sums, void* max_logits, void* tmp_out,
    void* query, void* key_cache, void* value_cache,
    int64_t num_kv_heads, double scale,
    void* block_tables, void* seq_lens,
    void* query_start_loc,      // nullable
    int64_t block_size, int64_t max_seq_len,
    void* alibi_slopes,         // nullable
    const char* kv_cache_dtype, // "auto" | "fp8" | "fp8_e4m3"
    void* k_scale, void* v_scale,
    void* fp8_out_scale,        // nullable
    const char* mfma_type,      // "fp16" | "fp8"
    int64_t max_num_blocks_per_seq,
    int64_t num_tokens, int64_t num_heads, int64_t head_size,
    int64_t max_num_partition);

#ifdef __cplusplus
}
#endif
