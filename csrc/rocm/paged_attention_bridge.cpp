// D017: C-ABI trampoline from the GNU-ldd host exe into the torch-registered
// paged_attention op (compiled in rocm_ops.dll via MSVC). model_runner.cpp
// calls paged_attention_rocm() with raw device pointers + layout metadata
// (no torch headers leak into the engine build — D014 cross-ABI boundary).
//
// Compiled by cl.exe so it can ingest the MSVC intrinsics torch pulls
// (<intrin.h>) that break hipcc's clang host pass (D013).  Only minimal
// at::/c10 headers for tensor wrapping + op dispatch are included.
#include <ATen/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/optional.h>
#include <cstring>
#include <string>
#include <stdexcept>
#include "rocm/ops.h"   // declares void paged_attention(...)

namespace {
at::Tensor wrap(void* p, at::ScalarType st, int64_t n) {
    auto opts = at::TensorOptions(st).device(c10::kHIP);
    return at::from_blob(p, c10::IntArrayRef({n}), c10::IntArrayRef({sizeof(void*)==8?0:0}),
                         opts, /*deleter=*/nullptr);
}
at::Tensor wrap2(void* p, at::ScalarType st, int64_t d0, int64_t d1) {
    auto opts = at::TensorOptions(st).device(c10::kHIP);
    int64_t stride1 = 1;
    int64_t stride0 = d1;
    return at::from_blob(p, {d0, d1}, {stride0, stride1}, opts, /*deleter=*/nullptr);
}
}  // namespace

extern "C" {

__declspec(dllexport) int paged_attention_rocm(
    void* out, void* exp_sums, void* max_logits, void* tmp_out,
    void* query, void* key_cache, void* value_cache,
    int64_t num_kv_heads, double scale,
    void* block_tables, void* seq_lens,
    void* query_start_loc, int64_t block_size, int64_t max_seq_len,
    void* alibi_slopes, const char* kv_cache_dtype,
    void* k_scale, void* v_scale, void* fp8_out_scale,
    const char* mfma_type, int64_t max_num_blocks_per_seq,
    int64_t num_tokens, int64_t num_heads, int64_t head_size,
    int64_t max_num_partition) {
  try {
    const at::cuda::OptionalCUDAGuard device_guard(query);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    (void)stream;

    int64_t nh = num_heads;
    int64_t nq = num_tokens;
    int64_t hs = head_size;
    // out: [num_tokens, num_heads, head_size] flattened row-major
    int64_t out_total = nq * nh * hs;
    auto t_out = wrap2(out, at::ScalarType::Half, nq * nh, hs);
    auto t_exp = wrap2(exp_sums, at::ScalarType::Float, nq * nh, max_num_partition);
    auto t_max = wrap2(max_logits, at::ScalarType::Float, nq * nh, max_num_partition);
    auto t_tmp = wrap2(tmp_out, at::ScalarType::Float, nq * nh,
                       max_num_partition * hs);
    auto t_q   = wrap2(query, at::ScalarType::Half, nq * nh, hs);
    int64_t kv_total = num_kv_heads * hs * block_size;
    auto t_k = wrap2(key_cache, at::ScalarType::Half, num_kv_heads, kv_total);
    auto t_v = wrap2(value_cache, at::ScalarType::Half, num_kv_heads, kv_total);
    auto t_bt = wrap2(block_tables, at::ScalarType::Int, nq, max_num_blocks_per_seq);
    auto t_sl = wrap(seq_lens, at::ScalarType::Int, nq);

    std::optional<at::Tensor> opt_qsl = query_start_loc
        ? std::optional<at::Tensor>(wrap2(query_start_loc, at::ScalarType::Int, nq, 1))
        : std::nullopt;
    std::optional<at::Tensor> opt_ali = alibi_slopes
        ? std::optional<at::Tensor>(wrap2(alibi_slopes, at::ScalarType::Float, num_heads, 1))
        : std::nullopt;
    std::optional<at::Tensor> opt_fp8 = fp8_out_scale
        ? std::optional<at::Tensor>(wrap(fp8_out_scale, at::ScalarType::Float, 1))
        : std::nullopt;
    auto t_kscale = (k_scale && k_scale != reinterpret_cast<void*>(&scale))
        ? at::from_blob(k_scale, {1}, at::TensorOptions(at::ScalarType::Float).device(c10::kHIP), nullptr)
        : at::from_blob(const_cast<double*>(&scale), {1}, at::TensorOptions(at::ScalarType::Float).device(c10::kHIP), nullptr);
    auto t_vscale = (v_scale && v_scale != reinterpret_cast<void*>(&scale))
        ? at::from_blob(v_scale, {1}, at::TensorOptions(at::ScalarType::Float).device(c10::kHIP), nullptr)
        : at::from_blob(const_cast<double*>(&scale), {1}, at::TensorOptions(at::ScalarType::Float).device(c10::kHIP), nullptr);
    (void)out_total; (void)nh; (void)nq; (void)hs;

    paged_attention(t_out, t_exp, t_max, t_tmp, t_q, t_k, t_v,
                    num_kv_heads, scale, t_bt, t_sl, opt_qsl, block_size,
                    max_seq_len, opt_ali, std::string(kv_cache_dtype),
                    t_kscale, t_vscale, opt_fp8, std::string(mfma_type));
    return 0;
  } catch (const std::exception& e) {
    return -1;
  }
}

}  // extern "C"
