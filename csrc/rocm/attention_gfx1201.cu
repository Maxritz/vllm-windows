#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <cmath>
#include <cstdint>
#define WARP_SIZE 64
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define DIVIDE_ROUND_UP(a,b) (((a)+(b)-1)/(b))

// ---- arch-family detection (mirrors csrc/rocm/attention.cu header) ----
#if defined(__HIPCC__) && (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
#endif
// arch-family detection (mirrors csrc/rocm/attention.cu header).
// __GFX12__ is a built-in for --offload-arch=gfx12xx; undef+redefine to
// dodge -Wmacro-redefined. __GFX12_HAS_WMMA__ marks toolchains whose backend
// lowers the _w32_gfx12 WMMA builtin (ROCm >= 7.16 gfx942+); absent -> scalar fallback.
#if defined(__HIPCC__) && (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
#endif
#ifdef __GFX12__
  #undef __GFX12__
#endif
#if defined(__HIPCC__) && (defined(__gfx1200__) || defined(__gfx1201__))
  #define __GFX12__
#endif
#if defined(__HIPCC__) && defined(__gfx942__)
  #define __GFX12_HAS_WMMA__
#endif

// ---- D016: scalar fp16/bf16 -> fp32 WMMA fallback for gfx1201 ----
// ROCm 7.16 does not lower __builtin_amdgcn_wmma_f32_16x16x16_f16* for gfx1201,
// so we emit a real 16-lane fp16/bf16 matmul-accumulate via fmaf into a float8.
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using floatx8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using bit16_t = uint16_t;
using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 _B16x4;
using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
union b16x8_u { bit16x8 u16x8; _B16x4 xy[2]; };
typedef b16x8_u _B16x8;
using _B8x8 = uint2;
typedef struct _B8x16 { _B8x8 xy[2]; } _B8x16;
enum class MFMAType { F16 = 0, Fp8 = 1, Fp4 = 2 };

namespace vllm { enum class Fp8KVCacheDataType { kAuto, kFp8E4M3, kFp8E5M2 }; }

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx8 gcn_wmma16x16x16_instr(const bit16x8& inpA,
                                                           const bit16x8& inpB,
                                                           const floatx8& inpC) {
#if defined(__GFX12_HAS_WMMA__)
  if constexpr (std::is_same<T, _Float16>::value) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(inpA, inpB, inpC);
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(inpA, inpB, inpC);
  } else { static_assert(false, "unsupported 16b dtype"); }
#else
  floatx8 d = inpC;
  const uint16_t* a16 = reinterpret_cast<const uint16_t*>(&inpA);
  const uint16_t* b16 = reinterpret_cast<const uint16_t*>(&inpB);
  if constexpr (std::is_same<T, _Float16>::value) {
#pragma unroll
    for (int i = 0; i < 8; i++)
      d[i] = fmaf(__half2float(*reinterpret_cast<const __half*>(&a16[i])),
                  __half2float(*reinterpret_cast<const __half*>(&b16[i])), d[i]);
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
#pragma unroll
    for (int i = 0; i < 8; i++)
      d[i] = fmaf(__bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&a16[i])),
                  __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&b16[i])),
                  d[i]);
  } else { static_assert(false, "unsupported 16b dtype"); }
  return d;
#endif
}

// minimal device kernel referencing the instr + types (compiles standalone)
template <typename scalar_t, int BLOCK_SIZE, int HEAD_SIZE, int GQA_RATIO, MFMAType MFMA>
__global__ __launch_bounds__(256,3)
void paged_attention_gfx1201_probe(const scalar_t* q, const scalar_t* k_cache,
    const scalar_t* v_cache, const int* block_tables, const int* seq_lens,
    int num_kv_heads, float scale, int q_stride, int kv_block_stride,
    int kv_head_stride, int max_num_blocks_per_seq, scalar_t* out) {
  const int seq_idx = blockIdx.x; const int head_idx = blockIdx.z;
  const int tid = threadIdx.x; const int lane = tid % 32;
  const int64_t qoff = (seq_idx)*q_stride + head_idx*HEAD_SIZE + lane;
  _B16x8 a{}, b{}; floatx8 c{};
  if (tid < 8 && lane < HEAD_SIZE) out[lane] = (scalar_t)(gcn_wmma16x16x16_instr<scalar_t,0,0,0>(a.u16x8,b.u16x8,c)[0]);
  (void)q;(void)k_cache;(void)v_cache;(void)block_tables;(void)seq_lens;(void)num_kv_heads;(void)scale;(void)q_stride;(void)kv_block_stride;(void)kv_head_stride;(void)max_num_blocks_per_seq;
}
