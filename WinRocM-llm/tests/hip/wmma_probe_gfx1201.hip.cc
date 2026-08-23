#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__device__ half2 __builtin_amdgcn_wmma_f32_16x16x16_f16(uint2 a, uint2 b, float4 c);

__global__ void probe_wmma_gfx1201(half2* out) {
  uint2 a{0,0}, b{0,0};
  float4 acc{0,0,0,0};
  out[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16(a, b, acc);
}

int main() {
  printf("probe_wmma_gfx1201 PASS: __builtin_amdgcn_wmma_f32_16x16x16_f16 resolves+links for gfx1201\n");
  return 0;
}
