// phase0_device_smoke.hip.cc — minimal HIP device validation (no cmath-dependent shfl).
// Toolchain-validated by smoke_hip.cmd. Compiled for RX 9070 XT (gfx1100 target).
// NOTE: gfx1031 (RDNA2) was avoided in favor of gfx1100 to match runtime RDNA3/RDNA4 dispatch.
#include <hip/hip_runtime.h>

__global__ void saxpy(const float* x, float* y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    hipSetDevice(0);
    constexpr int N = 4;
    float x[N] = {1,2,3,4}, y[N] = {0,0,0,0};
    float *dx, *dy;
    if (hipMalloc(&dx, sizeof(x)) != hipSuccess) { printf("hipMalloc failed dx\n"); return 1; }
    if (hipMalloc(&dy, sizeof(y)) != hipSuccess) { printf("hipMalloc failed dy\n"); return 1; }
    hipMemcpy(dx, x, sizeof(x), hipMemcpyHostToDevice);
    hipMemcpy(dy, y, sizeof(y), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(saxpy, dim3(1), dim3(N), 0, 0, dx, dy, 2.0f, N);
    hipDeviceSynchronize();
    hipMemcpy(y, dy, sizeof(y), hipMemcpyDeviceToHost);
    int ok = (y[0]==2.f && y[3]==8.f);
    printf("saxpy y[0]=%.0f y[3]=%.0f %s\n", y[0], y[3], ok ? "PASS" : "FAIL");
    hipFree(dx); hipFree(dy);
    return ok ? 0 : 1;
}
