// hip_malloc_smoke.hip.cc — real HIP runtime smoke: hipMalloc/hipFree/hipMemcpy round-trip.
// No stubs; allocates GPU memory, writes host->device, kernel-less memcpy back, frees.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("hipSetDevice(0) FAIL\n");
        return 2;
    }
    constexpr size_t N = 1024;
    float* dev = nullptr;
    hipError_t err = hipMalloc(&dev, N * sizeof(float));
    if (err != hipSuccess || dev == nullptr) {
        std::printf("hipMalloc FAIL: %s\n", hipGetErrorString(err));
        return 2;
    }
    float host[N];
    for (size_t i = 0; i < N; ++i) host[i] = static_cast<float>(i) * 0.5f;
    if (hipMemcpy(dev, host, N * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("hipMemcpy H2D FAIL\n");
        hipFree(dev);
        return 2;
    }
    float back[N] = {};
    if (hipMemcpy(back, dev, N * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("hipMemcpy D2H FAIL\n");
        hipFree(dev);
        return 2;
    }
    size_t mism = 0;
    for (size_t i = 0; i < N; ++i) if (back[i] != host[i]) ++mism;
    std::printf("hipMalloc round-trip: N=%zu mismatch=%zu\n", N, mism);
    hipFree(dev);
    // Memory info sanity.
    size_t free = 0, total = 0;
    if (hipMemGetInfo(&free, &total) == hipSuccess) {
        std::printf("VRAM total=%llu MB free=%llu MB\n",
            (unsigned long long)(total >> 20), (unsigned long long)(free >> 20));
    }
    std::printf(mism == 0 ? "hipMalloc PASS\n" : "hipMalloc FAIL mismatch\n");
    return mism == 0 ? 0 : 1;
}
