// D013/D017: cuda_runtime_api.h stub for cl.exe compiling torch headers on ROCm.
// Host-side bridge never calls cuda runtime APIs — device ops run via hipcc
// kernels in the same DLL.  Declares just the prototypes/signatures torch
// headers instantiate so cl.exe links against the shim dir, not HIP's GNU-attr
// headers.  Actual symbols resolve to no-ops (no runtime use here).
#pragma once
#include "cuda_runtime.h"
#include "cuda.h"
#define __CUDA_RUNTIME_API_H__ 1
#define CUDA_API_VERSION 12000
#define cudaStreamDefault 0x0
#define cudaStreamNonBlocking 0x1
#define cudaStreamLegacy 0x2
#define cudaStreamPerThread 0x3
#define cudaIpcMemHandleSize 128
#define cudaIpcAddMemHandle(ptr, handle) (cudaError_t(0))
#define cudaIpcGetMemHandle(handle, dev) (cudaError_t(0))
#define cudaIpcOpenMemHandle(dev, handle, flags) (cudaError_t(0))
#define cudaIpcCloseMemHandle(handle) (cudaError_t(0))
#define cudaIpcGetEventHandle(handle, event) (cudaError_t(0))
#define cudaIpcOpenEventHandle(event, handle) (cudaError_t(0))
#define cudaMemcpyDefault 0
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaMemcpyDeviceToDevice 3
#define cudaMemcpyPeer 4
#define cudaMemcpy3DArrayToArray 5
inline cudaError_t cudaMemcpy(void*, const void*, size_t, int) { return cudaError_t(0); }
inline cudaError_t cudaMemcpyAsync(void*, const void*, size_t, int, cudaStream_t) { return cudaError_t(0); }
inline cudaError_t cudaMemcpy2D(void*, size_t, const void*, size_t, size_t, int, int) { return cudaError_t(0); }
inline cudaError_t memcpy3D(...) { return cudaError_t(0); }
inline cudaError_t cudaEventCreate(cudaEvent_t*) { return cudaError_t(0); }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned int) { return cudaError_t(0); }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaError_t(0); }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaError_t(0); }
inline cudaError_t cudaStreamCreate(cudaStream_t*, unsigned int) { return cudaError_t(0); }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned int, int) { return cudaError_t(0); }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaError_t(0); }
inline cudaError_t cudaDeviceSynchronize() { return cudaError_t(0); }
inline cudaError_t cudaDeviceGetAttribute(int*, int, int) { return cudaError_t(0); }
inline cudaError_t cudaGetDevice(int*) { return cudaError_t(0); }
inline cudaError_t cudaGetDeviceCount(int*) { return cudaError_t(0); }
inline cudaError_t cudaSetDevice(int) { return cudaError_t(0); }
typedef struct cudaIpcMemHandle_st { uint8_t reserved[128]; } cudaIpcMemHandle_t;
