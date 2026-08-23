// cuda.h — D017 torch-on-ROCm compat (CUDA driver API surface).
// Shares the cuda<->hip handle aliases/typedefs from cuda_runtime.h so the
// two shims stay consistent for torch's c10/cuda/* headers.
#pragma once
#include "cuda_runtime.h"

// driver-API-style no-op inlines (torch may reference in headers):
typedef int CUresult;
inline CUresult cuCtxGetCurrent(CUcontext*) { return CUDA_SUCCESS; }
inline CUresult cuMemAlloc(CUdeviceptr*, unsigned long long, unsigned long long) { return CUDA_SUCCESS; }
inline CUresult cuCtxSetCurrent(CUcontext) { return CUDA_SUCCESS; }
inline CUresult cuCtxGetDevice(CUdevice*) { return CUDA_SUCCESS; }
inline CUresult cuMemGetAddressRange(CUdeviceptr*, unsigned long long*, CUdeviceptr) { return CUDA_SUCCESS; }
