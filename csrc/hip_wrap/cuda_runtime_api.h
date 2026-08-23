// cuda_runtime_api.h — D017 torch-on-ROCm compat.
// Delegates to the real ROCm HIP runtime headers (which provide all
// cudaStreamCaptureStatus/cudaMemcpyKind/cudaError_t enum members torch
// c10/cuda/* needs), then aliases the cuda->hip namespace.
#pragma once
#include "cuda_runtime.h"
