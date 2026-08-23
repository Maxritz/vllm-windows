// D013/D017: cuda_runtime.h shim for cl.exe compiling torch headers on Windows-ROCm.
// The bridge TU does NOT call any cuda/hip runtime host API — it only forwards
// device pointers to the C-ABI dispatch.  So a minimal stub declaring the
// handful of cuda* types torch's headers reference at compile time suffices,
// avoiding the GNU-attribute-heavy am_hip_vector_types.h that cl.exe can't
// parse (D013).  Device-side kernels stay in hipcc-compiled .obj.
#pragma once
#include <cstdint>
#include <cstddef>
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef void* cudaEvent;
typedef void* CUcontext;
typedef void* CUstream;
typedef void* CUevent;
typedef unsigned long long cudaIpcMemHandle_t;
template <typename T> struct cudaIpcMemHandle { T reserved[16]; };
inline cudaError_t cudaGetLastError() { return cudaError_t(0); }
inline const char* cudaGetErrorString(int) { return "cuda_runtime shim (no-op)"; }
struct cudaPointerAttributes { unsigned int isManaged : 1; };
