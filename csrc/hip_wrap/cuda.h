// D013/D017: cuda.h (CUDA driver API) shim for cl.exe compiling torch headers.
// Bridge TU never invokes driver APIs — only needs the type decls torch headers
// reference at compile time.  All symbols resolve to no-ops/empty so cl.exe
// never hits the GNU-attribute-laden HIP headers (D013).
#pragma once
#include <cstdint>
typedef int CUresult;
typedef void* CUcontext;
typedef void* CUstream;
typedef void* CUevent;
typedef void* CUdevice;
typedef unsigned long long CUdeviceptr;
typedef void* CUfunction;
typedef void* CUevent;
typedef void* CUgraph;
typedef void* CUgraphExec;
typedef void* CUmemGenericAllocationHandle;
struct CUipcMemHandle;
inline CUresult cuCtxGetCurrent(CUcontext*) { return CUresult(0); }
inline CUresult cuMemAlloc(CUdeviceptr*, unsigned long long, unsigned long long) { return CUresult(0); }
#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CU_MEM_GENERIC_ALLOC_FLAG_NONE 0
#define CU_POINTER_ATTRIBUTE_CONTEXT 1
#define CU_POINTER_ATTRIBUTE_MEMORY_RANGE_SIZE 2
#define CU_POINTER_ATTRIBUTE_IS_MANAGED 3
#define CU_POINTER_ATTRIBUTE_IS_CUMEM 4
#define CU_POINTER_ATTRIBUTE_DEVICE_ 5
#define CU_POINTER_ATTRIBUTE_IS_LEGACY_HSMA 6
#define CU_POINTER_ATTRIBUTE_ALLOW_ATOMICS 7
#define CU_POINTER_ATTRIBUTE_BRAND 8
#define CU_POINTER_ATTRIBUTE_MMU_FORMAT 9
#define CU_POINTER_ATTRIBUTE_MMU_ADDR_SIZE 10
#define CU_POINTER_ATTRIBUTE_ACCESS_FLAG 11
#define CU_POINTER_ATTRIBUTE_IS_DEFAULT 12
#define CUDA_VERSION 12000
#define __CU_DEVICE_ATTR_MAX_THREADS_PER_BLOCK 1
#define __CU_DEVICE_ATTR_MAX_THREADS_PER_MULTIPROCESSOR 2
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK 1
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 2
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR 3
#define CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK 4
#define CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY 5
#define CU_DEVICE_ATTRIBUTE_LOCAL_MEMORY_PER_BLOCK 6
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK 7
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR 8
#define CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK 9
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 10
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 11
#define CU_DEVICE_ATTRIBUTE_COMPUTE_MODE 12
