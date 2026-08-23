// cuda_runtime.h — D017/D018 torch-on-ROCm compat shim.
// Delegates to the real ROCm HIP runtime headers, then aliases the
// CUDA runtime namespace onto HIP so torch's c10/cuda/* headers resolve
// cudaStreamCaptureStatus / cudaMemcpyKind / cudaGraph_t etc. to the
// symbols HIP 7.16 actually defines.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

typedef hipError_t cudaError_t;
typedef hipStream_t cudaStream_t;
typedef hipEvent_t cudaEvent_t;
typedef hipEvent_t cudaEvent;
typedef hipIpcMemHandle_t cudaIpcMemHandle_t;

// enums (real HIP defines the hip* members):
typedef hipStreamCaptureStatus cudaStreamCaptureStatus;
typedef hipMemcpyKind cudaMemcpyKind;
typedef hipStreamCaptureMode cudaStreamCaptureMode;
#define cudaStreamCaptureStatusNone        hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive      hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusPreexisting hipStreamCaptureStatusPreexisting
#define cudaStreamCaptureStatusInvalidated hipStreamCaptureStatusInvalidated

#define cudaMemcpyHostToHost        hipMemcpyHostToHost
#define cudaMemcpyHostToDevice      hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice    hipMemcpyDeviceToDevice
#define cudaMemcpyDefault           hipMemcpyDefault

// stream flags:
#define cudaStreamDefault           hipStreamDefault
#define cudaStreamNonBlocking       hipStreamNonBlocking
#define cudaEventBlockingSync       hipEventBlockingSync
#define cudaEventDisableTiming      hipEventDisableTiming
#define cudaEventDisableHostBusy    hipEventDisableHostBusy

// stream / graph / capture API symbols:
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent   hipStreamWaitEvent
#define cudaStreamQuery       hipStreamQuery
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamGetPriority      hipStreamGetPriority
#define cudaStreamGetPriorityRange hipStreamGetPriorityRange
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode

#define cudaStreamCreate       hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy      hipStreamDestroy
#define cudaStreamDestroyAsync hipStreamDestroyAsync

#define cudaEventCreate        hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy       hipEventDestroy
#define cudaEventRecord        hipEventRecord
#define cudaEventQuery         hipEventQuery
#define cudaEventSynchronize   hipEventSynchronize

#define cudaMemcpy             hipMemcpy
#define cudaMemcpyAsync        hipMemcpyAsync
#define cudaMemcpy2D           hipMemcpy2D
#define cudaMemcpy2DAsync      hipMemcpy2DAsync
#define cudaMemcpy3D           hipMemcpy3D
#define cudaMemcpy3DAsync      hipMemcpy3DAsync

#define cudaGetLastError       hipGetLastError
#define cudaPeekAtLastError    hipPeekAtLastError
#define cudaGetErrorString     hipGetErrorString
#define cudaGetErrorName       hipGetErrorName

#define cudaDeviceSynchronize  hipDeviceSynchronize
#define cudaGetDevice          hipGetDevice
#define cudaGetDeviceCount     hipGetDeviceCount
#define cudaSetDevice          hipSetDevice
#define cudaSetDeviceFlags     hipSetDeviceFlags
#define cudaDeviceReset        hipDeviceReset
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceGetLimit     hipDeviceGetLimit

#define cudaStreamCaptureMode    hipStreamCaptureMode
#define cudaStreamCaptureModeNone         hipStreamCaptureModeNone
#define cudaStreamCaptureModeForeign      hipStreamCaptureModeForeign
#define cudaStreamCaptureModeRelaxed      hipStreamCaptureModeRelaxed

// graph types:
typedef hipGraph_t cudaGraph_t;
typedef hipGraphNode_t cudaGraphNode_t;
typedef hipGraphExec_t cudaGraphExec_t;
typedef hipHostFn_t cudaHostFn_t;
#define cudaGraphCreate               hipGraphCreate
#define cudaGraphInstantiate           hipGraphInstantiate
#define cudaGraphLaunch                hipGraphLaunch
#define cudaGraphDestroy               hipGraphDestroy
#define cudaUserObjectNoDestructorSync hipUserObjectNoDestructorSync
#define cudaGraphUserObjectMove        hipGraphUserObjectMove
#define cudaGraphAddChild              hipGraphAddChild
#define cudaGraphInstantiateWithFlags  hipGraphInstantiateWithFlags
#define cudaGraphInstantiateWithParams hipGraphInstantiateWithParams

// error constants:
#define cudaSuccess                 hipSuccess
#define cudaErrorInvalidValue       hipErrorInvalidValue
#define cudaErrorInvalidResourceHandle hipErrorInvalidResourceHandle
#define cudaErrorUnhandledHost    hipErrorUnhandledHost
#define cudaErrorNotInitialized   hipErrorNotInitialized

// cudaIpc:
inline cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t*, void*) { return hipSuccess; }
inline cudaError_t cudaIpcOpenMemHandle(void**, int, cudaIpcMemHandle_t, unsigned int) { return hipSuccess; }
inline cudaError_t cudaIpcCloseMemHandle(void*) { return hipSuccess; }
inline cudaError_t cudaIpcGetEventHandle(void*, void*) { return hipSuccess; }
inline cudaError_t cudaIpcOpenEventHandle(void*, void*) { return hipSuccess; }

struct cudaPointerAttributes { unsigned int isManaged : 1; };

// opaque handle aliases shared with cuda.h:
typedef void* CUcontext;
typedef void* CUstream;
typedef void* CUevent;
typedef void* CUdevice;
typedef unsigned long long CUdeviceptr;
typedef void* CUfunction;
typedef void* CUgraph;
typedef void* CUgraphExec;
typedef void* CUmemGenericAllocationHandle;
struct CUipcMemHandle;
#define CUDA_SUCCESS                0
#define CUDA_VERSION                12000
#define CU_MEM_GENERIC_ALLOC_FLAG_NONE 0
#define CU_POINTER_ATTRIBUTE_CONTEXT 1
#define CU_POINTER_ATTRIBUTE_MEMORY_RANGE_SIZE 2
#define CU_POINTER_ATTRIBUTE_IS_MANAGED 3
#define CU_POINTER_ATTRIBUTE_IS_CUMEM 4
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
#