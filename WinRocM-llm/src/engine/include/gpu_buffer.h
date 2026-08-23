#pragma once
/**
@file engine/gpu_buffer
@brief RAII GPU memory buffer backed by real hipMalloc/hipFree (no stubs).
@details
Owns a slice of device memory with RAII semantics. Move-only. Used by the
KV-cache block allocator (BlockTable) and the tensor transfer path. On Windows
there is no amdsmi handle to import; buffers are allocated through the HIP
runtime directly (hipMalloc). All error paths throw — never return a usable
null handle on allocation failure (fail-closed).
@requires D012 toolchain (GNU triple + _MSC_VER shim) for HIP headers.
@status done
@see knowledge/engine/core.dox.md
*/
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace vllm::engine {

class GPUBuffer {
public:
    GPUBuffer() noexcept = default;
    explicit GPUBuffer(size_t bytes, int device_id = 0);
    ~GPUBuffer();
    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;
    GPUBuffer(GPUBuffer&&) noexcept;
    GPUBuffer& operator=(GPUBuffer&&) noexcept;

    void* device_ptr() noexcept { return ptr_; }
    const void* device_ptr() const noexcept { return ptr_; }
    void* data() noexcept { return ptr_; }
    const void* data() const noexcept { return ptr_; }
    size_t bytes() const noexcept { return bytes_; }
    int device_id() const noexcept { return device_id_; }
    bool valid() const noexcept { return ptr_ != nullptr; }

    void resize(size_t new_bytes);
    /// Ensure the buffer holds at least `bytes` bytes (grows if smaller,
    /// idempotent if already large enough).  Throws on hipMalloc failure.
    void reserve(size_t bytes);
    void deallocate();

private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
    int device_id_ = 0;
};

} // namespace vllm::engine
