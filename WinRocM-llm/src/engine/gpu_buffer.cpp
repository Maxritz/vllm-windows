#include "gpu_buffer.h"

namespace vllm::engine {

GPUBuffer::GPUBuffer(size_t bytes, int device_id) : bytes_(bytes), device_id_(device_id) {
    if (bytes_ > 0) {
        if (hipSetDevice(device_id_) != hipSuccess) {
            throw std::runtime_error("GPUBuffer: hipSetDevice failed");
        }
        if (hipMalloc(&ptr_, bytes_) != hipSuccess || ptr_ == nullptr) {
            throw std::runtime_error("GPUBuffer: hipMalloc failed");
        }
    }
}

GPUBuffer::~GPUBuffer() { deallocate(); }

GPUBuffer::GPUBuffer(GPUBuffer&& other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_), device_id_(other.device_id_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
}

GPUBuffer& GPUBuffer::operator=(GPUBuffer&& other) noexcept {
    if (this != &other) {
        deallocate();
        ptr_ = other.ptr_;
        bytes_ = other.bytes_;
        device_id_ = other.device_id_;
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

void GPUBuffer::resize(size_t new_bytes) {
    if (new_bytes == bytes_ && ptr_ != nullptr) return;
    deallocate();
    bytes_ = new_bytes;
    if (bytes_ > 0) {
        if (hipSetDevice(device_id_) != hipSuccess) {
            throw std::runtime_error("GPUBuffer::resize: hipSetDevice failed");
        }
        if (hipMalloc(&ptr_, bytes_) != hipSuccess || ptr_ == nullptr) {
            throw std::runtime_error("GPUBuffer::resize: hipMalloc failed");
        }
    }
}

void GPUBuffer::reserve(size_t min_bytes) {
    if (bytes_ >= min_bytes) return;
    resize(min_bytes);
}

void GPUBuffer::deallocate() {
    if (ptr_ != nullptr) {
        hipSetDevice(device_id_);
        hipFree(ptr_);
        ptr_ = nullptr;
    }
    bytes_ = 0;
}

} // namespace vllm::engine
