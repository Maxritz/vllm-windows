// gpu_windows.h — native Windows GPU enumeration (replaces amdsmi on Windows).
// Uses HIP (hipGetDeviceProperties) for VRAM/device count, Win32 WMI for UUID/name
// fallback where HIP lacks them. Minimal surface: replaces vllm/platforms/rocm.py
// amdsmi_* calls with a C++ query used by the master control plane.
// See knowledge/gpu/windows_gpu.dox.md
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vllm::gpu {

struct GpuInfo {
    int device_id;            // 0-based logical index
    int pci_bus;              // PCI bus (from hipDeviceProp_t)
    int pci_device;
    int pci_function;
    uint64_t vram_bytes;      // total VRAM (bytes)
    uint64_t vram_used;       // used VRAM (bytes)
    std::string gcn_arch;     // e.g. "gfx1201" (parsed from hip runtime name)
    std::string device_name;  // human-readable
    std::string uuid;         // device UUID
};

// Enumerate all AMD/HIP GPUs. Returns empty vector if hipInit fails (no ROCm on host).
// numa_node omitted — Windows treats consumer GPU NUMA as 0 (no affinity for RDMA).
std::vector<GpuInfo> enumerate_gpus();

// True if a HIP-capable AMD GPU is present and the runtime initialized.
bool has_hip_gpu();

} // namespace vllm::gpu
