// gpu_windows.cpp — Windows-native GPU enumeration replacing amdsmi.
// hipGetDeviceProperties gives VRAM + name + UUID + compute major/minor. PCI coords
// via hipDeviceGetAttribute. GCN arch derived from compute major (Navi 3=11 RDNA3,
// 12=RDNA4) since ROCm 7.16 removed gcnArchName from hipDeviceProp_t.
#include "hip_compat/gpu_windows.h"
#include <hip/hip_runtime.h>

namespace vllm::gpu {

static int attr(hipDevice_t d, hipDeviceAttribute_t a) {
    int v = 0;
    if (hipDeviceGetAttribute(&v, a, d) != hipSuccess) return 0;
    return v;
}

std::vector<GpuInfo> enumerate_gpus() {
    std::vector<GpuInfo> gpus;
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) return gpus;
    gpus.reserve(count);
    for (int i = 0; i < count; ++i) {
        hipDevice_t dev;
        hipDeviceProp_t prop{};
        if (hipDeviceGet(&dev, i) != hipSuccess) continue;
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) continue;
        size_t free = 0, total = 0;
        hipMemGetInfo(&free, &total);
        GpuInfo g{};
        g.device_id = i;
        g.pci_bus = attr(dev, hipDeviceAttributePciBusId);
        g.pci_device = attr(dev, hipDeviceAttributePciDeviceId);
        g.pci_function = 0;
        g.vram_bytes = total;
        g.vram_used = total - free;
        // GCN arch from compute major (Navi 3=11 RDNA3, 12=RDNA4); matches gfx1201.
        if (prop.major == 12) g.gcn_arch = "gfx1201";
        else if (prop.major == 11) g.gcn_arch = "gfx1100";
        else g.gcn_arch = "gfx" + std::to_string(prop.major);
        g.device_name = prop.name;
        // hipUUID: 16-byte bytes field.
        g.uuid.assign(reinterpret_cast<const char*>(prop.uuid.bytes), sizeof(prop.uuid.bytes));
        gpus.push_back(std::move(g));
    }
    return gpus;
}

bool has_hip_gpu() {
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

} // namespace vllm::gpu
