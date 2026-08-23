/**
@file gpu/windows_gpu
@brief Native Windows GPU enumeration replacing amdsmi (D007).
@details
vLLM's Linux platform layer (`vllm/platforms/rocm.py`) queries GPU info via the
`amdsmi` Python binding — device count, ASN name, GCN arch, total VRAM, NUMA node, UUID.
amdsmi is **absent from the ROCm 7.16 Windows preview** (no `amdsmi.dll`).

WinRocM-llm is pure C++ and does NOT invoke the Python amdsmi path at all. Instead it
uses HIP's portable `hipGetDeviceCount`/`hipGetDeviceProperties`/`hipMemGetInfo`, which
are present in `amdhip64.lib`. These expose:
  - device count + PCI bus/device/function
  - total + used VRAM
  - GCN arch string (`gcnArchName`, e.g. "gfx1201")
  - device name (e.g. "AMD Radeon RX 9070 XT")
  - 16-byte device UUID (hipDeviceProp_t.uuid)

@windows_issue
- amdsmi is Linux-only; on Windows use HIP runtime + (optionally) Win32 WMI for
  cross-vendor adapter enumeration without initializing a HIP context.
- NUMA node: Windows consumer GPUs expose no real NUMA affinity (treated as 0).
  vLLM's `amdsmi_topo_get_numa_node_number` call is replaced with a constant `0`.
- UUID: `hipDeviceProp_t.uuid` (struct with .bytes[16]) is available on ROCm 7.16;
  legacy 10.x used a hex-string form — handle both by reading `.bytes`.

@port_replaces vllm/platforms/rocm.py:_query_total_memory_from_amdsmi /
  _query_gcn_arch_from_amdsmi / with_amdsmi_context / device_uuid / numa_node
@status done (Phase 0.3 stub validated with hipGetDeviceProperties on RX 9070 XT)
@see knowledge/hip/toolchain.dox.md
*/
