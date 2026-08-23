// hip_runtime.cpp — host-side HIP dispatch shim stub (Phase 0.4 stub).
// Real implementation links to prebuilt vllm._C.rocm via libtorch; for now
// just declare that GPU ops come from the prebuilt wheel, not compiled here.
namespace vllm { namespace hip { bool gpu_available() { return false; } }}
