// logger.cpp — minimal logging stub (Phase 0.4 stub).
#include <cstdio>
#include <cstdarg>
namespace vllm { namespace engine {
void log_info(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
}
}}
