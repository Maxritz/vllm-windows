# ==============================================================================
# Phase 0.2 — Host C++ build for WinRocM-llm under the ROCm-on-Windows toolchain.
#
# CMake HIP language is NOT enabled (D013): CMake 3.29's Windows-Clang platform
# module injects MSVC flags incompatible with the GNU-triple shim (D012). HIP
# kernels compile via cmake/smoke_hip.cmd (two-stage: hipcc -c  ->  lld link).
# This CMakeLists builds only the host C++ engine.
# ==============================================================================
cmake_minimum_required(VERSION 3.21)
project(WinRocM_engine LANGUAGES C CXX)

set(ROCM_HOME "G:/ROCM10RT-gfx1201")
set(MINGW_GCC_LIBDIR "C:/Strawberry/c/lib/gcc/x86_64-w64-mingw32/13.2.0")
set(MINGW_LIBDIR "C:/Strawberry/c/x86_64-w64-mingw32/lib")

# --- D012 shim flags (applied post-project() to override Windows-Clang.cmake) ---
add_compile_options(
    --target=x86_64-pc-windows-gnu
    -isystem ${ROCM_HOME}/include
    -D__HIP_PLATFORM_AMD__=1
    -D__NO_MATH_DEFINES=1
    -D_MSC_VER=1900
    -D_NATIVE_WCHAR_T_DEFINED=1
    -D_WCHAR_T_DEFINED
    -fshort-wchar
)
# Strip MSVC-debug runtime selectors injected by Windows-Clang.cmake for Debug.
foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    foreach(_l CXX C)
        string(REGEX REPLACE "-D_DEBUG|-D_DLL|-D_MT|-Xclang --dependent-lib=[a-z]+" "" CMAKE_${_l}_FLAGS_${_cfg} "${CMAKE_${_l}_FLAGS_${_cfg}}")
    endforeach()
endforeach()
# Kill MSVC-linker selectors Windows-Clang.cmake injects — they force
# -fuse-ld=lld-link (MSVC COFF lld), incompatible with GNU .a archives (D012).
set(CMAKE_CXX_LINK_FLAGS "")
set(CMAKE_C_LINK_FLAGS "")
set(CMAKE_CXX_LINK_OPTIONS "")
set(CMAKE_C_LINK_OPTIONS "")
set(CMAKE_EXE_LINKER_FLAGS "")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "")
# Override the Windows-clang LINK_EXECUTABLE template (which hardcodes /MANIFEST,
# /implib, /pdb) with a GNU-lld rule (D012 two-stage link style).
set(CMAKE_CXX_LINK_EXECUTABLE
    "<CMAKE_CXX_COMPILER> <FLAGS> <LINK_FLAGS> <LINK_OPTIONS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
set(CMAKE_C_LINK_EXECUTABLE
    "<CMAKE_C_COMPILER> <FLAGS> <LINK_FLAGS> <LINK_OPTIONS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
set(CMAKE_CXX_LINK_OPTIONS "")
set(CMAKE_C_LINK_OPTIONS "")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(_root "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_executable(vllm_engine
    ${_root}/src/engine/engine.cpp
    ${_root}/src/engine/logger.cpp
    ${_root}/src/engine/gpu_buffer.cpp
    ${_root}/src/engine/block_table.cpp
    ${_root}/src/engine/model_runner.cpp
    ${_root}/src/hip_wrap/hip_compat/gpu_windows.cpp
    ${_root}/src/cli/main.cpp
)

target_include_directories(vllm_engine PRIVATE
    ${_root}/src
    ${_root}/src/hip_wrap
    ${_root}/src/engine/include
    ${_root}
)

# --- Link: GNU lld (ld.lld, NOT lld-link) against MinGW runtime objects + amdhip64.lib (D012) ---
# Strip the MSVC-style -Xlinker /subsystem:console that Windows-Clang.cmake injects
# (it makes -fuse-ld=lld dispatch to lld-link, which can't consume GNU .a archives).
string(REGEX REPLACE "-Xlinker /[a-z:]+" "" CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS}")
string(REGEX REPLACE "-Xlinker /[a-z:]+" "" CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS}")
target_link_options(vllm_engine PRIVATE
    -fuse-ld=lld           # GNU lld (ld.lld) — accepts both GNU .a and MSVC .lib
    -nodefaultlibs
    -Wl,--subsystem,console
    -L${MINGW_GCC_LIBDIR}
    -L${MINGW_LIBDIR}
    -L${ROCM_HOME}/lib
)
# Order matters: MinGW runtime first, then ROCm lib, then Windows system libs.
target_link_libraries(vllm_engine PRIVATE
    "${MINGW_GCC_LIBDIR}/libgcc.a"
    "${MINGW_GCC_LIBDIR}/libgcc_eh.a"
    "${MINGW_LIBDIR}/libmingwex.a"
    "${MINGW_LIBDIR}/libmsvcrt.a"
    "${MINGW_LIBDIR}/libkernel32.a"
    "${MINGW_LIBDIR}/libuser32.a"
    "${MINGW_LIBDIR}/libadvapi32.a"
    "${ROCM_HOME}/lib/amdhip64.lib"
)

target_compile_definitions(vllm_engine PRIVATE
    AMDT_SCOPED_RESOURCES_DYNAMIC=0
)
