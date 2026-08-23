# ==============================================================================
# rocm-windows-toolchain.cmake — Phase 0.2
# CMake toolchain driving the Phase 0.1-validated HIP-on-Windows recipe (D012/D013).
#
# Two-shim compile recipe:
#   GNU triple (--target=x86_64-pc-windows-gnu)  +  -D_MSC_VER=1900
#     -> no MSVC <cmath> isgreater macro conflict
#     -> amd_warp_functions.h takes LLP64 #else branch (assert passes on Windows)
#
# Two-stage link (GNU lld against Strawberry MinGW runtime objects + amdhip64.lib):
#   hipcc --target=gnu -c  -> .o
#   clang++ --fuse-ld=lld -nodefaultlibs <mingw .a> <amdhip64.lib> <sdk .lib> -> exe
#
# CMake HIP language is NOT used (see D013) — HIP kernels compile via smoke_hip.cmd.
# This toolchain drives only host CXX for the engine control plane.
# ==============================================================================
get_filename_component(ROCM_HOME "G:/ROCM10RT-gfx1201" ABSOLUTE)

set(CMAKE_SYSTEM_NAME           Windows)
set(CMAKE_SYSTEM_PROCESSOR     x86_64)

# --- Compilers: ROCm clang++ (GNU dialect), NOT MSVC cl.exe (D001) ---
set(CMAKE_C_COMPILER            "${ROCM_HOME}/lib/llvm/bin/clang.exe" CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER          "${ROCM_HOME}/lib/llvm/bin/clang++.exe" CACHE FILEPATH "C++ compiler" FORCE)
set(CMAKE_C_COMPILER_FRONTEND_VARIANT     "GNU")
set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT   "GNU")
# Skip full compiler test (ROCm clang++ without --target shim during project()
# fails the GCC-ABI link test). We re-apply the actual flags via add_compile_options.
set(CMAKE_CXX_COMPILER_WORKS    TRUE)
set(CMAKE_C_COMPILER_WORKS      TRUE)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# RC resource compiler (Windows-Clang platform module requires llvm-rc).
set(CMAKE_RC_COMPILER           "${ROCM_HOME}/lib/llvm/bin/llvm-rc.exe")

# --- Target AMD ISA: gfx1201 = local RX 9070 XT (RDNA4) ---
set(VLLM_GPU_ARCH "gfx1201" CACHE STRING "Target AMD GPU architecture")

# --- ROCm device lib + runtime ---
set(ROCM_DEVICE_LIB     "${ROCM_HOME}/lib/llvm/amdgcn/bitcode")
set(ROCM_LINK_LIBS      "${ROCM_HOME}/lib")

# --- Strawberry MinGW runtime objects (for two-stage GNU-lld link) ---
set(MINGW_GCC_LIBDIR    "C:/Strawberry/c/lib/gcc/x86_64-w64-mingw32/13.2.0")
set(MINGW_LIBDIR        "C:/Strawberry/c/x86_64-w64-mingw32/lib")

# --- Compiler flags: GNU triple + shim defines that satisfy HIP headers (D012) ---
# Export cache var consumed by CMakeLists (post-project, Windows-Clang.cmake
# overrides CMAKE_<LANG>_FLAGS during project(), so we re-apply via add_compile_options).
set(_SHIM_FLAGS "--target=x86_64-pc-windows-gnu -isystem ${ROCM_HOME}/include -D__HIP_PLATFORM_AMD__=1 -D__NO_MATH_DEFINES=1 -D_MSC_VER=1900 -D_NATIVE_WCHAR_T_DEFINED=1 -D_WCHAR_T_DEFINED -fshort-wchar")
set(WINROCM_SHIM_FLAGS "${_SHIM_FLAGS}" CACHE STRING "Windows-ROCm compile flags (D012)" FORCE)
set(CMAKE_CXX_FLAGS_INIT     "${_SHIM_FLAGS}")
set(CMAKE_C_FLAGS_INIT       "${_SHIM_FLAGS}")
set(CMAKE_RC_FLAGS_INIT      "${_SHIM_FLAGS}")
# Strip MSVC-debug runtime + /subsystem selectors injected by Windows-Clang.cmake.
# These make -fuse-ld=lld dispatch to lld-link (MSVC COFF mode), incompatible with
# GNU .a archives. Applied via FORCE so project()'s platform module can't restore them.
set(CMAKE_C_FLAGS_DEBUG_INIT         "-O0 -g -D_DEBUG")
set(CMAKE_CXX_FLAGS_DEBUG_INIT       "-O0 -g -D_DEBUG")
set(CMAKE_C_FLAGS_RELEASE_INIT       "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE_INIT     "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-O2 -g -DNDEBUG")
# Nuking MSVC runtime selectors from the *cache* (which project() reads):
set(CMAKE_C_FLAGS               "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS             "" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_DEBUG         "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG       "" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELEASE       "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE     "" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELWITHDEBINFO "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "" CACHE STRING "" FORCE)
# Link flags: strip any /subsystem or --dependent-lib that Windows-Clang.cmake adds.
set(CMAKE_C_LINK_FLAGS          "" CACHE STRING "" FORCE)
set(CMAKE_CXX_LINK_FLAGS        "" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS   "" CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS   "" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS      "" CACHE STRING "" FORCE)
# Also pre-populate CMAKE_<LANG>_FLAGS_INIT for first-configure seed.
set(CMAKE_CXX_FLAGS_INIT "${_SHIM_FLAGS}")
set(CMAKE_C_FLAGS_INIT   "${_SHIM_FLAGS}")
set(CMAKE_RC_FLAGS_INIT  "${_SHIM_FLAGS}")

# --- Link: GNU lld, no implicit libs, explicit MinGW runtime .a + MSVC .lib ---
# NOTE: mirrors the working Phase 0.1 smoke link exactly.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld -nodefaultlibs -L${MINGW_GCC_LIBDIR} -L${MINGW_LIBDIR} -L${ROCM_LINK_LIBS}")
set(CMAKE_CXX_STANDARD_LIBRARIES_INIT
    "${MINGW_GCC_LIBDIR}/libgcc.a"
    "${MINGW_GCC_LIBDIR}/libgcc_eh.a"
    "${MINGW_LIBDIR}/libmingwex.a"
    "${MINGW_LIBDIR}/libmsvcrt.a"
    "${MINGW_LIBDIR}/libkernel32.a"
    "${MINGW_LIBDIR}/libuser32.a"
    "${MINGW_LIBDIR}/libadvapi32.a"
)

# ROCm root for find_package / hipconfig lookups.
set(ROCM_PATH "${ROCM_HOME}" CACHE PATH "ROCm root")
set(ENV{ROCM_PATH} "${ROCM_HOME}")
set(ENV{PATH} "${ROCM_HOME}/bin;${ROCM_HOME}/lib/llvm/bin;$ENV{PATH}")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_HOME}")

# Strip MSVC cl-style flags from PyTorch's imported targets (D010, f8ddc4c13).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
set(CMAKE_USER_MAKE_RULES_OVERRIDE "${CMAKE_CURRENT_LIST_DIR}/strip-msvc-flags.cmake")
