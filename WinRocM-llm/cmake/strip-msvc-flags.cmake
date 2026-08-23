# strip-msvc-flags.cmake
# ==============================================================================
# CMAKE_USER_MAKE_RULES_OVERRIDE hook (loaded by rocm-windows-toolchain.cmake).
# Strips MSVC cl-style flags that PyTorch's imported targets inject via
# INTERFACE_COMPILE_OPTIONS, which ROCm clang++ (GNU frontend) cannot parse.
# Mirrors the MaxRitz/vllm-windows strip-msvc-flags.cmake approach (f8ddc4c13).
# ==============================================================================

set(_MSVC_BAD_FLAGS
    "/bigobj"
    "/utf-8"
    "/permissive-"
    "/W3" "/W4" "/WX"
    "/GR" "/GR-"
    "/EHs-c-5"
    "/Zc:__cplusplus" "/Zc:preprocessor" "/Zc:preprocessorCompat-0"
    "/std:c++17" "/std:c++20"      # clang uses -std= not /std:
    "/EHsc"
    "/MD" "/MDd" "/MT" "/MTd"
    "/O2" "/Od" "/O1" "/Oy" "/Ob1" "/Ob2"
    "/GF" "/GS"
    "/RTC1" "/RTCs" "/RTCu"
    "/fp:precise" "/fp:fast"
    "/utf-8"
)

set(_FLAG_VARS
    CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_HIP_FLAGS
    CMAKE_C_FLAGS_DEBUG CMAKE_CXX_FLAGS_DEBUG CMAKE_HIP_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE CMAKE_HIP_FLAGS_RELEASE
    CMAKE_C_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_HIP_FLAGS_RELWITHDEBINFO
    CMAKE_C_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_HIP_FLAGS_MINSIZEREL
)

foreach(_var ${_FLAG_VARS})
    foreach(_flag ${_MSVC_BAD_FLAGS})
        string(REPLACE "${_flag}" "" ${_var} "${${_var}}")
        string(REPLACE "${_flag} " "" ${_var} "${${_var}}")
    endforeach()
    # Collapse double spaces left by stripping.
    string(REGEX REPLACE "  +" " " ${_var} "${${_var}}")
endforeach()
