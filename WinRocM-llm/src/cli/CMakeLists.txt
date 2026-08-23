add_executable(vllm_cli main.cpp)

# Phase 0 self-check CLI
target_sources(vllm_cli PRIVATE
    $<$<CONFIG:Debug>:src/cli/selfcheck.cpp>
)

# Expose as vllm_test alias for the gate target
if(NOT TARGET vllm_test)
    add_custom_target(vllm_test
        DEPENDS vllm_cli
        COMMAND $<TARGET_FILE:vllm_cli> --self-check
    )
endif()
