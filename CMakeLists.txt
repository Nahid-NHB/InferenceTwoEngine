cmake_minimum_required(VERSION 3.20)
project(tinyllm
    VERSION 0.1.0
    DESCRIPTION "Tiny LLM Inference Engine from scratch in modern C++"
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if nothing specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# Compile options
# -----------------------------------------------------------------------------
function(tinyllm_set_options target)
    target_compile_options(${target} PRIVATE
        -Wall -Wextra -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor -Wold-style-cast
        -Wcast-align -Wunused -Woverloaded-virtual
        -Wnull-dereference
        -Wformat=2
    )
    if(NOT MSVC)
        target_compile_options(${target} PRIVATE
            -fno-omit-frame-pointer
        )
    endif()
endfunction()

# -----------------------------------------------------------------------------
# tinyllm core library (header + impl split)
# -----------------------------------------------------------------------------
add_library(tinyllm_core
    src/tensor.cpp
    src/memory.cpp
    src/cpu_features.cpp
    src/matmul.cpp
    src/tokenizer.cpp
    src/gguf.cpp
    src/rmsnorm.cpp
    src/rope.cpp
    src/attention.cpp
    src/mlp.cpp
    src/llama_block.cpp
    src/model.cpp
    src/kv_cache.cpp
    src/sampler.cpp
    src/quantize.cpp
    src/llama_loader.cpp
)
target_include_directories(tinyllm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
tinyllm_set_options(tinyllm_core)

# Phase 15: SIMD is auto-detected at runtime. The compile-time flag
# is now just an upper-bound gate — pass -DTINYLLM_ENABLE_AVX2=OFF (or
# -DTINYLLM_ENABLE_AVX512=OFF) to *omit* the corresponding kernel code
# from the binary entirely (useful for size-limited or narrowly-portable
# builds). Without those overrides we default to ON whenever the
# compiler supports the ISA — the runtime CPUID probe then decides
# whether to actually use the kernel.
#
# Note: AVX-512 implies AVX2 on GCC/Clang (the `-mavx512f` flag turns
# on `__AVX2__` automatically), so disabling AVX2 also disables AVX-512
# even if the latter option is on. The runtime dispatch in src/matmul.cpp
# + src/quantize.cpp handles the rest.
option(TINYLLM_ENABLE_AVX2   "Compile AVX2 kernels into tinyllm_core" ON)
option(TINYLLM_ENABLE_AVX512 "Compile AVX-512 kernels into tinyllm_core" ON)
if(TINYLLM_ENABLE_AVX512 AND NOT TINYLLM_ENABLE_AVX2)
    message(FATAL_ERROR
        "TINYLLM_ENABLE_AVX512=ON requires TINYLLM_ENABLE_AVX2=ON "
        "(AVX-512 implies AVX2 on the toolchains we target).")
endif()

# Enable AVX2 + FMA on x86 if the compiler supports it.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i.86)")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2 -mfma -mbmi2" HAVE_AVX2_FLAG)
    if(TINYLLM_ENABLE_AVX2 AND HAVE_AVX2_FLAG)
        message(STATUS "Enabling AVX2 + FMA in tinyllm_core (runtime-dispatched)")
        target_compile_options(tinyllm_core PUBLIC -mavx2 -mfma -mbmi2)
        target_compile_definitions(tinyllm_core PUBLIC TINYLLM_ENABLE_AVX2=1)
    endif()

    # Phase 13: prefer AVX-512F+VL+BW+VBMI2 for the Q4_0 matvec kernel
    # when available. Runtime CPUID probe picks between AVX-512 / AVX2 /
    # dequant-bounce based on what the host CPU actually exposes. We
    # require VL+BW so we can use 512-bit byte/word shifts and the VBMI2
    # byte-shuffle instructions (vpcompressd is in VBMI2; we use it as a
    # fast deinterleave). VNNI is not required.
    check_cxx_compiler_flag(
        "-mavx512f -mavx512vl -mavx512bw -mavx512vbmi2" HAVE_AVX512_FLAG)
    if(TINYLLM_ENABLE_AVX512 AND HAVE_AVX512_FLAG)
        message(STATUS
            "Enabling AVX-512F+VL+BW+VBMI2 in tinyllm_core (runtime-dispatched)")
        target_compile_options(tinyllm_core PUBLIC
            -mavx512f -mavx512vl -mavx512bw -mavx512vbmi2)
        target_compile_definitions(tinyllm_core PUBLIC
            TINYLLM_ENABLE_AVX512=1)
    endif()
endif()

# -----------------------------------------------------------------------------
# Executables
# -----------------------------------------------------------------------------
# main CLI (placeholder for later phases; today it runs tensor sanity checks)
add_executable(tinyllm src/main.cpp)
target_link_libraries(tinyllm PRIVATE tinyllm_core)
tinyllm_set_options(tinyllm)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
option(TINYLLM_BUILD_TESTS "Build unit tests" ON)
if(TINYLLM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------
option(TINYLLM_BUILD_BENCHMARKS "Build benchmarks" ON)
if(TINYLLM_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------
option(TINYLLM_BUILD_EXAMPLES "Build examples" ON)
if(TINYLLM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "tinyllm configuration")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build tests:       ${TINYLLM_BUILD_TESTS}")
message(STATUS "  Build benchmarks:  ${TINYLLM_BUILD_BENCHMARKS}")
message(STATUS "  AVX2 enabled:      ${TINYLLM_ENABLE_AVX2}")
message(STATUS "  AVX-512 enabled:   ${TINYLLM_ENABLE_AVX512}")
message(STATUS "  Dispatch:          runtime CPUID (Phase 15)")
message(STATUS "")
