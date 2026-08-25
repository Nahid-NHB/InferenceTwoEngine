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
)
target_include_directories(tinyllm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
tinyllm_set_options(tinyllm_core)

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
message(STATUS "")
