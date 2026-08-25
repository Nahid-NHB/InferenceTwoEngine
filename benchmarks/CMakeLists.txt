include(${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_helpers.cmake)

function(tinyllm_add_bench name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE tinyllm_core)
    tinyllm_set_options(${name})
endfunction()

tinyllm_add_bench(bench_tensor bench_tensor.cpp)
tinyllm_add_bench(bench_matmul bench_matmul.cpp)
tinyllm_add_bench(bench_tokenizer bench_tokenizer.cpp)
tinyllm_add_bench(bench_quantize bench_quantize.cpp)
tinyllm_add_bench(bench_transformer bench_transformer.cpp)