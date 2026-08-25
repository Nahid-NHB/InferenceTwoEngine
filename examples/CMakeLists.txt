include(${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_helpers.cmake)

function(tinyllm_add_example name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE tinyllm_core)
    tinyllm_set_options(${name})
endfunction()

tinyllm_add_example(phase1_demo phase1_demo.cpp)