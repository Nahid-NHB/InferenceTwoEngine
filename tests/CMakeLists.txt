include(${CMAKE_CURRENT_SOURCE_DIR}/test_helpers.cmake)

# Each test file is compiled into a tinyllm_test_cases library. The runner
# pulls them all in via a single executable.
add_library(tinyllm_test_cases OBJECT
    test_tensor.cpp
    test_matmul.cpp
)
target_include_directories(tinyllm_test_cases PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(tinyllm_test_cases PUBLIC tinyllm_core)
tinyllm_set_options(tinyllm_test_cases)

# Runner that pulls the test cases together and calls main().
add_executable(run_all_tests test_main.cpp)
target_link_libraries(run_all_tests PRIVATE tinyllm_test_cases tinyllm_core)
tinyllm_set_options(run_all_tests)
tinyllm_register_test(run_all_tests)