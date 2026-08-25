# tests/test_helpers.cmake
# -----------------------------------------------------------------------------
# Helper to register CTest entries.
# -----------------------------------------------------------------------------

function(tinyllm_register_test name)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES TIMEOUT 120)
endfunction()