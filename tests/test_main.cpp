// tests/test_main.cpp
// -----------------------------------------------------------------------------
// Test runner: pulls in the auto-registered test cases from each file via the
// header trick, then runs them all.
//
// Each test_*.cpp file uses TEST_CASE to register itself into tltest::registry.
// We don't need to call anything from those files directly — the static
// initializers do the work. But we DO need to force-link them; that's what the
// CMake helper does via --whole-archive or by referencing any symbol.
// -----------------------------------------------------------------------------
#include "test_helpers.hpp"

int main() {
    return tltest::run_all();
}