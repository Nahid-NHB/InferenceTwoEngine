// tests/test_helpers.hpp
// -----------------------------------------------------------------------------
// Minimal header-only test harness.
//
// We don't pull in GoogleTest / Catch2 to keep the project dependency-free
// for Phase 1. Tests register themselves in static initializers; the main()
// below runs them and reports.
//
// Usage:
//   TEST_CASE("my test") {
//       REQUIRE(x == 1);
//       REQUIRE_NEAR(a, b, 1e-5);
//   }
// -----------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace tltest {

struct Case {
    const char* name;
    void (*fn)();
};

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

struct Register {
    Register(const char* n, void (*f)()) { registry().push_back({n, f}); }
};

struct AssertionFailure : std::exception {
    std::string msg;
    explicit AssertionFailure(std::string m) : msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

inline int run_all() {
    int failed = 0;
    int total  = 0;
    for (const auto& c : registry()) {
        ++total;
        std::printf("[ RUN      ] %s\n", c.name);
        try {
            c.fn();
            std::printf("[       OK ] %s\n", c.name);
        } catch (const AssertionFailure& e) {
            std::printf("[  FAILED  ] %s\n  %s\n", c.name, e.what());
            ++failed;
        } catch (const std::exception& e) {
            std::printf("[  FAILED  ] %s\n  unexpected: %s\n", c.name, e.what());
            ++failed;
        }
    }
    std::printf("\n[====] %d tests, %d passed, %d failed\n",
                total, total - failed, failed);
    return failed == 0 ? 0 : 1;
}

}  // namespace tltest

#define TEST_CASE(name)                                                           \
    static void test_##name();                                                    \
    static ::tltest::Register reg_##name(#name, &test_##name);                    \
    static void test_##name()

#define REQUIRE(cond)                                                             \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::string m = std::string("REQUIRE failed: ") + #cond              \
                          + " @ " __FILE__ ":" + std::to_string(__LINE__);        \
            throw ::tltest::AssertionFailure(m);                                  \
        }                                                                         \
    } while (0)

#define REQUIRE_THROWS(expr)                                                      \
    do {                                                                          \
        bool threw = false;                                                       \
        try { (void)(expr); } catch (...) { threw = true; }                       \
        if (!threw) {                                                             \
            std::string m = std::string("REQUIRE_THROWS failed: ") + #expr        \
                          + " @ " __FILE__ ":" + std::to_string(__LINE__);        \
            throw ::tltest::AssertionFailure(m);                                  \
        }                                                                         \
    } while (0)

inline bool nearly_equal(float a, float b, float tol) {
    return std::fabs(a - b) <= tol;
}

#define REQUIRE_NEAR(a, b, tol)                                                   \
    do {                                                                          \
        double _aa = double(a);                                                   \
        double _bb = double(b);                                                   \
        if (std::fabs(_aa - _bb) > (tol)) {                                       \
            std::string m = std::string("REQUIRE_NEAR failed: ") + #a "("         \
                          + std::to_string(_aa) + ") vs " + #b "("                \
                          + std::to_string(_bb) + "), tol=" + std::to_string(tol) \
                          + " @ " __FILE__ ":" + std::to_string(__LINE__);        \
            throw ::tltest::AssertionFailure(m);                                  \
        }                                                                         \
    } while (0)
