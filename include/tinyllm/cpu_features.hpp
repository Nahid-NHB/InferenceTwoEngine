// include/tinyllm/cpu_features.hpp
// -----------------------------------------------------------------------------
// Runtime CPU feature detection.
//
// Until Phase 15 the engine used compile-time flags
// (TINYLLM_ENABLE_AVX2 / TINYLLM_ENABLE_AVX512) to decide which kernels to
// emit. That made the dispatch correct but inflexible — a binary built
// without AVX2 enabled could never pick the AVX2 path, even on hardware
// that has it.
//
// Phase 15 keeps the compile-time flags as upper-bound gates (a user can
// still pass -DTINYLLM_ENABLE_AVX2=OFF to omit AVX2 code from the binary
// entirely) but lets the runtime pick between AVX2 / AVX-512 / scalar
// within whatever the build emitted. The pick happens once, lazily, the
// first time `have_avx2()` / `have_avx512()` is called.
//
// Implementations:
//   - x86 / x86_64 : CPUID leaves 1 (family/model/feature bits) and 7
//     (extended features) via the `__cpuid_count` intrinsic (MSVC) or
//     inline asm (GCC/Clang). On non-x86 we return false for everything;
//     no surprise behavior.
//   - hardware_threads : std::thread::hardware_concurrency(), clamped to
//     >= 1 (matches what we used pre-Phase 15).
//
// All functions are noexcept and thread-safe (the underlying `CpuInfo`
// singleton uses a function-local static, which the standard guarantees
// is initialized at most once, even under concurrency).
// -----------------------------------------------------------------------------
#pragma once

#include <cstdint>

namespace tinyllm {

// Probe results live here. The struct itself is internal; consumers use
// the free functions below.
struct CpuInfo {
    bool sse2        = false;
    bool sse4_2      = false;
    bool avx         = false;
    bool avx2        = false;
    bool avx512f     = false;
    bool avx512vl    = false;
    bool avx512bw    = false;
    bool avx512vbmi2 = false;
    bool fma         = false;
    bool bmi2        = false;
    bool popcnt      = false;
    int  hardware_threads = 1;
};

// Snapshot the host CPU. The result is memoized the first time it's
// requested; subsequent calls return the same object. Thread-safe.
const CpuInfo& cpu_info() noexcept;

// True iff CPUID says AVX2 + FMA + BMI2 are usable by this process.
// On non-x86 returns false.
bool have_sse2()   noexcept;
bool have_sse4_2() noexcept;
bool have_avx()    noexcept;
bool have_avx2()   noexcept;
bool have_fma()    noexcept;
bool have_bmi2()   noexcept;
bool have_popcnt() noexcept;

// True iff the full AVX-512 feature floor (F + VL + BW) is present. The
// Q4_0 matvec kernel also benefits from VBMI2 but treats it as optional.
// On non-x86 returns false.
bool have_avx512f()  noexcept;
bool have_avx512vl() noexcept;
bool have_avx512bw() noexcept;
bool have_avx512()   noexcept;  // F + VL + BW

// hardware_concurrency, clamped to >= 1. Matches the pre-Phase-15
// semantics used by the threaded matmul.
int hardware_threads() noexcept;

// Bit-budget `cpu_info()` into a short human-readable summary, e.g.
//   "avx2 avx512bw(4.threads)"
// Useful for the benchmarking suite to print what was picked.
const char* cpu_feature_summary() noexcept;

}  // namespace tinyllm
