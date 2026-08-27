// src/cpu_features.cpp
// -----------------------------------------------------------------------------
// Implementation of the runtime CPUID probe declared in cpu_features.hpp.
//
// We support x86_64 (and x86) via two routes:
//   - MSVC:  __cpuid / __cpuidex intrinsics in <intrin.h>
//   - GCC / Clang on x86: __get_cpuid_count from <cpuid.h>
//
// Anything else (ARM, unknown arch) returns a CpuInfo with every feature
// bit false and hardware_threads = 1; the kernels that need a feature
// gate to runtime-detect will then take the scalar fallback.
//
// Notes on the bits we read:
//   CPUID leaf 1, ECX:
//     bit 28 = AVX       (YMM state)
//     bit 12 = FMA       (3-operand FMA on YMM)
//     bit 23 = POPCNT
//     bit 20 = SSE4.2
//     bit 26 = SSE2
//   CPUID leaf 1, EBX:
//     bits 16..23 = the *logical core count for this package* (we read
//     this only as a sanity floor; std::thread::hardware_concurrency()
//     is the real source for the threading decision).
//   CPUID leaf 7, sub-leaf 0, EBX:
//     bit  5 = AVX2
//     bit  8 = BMI2
//     bit 16 = AVX-512F
//     bit 17 = AVX-512DQ
//     bit 28 = AVX-512CD
//     bit 30 = AVX-512BW
//     bit 31 = AVX-512VL
//   CPUID leaf 7, sub-leaf 0, ECX:
//     bit  6 = AVX-512VBMI2
//
// We don't read leaf 0 to detect the maximum supported leaf — instead we
// guard each call with a hand-rolled "is leaf N supported?" check that
// calls CPUID with EAX=0 first and compares N to the returned max.
// -----------------------------------------------------------------------------
#include "tinyllm/cpu_features.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <thread>

#if defined(_MSC_VER)
  #include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
  #include <cpuid.h>
#endif

namespace tinyllm {

namespace {

// CPUID wrapper that hides the MSVC vs GCC/Clang differences. `leaf` is
// the value passed in EAX; for leaf 7 (extended features) use `cpuid` with
// ECX = `subleaf` (the input register for sub-leaf selection).
struct CpuidResult {
    unsigned eax, ebx, ecx, edx;
};

#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)

CpuidResult cpuid(unsigned leaf, unsigned subleaf = 0) {
    CpuidResult r{0, 0, 0, 0};
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<unsigned>(regs[0]);
    r.ebx = static_cast<unsigned>(regs[1]);
    r.ecx = static_cast<unsigned>(regs[2]);
    r.edx = static_cast<unsigned>(regs[3]);
#else
    // __get_cpuid_count returns 0 if the requested leaf isn't supported.
    // We probe with subleaf because leaf 7 explicitly needs it.
    unsigned a, b, c, d;
    if (__get_cpuid_count(leaf, subleaf, &a, &b, &c, &d)) {
        r.eax = a; r.ebx = b; r.ecx = c; r.edx = d;
    }
#endif
    return r;
}

#endif  // x86

CpuInfo probe() {
    CpuInfo info;
    info.hardware_threads =
        std::max<unsigned>(1, std::thread::hardware_concurrency());

#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
    // Leaf 0 returns the maximum supported basic leaf in EAX. Anything
    // larger than 0 is supported.
    const CpuidResult l0 = cpuid(0);
    if (l0.eax >= 1) {
        const CpuidResult l1 = cpuid(1);
        info.sse2   = (l1.edx & (1u << 26)) != 0;
        info.sse4_2 = (l1.ecx & (1u << 20)) != 0;
        info.avx    = (l1.ecx & (1u << 28)) != 0;
        info.fma    = (l1.ecx & (1u << 12)) != 0;
        info.popcnt = (l1.ecx & (1u << 23)) != 0;
    }
    // Leaf 7 sub-leaf 0 carries AVX2 / BMI2 / AVX-512F/BW/CD/VL/DQ.
    if (l0.eax >= 7) {
        const CpuidResult l7 = cpuid(7, 0);
        info.avx2        = (l7.ebx & (1u <<  5)) != 0;
        info.bmi2        = (l7.ebx & (1u <<  8)) != 0;
        info.avx512f     = (l7.ebx & (1u << 16)) != 0;
        info.avx512bw    = (l7.ebx & (1u << 30)) != 0;
        info.avx512vl    = (l7.ebx & (1u << 31)) != 0;
        info.avx512vbmi2 = (l7.ecx & (1u <<  6)) != 0;
    }
#endif

    return info;
}

const CpuInfo& cpu_info_impl() {
    // function-local static: thread-safe per [stmt.dcl]/4.
    static const CpuInfo info = probe();
    return info;
}

}  // namespace

const CpuInfo& cpu_info() noexcept { return cpu_info_impl(); }

bool have_sse2()   noexcept { return cpu_info().sse2;   }
bool have_sse4_2() noexcept { return cpu_info().sse4_2; }
bool have_avx()    noexcept { return cpu_info().avx;    }
bool have_avx2()   noexcept { return cpu_info().avx2;   }
bool have_fma()    noexcept { return cpu_info().fma;    }
bool have_bmi2()   noexcept { return cpu_info().bmi2;   }
bool have_popcnt() noexcept { return cpu_info().popcnt; }

bool have_avx512f()  noexcept { return cpu_info().avx512f;  }
bool have_avx512vl() noexcept { return cpu_info().avx512vl; }
bool have_avx512bw() noexcept { return cpu_info().avx512bw; }
bool have_avx512()   noexcept {
    const CpuInfo& c = cpu_info();
    return c.avx512f && c.avx512vl && c.avx512bw;
}

int hardware_threads() noexcept { return cpu_info().hardware_threads; }

// -----------------------------------------------------------------------------
// cpu_feature_summary — short string for bench headers.
// Cached on first call. Bounded size (128 bytes worst case) so callers
// can store it in a fixed buffer.
// -----------------------------------------------------------------------------
const char* cpu_feature_summary() noexcept {
    static const std::array<char, 128> buf = []() {
        std::array<char, 128> b{};
        std::size_t w = 0;
        auto put = [&](const char* s) {
            for (std::size_t i = 0; s[i] && w + 1 < b.size(); ++i) {
                b[w++] = s[i];
            }
        };
        auto maybe = [&](bool have, const char* name) {
            if (have) {
                if (w > 0 && b[w - 1] != '(') put(" ");
                put(name);
            }
        };
        const CpuInfo& c = cpu_info();
        maybe(c.avx2,           "avx2");
        maybe(c.avx512f,        "avx512f");
        maybe(c.avx512bw,       "avx512bw");
        maybe(c.avx512vl,       "avx512vl");
        maybe(c.avx512vbmi2,    "avx512vbmi2");
        maybe(c.fma,            "fma");
        maybe(c.bmi2,           "bmi2");
        maybe(c.sse4_2,         "sse4_2");
        maybe(c.avx,            "avx");
        if (w == 0) put("(scalar)");
        // Append "(N threads)" suffix.
        put("(");
        int n = c.hardware_threads;
        char num[16];
        int len = 0;
        if (n == 0) { num[len++] = '0'; }
        else {
            int tmp = n;
            while (tmp > 0 && len < 15) { num[len++] = '0' + static_cast<char>(tmp % 10); tmp /= 10; }
            std::reverse(num, num + len);
        }
        for (int i = 0; i < len && w + 1 < b.size(); ++i) b[w++] = num[i];
        put(" threads)");
        b[w] = '\0';
        return b;
    }();
    return buf.data();
}

}  // namespace tinyllm
