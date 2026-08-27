// benchmarks/bench_matmul.cpp
// -----------------------------------------------------------------------------
// Compare all matmul variants at a range of sizes.
//
// For each size N x N:
//   - naive, blocked, avx2, [avx512], threaded
//   - warmup once
//   - report ms and GFLOPS
//
// Output is CSV-style so it can be diffed across runs / machines.
// -----------------------------------------------------------------------------
#include "tinyllm/cpu_features.hpp"
#include "tinyllm/matmul.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace tinyllm;
using clk = std::chrono::high_resolution_clock;

static double gflops(int64_t M, int64_t N, int64_t K, double ms) {
    return 2.0 * static_cast<double>(M)
              * static_cast<double>(N)
              * static_cast<double>(K) / (ms * 1e-3) / 1e9;
}

struct Bench {
    std::string label;
    double ms;
    double gf;
};

int main() {
    std::printf("# runtime: %s\n", tinyllm::cpu_feature_summary());
    std::printf("# dispatch: avx2=%d avx-512=%d threads=%d\n",
                ops::have_avx2() ? 1 : 0,
                ops::have_avx512() ? 1 : 0,
                ops::hardware_threads());

    std::vector<int64_t> sizes = {64, 128, 256, 512, 1024};
    std::vector<ops::MatmulVariant> variants = {
        ops::MatmulVariant::Naive,
        ops::MatmulVariant::Blocked,
        ops::MatmulVariant::Avx2,
        ops::MatmulVariant::Threaded,
    };
#if TINYLLM_ENABLE_AVX512
    // Phase 16: include the AVX-512 fused F32 kernel in the sweep when
    // the build emitted it. The `ops::have_avx512()` runtime probe (set
    // via the Phase 15 CPUID module) decides whether to actually call
    // the kernel; on non-AVX-512 hosts the path silently falls back to
    // the blocked scalar tile, so we don't have to special-case the
    // bench loop here. The output line just reads "avx512,..." even on
    // hosts where it ran the scalar fallback (look at the dispatch
    // header line above to know which).
    variants.insert(variants.begin() + 3, ops::MatmulVariant::Avx512);
#endif

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::printf("variant,size,ms,gflops\n");

    for (int64_t n : sizes) {
        std::size_t nn = static_cast<std::size_t>(n);
        std::vector<float> da(nn * nn), db(nn * nn);
        for (auto& v : da) v = dist(rng);
        for (auto& v : db) v = dist(rng);
        Tensor A({n, n}, DType::Float32, da.data(), da.size() * sizeof(float));
        Tensor B({n, n}, DType::Float32, db.data(), db.size() * sizeof(float));

        for (auto v : variants) {
            // Warmup
            auto warm = ops::matmul(A, B, v);
            (void)warm;

            auto t0 = clk::now();
            auto r  = ops::matmul(A, B, v);
            auto t1 = clk::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double gf = gflops(n, n, n, ms);
            std::printf("%s,%ld,%.3f,%.3f\n",
                        std::string(ops::variant_name(v)).c_str(),
                        n, ms, gf);
            (void)r;
        }
    }

    std::printf("\n");
    std::printf("auto-pick = %s\n",
                std::string(ops::variant_name(ops::last_picked_variant())).c_str());
    return 0;
}