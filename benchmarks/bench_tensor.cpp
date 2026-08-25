// benchmarks/bench_tensor.cpp
// -----------------------------------------------------------------------------
// Benchmark for Phase 1 tensor ops.
// Runs naive matmul at a few sizes, prints GFLOPS and wall time.
// -----------------------------------------------------------------------------
#include "tinyllm/tensor.hpp"
#include "tinyllm/matmul.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace tinyllm;
using clk = std::chrono::high_resolution_clock;

static double gflops_matmul(int64_t M, int64_t N, int64_t K, double ms) {
    double ops = 2.0 * static_cast<double>(M)
                     * static_cast<double>(N)
                     * static_cast<double>(K);  // mul + add per element
    return ops / (ms * 1e-3) / 1e9;
}

int main() {
    std::vector<int64_t> sizes = {64, 128, 256, 512};

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::printf("op,size,ms,gflops\n");
    for (int64_t n : sizes) {
        std::size_t nn = static_cast<std::size_t>(n);
        std::vector<float> da(nn * nn), db(nn * nn);
        for (auto& x : da) x = dist(rng);
        for (auto& x : db) x = dist(rng);
        Tensor A({n, n}, DType::Float32, da.data(), da.size() * sizeof(float));
        Tensor B({n, n}, DType::Float32, db.data(), db.size() * sizeof(float));

        // warmup
        auto Cwarm = ops::matmul(A, B);
        (void)Cwarm;

        auto t0 = clk::now();
        auto C2 = ops::matmul(A, B);
        auto t1 = clk::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gf = gflops_matmul(n, n, n, ms);
        std::printf("matmul,%ld,%.3f,%.3f\n", n, ms, gf);
        (void)C2;
    }

    // softmax benchmark
    {
        Tensor t({1024, 1024}, DType::Float32);
        for (int64_t i = 0; i < t.numel(); ++i) {
            t.at_flat(i) = dist(rng);
        }
        auto warm = ops::softmax(t);
        auto t0 = clk::now();
        auto r = ops::softmax(t);
        auto t1 = clk::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("softmax,1024x1024,%.3f,-\n", ms);
        (void)warm; (void)r;
    }

    std::printf("\nBenchmark done.\n");
    return 0;
}