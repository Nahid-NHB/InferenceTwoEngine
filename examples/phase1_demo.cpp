// examples/phase1_demo.cpp
// -----------------------------------------------------------------------------
// A tiny end-to-end demo of the tensor library:
//  - make a one-hot-like input row,
//  - apply a (random) 2-layer "MLP" (matmul + relu + matmul),
//  - softmax the result,
//  - print the argmax.
//
// Builds against tinyllm_core. Useful as a smoke test and as an example of
// how to compose the tensor primitives.
// -----------------------------------------------------------------------------
#include "tinyllm/tensor.hpp"
#include "tinyllm/matmul.hpp"

#include <cstdio>
#include <random>
#include <vector>

using namespace tinyllm;

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Input: a one-hot-like vector of size 8 with a single 1.0
    std::vector<float> x_in(8, 0.0f);
    x_in[3] = 1.0f;

    // First matmul: [1, 8] x [8, 16] -> [1, 16]
    std::vector<float> w1(8 * 16);
    for (auto& v : w1) v = dist(rng);
    Tensor x({1, 8}, DType::Float32, x_in.data(), x_in.size() * sizeof(float));
    Tensor W1({8, 16}, DType::Float32, w1.data(), w1.size() * sizeof(float));

    auto h = ops::matmul(x, W1);                              // [1, 16]

    // ReLU emulation (no maximum op yet) — clamp negatives to 0.
    Tensor relu(h.shape(), DType::Float32);
    for (int64_t i = 0; i < h.numel(); ++i) {
        float v = h.at_flat(i);
        relu.at_flat(i) = v > 0.0f ? v : 0.0f;
    }

    // Second matmul: [1, 16] x [16, 4] -> [1, 4]
    std::vector<float> w2(16 * 4);
    for (auto& v : w2) v = dist(rng);
    Tensor W2({16, 4}, DType::Float32, w2.data(), w2.size() * sizeof(float));
    auto logits = ops::matmul(relu, W2);

    auto probs = ops::softmax(logits);

    int best = 0;
    float best_p = probs.at_flat(0);
    for (int64_t i = 1; i < probs.numel(); ++i) {
        if (probs.at_flat(i) > best_p) {
            best_p = probs.at_flat(i);
            best = static_cast<int>(i);
        }
    }

    std::printf("input argmax=3, output argmax=%d (p=%.4f)\n", best, best_p);
    return 0;
}