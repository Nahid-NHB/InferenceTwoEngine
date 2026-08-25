// src/main.cpp
// -----------------------------------------------------------------------------
// tinyllm CLI entry point.
//
// Phase 1: a smoke test for the tensor library. The CLI is not yet wired to
// model loading — that arrives in Phase 4.
// -----------------------------------------------------------------------------
#include "tinyllm/tensor.hpp"
#include "tinyllm/matmul.hpp"

#include <iostream>

using namespace tinyllm;

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    std::cout << "tinyllm " << 0.1 << " (Phase 1: tensor library)\n";

    // Smoke test
    Tensor a({2, 3}, DType::Float32);
    a.fill(1.0f);
    Tensor b({2, 3}, DType::Float32);
    b.fill(2.0f);

    auto c = a + b;
    auto d = ops::matmul(a, Tensor({3, 2}, DType::Float32));  // d is undefined numbers

    std::cout << "  a + b =\n" << c.to_string();
    std::cout << "  a.numel()=" << a.numel()
              << ", storage.use_count()=" << a.storage().use_count() << "\n";
    std::cout << "  d.numel()=" << d.numel() << "\n";

    std::cout << "OK\n";
    return 0;
}