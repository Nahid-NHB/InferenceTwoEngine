// src/rmsnorm.cpp
#include "tinyllm/rmsnorm.hpp"

#include <cmath>
#include <stdexcept>

namespace tinyllm {

Tensor rmsnorm(const Tensor& x, const Tensor& gamma, float eps) {
    if (x.dtype() != DType::Float32) {
        throw std::runtime_error("rmsnorm: only Float32 supported");
    }
    if (gamma.dtype() != DType::Float32) {
        throw std::runtime_error("rmsnorm: gamma must be Float32");
    }
    if (gamma.ndim() != 1) {
        throw std::runtime_error("rmsnorm: gamma must be 1-D");
    }
    const int64_t d = gamma.shape()[0];
    if (d == 0) {
        throw std::runtime_error("rmsnorm: gamma has zero length");
    }
    if (x.ndim() < 1 || x.shape().back() != d) {
        throw std::runtime_error(
            "rmsnorm: last dim of x must match gamma length");
    }

    const float* xp = x.data_float();
    const float* gp = gamma.data_float();

    // Compute the outer-dim count (product of all but the last dim).
    int64_t outer = 1;
    for (std::size_t i = 0; i + 1 < x.shape().size(); ++i) {
        outer *= x.shape()[i];
    }

    Tensor out(x.shape(), DType::Float32);
    float* yp = out.data_float();

    for (int64_t i = 0; i < outer; ++i) {
        const float* row = xp + i * d;

        // mean of squares (in double for stability; values can be large).
        double sumsq = 0.0;
        for (int64_t j = 0; j < d; ++j) {
            double v = static_cast<double>(row[j]);
            sumsq += v * v;
        }
        double mean_sq = sumsq / static_cast<double>(d);
        float  rms_inv = 1.0f / static_cast<float>(std::sqrt(mean_sq + eps));

        float* yrow = yp + i * d;
        for (int64_t j = 0; j < d; ++j) {
            yrow[j] = row[j] * rms_inv * gp[j];
        }
    }
    return out;
}

}  // namespace tinyllm