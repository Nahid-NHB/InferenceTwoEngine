// src/rope.cpp
#include "tinyllm/rope.hpp"

#include <cmath>
#include <stdexcept>

namespace tinyllm {

namespace {

// Build [seq_len, half_dim] cos/sin tables.
// half_dim = head_dim / 2.
// For position p (0..seq_len-1), frequency index i (0..half_dim-1):
//     freq_i   = 1 / theta_base^(2i / head_dim)
//     angle_pi = (start_pos + p) * freq_i
void fill_tables(Tensor& cos, Tensor& sin,
                 int64_t seq_len, int64_t head_dim,
                 int64_t start_pos, float theta_base) {
    const int64_t half = head_dim / 2;
    float* cp = cos.data_float();
    float* sp = sin.data_float();
    for (int64_t p = 0; p < seq_len; ++p) {
        const double pos = static_cast<double>(start_pos + p);
        for (int64_t i = 0; i < half; ++i) {
            // freq = theta_base^(-2i / head_dim)
            double exponent = -2.0 * static_cast<double>(i) / static_cast<double>(head_dim);
            double freq = std::pow(static_cast<double>(theta_base), exponent);
            double angle = pos * freq;
            cp[p * half + i] = static_cast<float>(std::cos(angle));
            sp[p * half + i] = static_cast<float>(std::sin(angle));
        }
    }
}

void check_rope_input(const Tensor& x) {
    if (x.dtype() != DType::Float32) {
        throw std::runtime_error("rope: only Float32 supported");
    }
    if (x.ndim() < 2) {
        throw std::runtime_error("rope: tensor must have at least 2 dims");
    }
    int64_t d = x.shape().back();
    if (d % 2 != 0) {
        throw std::runtime_error("rope: head_dim must be even");
    }
}

}  // namespace

RopeTables precompute_rope_tables(int64_t seq_len, int64_t head_dim,
                                   int64_t start_pos, float theta_base) {
    if (head_dim % 2 != 0) {
        throw std::runtime_error("rope: head_dim must be even");
    }
    RopeTables t;
    t.cos = Tensor({seq_len, head_dim / 2}, DType::Float32);
    t.sin = Tensor({seq_len, head_dim / 2}, DType::Float32);
    fill_tables(t.cos, t.sin, seq_len, head_dim, start_pos, theta_base);
    return t;
}

void rope_inplace_with_tables(Tensor& x, const RopeTables& tables) {
    check_rope_input(x);
    const int64_t seq = x.shape()[0];
    const int64_t head_dim = x.shape().back();
    const int64_t half = head_dim / 2;

    if (tables.cos.shape()[0] != seq || tables.cos.shape()[1] != half) {
        throw std::runtime_error("rope: tables shape mismatch");
    }

    // Outer count = product of all dims except the first and last.
    int64_t outer = 1;
    for (std::size_t i = 1; i + 1 < x.shape().size(); ++i) {
        outer *= x.shape()[i];
    }

    const float* cp = tables.cos.data_float();
    const float* sp = tables.sin.data_float();
    float*       xp = x.data_float();

    for (int64_t p = 0; p < seq; ++p) {
        const float* cos_row = cp + p * half;
        const float* sin_row = sp + p * half;
        for (int64_t h = 0; h < outer; ++h) {
            // Pointer to row p, head h.
            float* row = xp + (p * outer + h) * head_dim;
            // For each pair (i, i+half):
            //   new_lo = row[i]    * cos - row[i+half] * sin
            //   new_hi = row[i]    * sin + row[i+half] * cos
            for (int64_t i = 0; i < half; ++i) {
                float x_lo = row[i];
                float x_hi = row[i + half];
                float c    = cos_row[i];
                float s    = sin_row[i];
                row[i]      = x_lo * c - x_hi * s;
                row[i+half] = x_lo * s + x_hi * c;
            }
        }
    }
}

void rope_inplace(Tensor& x, int64_t start_pos, float theta_base) {
    check_rope_input(x);
    int64_t seq = x.shape()[0];
    int64_t head_dim = x.shape().back();
    auto t = precompute_rope_tables(seq, head_dim, start_pos, theta_base);
    rope_inplace_with_tables(x, t);
}

}  // namespace tinyllm