// src/attention.cpp
#include "tinyllm/attention.hpp"

#include "tinyllm/matmul.hpp"
#include "tinyllm/rope.hpp"
#include "tinyllm/tensor.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tinyllm {

namespace {

void check_config(const AttentionConfig& cfg) {
    if (cfg.hidden <= 0 || cfg.n_heads <= 0 || cfg.n_kv_heads <= 0 ||
        cfg.head_dim <= 0) {
        throw std::runtime_error("attention: invalid config");
    }
    if (cfg.n_heads % cfg.n_kv_heads != 0) {
        throw std::runtime_error("attention: n_heads must be a multiple of n_kv_heads");
    }
    if (cfg.hidden != cfg.n_heads * cfg.head_dim) {
        throw std::runtime_error("attention: hidden must equal n_heads * head_dim");
    }
}

void check_shape(const Tensor& x, const AttentionWeights& w,
                 const AttentionConfig& cfg) {
    if (x.dtype() != DType::Float32) {
        throw std::runtime_error("attention: x must be Float32");
    }
    if (x.ndim() != 2 || x.shape()[1] != cfg.hidden) {
        throw std::runtime_error("attention: x must be [seq, hidden]");
    }
    int64_t hd = cfg.hidden;
    int64_t kvd = cfg.n_kv_heads * cfg.head_dim;
    if (w.Wq.shape()[0] != hd || w.Wq.shape()[1] != hd) {
        throw std::runtime_error("attention: Wq shape mismatch");
    }
    if (w.Wk.shape()[0] != hd || w.Wk.shape()[1] != kvd) {
        throw std::runtime_error("attention: Wk shape mismatch");
    }
    if (w.Wv.shape()[0] != hd || w.Wv.shape()[1] != kvd) {
        throw std::runtime_error("attention: Wv shape mismatch");
    }
    if (w.Wo.shape()[0] != hd || w.Wo.shape()[1] != hd) {
        throw std::runtime_error("attention: Wo shape mismatch");
    }
}

// Reshape a [seq, n*D] tensor to [seq, n, D] by returning a NEW contiguous
// tensor with that shape (no copy-in-place reinterpret since strides differ).
Tensor to_3d(const Tensor& x, int64_t seq, int64_t n, int64_t d) {
    return x.reshape({seq, n, d}).contiguous();
}

// Expand K or V from [seq, n_kv_heads, head_dim] to [seq, n_heads, head_dim]
// by repeating each kv head (n_heads / n_kv_heads) times.
Tensor broadcast_kv(const Tensor& kv, int64_t n_heads, int64_t n_kv_heads) {
    int64_t seq = kv.shape()[0];
    int64_t head_dim = kv.shape()[2];
    int64_t repeat = n_heads / n_kv_heads;
    Tensor out({seq, n_heads, head_dim}, DType::Float32);
    const float* kp = kv.data_float();
    float*       op = out.data_float();
    for (int64_t s = 0; s < seq; ++s) {
        for (int64_t kh = 0; kh < n_kv_heads; ++kh) {
            for (int64_t r = 0; r < repeat; ++r) {
                std::memcpy(op + ((s * n_heads + kh * repeat + r) * head_dim),
                            kp + ((s * n_kv_heads + kh) * head_dim),
                            static_cast<std::size_t>(head_dim) * sizeof(float));
            }
        }
    }
    return out;
}

}  // namespace

Tensor attention_forward(const Tensor& x,
                         const AttentionWeights& w,
                         const AttentionConfig& cfg,
                         int64_t start_pos) {
    check_config(cfg);
    check_shape(x, w, cfg);

    int64_t seq = x.shape()[0];
    int64_t hd  = cfg.hidden;

    // 1) Project.
    Tensor Q2 = ops::matmul(x, w.Wq);   // [seq, n_heads   * head_dim]
    Tensor K2 = ops::matmul(x, w.Wk);   // [seq, n_kv_heads * head_dim]
    Tensor V2 = ops::matmul(x, w.Wv);   // [seq, n_kv_heads * head_dim]

    // 2) Reshape to 3-D.
    Tensor Q = to_3d(Q2, seq, cfg.n_heads,    cfg.head_dim);
    Tensor K = to_3d(K2, seq, cfg.n_kv_heads, cfg.head_dim);
    Tensor V = to_3d(V2, seq, cfg.n_kv_heads, cfg.head_dim);

    // 3) RoPE.
    rope_inplace(Q, start_pos, cfg.theta_base);
    rope_inplace(K, start_pos, cfg.theta_base);

    // 4) Broadcast K, V to n_heads.
    Tensor Kb = broadcast_kv(K, cfg.n_heads, cfg.n_kv_heads);
    Tensor Vb = broadcast_kv(V, cfg.n_heads, cfg.n_kv_heads);

    // 5) For each head: scores = Q_h @ K_hᵀ / sqrt(d), softmax with causal
    //    mask, ctx_h = weights @ V_h.
    float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
    Tensor ctx({seq, cfg.n_heads, cfg.head_dim}, DType::Float32);
    float* ctxp = ctx.data_float();

    // We could batch heads into one big matmul, but for clarity (and to
    // avoid transpose gymnastics) we do per-head here. Phase 9 will
    // vectorize across heads.
    for (int64_t h = 0; h < cfg.n_heads; ++h) {
        // Slice Q_h : [seq, head_dim]
        Tensor Qslice({seq, cfg.head_dim}, DType::Float32);
        for (int64_t s = 0; s < seq; ++s) {
            std::memcpy(Qslice.data_float() + s * cfg.head_dim,
                        Q.data_float() + (s * cfg.n_heads + h) * cfg.head_dim,
                        static_cast<std::size_t>(cfg.head_dim) * sizeof(float));
        }
        // Slice K_h : [seq, head_dim]
        Tensor Kslice({seq, cfg.head_dim}, DType::Float32);
        for (int64_t s = 0; s < seq; ++s) {
            std::memcpy(Kslice.data_float() + s * cfg.head_dim,
                        Kb.data_float() + (s * cfg.n_heads + h) * cfg.head_dim,
                        static_cast<std::size_t>(cfg.head_dim) * sizeof(float));
        }
        Tensor Vslice({seq, cfg.head_dim}, DType::Float32);
        for (int64_t s = 0; s < seq; ++s) {
            std::memcpy(Vslice.data_float() + s * cfg.head_dim,
                        Vb.data_float() + (s * cfg.n_heads + h) * cfg.head_dim,
                        static_cast<std::size_t>(cfg.head_dim) * sizeof(float));
        }

        // Ksliceᵀ : [head_dim, seq]
        Tensor Kt = Kslice.transpose();
        // scores : [seq, seq]
        Tensor scores = ops::matmul(Qslice, Kt);
        // scale + causal mask
        const float NEG_INF = -std::numeric_limits<float>::infinity();
        float* sp = scores.data_float();
        for (int64_t q = 0; q < seq; ++q) {
            for (int64_t kp_idx = 0; kp_idx < seq; ++kp_idx) {
                sp[q * seq + kp_idx] *= scale;
                // For prefill (seq_q == seq_k) we mask positions where
                // kp_idx > q (the key is "in the future"). In Phase 6 with
                // a real KV cache we'll generalize to absolute positions.
                if (kp_idx > q) {
                    sp[q * seq + kp_idx] = NEG_INF;
                }
            }
        }
        Tensor weights = ops::softmax(scores, /*axis=*/-1);

        // ctx_h = weights @ Vslice : [seq, head_dim]
        Tensor head_ctx = ops::matmul(weights, Vslice);
        for (int64_t s = 0; s < seq; ++s) {
            std::memcpy(ctxp + (s * cfg.n_heads + h) * cfg.head_dim,
                        head_ctx.data_float() + s * cfg.head_dim,
                        static_cast<std::size_t>(cfg.head_dim) * sizeof(float));
        }
    }

    // 6) Concatenate heads: [seq, n_heads * head_dim] = [seq, hidden].
    Tensor ctx2d({seq, hd}, DType::Float32);
    for (int64_t s = 0; s < seq; ++s) {
        std::memcpy(ctx2d.data_float() + s * hd,
                    ctxp + s * hd,
                    static_cast<std::size_t>(hd) * sizeof(float));
    }

    // 7) Output projection.
    return ops::matmul(ctx2d, w.Wo);
}

}  // namespace tinyllm