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

// Shared per-head kernel. For each head h:
//   - Qslice: [seq_q, head_dim]  (the new queries)
//   - Kslice: [seq_k, head_dim]  (the keys; full history)
//   - Vslice: [seq_k, head_dim]  (the values; full history)
//   - kpos[k]: the absolute position of key k (length seq_k)
//   - qpos[q] = kpos[seq_k - seq_q] + q   (start_pos + q, basically)
// Compute Q @ Kᵀ / sqrt(d), mask with kpos > qpos, softmax, ctx = weights @ V.
//
// `head_out` is [seq_q, head_dim]; receives the per-head context for each q.
// `ctx_buf` is [seq_q, n_heads, head_dim] and is filled by the caller.
void run_head(const AttentionConfig& cfg,
              int64_t seq_q, int64_t seq_k, int64_t head_h,
              const float* Q_all,        // [seq_q, n_heads, head_dim]
              const float* K_all,        // [seq_k, n_heads, head_dim]   (broadcast)
              const float* V_all,        // [seq_k, n_heads, head_dim]   (broadcast)
              const int64_t* kpos,       // [seq_k]
              int64_t start_pos,
              float* ctx_buf) {          // [seq_q, n_heads, head_dim]
    // Extract per-head slices as 2-D [seq, head_dim] tensors so we can
    // reuse ops::matmul. We allocate scratch each call — Phase 9 will
    // fuse this away.
    Tensor Qslice({seq_q, cfg.head_dim}, DType::Float32);
    for (int64_t s = 0; s < seq_q; ++s) {
        std::memcpy(Qslice.data_float() + s * cfg.head_dim,
                    Q_all + (s * cfg.n_heads + head_h) * cfg.head_dim,
                    static_cast<std::size_t>(cfg.head_dim) * sizeof(float));
    }
    Tensor Kslice({seq_k, cfg.head_dim}, DType::Float32);
    for (int64_t s = 0; s < seq_k; ++s) {
        std::memcpy(Kslice.data_float() + s * cfg.head_dim,
                    K_all + (s * cfg.n_heads + head_h) * cfg.head_dim,
                    static_cast<std::size_t>(cfg.head_dim) * sizeof(float));
    }
    Tensor Vslice({seq_k, cfg.head_dim}, DType::Float32);
    for (int64_t s = 0; s < seq_k; ++s) {
        std::memcpy(Vslice.data_float() + s * cfg.head_dim,
                    V_all + (s * cfg.n_heads + head_h) * cfg.head_dim,
                    static_cast<std::size_t>(cfg.head_dim) * sizeof(float));
    }

    Tensor scores = ops::matmul(Qslice, Kslice.transpose());   // [seq_q, seq_k]
    float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    float* sp = scores.data_float();
    // Absolute causal mask: key position kpos[k] > query position
    // (start_pos + q) is masked.
    for (int64_t q = 0; q < seq_q; ++q) {
        int64_t qpos = start_pos + q;
        for (int64_t k = 0; k < seq_k; ++k) {
            float v = sp[q * seq_k + k] * scale;
            sp[q * seq_k + k] = (kpos[k] > qpos) ? NEG_INF : v;
        }
    }
    Tensor weights = ops::softmax(scores, /*axis=*/-1);
    Tensor head_ctx = ops::matmul(weights, Vslice);             // [seq_q, head_dim]

    // Scatter into the [seq_q, n_heads, head_dim] context buffer.
    for (int64_t s = 0; s < seq_q; ++s) {
        std::memcpy(ctx_buf + (s * cfg.n_heads + head_h) * cfg.head_dim,
                    head_ctx.data_float() + s * cfg.head_dim,
                    static_cast<std::size_t>(cfg.head_dim) * sizeof(float));
    }
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

    // 5) Per-head kernel.
    Tensor ctx({seq, cfg.n_heads, cfg.head_dim}, DType::Float32);
    std::vector<int64_t> kpos(seq);
    for (int64_t k = 0; k < seq; ++k) kpos[k] = start_pos + k;
    for (int64_t h = 0; h < cfg.n_heads; ++h) {
        run_head(cfg, seq, seq, h,
                 Q.data_float(), Kb.data_float(), Vb.data_float(),
                 kpos.data(), start_pos, ctx.data_float());
    }

    // 6) Concatenate heads and project.
    Tensor ctx2d({seq, hd}, DType::Float32);
    for (int64_t s = 0; s < seq; ++s) {
        std::memcpy(ctx2d.data_float() + s * hd,
                    ctx.data_float() + s * hd,
                    static_cast<std::size_t>(hd) * sizeof(float));
    }
    return ops::matmul(ctx2d, w.Wo);
}

Tensor attention_forward_cached(const Tensor& x,
                                const AttentionWeights& w,
                                const AttentionConfig& cfg,
                                KvCache& cache,
                                int64_t start_pos) {
    check_config(cfg);
    check_shape(x, w, cfg);
    if (cache.K.shape()[1] != cfg.n_kv_heads ||
        cache.K.shape()[2] != cfg.head_dim) {
        throw std::runtime_error("attention_forward_cached: cache dim mismatch");
    }

    int64_t seq_q = x.shape()[0];
    int64_t hd    = cfg.hidden;

    // 1) Project.
    Tensor Q2 = ops::matmul(x, w.Wq);
    Tensor K2 = ops::matmul(x, w.Wk);
    Tensor V2 = ops::matmul(x, w.Wv);

    // 2) Reshape to 3-D.
    Tensor Q = to_3d(Q2, seq_q, cfg.n_heads,    cfg.head_dim);
    Tensor K = to_3d(K2, seq_q, cfg.n_kv_heads, cfg.head_dim);
    Tensor V = to_3d(V2, seq_q, cfg.n_kv_heads, cfg.head_dim);

    // 3) RoPE on the new Q/K/V slices.
    rope_inplace(Q, start_pos, cfg.theta_base);
    rope_inplace(K, start_pos, cfg.theta_base);

    // 4) Append K, V to the cache, then read back the full K[0:start_pos+seq_q].
    cache.append(K, V);
    int64_t seq_k = cache.length;

    // Slice the cached K and V into a [seq_k, n_kv_heads, head_dim] view
    // (no copy needed — both are contiguous with row stride = n_kv_heads * d).
    Tensor Kcached = cache.K;
    Tensor Vcached = cache.V;

    // 5) Broadcast to n_heads.
    Tensor Kb = broadcast_kv(Kcached, cfg.n_heads, cfg.n_kv_heads);
    Tensor Vb = broadcast_kv(Vcached, cfg.n_heads, cfg.n_kv_heads);

    // 6) Per-head kernel.
    Tensor ctx({seq_q, cfg.n_heads, cfg.head_dim}, DType::Float32);
    std::vector<int64_t> kpos(seq_k);
    for (int64_t k = 0; k < seq_k; ++k) kpos[k] = k;   // absolute positions
    for (int64_t h = 0; h < cfg.n_heads; ++h) {
        run_head(cfg, seq_q, seq_k, h,
                 Q.data_float(), Kb.data_float(), Vb.data_float(),
                 kpos.data(), start_pos, ctx.data_float());
    }

    // 7) Concat heads and project.
    Tensor ctx2d({seq_q, hd}, DType::Float32);
    for (int64_t s = 0; s < seq_q; ++s) {
        std::memcpy(ctx2d.data_float() + s * hd,
                    ctx.data_float() + s * hd,
                    static_cast<std::size_t>(hd) * sizeof(float));
    }
    return ops::matmul(ctx2d, w.Wo);
}

}  // namespace tinyllm