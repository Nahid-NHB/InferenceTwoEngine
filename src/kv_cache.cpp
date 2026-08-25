// src/kv_cache.cpp
#include "tinyllm/kv_cache.hpp"

#include <cstring>
#include <stdexcept>

namespace tinyllm {

KvCache::KvCache(int64_t max_seq, int64_t n_kv_heads, int64_t head_dim) {
    if (max_seq <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("KvCache: invalid dimensions");
    }
    K = Tensor({max_seq, n_kv_heads, head_dim}, DType::Float32);
    V = Tensor({max_seq, n_kv_heads, head_dim}, DType::Float32);
    length = 0;
}

void KvCache::append(const Tensor& K_in, const Tensor& V_in) {
    if (K_in.dtype() != DType::Float32 || V_in.dtype() != DType::Float32) {
        throw std::runtime_error("KvCache::append: K/V must be Float32");
    }
    if (K_in.ndim() != 3 || V_in.ndim() != 3) {
        throw std::runtime_error("KvCache::append: K/V must be 3-D");
    }
    int64_t seq_len    = K_in.shape()[0];
    int64_t n_kv_h     = K_in.shape()[1];
    int64_t head_dim   = K_in.shape()[2];
    if (V_in.shape()[0] != seq_len || V_in.shape()[1] != n_kv_h ||
        V_in.shape()[2] != head_dim) {
        throw std::runtime_error("KvCache::append: K/V shape mismatch");
    }
    if (n_kv_h != K.shape()[1] || head_dim != K.shape()[2]) {
        throw std::runtime_error("KvCache::append: K/V dim mismatch with cache");
    }
    if (length + seq_len > capacity()) {
        throw std::runtime_error("KvCache::append: cache capacity exceeded");
    }

    // Copy row-by-row. K and V have row stride = n_kv_heads * head_dim.
    int64_t row_bytes = head_dim * static_cast<int64_t>(sizeof(float));
    const float* kp = K_in.data_float();
    const float* vp = V_in.data_float();
    float*       Kdst = K.data_float() + length * (K.shape()[1] * K.shape()[2]);
    float*       Vdst = V.data_float() + length * (V.shape()[1] * V.shape()[2]);
    for (int64_t s = 0; s < seq_len; ++s) {
        std::memcpy(Kdst + s * (K.shape()[1] * K.shape()[2]),
                    kp   + s * (n_kv_h * head_dim),
                    static_cast<std::size_t>(row_bytes) *
                        static_cast<std::size_t>(n_kv_h));
        std::memcpy(Vdst + s * (V.shape()[1] * V.shape()[2]),
                    vp   + s * (n_kv_h * head_dim),
                    static_cast<std::size_t>(row_bytes) *
                        static_cast<std::size_t>(n_kv_h));
    }
    length += seq_len;
}

}  // namespace tinyllm