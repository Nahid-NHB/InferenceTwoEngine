// include/tinyllm/kv_cache.hpp
// -----------------------------------------------------------------------------
// Per-layer K/V cache for autoregressive decoding.
//
// Layout: K and V are stored as a single block of two tensors, each of
// shape [max_seq, n_kv_heads, head_dim]. The cache also tracks how many
// positions have been written so far (length_).
//
// The cache is owned by LlamaModel and passed to each layer. Each layer
// has its own cache.
//
// Phase 6 design:
//   - One cache per layer (not shared across layers).
//   - Float32 storage; Phase 8 will quantize the cache values to
//     reduce memory.
//   - We don't drop entries from the cache once written; capacity is
//     fixed at construction. If length_ == max_seq, appending throws.
// -----------------------------------------------------------------------------
#pragma once

#include "tinyllm/tensor.hpp"

#include <cstdint>

namespace tinyllm {

struct KvCache {
    Tensor K;            // [max_seq, n_kv_heads, head_dim]
    Tensor V;            // [max_seq, n_kv_heads, head_dim]
    int64_t length = 0;  // number of positions currently filled

    // Build a cache of the given capacity. head_dim > 0, n_kv_heads > 0,
    // max_seq > 0.
    KvCache() = default;
    KvCache(int64_t max_seq, int64_t n_kv_heads, int64_t head_dim);

    // How many positions are currently filled.
    int64_t size() const noexcept { return length; }

    // How many positions the cache can hold.
    int64_t capacity() const noexcept { return K.shape()[0]; }

    // Append `seq_len` rows of K and V at positions [length, length+seq_len).
    // K_in / V_in shape: [seq_len, n_kv_heads, head_dim].
    // Throws if length + seq_len > capacity().
    void append(const Tensor& K_in, const Tensor& V_in);

    // Reset the cache (start over). Keeps the storage allocated.
    void clear() noexcept { length = 0; }
};

}  // namespace tinyllm