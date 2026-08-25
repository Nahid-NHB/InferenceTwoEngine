// include/tinyllm/memory.hpp
// -----------------------------------------------------------------------------
// Aligned, ref-counted storage for tensors.
//
// We separate "storage" (a buffer with a refcount) from "tensor" (a view into
// a buffer with shape + stride). This lets us slice, transpose and broadcast
// without copying data — multiple tensors can share the same underlying buffer.
//
// Memory is aligned to 64 bytes to support future SIMD paths (AVX-512) without
// any extra work at the call sites.
// -----------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <memory>
#include <utility>

namespace tinyllm {

// Forward declaration; the deleter needs the complete type.
class Storage;

namespace detail {

// Aligned deleter for std::unique_ptr. Uses ::operator delete with the matching
// alignment so we don't leak the alignment promise back to the system allocator.
struct AlignedDeleter {
    void operator()(Storage* p) const noexcept;
};

}  // namespace detail

// Storage owns a heap buffer and is ref-counted via shared_ptr-like semantics.
// We implement our own minimal refcount (instead of std::shared_ptr) because:
//   - we want to embed the control block + buffer in one allocation sometimes
//   - we want explicit alignment guarantees visible in the type
//   - this is a learning project; using std::shared_ptr is also fine.
class Storage {
public:
    // Allocate `n_bytes` aligned to `alignment`.
    static std::shared_ptr<Storage> allocate(std::size_t n_bytes,
                                             std::size_t alignment = kAlignment);

    // Accessors
    void*       data()       noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return n_bytes_; }
    std::size_t alignment() const noexcept { return alignment_; }

    // Alignment used for all tensor allocations. 64 bytes covers AVX-512.
    static constexpr std::size_t kAlignment = 64;

private:
    friend struct detail::AlignedDeleter;
    Storage(void* data, std::size_t n_bytes, std::size_t alignment) noexcept;

    void*        data_;
    std::size_t  n_bytes_;
    std::size_t  alignment_;
};

// Convenience alias used by Tensor. shared_ptr gives us refcounting + custom
// deleter in one line.
using StoragePtr = std::shared_ptr<Storage>;

}  // namespace tinyllm
