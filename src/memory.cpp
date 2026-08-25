// src/memory.cpp
// -----------------------------------------------------------------------------
// Ref-counted aligned storage for tensors.
//
// Layout of one allocation (all in one posix_memalign'd block):
//
//   [ padding | Storage object (header_size bytes) | aligned user data ]
//
// The Storage object is placement-new'd at the start. The user-visible pointer
// (data()) sits at the first address that satisfies the requested alignment.
// On destruction, we run the Storage destructor and free() the original raw
// pointer.
//
// We embed the header + data in a single allocation to:
//   1. keep the header in cache when we touch the data,
//   2. save one malloc/free pair per tensor,
//   3. guarantee that data_ has a well-defined lifetime tied to the Storage.
// -----------------------------------------------------------------------------
#include "tinyllm/memory.hpp"

#include <cstdlib>
#include <new>
#include <stdexcept>

#if defined(_WIN32)
  #include <malloc.h>
#endif

namespace tinyllm {

namespace {

// Layout hint: Storage holds 3 pointers. We round sizeof(Storage) up to a
// multiple of alignof(void*) just to keep things tidy; the real alignment of
// the data region is whatever was requested.
constexpr std::size_t kHeaderAlign = alignof(void*);

inline std::size_t align_up(std::size_t x, std::size_t a) noexcept {
    return (x + a - 1) & ~(a - 1);
}

}  // namespace

Storage::Storage(void* data, std::size_t n_bytes, std::size_t alignment) noexcept
    : data_(data), n_bytes_(n_bytes), alignment_(alignment) {}

StoragePtr Storage::allocate(std::size_t n_bytes, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Storage::allocate: alignment must be a power of two");
    }
    // Allow zero-byte allocations; give them 1 byte so data() is non-null and
    // we can still index (just don't actually write).
    if (n_bytes == 0) n_bytes = 1;

    const std::size_t header_size = align_up(sizeof(Storage), kHeaderAlign);
    const std::size_t offset      = align_up(header_size, alignment);
    const std::size_t total_bytes = align_up(offset + n_bytes, alignment);

    void* raw = nullptr;
#if defined(_WIN32)
    raw = _aligned_malloc(total_bytes, alignment);
#else
    if (posix_memalign(&raw, alignment, total_bytes) != 0) {
        raw = nullptr;
    }
#endif
    if (!raw) {
        throw std::bad_alloc();
    }

    void*       data_ptr = static_cast<char*>(raw) + offset;
    Storage*    storage  = new (raw) Storage(data_ptr, n_bytes, alignment);

    // Stateful deleter captures the original raw allocation so we can free() it.
    auto deleter = [raw](Storage* p) noexcept {
        if (!p) return;
        p->~Storage();
#if defined(_WIN32)
        _aligned_free(raw);
#else
        std::free(raw);
#endif
    };
    return std::shared_ptr<Storage>(storage, deleter);
}

namespace detail {
void AlignedDeleter::operator()(Storage* /*p*/) const noexcept {
    // Reserved for future use; Storage::allocate builds its own stateful deleter.
}
}  // namespace detail

}  // namespace tinyllm