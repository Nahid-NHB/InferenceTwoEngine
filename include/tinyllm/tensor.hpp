// include/tinyllm/tensor.hpp
// -----------------------------------------------------------------------------
// Tensor: a multi-dimensional array view over a ref-counted, aligned storage.
//
// Key ideas
// ---------
//  * Row-major (C-style) layout by default. The last dimension is the
//    fastest-varying one in memory.
//  * `shape`   is a vector<int64_t> giving the size along each axis.
//  * `strides` is a vector<int64_t> giving the step (in ELEMENTS, not bytes)
//    along each axis. For a contiguous tensor, stride[i] = product(shape[i+1:]).
//  * `storage` is a shared_ptr to a heap buffer. Multiple Tensors can share
//    the same storage (e.g. views, slices, transposes) — no copy.
//  * `offset`  is the element offset into storage where this Tensor starts.
//    (We keep it as a separate field rather than baking it into data_ because
//    it keeps storage reusable for many views.)
//
// Views do NOT copy. Operations that allocate new memory (e.g. add, matmul)
// return a brand-new Tensor with its own storage.
//
// Data types supported in Phase 1: float32, int32.
// -----------------------------------------------------------------------------
#pragma once

#include "tinyllm/memory.hpp"

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tinyllm {

// -----------------------------------------------------------------------------
// Data type
// -----------------------------------------------------------------------------
enum class DType : uint8_t {
    Unknown = 0,
    Float32,
    Int32,
    Float16,  // stored as IEEE-754 binary16; reserved for Phase 4
};

std::string_view dtype_name(DType dt) noexcept;
std::size_t      dtype_sizeof(DType dt) noexcept;  // bytes per element

// -----------------------------------------------------------------------------
// Tensor
// -----------------------------------------------------------------------------
class Tensor {
public:
    // ---- Constructors ------------------------------------------------------
    Tensor();  // empty scalar (0-d, 1 element) — useful default

    // Allocate a tensor with given shape (default F32, contiguous row-major).
    explicit Tensor(std::vector<int64_t> shape, DType dtype = DType::Float32);

    // Allocate from raw data (copies into a new buffer).
    Tensor(std::vector<int64_t> shape, DType dtype,
           const void* data, std::size_t n_bytes);

    // Wrap existing storage as a view. Used internally by views/transposes.
    Tensor(StoragePtr storage, std::vector<int64_t> shape,
           std::vector<int64_t> strides, int64_t offset, DType dtype);

    // Convenience: 1-D, 2-D, 3-D ctors
    explicit Tensor(std::initializer_list<int64_t> shape, DType dtype = DType::Float32);

    // ---- Accessors ---------------------------------------------------------
    const std::vector<int64_t>& shape()   const noexcept { return shape_; }
    const std::vector<int64_t>& strides() const noexcept { return strides_; }
    int64_t                     offset()  const noexcept { return offset_; }
    DType                       dtype()   const noexcept { return dtype_; }
    std::size_t                 ndim()    const noexcept { return shape_.size(); }
    int64_t                     numel()   const noexcept { return numel_; }
    std::size_t                 nbytes()  const noexcept {
        return static_cast<std::size_t>(numel_) * dtype_sizeof(dtype_);
    }
    const StoragePtr& storage()  const noexcept { return storage_; }

    // Element count along a single axis (or 1 if axis is out of range).
    int64_t dim(std::ptrdiff_t axis) const noexcept;

    // ---- Typed data pointers ----------------------------------------------
    // Valid only when dtype matches. Returns offset-applied pointer.
    float*       data_float()       noexcept;
    const float* data_float() const noexcept;
    int32_t*     data_int()       noexcept;
    const int32_t* data_int() const noexcept;

    // Raw pointer for generic access (e.g. memcpy).
    void*       raw()       noexcept { return storage_->data(); }
    const void* raw() const noexcept { return storage_->data(); }

    // ---- Views (zero-copy) --------------------------------------------------
    // Transpose two axes (default: full reverse). Returns a non-contiguous view.
    Tensor transpose(std::vector<int64_t> perm = {}) const;

    // Reshape to a new shape. Requires that the tensor is contiguous.
    Tensor reshape(std::vector<int64_t> new_shape) const;

    // Make this tensor contiguous (copy if needed).
    Tensor contiguous() const;

    bool is_contiguous() const noexcept;

    // Squeeze / unsqueeze
    Tensor squeeze(std::optional<int64_t> axis = std::nullopt) const;
    Tensor unsqueeze(int64_t axis) const;

    // Element access (1-D indexing in row-major order of the *contiguous*
    // representation). For non-contiguous tensors use at_multi(...) or read
    // via data_float() + strides.
    float&       at_flat(int64_t i)       noexcept;
    const float& at_flat(int64_t i) const noexcept;

    // Multi-dimensional element access. Args are integer coordinates along
    // each axis (row-major). Returns the value at that logical coordinate.
    float at(std::initializer_list<int64_t> coords) const;

    // Fill with a constant.
    void fill(float value) noexcept;

    // Pretty-print.
    std::string to_string() const;

private:
    // Recompute numel_ from shape_; validate that strides make sense.
    void recompute_numel();
    // Default contiguous strides for the current shape.
    static std::vector<int64_t> default_strides(const std::vector<int64_t>& shape);

    StoragePtr             storage_;
    std::vector<int64_t>   shape_;
    std::vector<int64_t>   strides_;
    int64_t                offset_ = 0;
    DType                  dtype_  = DType::Float32;
    int64_t                numel_  = 0;
};

// -----------------------------------------------------------------------------
// Element-wise / reduction ops
// -----------------------------------------------------------------------------
namespace ops {

Tensor add      (const Tensor& a, const Tensor& b);
Tensor subtract (const Tensor& a, const Tensor& b);
Tensor multiply (const Tensor& a, const Tensor& b);
Tensor divide   (const Tensor& a, const Tensor& b);

// Scalar variants (broadcast scalar across tensor).
Tensor add_scalar(const Tensor& a, float s);
Tensor mul_scalar(const Tensor& a, float s);

Tensor sum    (const Tensor& a, std::optional<int64_t> axis = std::nullopt,
               bool keepdims = false);
Tensor mean   (const Tensor& a, std::optional<int64_t> axis = std::nullopt,
               bool keepdims = false);

// In-place softmax along an axis (numerically stable).
Tensor softmax(const Tensor& a, int64_t axis = -1);

// Matrix multiply: 2-D x 2-D → 2-D. Defined in matmul.hpp (Phase 2).
//   #include "tinyllm/matmul.hpp" to use it.

// Reshape helpers (also exposed as Tensor methods).
Tensor reshape (const Tensor& a, std::vector<int64_t> shape);
Tensor transpose(const Tensor& a, std::vector<int64_t> perm = {});

// Broadcasting: returns a tensor with the broadcasted shape.
Tensor broadcast_to(const Tensor& a, std::vector<int64_t> target_shape);

}  // namespace ops

// -----------------------------------------------------------------------------
// Element-wise operators for ergonomics.
// -----------------------------------------------------------------------------
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);

}  // namespace tinyllm