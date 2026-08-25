// src/tensor.cpp
// -----------------------------------------------------------------------------
// Tensor + elementwise / matmul / softmax implementations.
//
// Numerical notes
// ---------------
//   * softmax uses the max-subtraction trick for numerical stability.
//   * matmul starts as naive O(n^3) — we optimize in Phase 2 with blocking +
//     SIMD + threading, but never without a correctness benchmark first.
// -----------------------------------------------------------------------------
#include "tinyllm/tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace tinyllm {

// -----------------------------------------------------------------------------
// DType helpers
// -----------------------------------------------------------------------------
std::string_view dtype_name(DType dt) noexcept {
    switch (dt) {
        case DType::Float32: return "float32";
        case DType::Int32:   return "int32";
        case DType::Float16: return "float16";
        default:             return "unknown";
    }
}
std::size_t dtype_sizeof(DType dt) noexcept {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Int32:   return 4;
        case DType::Float16: return 2;
        default:             return 0;
    }
}

// -----------------------------------------------------------------------------
// Tensor construction
// -----------------------------------------------------------------------------
Tensor::Tensor() : Tensor(std::vector<int64_t>{}, DType::Float32) {}

std::vector<int64_t> Tensor::default_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> s(shape.size());
    int64_t acc = 1;
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i) {
        s[i] = acc;
        acc *= shape[i];
    }
    return s;
}

void Tensor::recompute_numel() {
    int64_t n = 1;
    for (int64_t d : shape_) n *= d;
    numel_ = n;
}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype)
    : shape_(std::move(shape)), strides_(default_strides(shape_)),
      offset_(0), dtype_(dtype) {
    recompute_numel();
    storage_ = Storage::allocate(static_cast<std::size_t>(numel_) * dtype_sizeof(dtype_),
                                 Storage::kAlignment);
}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype,
               const void* data, std::size_t n_bytes)
    : Tensor(std::move(shape), dtype) {
    if (data == nullptr) {
        throw std::invalid_argument("Tensor ctor: data pointer is null");
    }
    const std::size_t expected = static_cast<std::size_t>(numel_) * dtype_sizeof(dtype_);
    if (n_bytes < expected) {
        throw std::invalid_argument("Tensor ctor: data buffer too small");
    }
    std::memcpy(storage_->data(), data, expected);
}

Tensor::Tensor(std::initializer_list<int64_t> shape, DType dtype)
    : Tensor(std::vector<int64_t>(shape), dtype) {}

Tensor::Tensor(StoragePtr storage, std::vector<int64_t> shape,
               std::vector<int64_t> strides, int64_t offset, DType dtype)
    : storage_(std::move(storage)), shape_(std::move(shape)),
      strides_(std::move(strides)), offset_(offset), dtype_(dtype) {
    if (shape_.size() != strides_.size()) {
        throw std::invalid_argument("Tensor ctor: shape/stride rank mismatch");
    }
    recompute_numel();
}

// -----------------------------------------------------------------------------
// Accessors
// -----------------------------------------------------------------------------
int64_t Tensor::dim(std::ptrdiff_t axis) const noexcept {
    auto n = static_cast<std::ptrdiff_t>(shape_.size());
    if (axis < 0) axis += n;
    if (axis < 0 || axis >= n) return 1;
    return shape_[axis];
}

float* Tensor::data_float() noexcept {
    return reinterpret_cast<float*>(storage_->data()) + offset_;
}
const float* Tensor::data_float() const noexcept {
    return reinterpret_cast<const float*>(storage_->data()) + offset_;
}
int32_t* Tensor::data_int() noexcept {
    return reinterpret_cast<int32_t*>(storage_->data()) + offset_;
}
const int32_t* Tensor::data_int() const noexcept {
    return reinterpret_cast<const int32_t*>(storage_->data()) + offset_;
}

float& Tensor::at_flat(int64_t i) noexcept {
    assert(i >= 0 && i < numel_);
    return data_float()[i];
}
const float& Tensor::at_flat(int64_t i) const noexcept {
    assert(i >= 0 && i < numel_);
    return data_float()[i];
}

float Tensor::at(std::initializer_list<int64_t> coords) const {
    if (coords.size() != shape_.size()) {
        throw std::invalid_argument("at: coordinate rank mismatch");
    }
    int64_t off = 0;
    std::size_t d = 0;
    for (int64_t c : coords) {
        if (c < 0 || c >= shape_[d]) {
            throw std::out_of_range("at: coordinate out of range");
        }
        off += c * strides_[d];
        ++d;
    }
    return data_float()[off];
}

void Tensor::fill(float value) noexcept {
    float* p = data_float();
    for (int64_t i = 0; i < numel_; ++i) p[i] = value;
}

// -----------------------------------------------------------------------------
// Contiguity / views
// -----------------------------------------------------------------------------
bool Tensor::is_contiguous() const noexcept {
    int64_t expected = 1;
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape_.size()) - 1; i >= 0; --i) {
        if (shape_[i] == 1) continue;            // broadcast dim doesn't matter
        if (strides_[i] != expected) return false;
        expected *= shape_[i];
    }
    return true;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;            // already contiguous: share
    Tensor out(shape_, dtype_);                   // allocate a fresh contiguous buffer
    const int64_t N = numel_;
    const float* src_base = data_float();
    float*       dst_base = out.data_float();
    std::vector<int64_t> coord(shape_.size(), 0);
    for (int64_t i = 0; i < N; ++i) {
        int64_t src_off = 0;
        for (std::size_t d = 0; d < shape_.size(); ++d) {
            src_off += coord[d] * strides_[d];
        }
        dst_base[i] = src_base[src_off];
        ++coord[coord.size() - 1];
        for (std::ptrdiff_t k = static_cast<std::ptrdiff_t>(coord.size()) - 1; k > 0; --k) {
            auto ku  = static_cast<std::size_t>(k);
            auto ku1 = static_cast<std::size_t>(k - 1);
            if (coord[ku] >= shape_[ku]) { coord[ku] = 0; ++coord[ku1]; } else break;
        }
    }
    return out;
}

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
    int64_t new_numel = 1;
    for (int64_t d : new_shape) new_numel *= d;
    if (new_numel != numel_) {
        throw std::invalid_argument("reshape: total element count mismatch");
    }
    if (!is_contiguous()) {
        // Make it contiguous first.
        Tensor c = contiguous();
        return Tensor(c.storage_, std::move(new_shape),
                     default_strides(new_shape), 0, dtype_);
    }
    return Tensor(storage_, std::move(new_shape),
                  default_strides(new_shape), offset_, dtype_);
}

Tensor Tensor::transpose(std::vector<int64_t> perm) const {
    if (perm.empty()) {
        perm.resize(shape_.size());
        for (std::size_t i = 0; i < shape_.size(); ++i) {
            perm[i] = static_cast<int64_t>(shape_.size() - 1 - i);
        }
    }
    if (perm.size() != shape_.size()) {
        throw std::invalid_argument("transpose: rank mismatch");
    }
    std::vector<int64_t> new_shape(shape_.size());
    std::vector<int64_t> new_strides(shape_.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        new_shape[i]   = shape_[perm[i]];
        new_strides[i] = strides_[perm[i]];
    }
    return Tensor(storage_, std::move(new_shape), std::move(new_strides),
                  offset_, dtype_);
}

Tensor Tensor::squeeze(std::optional<int64_t> axis) const {
    std::vector<int64_t> ns, nst;
    if (axis) {
        int64_t ax = *axis;
        if (ax < 0) ax += static_cast<int64_t>(shape_.size());
        for (std::size_t i = 0; i < shape_.size(); ++i) {
            if (static_cast<int64_t>(i) == ax) {
                if (shape_[i] != 1) {
                    throw std::invalid_argument("squeeze: dim != 1");
                }
                continue;
            }
            ns.push_back(shape_[i]);
            nst.push_back(strides_[i]);
        }
    } else {
        for (std::size_t i = 0; i < shape_.size(); ++i) {
            if (shape_[i] == 1) continue;
            ns.push_back(shape_[i]);
            nst.push_back(strides_[i]);
        }
    }
    return Tensor(storage_, std::move(ns), std::move(nst), offset_, dtype_);
}

Tensor Tensor::unsqueeze(int64_t axis) const {
    if (axis < 0) axis += static_cast<int64_t>(shape_.size()) + 1;
    std::vector<int64_t> ns(shape_.size() + 1);
    std::vector<int64_t> nst(shape_.size() + 1);
    for (std::size_t i = 0; i < ns.size(); ++i) {
        if (static_cast<int64_t>(i) < axis) {
            ns[i]  = shape_[i];
            nst[i] = strides_[i];
        } else if (static_cast<int64_t>(i) == axis) {
            ns[i]  = 1;
            nst[i] = 0;
        } else {
            ns[i]  = shape_[i - 1];
            nst[i] = strides_[i - 1];
        }
    }
    return Tensor(storage_, std::move(ns), std::move(nst), offset_, dtype_);
}

// -----------------------------------------------------------------------------
// Pretty printing
// -----------------------------------------------------------------------------
namespace {
void print_recursive(std::ostringstream& os, const float* base,
                     const std::vector<int64_t>& shape,
                     const std::vector<int64_t>& strides,
                     int64_t offset, std::size_t dim, int indent) {
    if (dim == shape.size() - 1) {
        os << "[";
        for (int64_t i = 0; i < shape[dim]; ++i) {
            if (i) os << ", ";
            os << std::setprecision(6) << base[offset + i * strides[dim]];
        }
        os << "]";
        return;
    }
    os << "[";
    for (int64_t i = 0; i < shape[dim]; ++i) {
        if (i) {
            os << ",\n";
            os << std::string(indent + 1, ' ');
        }
        print_recursive(os, base, shape, strides, offset + i * strides[dim],
                        dim + 1, indent + 1);
    }
    os << "]";
}
}  // namespace

std::string Tensor::to_string() const {
    if (numel_ == 0) return "[]";
    std::ostringstream os;
    os << "Tensor<" << dtype_name(dtype_) << ">(";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (i) os << "x";
        os << shape_[i];
    }
    os << ")\n";
    if (dtype_ != DType::Float32) {
        os << "  <non-float32: " << numel_ << " elements>\n";
        return os.str();
    }
    if (!is_contiguous()) {
        // Force a contiguous copy for readable printing.
        Tensor c = contiguous();
        print_recursive(os, c.data_float(), c.shape_, c.strides_, 0, 0, 2);
        os << "\n";
        return os.str();
    }
    print_recursive(os, data_float(), shape_, strides_, 0, 0, 2);
    os << "\n";
    return os.str();
}

// =============================================================================
// ops:: implementations
// -----------------------------------------------------------------------------
// We implement broadcasting carefully: two tensors can have different shapes if
// they are broadcast-compatible (same shape from the right, or one of them ==1).
// For Phase 1 we support broadcasts up to N-D by computing the broadcast shape
// up front and then iterating with per-tensor coord → offset conversion.
// =============================================================================
namespace ops {

namespace {

// Compute the broadcast shape of two shapes (right-align, dim == 1 broadcasts).
std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                     const std::vector<int64_t>& b) {
    std::vector<int64_t> out;
    out.reserve(std::max(a.size(), b.size()));
    std::size_t n = std::max(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        int64_t da = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        int64_t db = (i < b.size()) ? b[b.size() - 1 - i] : 1;
        if (da == db || da == 1 || db == 1) {
            out.push_back(std::max(da, db));
        } else {
            throw std::invalid_argument("broadcast: shapes incompatible");
        }
    }
    std::reverse(out.begin(), out.end());
    return out;
}

// Given broadcast shape and the smaller shape, return strides that match the
// broadcast shape (with stride 0 for broadcast dims).
std::vector<int64_t> broadcast_strides(const std::vector<int64_t>& s_shape,
                                       const std::vector<int64_t>& s_strides,
                                       const std::vector<int64_t>& out_shape) {
    std::vector<int64_t> out(out_shape.size(), 0);
    std::ptrdiff_t pad = static_cast<std::ptrdiff_t>(out_shape.size()) -
                         static_cast<std::ptrdiff_t>(s_shape.size());
    for (std::size_t i = 0; i < s_shape.size(); ++i) {
        std::size_t oi = static_cast<std::size_t>(pad + i);
        out[oi] = (s_shape[i] == 1) ? 0 : s_strides[i];
    }
    return out;
}

template <typename Op>
Tensor elementwise(const Tensor& a, const Tensor& b, Op op) {
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("elementwise: dtype mismatch");
    }
    auto shape = broadcast_shape(a.shape(), b.shape());
    auto a_strides = broadcast_strides(a.shape(), a.strides(), shape);
    auto b_strides = broadcast_strides(b.shape(), b.strides(), shape);
    int64_t n = 1; for (int64_t d : shape) n *= d;
    Tensor out(shape, a.dtype());
    const float* ap = a.data_float();
    const float* bp = b.data_float();
    float*       op_ = out.data_float();
    std::vector<int64_t> coord(shape.size(), 0);
    for (int64_t i = 0; i < n; ++i) {
        int64_t ao = 0, bo = 0;
        for (std::size_t d = 0; d < shape.size(); ++d) {
            ao += coord[d] * a_strides[d];
            bo += coord[d] * b_strides[d];
        }
        op_[i] = op(ap[ao], bp[bo]);
        ++coord[coord.size() - 1];
        for (std::ptrdiff_t k = static_cast<std::ptrdiff_t>(coord.size()) - 1; k > 0; --k) {
            auto ku = static_cast<std::size_t>(k);
            auto ku1 = static_cast<std::size_t>(k - 1);
            if (coord[ku] >= shape[ku]) { coord[ku] = 0; ++coord[ku1]; } else break;
        }
    }
    return out;
}

struct AddOp      { float operator()(float x, float y) const noexcept { return x + y; } };
struct SubOp      { float operator()(float x, float y) const noexcept { return x - y; } };
struct MulOp      { float operator()(float x, float y) const noexcept { return x * y; } };
struct DivOp      { float operator()(float x, float y) const noexcept { return x / y; } };

}  // namespace

Tensor add      (const Tensor& a, const Tensor& b) { return elementwise(a, b, AddOp{}); }
Tensor subtract (const Tensor& a, const Tensor& b) { return elementwise(a, b, SubOp{}); }
Tensor multiply (const Tensor& a, const Tensor& b) { return elementwise(a, b, MulOp{}); }
Tensor divide   (const Tensor& a, const Tensor& b) { return elementwise(a, b, DivOp{}); }

Tensor add_scalar(const Tensor& a, float s) {
    Tensor out(a.shape(), a.dtype());
    const float* ap = a.data_float();
    float* op_ = out.data_float();
    for (int64_t i = 0; i < a.numel(); ++i) op_[i] = ap[i] + s;
    return out;
}
Tensor mul_scalar(const Tensor& a, float s) {
    Tensor out(a.shape(), a.dtype());
    const float* ap = a.data_float();
    float* op_ = out.data_float();
    for (int64_t i = 0; i < a.numel(); ++i) op_[i] = ap[i] * s;
    return out;
}

// -----------------------------------------------------------------------------
// Reduction (sum / mean)
// -----------------------------------------------------------------------------
namespace {

Tensor reduce(const Tensor& a, std::optional<int64_t> axis, bool keepdims, bool avg) {
    if (a.dtype() != DType::Float32) {
        throw std::invalid_argument("reduce: only float32 supported in Phase 1");
    }
    if (!axis) {
        // full reduction → scalar tensor
        const float* p = a.data_float();
        double s = 0.0;
        for (int64_t i = 0; i < a.numel(); ++i) s += static_cast<double>(p[i]);
        if (avg) s /= static_cast<double>(a.numel());
        Tensor out(std::vector<int64_t>{}, DType::Float32);
        out.data_float()[0] = static_cast<float>(s);
        return out;
    }
    int64_t ax = *axis;
    if (ax < 0) ax += static_cast<int64_t>(a.shape().size());
    if (ax < 0 || ax >= static_cast<int64_t>(a.shape().size())) {
        throw std::invalid_argument("reduce: axis out of range");
    }

    // Compute output shape (collapse the reduced dim).
    std::vector<int64_t> out_shape;
    int64_t inner = 1, outer = 1, dim_size = a.shape()[ax];
    for (std::size_t i = 0; i < a.shape().size(); ++i) {
        int64_t s = a.shape()[i];
        if (static_cast<int64_t>(i) < ax) outer *= s;
        else if (static_cast<int64_t>(i) == ax) { dim_size = s; }
        else inner *= s;
    }
    for (std::size_t i = 0; i < a.shape().size(); ++i) {
        if (static_cast<int64_t>(i) == ax) {
            if (keepdims) out_shape.push_back(1);
        } else out_shape.push_back(a.shape()[i]);
    }

    // Make a contiguous copy of `a` so we can use stride arithmetic.
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    const float* p = ac.data_float();
    const std::vector<int64_t>& sh = ac.shape();
    const std::vector<int64_t>& st = ac.strides();

    Tensor out(out_shape, DType::Float32);
    float* op = out.data_float();

    // Iterate over (outer, inner) and accumulate over `dim_size`.
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            double s = 0.0;
            for (int64_t k = 0; k < dim_size; ++k) {
                int64_t idx = 0;
                // Compute multi-dim index from (o, k, i).
                // (We could precompute linear outer stride; keep it general.)
                int64_t oo = o, ii = i, kk = k;
                for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(sh.size()) - 1; d >= 0; --d) {
                    int64_t v;
                    if (static_cast<int64_t>(d) < ax) { v = oo % sh[d]; oo /= sh[d]; }
                    else if (static_cast<int64_t>(d) == ax) { v = kk; kk = 0; }
                    else { v = ii % sh[d]; ii /= sh[d]; }
                    idx += v * st[d];
                }
                s += static_cast<double>(p[idx]);
            }
            if (avg) s /= static_cast<double>(dim_size);
            op[o * inner + i] = static_cast<float>(s);
        }
    }
    return out;
}

}  // namespace

Tensor sum (const Tensor& a, std::optional<int64_t> axis, bool keepdims) {
    return reduce(a, axis, keepdims, /*avg=*/false);
}
Tensor mean(const Tensor& a, std::optional<int64_t> axis, bool keepdims) {
    return reduce(a, axis, keepdims, /*avg=*/true);
}

// -----------------------------------------------------------------------------
// softmax
// -----------------------------------------------------------------------------
Tensor softmax(const Tensor& a, int64_t axis) {
    if (a.dtype() != DType::Float32) {
        throw std::invalid_argument("softmax: only float32 supported in Phase 1");
    }
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor out(a.shape(), DType::Float32);
    int64_t nd = static_cast<int64_t>(a.shape().size());
    if (axis < 0) axis += nd;
    if (axis < 0 || axis >= nd) {
        throw std::invalid_argument("softmax: axis out of range");
    }
    // Compute outer/inner/dim sizes (like reduce).
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < nd; ++i) {
        if (i < axis) outer *= a.shape()[i];
        else if (i == axis) { /* skip */ }
        else inner *= a.shape()[i];
    }
    int64_t dim_size = a.shape()[axis];
    const float* p = ac.data_float();
    float* op = out.data_float();
    const std::vector<int64_t>& sh = ac.shape();
    const std::vector<int64_t>& st = ac.strides();
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            // 1. find max for numerical stability
            float m = -std::numeric_limits<float>::infinity();
            for (int64_t k = 0; k < dim_size; ++k) {
                int64_t idx = 0;
                int64_t oo = o, ii = i, kk = k;
                for (int64_t d = nd - 1; d >= 0; --d) {
                    int64_t v;
                    if (d < axis) { v = oo % sh[d]; oo /= sh[d]; }
                    else if (d == axis) { v = kk; kk = 0; }
                    else { v = ii % sh[d]; ii /= sh[d]; }
                    idx += v * st[d];
                }
                if (p[idx] > m) m = p[idx];
            }
            // 2. exp(x - m) and sum
            float sum = 0.0f;
            std::vector<float> tmp(dim_size);
            for (int64_t k = 0; k < dim_size; ++k) {
                int64_t idx = 0;
                int64_t oo = o, ii = i, kk = k;
                for (int64_t d = nd - 1; d >= 0; --d) {
                    int64_t v;
                    if (d < axis) { v = oo % sh[d]; oo /= sh[d]; }
                    else if (d == axis) { v = kk; kk = 0; }
                    else { v = ii % sh[d]; ii /= sh[d]; }
                    idx += v * st[d];
                }
                tmp[k] = std::exp(p[idx] - m);
                sum += tmp[k];
            }
            float inv = 1.0f / sum;
            for (int64_t k = 0; k < dim_size; ++k) {
                int64_t idx = 0;
                int64_t oo = o, ii = i, kk = k;
                for (int64_t d = nd - 1; d >= 0; --d) {
                    int64_t v;
                    if (d < axis) { v = oo % sh[d]; oo /= sh[d]; }
                    else if (d == axis) { v = kk; kk = 0; }
                    else { v = ii % sh[d]; ii /= sh[d]; }
                    idx += v * st[d];
                }
                op[idx] = tmp[k] * inv;
            }
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// matmul (2-D x 2-D, naive O(n^3))
// We optimize this in Phase 2 with cache blocking / SIMD / threads.
// -----------------------------------------------------------------------------
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
        throw std::invalid_argument("matmul: only float32 supported in Phase 1");
    }
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::invalid_argument("matmul: inputs must be 2-D in Phase 1");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("matmul: shape mismatch");
    }
    int64_t M = a.shape()[0];
    int64_t K = a.shape()[1];
    int64_t N = b.shape()[1];
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    Tensor out({M, N}, DType::Float32);
    const float* A = ac.data_float();
    const float* B = bc.data_float();
    float*       C = out.data_float();
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                s += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// reshape / transpose passthroughs
// -----------------------------------------------------------------------------
Tensor reshape(const Tensor& a, std::vector<int64_t> shape) { return a.reshape(std::move(shape)); }
Tensor transpose(const Tensor& a, std::vector<int64_t> perm) { return a.transpose(std::move(perm)); }

Tensor broadcast_to(const Tensor& a, std::vector<int64_t> target_shape) {
    auto a_strides = broadcast_strides(a.shape(), a.strides(), target_shape);
    return Tensor(a.storage(), std::move(target_shape), std::move(a_strides),
                  a.offset(), a.dtype());
}

}  // namespace ops

// -----------------------------------------------------------------------------
// Operator overloads
// -----------------------------------------------------------------------------
Tensor operator+(const Tensor& a, const Tensor& b) { return ops::add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return ops::subtract(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return ops::multiply(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return ops::divide(a, b); }

}  // namespace tinyllm