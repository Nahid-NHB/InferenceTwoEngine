// include/tinyllm/gguf.hpp
// -----------------------------------------------------------------------------
// GGUF file parser (https://github.com/ggerganov/ggml/blob/master/docs/gguf.md).
//
// We support reading GGUF v3 files. The format is:
//
//   header:
//       magic[4] = "GGUF"
//       version  : u32     (we accept v3; v2 also tolerated)
//       n_tensors: u64
//       n_kv     : u64
//   metadata KV pairs (n_kv of them):
//       key   : length-prefixed UTF-8 string
//       value : type-tagged scalar or array (see GgufValueType)
//   tensor infos (n_tensors of them):
//       name   : length-prefixed UTF-8 string
//       n_dims : u32
//       dims   : u64[n_dims]   (in reverse; the first stored dim is the
//                               slowest-varying in memory, matching our
//                               row-major convention)
//       dtype  : u32 (GgufTensorType)
//       offset : u64            (offset into the tensor-data section)
//   tensor data:
//       tensors are placed at their offset; offsets are multiples of
//       alignment (default 32 in v3; we report any value we encounter)
//
// We expose:
//   - read the whole file with GgufFile::open(path)
//   - iterate metadata via kv_pairs()
//   - iterate tensors via tensors()
//   - load a tensor's data into a Tensor via load_tensor(idx)
//
// We only fully support tensor types F32 and F16 for now. Q-tensors parse
// but raise on load; quantization support is in Phase 8.
// -----------------------------------------------------------------------------
#pragma once

#include "tinyllm/tensor.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// Forward decl (full def in gguf.cpp).
struct GgufReader;

namespace tinyllm {

enum class GgufValueType : uint32_t {
    Array   = 0,
    Uint8   = 1,
    Int8    = 2,
    Uint16  = 3,
    Int16   = 4,
    Uint32  = 5,
    Int32   = 6,
    Float32 = 7,
    Bool    = 8,
    String  = 9,
    Array2  = 10,   // GGUF v2-only, distinct from v3's "array"
    Uint64  = 11,
    Int64   = 12,
    Float64 = 13,
};

enum class GgufTensorType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    // ... many quantization types; see ggml.h
    Q8_0    = 8,
    // ...
};

std::string_view gguf_value_type_name(GgufValueType t) noexcept;
std::string_view gguf_tensor_type_name(GgufTensorType t) noexcept;
std::size_t      gguf_tensor_type_size(GgufTensorType t) noexcept;  // bytes per element

// -----------------------------------------------------------------------------
// Metadata value: a tagged union over the supported types.
// -----------------------------------------------------------------------------
class GgufArray;

class GgufValue {
public:
    using Storage = std::variant<
        uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
        uint64_t, int64_t, float, double, bool, std::string,
        std::shared_ptr<GgufArray>>;

    GgufValue() = default;
    template <typename T>
    GgufValue(T v) : s_(std::move(v)) {}

    // Build from type + storage directly. Public so the parser in gguf.cpp
    // doesn't need friend declarations.
    static GgufValue make(GgufValueType t, Storage s) {
        GgufValue v;
        v.type_ = t;
        v.s_    = std::move(s);
        return v;
    }

    GgufValueType type() const noexcept { return type_; }
    const Storage& storage() const noexcept { return s_; }

private:
    friend class GgufFile;

    GgufValueType type_ = GgufValueType::Uint8;
    Storage s_;
};

class GgufArray {
public:
    GgufValueType element_type;
    std::vector<GgufValue> elements;
};

// Convenience accessors for the common types.
inline std::optional<uint32_t> as_uint32(const GgufValue& v) {
    if (auto* p = std::get_if<uint32_t>(&v.storage())) return *p;
    if (auto* p = std::get_if<int64_t>(&v.storage())) {
        if (*p >= 0 && *p <= 0xFFFFFFFFu) return static_cast<uint32_t>(*p);
    }
    if (auto* p = std::get_if<uint64_t>(&v.storage())) {
        if (*p <= 0xFFFFFFFFu) return static_cast<uint32_t>(*p);
    }
    return std::nullopt;
}
inline std::optional<int32_t> as_int32(const GgufValue& v) {
    if (auto* p = std::get_if<int32_t>(&v.storage())) return *p;
    return std::nullopt;
}
inline std::optional<float> as_float32(const GgufValue& v) {
    if (auto* p = std::get_if<float>(&v.storage())) return *p;
    if (auto* p = std::get_if<double>(&v.storage())) return static_cast<float>(*p);
    return std::nullopt;
}
inline std::optional<std::string> as_string(const GgufValue& v) {
    if (auto* p = std::get_if<std::string>(&v.storage())) return *p;
    return std::nullopt;
}

// -----------------------------------------------------------------------------
// Tensor info: parsed from the header, before we read the data.
// -----------------------------------------------------------------------------
struct GgufTensorInfo {
    std::string          name;
    std::vector<uint64_t> dims;          // in GGUF order (slowest first)
    GgufTensorType        type = GgufTensorType::F32;
    uint64_t              offset = 0;    // offset into the tensor-data section
};

// -----------------------------------------------------------------------------
// GGUF file
// -----------------------------------------------------------------------------

// Forward declaration of the parser-side helper; full def in gguf.cpp.
struct GgufReader;

class GgufFile {
public:
    GgufFile() = default;

    // Open and parse. Throws std::runtime_error on bad magic / truncated file.
    static GgufFile open(const std::string& path);

    // Read-only access to the parsed metadata.
    const std::vector<std::pair<std::string, GgufValue>>& kv_pairs() const noexcept {
        return kv_;
    }

    // Lookup helpers.
    const GgufValue* get_kv(const std::string& key) const;

    // Tensor info (parsed but not loaded).
    const std::vector<GgufTensorInfo>& tensor_infos() const noexcept { return tinfos_; }

    // Load a single tensor by index. Throws if dtype is unsupported or
    // the read would go past the file.
    Tensor load_tensor(std::size_t idx) const;

    // Metadata about the file.
    uint32_t version() const noexcept { return version_; }
    std::size_t n_tensors() const noexcept { return tinfos_.size(); }
    std::size_t n_kv()     const noexcept { return kv_.size(); }

private:
    GgufFile(const std::string& path, std::vector<std::pair<std::string, GgufValue>> kv,
             std::vector<GgufTensorInfo> tinfos,
             uint64_t data_section_offset, uint64_t alignment, uint32_t version)
        : path_(path), kv_(std::move(kv)), tinfos_(std::move(tinfos)),
          data_section_offset_(data_section_offset),
          alignment_(alignment), version_(version) {}

    std::string path_;
    std::vector<std::pair<std::string, GgufValue>> kv_;
    std::vector<GgufTensorInfo>                    tinfos_;
    uint64_t data_section_offset_ = 0;
    uint64_t alignment_           = 32;
    uint32_t version_             = 3;
};

// Full definition of the byte reader (used in gguf.cpp).
struct GgufReader {
    const char* p;
    const char* end;

    GgufReader(const char* pp, const char* pe) : p(pp), end(pe) {}
    void need(std::size_t n, const char* what) const;
    template <typename T> T read_le(const char* what);
    std::string read_str(const char* what);
};

}  // namespace tinyllm