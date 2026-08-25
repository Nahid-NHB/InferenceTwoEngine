// src/gguf.cpp
// -----------------------------------------------------------------------------
// GGUF parser implementation.
//
// Endianness: GGUF uses little-endian on disk. We assume the host is also
// little-endian (true for x86 / ARM in their usual configurations). On a
// big-endian host we'd need to byte-swap, but that's not our target.
//
// We don't mmap; we read the whole file into memory. For a 4 GB Q4_K file
// that's a lot, but a tiny model like TinyLlama-1.1B (Q4) is ~700 MB and
// fits in RAM. A bigger model would warrant mmap — we'll revisit in Phase 9.
//
// All reads from `buf` check bounds and throw on truncation.
// -----------------------------------------------------------------------------
#include "tinyllm/gguf.hpp"

#include "tinyllm/quantize.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tinyllm {

// -----------------------------------------------------------------------------
// Names + sizes
// -----------------------------------------------------------------------------
std::string_view gguf_value_type_name(GgufValueType t) noexcept {
    switch (t) {
        case GgufValueType::Array:   return "array";
        case GgufValueType::Uint8:   return "uint8";
        case GgufValueType::Int8:    return "int8";
        case GgufValueType::Uint16:  return "uint16";
        case GgufValueType::Int16:   return "int16";
        case GgufValueType::Uint32:  return "uint32";
        case GgufValueType::Int32:   return "int32";
        case GgufValueType::Float32: return "float32";
        case GgufValueType::Bool:    return "bool";
        case GgufValueType::String:  return "string";
        case GgufValueType::Array2:  return "array2";
        case GgufValueType::Uint64:  return "uint64";
        case GgufValueType::Int64:   return "int64";
        case GgufValueType::Float64: return "float64";
    }
    return "unknown";
}

std::string_view gguf_tensor_type_name(GgufTensorType t) noexcept {
    switch (t) {
        case GgufTensorType::F32:  return "F32";
        case GgufTensorType::F16:  return "F16";
        case GgufTensorType::Q4_0: return "Q4_0";
        case GgufTensorType::Q4_1: return "Q4_1";
        case GgufTensorType::Q8_0: return "Q8_0";
    }
    return "Q??";
}

std::size_t gguf_tensor_type_size(GgufTensorType t) noexcept {
    switch (t) {
        case GgufTensorType::F32:  return 4;
        case GgufTensorType::F16:  return 2;
        default: return 0;
    }
}

// -----------------------------------------------------------------------------
// GgufReader: methods defined out-of-line so templates don't have to live in
// the header.
// -----------------------------------------------------------------------------
void GgufReader::need(std::size_t n, const char* what) const {
    if (static_cast<std::size_t>(end - p) < n) {
        throw std::runtime_error(std::string("GGUF: truncated read: ") + what);
    }
}

template <typename T>
T GgufReader::read_le(const char* what) {
    need(sizeof(T), what);
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}

std::string GgufReader::read_str(const char* what) {
    uint64_t n = read_le<uint64_t>(what);
    need(static_cast<std::size_t>(n), what);
    std::string s(p, static_cast<std::size_t>(n));
    p += n;
    return s;
}

// -----------------------------------------------------------------------------
// Parse a single metadata value.
// -----------------------------------------------------------------------------
GgufValue parse_value_friend(GgufReader& r) {
    uint32_t t = r.read_le<uint32_t>("value type");
    GgufValueType vt = static_cast<GgufValueType>(t);
    GgufValue::Storage s;
    switch (vt) {
        case GgufValueType::Uint8:   s = r.read_le<uint8_t>("uint8");   break;
        case GgufValueType::Int8:    s = r.read_le<int8_t>("int8");     break;
        case GgufValueType::Uint16:  s = r.read_le<uint16_t>("uint16"); break;
        case GgufValueType::Int16:   s = r.read_le<int16_t>("int16");   break;
        case GgufValueType::Uint32:  s = r.read_le<uint32_t>("uint32"); break;
        case GgufValueType::Int32:   s = r.read_le<int32_t>("int32");   break;
        case GgufValueType::Uint64:  s = r.read_le<uint64_t>("uint64"); break;
        case GgufValueType::Int64:   s = r.read_le<int64_t>("int64");   break;
        case GgufValueType::Float32: s = r.read_le<float>("float32");   break;
        case GgufValueType::Float64: s = r.read_le<double>("float64");  break;
        case GgufValueType::Bool:    s = static_cast<bool>(r.read_le<uint8_t>("bool")); break;
        case GgufValueType::String:  s = r.read_str("string");          break;
        case GgufValueType::Array: {
            uint32_t et  = r.read_le<uint32_t>("array elem type");
            uint64_t len = r.read_le<uint64_t>("array len");
            auto arr = std::make_shared<GgufArray>();
            arr->element_type = static_cast<GgufValueType>(et);
            arr->elements.reserve(static_cast<std::size_t>(len));
            for (uint64_t i = 0; i < len; ++i) {
                arr->elements.push_back(parse_value_friend(r));
            }
            s = arr;
            break;
        }
        default:
            throw std::runtime_error("GGUF: unsupported metadata value type id " +
                                     std::to_string(t));
    }
    return GgufValue::make(vt, std::move(s));
}

// -----------------------------------------------------------------------------
// Open + parse
// -----------------------------------------------------------------------------
GgufFile GgufFile::open(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("GgufFile: cannot open: " + path);

    std::ostringstream ss;
    ss << f.rdbuf();
    std::string buf = ss.str();
    if (buf.size() < 24) throw std::runtime_error("GgufFile: file too small: " + path);

    GgufReader rd(buf.data(), buf.data() + buf.size());

    // Magic.
    if (std::memcmp(rd.p, "GGUF", 4) != 0) {
        throw std::runtime_error("GgufFile: bad magic (expected 'GGUF'): " + path);
    }
    rd.p += 4;

    uint32_t version = rd.read_le<uint32_t>("version");
    if (version != 3 && version != 2) {
        throw std::runtime_error("GgufFile: unsupported version " +
                                 std::to_string(version) + " (only 2 and 3)");
    }
    uint64_t n_tensors = rd.read_le<uint64_t>("n_tensors");
    uint64_t n_kv      = rd.read_le<uint64_t>("n_kv");

    std::vector<std::pair<std::string, GgufValue>> kv;
    kv.reserve(static_cast<std::size_t>(n_kv));
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = rd.read_str("kv key");
        GgufValue val = parse_value_friend(rd);
        kv.emplace_back(std::move(key), std::move(val));
    }

    std::vector<GgufTensorInfo> tinfos;
    tinfos.reserve(static_cast<std::size_t>(n_tensors));
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensorInfo ti;
        ti.name = rd.read_str("tensor name");
        uint32_t ndims = rd.read_le<uint32_t>("tensor n_dims");
        ti.dims.resize(ndims);
        for (uint32_t d = 0; d < ndims; ++d) {
            ti.dims[d] = rd.read_le<uint64_t>("tensor dim");
        }
        uint32_t dt = rd.read_le<uint32_t>("tensor type");
        ti.type   = static_cast<GgufTensorType>(dt);
        ti.offset = rd.read_le<uint64_t>("tensor offset");
        tinfos.push_back(std::move(ti));
    }

    uint64_t alignment = 32;
    if (version == 3) {
        if (static_cast<std::size_t>(rd.end - rd.p) >= sizeof(uint64_t)) {
            alignment = rd.read_le<uint64_t>("alignment");
        }
    }

    // The data section begins at an alignment-multiple position. The byte
    // right after the alignment field may not itself be aligned (e.g. 8 mod
    // 32), so we round up.
    uint64_t data_section_offset = static_cast<uint64_t>(rd.p - buf.data());
    if (alignment > 0) {
        data_section_offset = (data_section_offset + alignment - 1) & ~(alignment - 1);
    }

    return GgufFile(path, std::move(kv), std::move(tinfos),
                    data_section_offset, alignment, version);
}

const GgufValue* GgufFile::get_kv(const std::string& key) const {
    for (const auto& [k, v] : kv_) {
        if (k == key) return &v;
    }
    return nullptr;
}

// -----------------------------------------------------------------------------
// Load a tensor's data
// -----------------------------------------------------------------------------
Tensor GgufFile::load_tensor(std::size_t idx) const {
    if (idx >= tinfos_.size()) {
        throw std::out_of_range("GgufFile::load_tensor: index out of range");
    }
    const GgufTensorInfo& ti = tinfos_[idx];

    std::ifstream f(path_, std::ios::binary);
    if (!f) throw std::runtime_error("GgufFile::load_tensor: cannot reopen: " + path_);
    f.seekg(static_cast<std::streamoff>(data_section_offset_ + ti.offset));

    std::vector<int64_t> shape;
    shape.reserve(ti.dims.size());
    for (uint64_t d : ti.dims) shape.push_back(static_cast<int64_t>(d));

    int64_t n = 1;
    for (auto d : shape) n *= d;

    switch (ti.type) {
        case GgufTensorType::F32: {
            std::vector<float> data(static_cast<std::size_t>(n));
            f.read(reinterpret_cast<char*>(data.data()),
                   static_cast<std::streamsize>(n * sizeof(float)));
            if (static_cast<int64_t>(f.gcount()) != n * static_cast<int64_t>(sizeof(float))) {
                throw std::runtime_error("GgufFile::load_tensor: truncated F32 read for " + ti.name);
            }
            return Tensor(std::move(shape), DType::Float32,
                          data.data(), data.size() * sizeof(float));
        }
        case GgufTensorType::F16: {
            std::vector<uint16_t> raw(static_cast<std::size_t>(n));
            f.read(reinterpret_cast<char*>(raw.data()),
                   static_cast<std::streamsize>(n * sizeof(uint16_t)));
            if (static_cast<int64_t>(f.gcount()) != n * static_cast<int64_t>(sizeof(uint16_t))) {
                throw std::runtime_error("GgufFile::load_tensor: truncated F16 read for " + ti.name);
            }
            std::vector<float> f32(static_cast<std::size_t>(n));
            for (int64_t i = 0; i < n; ++i) {
                uint16_t h = raw[static_cast<std::size_t>(i)];
                uint32_t sign     = (h >> 15) & 0x1;
                uint32_t exponent = (h >> 10) & 0x1F;
                uint32_t mantissa = h & 0x3FF;
                uint32_t bits;
                if (exponent == 0) {
                    if (mantissa == 0) {
                        bits = sign << 31;
                    } else {
                        int e = -14;
                        while ((mantissa & 0x400) == 0) { mantissa <<= 1; ++e; }
                        mantissa &= 0x3FF;
                        bits = (sign << 31) | ((e + 127) << 23) | (mantissa << 13);
                    }
                } else if (exponent == 31) {
                    bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
                } else {
                    bits = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
                }
                std::memcpy(&f32[static_cast<std::size_t>(i)], &bits, sizeof(float));
            }
            return Tensor(std::move(shape), DType::Float32,
                          f32.data(), f32.size() * sizeof(float));
        }
        case GgufTensorType::Q8_0: {
            if (n % kQ8_0BlockSize != 0) {
                throw std::runtime_error("GgufFile::load_tensor: Q8_0 tensor " +
                                         ti.name + " size " + std::to_string(n) +
                                         " not divisible by 32");
            }
            int64_t nblocks = n / kQ8_0BlockSize;
            std::vector<uint8_t> packed(static_cast<std::size_t>(nblocks * kQ8_0BlockBytes));
            f.read(reinterpret_cast<char*>(packed.data()),
                   static_cast<std::streamsize>(nblocks * kQ8_0BlockBytes));
            if (static_cast<int64_t>(f.gcount()) != nblocks *
                                                    static_cast<int64_t>(kQ8_0BlockBytes)) {
                throw std::runtime_error("GgufFile::load_tensor: truncated Q8_0 read for " + ti.name);
            }
            std::vector<float> f32(static_cast<std::size_t>(n));
            dequantize_q8_0(packed.data(), n, f32.data());
            return Tensor(std::move(shape), DType::Float32,
                          f32.data(), f32.size() * sizeof(float));
        }
        case GgufTensorType::Q4_0: {
            if (n % kQ4_0BlockSize != 0) {
                throw std::runtime_error("GgufFile::load_tensor: Q4_0 tensor " +
                                         ti.name + " size " + std::to_string(n) +
                                         " not divisible by 32");
            }
            int64_t nblocks = n / kQ4_0BlockSize;
            std::vector<uint8_t> packed(static_cast<std::size_t>(nblocks * kQ4_0BlockBytes));
            f.read(reinterpret_cast<char*>(packed.data()),
                   static_cast<std::streamsize>(nblocks * kQ4_0BlockBytes));
            if (static_cast<int64_t>(f.gcount()) != nblocks *
                                                    static_cast<int64_t>(kQ4_0BlockBytes)) {
                throw std::runtime_error("GgufFile::load_tensor: truncated Q4_0 read for " + ti.name);
            }
            std::vector<float> f32(static_cast<std::size_t>(n));
            dequantize_q4_0(packed.data(), n, f32.data());
            return Tensor(std::move(shape), DType::Float32,
                          f32.data(), f32.size() * sizeof(float));
        }
        default:
            throw std::runtime_error("GgufFile::load_tensor: unsupported dtype " +
                                     std::string(gguf_tensor_type_name(ti.type)) +
                                     " for tensor " + ti.name);
    }
}

}  // namespace tinyllm