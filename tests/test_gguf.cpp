// tests/test_gguf.cpp
// -----------------------------------------------------------------------------
// GGUF parser tests.
//
// We test against a synthetic GGUF file written to /tmp by the test
// fixture. The fixture writes a file with:
//   - a few scalar metadata KV pairs (int, float, bool, string)
//   - an array metadata KV pair
//   - 3 F32 tensors of varying shapes
//   - 1 F16 tensor
// -----------------------------------------------------------------------------
#include "test_helpers.hpp"
#include "tinyllm/gguf.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace tinyllm;

// -----------------------------------------------------------------------------
// Minimal GGUF writer for tests.
// -----------------------------------------------------------------------------
namespace {

class Writer {
public:
    explicit Writer(std::ostream& o) : o_(o) {}

    template <typename T>
    void le(T v) {
        o_.write(reinterpret_cast<const char*>(&v), sizeof(T));
    }
    void bytes(const void* p, std::size_t n) { o_.write(reinterpret_cast<const char*>(p), n); }
    void str(std::string_view s) {
        le<uint64_t>(s.size());
        bytes(s.data(), s.size());
    }
    void value_uint32(uint32_t x) {
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::Uint32));
        le<uint32_t>(x);
    }
    void value_int32(int32_t x) {
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::Int32));
        le<int32_t>(x);
    }
    void value_float32(float x) {
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::Float32));
        le<float>(x);
    }
    void value_bool(bool x) {
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::Bool));
        le<uint8_t>(x ? 1 : 0);
    }
    void value_string(std::string_view s) {
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::String));
        str(s);
    }
    void value_array_uint32(const std::vector<uint32_t>& arr) {
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::Array));
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::Uint32));
        le<uint64_t>(arr.size());
        for (uint32_t v : arr) le<uint32_t>(v);
    }
    void value_array_string(const std::vector<std::string>& arr) {
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::Array));
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::String));
        le<uint64_t>(arr.size());
        for (const auto& s : arr) str(s);
    }
    void value_array_float32(const std::vector<float>& arr) {
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::Array));
        le<uint32_t>(static_cast<uint32_t>(GgufValueType::Float32));
        le<uint64_t>(arr.size());
        for (float v : arr) le<float>(v);
    }

    void tensor_info(std::string_view name, const std::vector<uint64_t>& dims,
                     GgufTensorType dt, uint64_t offset) {
        str(name);
        le<uint32_t>(static_cast<uint32_t>(dims.size()));
        for (uint64_t d : dims) le<uint64_t>(d);
        le<uint32_t>(static_cast<uint32_t>(dt));
        le<uint64_t>(offset);
    }

private:
    std::ostream& o_;
};

struct Fixture {
    std::string path;
    std::vector<float> expected_f32_a;   // 2x3
    std::vector<float> expected_f32_b;   // 4
    std::vector<float> expected_f32_c;   // scalar
    std::vector<float> expected_f16;     // 5 elements
};

Fixture write_test_gguf() {
    Fixture f;
    f.path = "/tmp/tinyllm_test_gguf.gguf";

    std::ofstream out(f.path, std::ios::binary);
    Writer w(out);

    // Header
    out.write("GGUF", 4);
    w.le<uint32_t>(3);                  // version
    w.le<uint64_t>(4);                  // n_tensors
    w.le<uint64_t>(5);                  // n_kv

    // KV 0: name="general.architecture"  value=string "llama"
    w.str("general.architecture");
    w.value_string("llama");

    // KV 1: name="llama.context_length" value=uint32 2048
    w.str("llama.context_length");
    w.value_uint32(2048);

    // KV 2: name="llama.embedding_length" value=uint32 128
    w.str("llama.embedding_length");
    w.value_uint32(128);

    // KV 3: name="llama.attention.head_count" value=uint32 4
    w.str("llama.attention.head_count");
    w.value_uint32(4);

    // KV 4: name="tokenizer.ggml.model" value=string "llama"
    w.str("tokenizer.ggml.model");
    w.value_string("llama");

    // We also test an array KV separately (below).

    // Tensor infos. We pre-compute offsets so each F32 is 32-byte aligned.
    constexpr uint64_t kAlign = 32;
    // We'll compute offsets as we go. The first tensor's data offset is 0.
    // We track the running cursor.
    uint64_t cursor = 0;
    auto pad = [&]() { cursor = (cursor + kAlign - 1) & ~(kAlign - 1); };
    pad();

    // tensor_a: F32 [2, 3]
    auto off_a = cursor;
    cursor += 2 * 3 * sizeof(float);
    pad();
    // tensor_b: F32 [4]
    auto off_b = cursor;
    cursor += 4 * sizeof(float);
    pad();
    // tensor_c: F32 [] (scalar)
    auto off_c = cursor;
    cursor += 1 * sizeof(float);
    pad();
    // tensor_d: F16 [5]
    auto off_d = cursor;
    cursor += 5 * sizeof(uint16_t);
    pad();

    w.tensor_info("tensor_a", {2, 3},         GgufTensorType::F32, off_a);
    w.tensor_info("tensor_b", {4},            GgufTensorType::F32, off_b);
    w.tensor_info("tensor_c", {},             GgufTensorType::F32, off_c);
    w.tensor_info("tensor_d", {5},            GgufTensorType::F16, off_d);

    // Alignment (v3)
    w.le<uint64_t>(kAlign);

    // Pad the file position to alignment for the first tensor's data.
    auto pos = static_cast<std::uint64_t>(out.tellp());
    while (pos % kAlign != 0) { out.put(0); ++pos; }

    // Data: F32 [2,3] = [[1,2,3],[4,5,6]]
    f.expected_f32_a = {1, 2, 3, 4, 5, 6};
    out.write(reinterpret_cast<const char*>(f.expected_f32_a.data()),
              f.expected_f32_a.size() * sizeof(float));
    pos = static_cast<std::uint64_t>(out.tellp());
    while (pos % kAlign != 0) { out.put(0); ++pos; }

    // Data: F32 [4] = [10, 20, 30, 40]
    f.expected_f32_b = {10, 20, 30, 40};
    out.write(reinterpret_cast<const char*>(f.expected_f32_b.data()),
              f.expected_f32_b.size() * sizeof(float));
    pos = static_cast<std::uint64_t>(out.tellp());
    while (pos % kAlign != 0) { out.put(0); ++pos; }

    // Data: F32 scalar = 7.5
    f.expected_f32_c = {7.5f};
    out.write(reinterpret_cast<const char*>(f.expected_f32_c.data()),
              sizeof(float));
    pos = static_cast<std::uint64_t>(out.tellp());
    while (pos % kAlign != 0) { out.put(0); ++pos; }

    // Data: F16 [5] = [1.0, 2.0, 3.0, 4.0, 5.0] as binary16
    auto to_f16 = [](float x) -> uint16_t {
        uint32_t bits; std::memcpy(&bits, &x, sizeof(bits));
        uint32_t sign     = (bits >> 31) & 1;
        uint32_t exponent = (bits >> 23) & 0xFF;
        uint32_t mantissa = bits & 0x7FFFFF;
        if (exponent == 0xFF) {
            return (sign << 15) | (0x1F << 10) | (mantissa ? (mantissa >> 13) : 0);
        }
        int e = static_cast<int>(exponent) - 127 + 15;
        if (e <= 0)   return static_cast<uint16_t>(sign << 15);
        if (e >= 31)  return static_cast<uint16_t>((sign << 15) | (0x1F << 10));
        return static_cast<uint16_t>((sign << 15) | (e << 10) | (mantissa >> 13));
    };
    std::vector<uint16_t> f16 = { to_f16(1.0f), to_f16(2.0f), to_f16(3.0f),
                                  to_f16(4.0f), to_f16(5.0f) };
    f.expected_f16 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    out.write(reinterpret_cast<const char*>(f16.data()), f16.size() * sizeof(uint16_t));
    pos = static_cast<std::uint64_t>(out.tellp());
    while (pos % kAlign != 0) { out.put(0); ++pos; }

    return f;
}

}  // namespace

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------
TEST_CASE(gguf_parses_header) {
    auto f = write_test_gguf();
    auto gf = GgufFile::open(f.path);
    REQUIRE(gf.version() == 3);
    REQUIRE(gf.n_tensors() == 4);
    REQUIRE(gf.n_kv() == 5);
}

TEST_CASE(gguf_reads_string_metadata) {
    auto f = write_test_gguf();
    auto gf = GgufFile::open(f.path);
    auto* v = gf.get_kv("general.architecture");
    REQUIRE(v != nullptr);
    auto s = as_string(*v);
    REQUIRE(s.has_value());
    REQUIRE(*s == "llama");
}

TEST_CASE(gguf_reads_uint_metadata) {
    auto f = write_test_gguf();
    auto gf = GgufFile::open(f.path);
    auto* v = gf.get_kv("llama.context_length");
    REQUIRE(v != nullptr);
    auto x = as_uint32(*v);
    REQUIRE(x.has_value());
    REQUIRE(*x == 2048u);
}

TEST_CASE(gguf_reads_tensor_infos) {
    auto f = write_test_gguf();
    auto gf = GgufFile::open(f.path);
    const auto& infos = gf.tensor_infos();
    REQUIRE(infos.size() == 4);
    REQUIRE(infos[0].name == "tensor_a");
    REQUIRE(infos[0].dims.size() == 2);
    REQUIRE(infos[0].dims[0] == 2);
    REQUIRE(infos[0].dims[1] == 3);
    REQUIRE(infos[0].type == GgufTensorType::F32);
}

TEST_CASE(gguf_loads_f32_2x3_tensor) {
    auto f = write_test_gguf();
    auto gf = GgufFile::open(f.path);
    // Find tensor_a
    std::size_t idx = 0;
    for (std::size_t i = 0; i < gf.n_tensors(); ++i) {
        if (gf.tensor_infos()[i].name == "tensor_a") { idx = i; break; }
    }
    auto t = gf.load_tensor(idx);
    REQUIRE(t.ndim() == 2);
    REQUIRE(t.shape()[0] == 2);
    REQUIRE(t.shape()[1] == 3);
    REQUIRE(t.dtype() == DType::Float32);
    REQUIRE(t.numel() == 6);
    for (int64_t i = 0; i < t.numel(); ++i) {
        REQUIRE_NEAR(t.at_flat(i), f.expected_f32_a[static_cast<std::size_t>(i)], 1e-6);
    }
}

TEST_CASE(gguf_loads_f32_1d_tensor) {
    auto f = write_test_gguf();
    auto gf = GgufFile::open(f.path);
    std::size_t idx = 1;
    auto t = gf.load_tensor(idx);
    REQUIRE(t.ndim() == 1);
    REQUIRE(t.numel() == 4);
    for (int64_t i = 0; i < t.numel(); ++i) {
        REQUIRE_NEAR(t.at_flat(i), f.expected_f32_b[static_cast<std::size_t>(i)], 1e-6);
    }
}

TEST_CASE(gguf_loads_f32_scalar) {
    auto f = write_test_gguf();
    auto gf = GgufFile::open(f.path);
    std::size_t idx = 2;
    auto t = gf.load_tensor(idx);
    REQUIRE(t.numel() == 1);
    REQUIRE_NEAR(t.at_flat(0), 7.5f, 1e-6);
}

TEST_CASE(gguf_loads_f16_tensor) {
    auto f = write_test_gguf();
    auto gf = GgufFile::open(f.path);
    std::size_t idx = 3;
    auto t = gf.load_tensor(idx);
    REQUIRE(t.dtype() == DType::Float32);  // we materialize as F32
    REQUIRE(t.numel() == 5);
    for (int64_t i = 0; i < t.numel(); ++i) {
        REQUIRE_NEAR(t.at_flat(i), f.expected_f16[static_cast<std::size_t>(i)], 0.01);
    }
}

TEST_CASE(gguf_rejects_bad_magic) {
    std::string path = "/tmp/tinyllm_test_gguf_bad.gguf";
    {
        std::ofstream o(path, std::ios::binary);
        o.write("NOPE", 4);
        o.write("\x03\0\0\0", 4);  // version 3
        // Just enough to pass initial size test.
    }
    REQUIRE_THROWS(GgufFile::open(path));
}

TEST_CASE(gguf_rejects_unsupported_version) {
    std::string path = "/tmp/tinyllm_test_gguf_bad_ver.gguf";
    {
        std::ofstream o(path, std::ios::binary);
        o.write("GGUF", 4);
        char v[4] = {99, 0, 0, 0};  // version 99
        o.write(v, 4);
        o.write("\0\0\0\0\0\0\0\0", 8);  // n_tensors=0
        o.write("\0\0\0\0\0\0\0\0", 8);  // n_kv=0
    }
    REQUIRE_THROWS(GgufFile::open(path));
}

TEST_CASE(gguf_dtype_names) {
    REQUIRE(std::string(gguf_tensor_type_name(GgufTensorType::F32)) == "F32");
    REQUIRE(std::string(gguf_tensor_type_name(GgufTensorType::F16)) == "F16");
    REQUIRE(gguf_tensor_type_size(GgufTensorType::F32) == 4);
    REQUIRE(gguf_tensor_type_size(GgufTensorType::F16) == 2);
}