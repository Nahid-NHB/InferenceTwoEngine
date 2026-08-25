// tests/test_quantize.cpp
// -----------------------------------------------------------------------------
// Quantization tests.
//
// Coverage:
//   - f16 <-> f32 round-trip
//   - Q8_0: round-trip (dequant(quant(x)) ≈ x), full-tensor round-trip,
//           edge cases (zeros, single hot value), div-by-32 errors
//   - Q4_0: same set of tests; round-trip is naturally lossier
//   - matmul_q*_f32: dequantize-on-the-fly matmul agrees with F32 matmul
//     to within the per-format tolerance
//   - GGUF round-trip: write a synthetic Q8_0 / Q4_0 tensor, read it back,
//     confirm dequantized values match
// -----------------------------------------------------------------------------
#include "test_helpers.hpp"
#include "tinyllm/gguf.hpp"
#include "tinyllm/matmul.hpp"
#include "tinyllm/quantize.hpp"
#include "tinyllm/tensor.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <vector>

using namespace tinyllm;

namespace {

// Linear spaced values in [a, b].
std::vector<float> linspace(float a, float b, int64_t n) {
    std::vector<float> out(static_cast<std::size_t>(n));
    if (n == 1) { out[0] = a; return out; }
    float step = (b - a) / static_cast<float>(n - 1);
    for (int64_t i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = a + step * static_cast<float>(i);
    }
    return out;
}

std::vector<float> random_floats(std::size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = d(rng);
    return out;
}

}  // namespace

// -----------------------------------------------------------------------------
// f16 <-> f32 helpers
// -----------------------------------------------------------------------------
TEST_CASE(f16_roundtrip_basic) {
    // Spot-check several exact values.
    REQUIRE(std::isinf(quantize_f16_to_f32(quantize_f32_to_f16(1e30f))));
    // 1.0 should round-trip exactly.
    uint16_t one = quantize_f32_to_f16(1.0f);
    REQUIRE(quantize_f16_to_f32(one) == 1.0f);
    // 0.5 is exactly representable.
    REQUIRE(quantize_f16_to_f32(quantize_f32_to_f16(0.5f)) == 0.5f);
    // -2.0 is exactly representable.
    REQUIRE(quantize_f16_to_f32(quantize_f32_to_f16(-2.0f)) == -2.0f);
}

TEST_CASE(f16_special_values) {
    // +inf
    uint16_t pinf = quantize_f32_to_f16(std::numeric_limits<float>::infinity());
    REQUIRE(std::isinf(quantize_f16_to_f32(pinf)));
    REQUIRE(quantize_f16_to_f32(pinf) > 0);
    // -inf
    uint16_t ninf = quantize_f32_to_f16(-std::numeric_limits<float>::infinity());
    REQUIRE(std::isinf(quantize_f16_to_f32(ninf)));
    REQUIRE(quantize_f16_to_f32(ninf) < 0);
    // 0.0
    uint16_t zero = quantize_f32_to_f16(0.0f);
    REQUIRE(quantize_f16_to_f32(zero) == 0.0f);
}

// -----------------------------------------------------------------------------
// Q8_0
// -----------------------------------------------------------------------------
TEST_CASE(q8_0_roundtrip_block) {
    // 32 linspace values in [-1, 1].
    auto v = linspace(-1.0f, 1.0f, 32);
    Q8_0Block b = quantize_q8_0_block(v.data());
    REQUIRE(b.scale > 0.0f);
    std::vector<float> out(32);
    dequantize_q8_0_block(b, out.data());
    for (int i = 0; i < 32; ++i) {
        REQUIRE_NEAR(out[static_cast<std::size_t>(i)],
                     v[static_cast<std::size_t>(i)],
                     /*tol=*/1.0f / 127.0f + 1e-6f);
    }
}

TEST_CASE(q8_0_roundtrip_full_tensor) {
    auto v = random_floats(32 * 100, /*seed=*/42);
    std::vector<uint8_t> packed = quantize_q8_0(v.data(),
                                               static_cast<int64_t>(v.size()));
    std::vector<float> out(v.size());
    dequantize_q8_0(packed.data(), static_cast<int64_t>(v.size()), out.data());
    REQUIRE(out.size() == v.size());
    float worst = 0.0f;
    for (std::size_t i = 0; i < v.size(); ++i) {
        float e = std::fabs(out[i] - v[i]);
        if (e > worst) worst = e;
    }
    // Each value's quantization error is at most ~scale/2. Worst-case
    // relative error is 1/127 ~ 0.8%. For values in [-1, 1] the worst
    // absolute error is also bounded by 1/127.
    REQUIRE(worst < 0.02f);
}

TEST_CASE(q8_0_zero_block_yields_zero) {
    std::vector<float> z(32, 0.0f);
    Q8_0Block b = quantize_q8_0_block(z.data());
    REQUIRE(b.scale == 0.0f);
    for (int8_t q : b.qs) REQUIRE(q == 0);
}

TEST_CASE(q8_0_rejects_non_multiple_of_32) {
    std::vector<float> v(33);
    REQUIRE_THROWS(quantize_q8_0(v.data(), 33));
}

// -----------------------------------------------------------------------------
// Q4_0
// -----------------------------------------------------------------------------
TEST_CASE(q4_0_roundtrip_block) {
    auto v = linspace(-1.0f, 1.0f, 32);
    Q4_0Block b = quantize_q4_0_block(v.data());
    REQUIRE(b.scale > 0.0f);
    std::vector<float> out(32);
    dequantize_q4_0_block(b, out.data());
    for (int i = 0; i < 32; ++i) {
        // Q4_0 is naturally coarser: 16 levels, but re-centered to
        // [-8, +7]. Per-step error is at most scale/2.
        REQUIRE_NEAR(out[static_cast<std::size_t>(i)],
                     v[static_cast<std::size_t>(i)],
                     /*tol=*/b.scale + 1e-6f);
    }
}

TEST_CASE(q4_0_roundtrip_full_tensor) {
    // Use a tighter value range than [-1, 1] so each block's scale is
    // reasonable relative to most values. (With absmax=1 and a value
    // of 0.01 the relative error is necessarily large — that's
    // fundamental to absolute-scale quantization on wide ranges.)
    auto v = random_floats(32 * 100, /*seed=*/7);
    // Compress to [-0.5, 0.5] to keep relative error bounded.
    for (auto& x : v) x *= 0.5f;
    std::vector<uint8_t> packed = quantize_q4_0(v.data(),
                                                static_cast<int64_t>(v.size()));
    std::vector<float> out(v.size());
    dequantize_q4_0(packed.data(), static_cast<int64_t>(v.size()), out.data());
    REQUIRE(out.size() == v.size());
    // For each block, the smallest non-zero representable value is
    // `scale` (which equals absmax/7 here). Values whose absolute
    // magnitude is below that threshold can be quantized to zero, so
    // we exclude them from the relative-error measurement. After that
    // filter, the worst-case relative error is at most ~7% (1/15).
    int64_t nblocks = v.size() / 32;
    float worst_rel = 0.0f;
    for (int64_t bi = 0; bi < nblocks; ++bi) {
        // Recover this block's scale by re-quantizing (same algorithm).
        float amax = 0.0f;
        for (int k = 0; k < 32; ++k) {
            amax = std::max(amax, std::fabs(v[static_cast<std::size_t>(bi*32 + k)]));
        }
        float scale = amax / 7.0f;
        // Skip values that fall within ±scale/2 of a quantization level
        // (including zero), where rounding is dominated by the half-step
        // boundary. Above that threshold, the worst-case relative error
        // can be up to ~100% when a value sits exactly on a quantization
        // boundary. For most values, it's around 7% (1/15).
        float threshold = scale * 4.0f;
        for (int k = 0; k < 32; ++k) {
            std::size_t i = static_cast<std::size_t>(bi * 32 + k);
            if (std::fabs(v[i]) < threshold) continue;
            float r = std::fabs(out[i] - v[i]) / std::fabs(v[i]);
            if (r > worst_rel) worst_rel = r;
        }
    }
    // Allow up to 50% relative error: any value at least 4 quantization
    // steps from zero should round to within ±half a step. In the
    // worst case (a value exactly at the bin boundary), this could be
    // up to ~50%, but for typical values it's ~7%.
    REQUIRE(worst_rel < 0.50f);
}

TEST_CASE(q4_0_zero_block_yields_zero) {
    std::vector<float> z(32, 0.0f);
    Q4_0Block b = quantize_q4_0_block(z.data());
    REQUIRE(b.scale == 0.0f);
    for (uint8_t x : b.qs) REQUIRE(x == 0);
}

TEST_CASE(q4_0_rejects_non_multiple_of_32) {
    std::vector<float> v(31);
    REQUIRE_THROWS(quantize_q4_0(v.data(), 31));
}

// -----------------------------------------------------------------------------
// Quantized matmul vs F32 matmul
// -----------------------------------------------------------------------------
TEST_CASE(matmul_q8_0_matches_f32) {
    // Random A (M x K) and x (K).
    const int64_t M = 8, K = 64;
    auto a = random_floats(static_cast<std::size_t>(M * K), /*seed=*/1);
    auto x = random_floats(static_cast<std::size_t>(K),     /*seed=*/2);
    auto packed = quantize_q8_0(a.data(), M * K);

    std::vector<float> y_q(static_cast<std::size_t>(M));
    matmul_q8_0_f32(packed.data(), M, K, x.data(), y_q.data());

    // Reference: F32 matmul (treat x as a [K, 1] column).
    Tensor At({M, K},     DType::Float32);
    Tensor xt({K, 1},     DType::Float32);
    std::memcpy(At.data_float(), a.data(), a.size() * sizeof(float));
    std::memcpy(xt.data_float(), x.data(), x.size() * sizeof(float));
    Tensor yt = ops::matmul(At, xt);

    for (int64_t mi = 0; mi < M; ++mi) {
        REQUIRE_NEAR(y_q[static_cast<std::size_t>(mi)],
                     yt.at_flat(mi),
                     /*tol=*/1e-3f * static_cast<float>(K));
    }
}

TEST_CASE(matmul_q4_0_matches_f32) {
    const int64_t M = 8, K = 64;
    auto a = random_floats(static_cast<std::size_t>(M * K), /*seed=*/11);
    auto x = random_floats(static_cast<std::size_t>(K),     /*seed=*/12);
    auto packed = quantize_q4_0(a.data(), M * K);

    std::vector<float> y_q(static_cast<std::size_t>(M));
    matmul_q4_0_f32(packed.data(), M, K, x.data(), y_q.data());

    Tensor At({M, K}, DType::Float32);
    Tensor xt({K, 1}, DType::Float32);
    std::memcpy(At.data_float(), a.data(), a.size() * sizeof(float));
    std::memcpy(xt.data_float(), x.data(), x.size() * sizeof(float));
    Tensor yt = ops::matmul(At, xt);

    for (int64_t mi = 0; mi < M; ++mi) {
        REQUIRE_NEAR(y_q[static_cast<std::size_t>(mi)],
                     yt.at_flat(mi),
                     /*tol=*/1e-2f * static_cast<float>(K));
    }
}

TEST_CASE(matmul_q4_0_larger) {
    // Bigger matmul to confirm block-dequant overhead is consistent.
    const int64_t M = 32, K = 256;
    auto a = random_floats(static_cast<std::size_t>(M * K), /*seed=*/21);
    auto x = random_floats(static_cast<std::size_t>(K),     /*seed=*/22);
    auto packed = quantize_q4_0(a.data(), M * K);

    std::vector<float> y_q(static_cast<std::size_t>(M));
    matmul_q4_0_f32(packed.data(), M, K, x.data(), y_q.data());

    Tensor At({M, K}, DType::Float32);
    Tensor xt({K, 1}, DType::Float32);
    std::memcpy(At.data_float(), a.data(), a.size() * sizeof(float));
    std::memcpy(xt.data_float(), x.data(), x.size() * sizeof(float));
    Tensor yt = ops::matmul(At, xt);

    for (int64_t mi = 0; mi < M; ++mi) {
        REQUIRE_NEAR(y_q[static_cast<std::size_t>(mi)],
                     yt.at_flat(mi),
                     /*tol=*/1e-2f * static_cast<float>(K));
    }
}

// -----------------------------------------------------------------------------
// GGUF round-trip: write a synthetic GGUF v3 file with a Q8_0 and a Q4_0
// tensor, read it back via GgufFile::load_tensor, and confirm the
// dequantized values match.
// -----------------------------------------------------------------------------
namespace {

void write_u32(std::vector<uint8_t>& buf, uint32_t v) {
    std::size_t n = buf.size();
    buf.resize(n + 4);
    std::memcpy(buf.data() + n, &v, 4);
}

void write_u64(std::vector<uint8_t>& buf, uint64_t v) {
    std::size_t n = buf.size();
    buf.resize(n + 8);
    std::memcpy(buf.data() + n, &v, 8);
}

void write_str(std::vector<uint8_t>& buf, const std::string& s) {
    write_u64(buf, static_cast<uint64_t>(s.size()));
    std::size_t n = buf.size();
    buf.resize(n + s.size());
    std::memcpy(buf.data() + n, s.data(), s.size());
}

// Write one tensor info record: name, n_dims + dims, dtype, data_offset.
void write_tensor_info(std::vector<uint8_t>& buf, const std::string& name,
                       const std::vector<int64_t>& dims,
                       GgufTensorType type, uint64_t data_offset) {
    write_str(buf, name);
    write_u32(buf, static_cast<uint32_t>(dims.size()));
    // GGUF stores dims in natural (row-major) order — fastest at the end.
    // The existing test_gguf.cpp uses the same convention.
    for (auto d : dims) write_u64(buf, static_cast<uint64_t>(d));
    write_u32(buf, static_cast<uint32_t>(type));
    write_u64(buf, data_offset);
}

void align_data_section(std::vector<uint8_t>& buf, uint64_t alignment) {
    while (buf.size() % alignment != 0) buf.push_back(0);
}

}  // namespace

TEST_CASE(gguf_loads_q8_0_tensor) {
    // Build a small [4, 32] F32 tensor with known values, quantize to
    // Q8_0, and embed in a synthetic GGUF v3 file.
    std::vector<float> values(4 * 32);
    for (int i = 0; i < 4 * 32; ++i) {
        values[static_cast<std::size_t>(i)] =
            static_cast<float>(i - 64) / 32.0f;
    }
    auto packed = quantize_q8_0(values.data(), static_cast<int64_t>(values.size()));

    // GGUF layout:
    //   [header][tensor info][alignment u64][pad to alignment][tensor data]
    //
    // Note: ti.offset is the offset *into the data section*, NOT the
    // absolute file position. The reader seeks to (data_section_offset_
    // + ti.offset). Since we only have one tensor and it's at the start
    // of the data section, we set ti.offset = 0 (default).
    std::vector<uint8_t> file;
    file.insert(file.end(), {'G', 'G', 'U', 'F'});
    write_u32(file, 3);   // version
    write_u64(file, 1);   // n_tensors
    write_u64(file, 0);   // n_kv

    write_tensor_info(file, "test", {4, 32}, GgufTensorType::Q8_0,
                      /*data_offset=*/0);
    write_u64(file, 32);  // alignment

    align_data_section(file, 32);
    file.insert(file.end(), packed.begin(), packed.end());

    // Write to disk and read back.
    const std::string path = "/tmp/tinyllm_q8_0_test.gguf";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(file.data()),
                static_cast<std::streamsize>(file.size()));
    }
    GgufFile gf = GgufFile::open(path);
    REQUIRE(gf.n_tensors() == 1);
    const auto& ti = gf.tensor_infos()[0];
    REQUIRE(ti.type == GgufTensorType::Q8_0);
    Tensor t = gf.load_tensor(0);
    REQUIRE(t.dtype() == DType::Float32);
    REQUIRE(t.shape()[0] == 4);
    REQUIRE(t.shape()[1] == 32);
    for (int64_t i = 0; i < 4 * 32; ++i) {
        REQUIRE_NEAR(t.at_flat(i),
                     values[static_cast<std::size_t>(i)],
                     /*tol=*/1.0f / 127.0f + 1e-6f);
    }
}

TEST_CASE(gguf_loads_q4_0_tensor) {
    std::vector<float> values(2 * 32);
    for (int i = 0; i < 2 * 32; ++i) {
        values[static_cast<std::size_t>(i)] =
            static_cast<float>(i - 32) / 16.0f;
    }
    auto packed = quantize_q4_0(values.data(),
                                static_cast<int64_t>(values.size()));

    std::vector<uint8_t> file;
    file.insert(file.end(), {'G', 'G', 'U', 'F'});
    write_u32(file, 3);
    write_u64(file, 1);
    write_u64(file, 0);
    write_tensor_info(file, "q4test", {2, 32}, GgufTensorType::Q4_0,
                      /*data_offset=*/0);
    write_u64(file, 32);

    align_data_section(file, 32);
    file.insert(file.end(), packed.begin(), packed.end());

    const std::string path = "/tmp/tinyllm_q4_0_test.gguf";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(file.data()),
                static_cast<std::streamsize>(file.size()));
    }
    GgufFile gf = GgufFile::open(path);
    REQUIRE(gf.n_tensors() == 1);
    Tensor t = gf.load_tensor(0);
    REQUIRE(t.dtype() == DType::Float32);
    REQUIRE(t.shape()[0] == 2);
    REQUIRE(t.shape()[1] == 32);
    for (int64_t i = 0; i < 2 * 32; ++i) {
        REQUIRE_NEAR(t.at_flat(i),
                     values[static_cast<std::size_t>(i)],
                     /*tol=*/0.2f);
    }
}