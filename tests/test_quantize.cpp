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
#include <algorithm>

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

#if TINYLLM_ENABLE_AVX2 && TINYLLM_ENABLE_AVX512
// Phase 13: AVX-512 fused Q4_0 kernel must agree with the AVX2 kernel
// on the same input. The two kernels have different reduction trees
// (AVX2: 4 vectors with separate mul + 4-way add; AVX-512: 1 FMA + 1
// mul + add per lo/hi pair, but a wider hsum). That means fp rounding
// can diverge by a few ULPs; the absolute tolerance below is the same
// as the AVX2-vs-F32 tolerance used elsewhere in this file.
TEST_CASE(matvec_q4_0_avx2_vs_avx512) {
    const int64_t M = 16, K = 512;
    auto a = random_floats(static_cast<std::size_t>(M * K), /*seed=*/31);
    auto x = random_floats(static_cast<std::size_t>(K),     /*seed=*/32);
    auto packed = quantize_q4_0(a.data(), M * K);

    std::vector<float> y_avx2(static_cast<std::size_t>(M));
    std::vector<float> y_avx512(static_cast<std::size_t>(M));
    ops::matvec_q4_0_f32_avx2(packed.data(), M, K, x.data(), y_avx2.data());
    ops::matvec_q4_0_f32_avx512(packed.data(), M, K, x.data(), y_avx512.data());

    for (int64_t mi = 0; mi < M; ++mi) {
        REQUIRE_NEAR(y_avx2[static_cast<std::size_t>(mi)],
                     y_avx512[static_cast<std::size_t>(mi)],
                     /*tol=*/1e-2f * static_cast<float>(K));
    }
}
#endif  // TINYLLM_ENABLE_AVX2 && TINYLLM_ENABLE_AVX512

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

// =============================================================================
// K-quant tests (Q4_K, Q5_K, Q6_K).
//
// We test by constructing a synthetic super-block with hand-picked
// values, dequantizing it, and verifying the output matches the formula:
//   Q4_K: y[i] = d * sc * q - dmin * m
//   Q5_K: y[i] = d * sc * (ql + (qh_bit ? 16 : 0)) - dmin * m
//   Q6_K: y[i] = d * sc[is] * (ql|qh_hi2 - 32)
// where (sc, m) come from get_scale_min_k4 unpacking of the 12-byte
// scales array.
//
// For GGUF-loading tests we write a synthetic file with a known tensor
// and verify load_tensor returns the expected dequantized values.
// =============================================================================

// Build a 12-byte scales buffer using the same packing scheme as
// llama.cpp's quantize_row_q4_K_ref (lines 1586-1599 in ggml-quants.c).
// `Ls[j]` = scale index (0..63), `Lm[j]` = min index (0..63) for
// sub-block j (j = 0..7). The result goes into 12 bytes.
static void pack_q4_K_scales(uint8_t out[12],
                             const uint8_t Ls[8], const uint8_t Lm[8]) {
    for (int i = 0; i < 12; ++i) out[i] = 0;
    for (int j = 0; j < 8; ++j) {
        uint8_t ls = Ls[j];
        uint8_t lm = Lm[j];
        if (j < 4) {
            out[j]     = ls;
            out[j + 4] = lm;
        } else {
            out[j + 4] = static_cast<uint8_t>((ls & 0xF) | ((lm & 0xF) << 4));
            out[j - 4] = static_cast<uint8_t>(out[j - 4] | ((ls >> 4) << 6));
            out[j - 0] = static_cast<uint8_t>(out[j - 0] | ((lm >> 4) << 6));
        }
    }
}

// Reference `get_scale_min_k4` mirrored here so tests can predict the
// (sc, m) values.
static void get_scale_min_k4_ref(int j, const uint8_t* q,
                                 uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

TEST_CASE(q4_K_dequant_matches_formula) {
    // Construct one Q4_K super-block (256 elements) with d = 0.5,
    // dmin = 0.25, scales[0..7] = 5..12, mins[0..7] = 1..8, and qs
    // containing a 4-bit pattern (lo = i%16, hi = (i+8)%16).
    float d   = 0.5f;
    float dm  = 0.25f;
    uint8_t Ls[8] = {5, 6, 7, 8, 9, 10, 11, 12};
    uint8_t Lm[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t scales[12];
    pack_q4_K_scales(scales, Ls, Lm);

    std::vector<uint8_t> packed(kQ4_KBlockBytes);
    std::memcpy(packed.data(), &d,  sizeof(float));
    std::memcpy(packed.data() + 4, &dm, sizeof(float));
    // overwrite with f16 versions (the loader expects f16, not f32)
    uint16_t d_h = quantize_f32_to_f16(d);
    uint16_t dm_h = quantize_f32_to_f16(dm);
    std::memcpy(packed.data(),     &d_h,  sizeof(uint16_t));
    std::memcpy(packed.data() + 2, &dm_h, sizeof(uint16_t));
    std::memcpy(packed.data() + 4, scales, 12);

    // Fill qs with the pattern. dequantize_q4_K reads 32 bytes per 64
    // output elements (low nibble + high nibble), so 4 iterations of 32
    // bytes = 128 bytes total cover the 256-element super-block.
    uint8_t* qs = packed.data() + 4 + 12;
    for (int chunk = 0; chunk < 4; ++chunk) {
        // Each chunk of 64 elements (32 bytes of qs, low+high nibbles).
        for (int j = 0; j < 32; ++j) {
            int i = chunk * 64 + j;        // low nibble index
            int k = chunk * 64 + j + 32;   // high nibble index
            uint8_t lo = static_cast<uint8_t>(i % 16);
            uint8_t hi = static_cast<uint8_t>(k % 16);
            qs[chunk * 32 + j] = static_cast<uint8_t>(lo | (hi << 4));
        }
    }

    std::vector<float> out(kQK_K);
    dequantize_q4_K(packed.data(), kQK_K, out.data());

    // Now compute the expected output using the same formula.
    uint8_t sc_l, m_l;
    for (int chunk = 0; chunk < 4; chunk += 1) {
        int is_lo = chunk * 2;     // sub-block index for low nibbles
        int is_hi = chunk * 2 + 1; // sub-block index for high nibbles
        get_scale_min_k4_ref(is_lo, scales, &sc_l, &m_l);
        float d_lo = d * sc_l;
        float m_lo = dm * m_l;
        get_scale_min_k4_ref(is_hi, scales, &sc_l, &m_l);
        float d_hi = d * sc_l;
        float m_hi = dm * m_l;
        for (int j = 0; j < 32; ++j) {
            int idx_lo = chunk * 64 + j;
            int idx_hi = chunk * 64 + j + 32;
            uint8_t lo = static_cast<uint8_t>(idx_lo % 16);
            uint8_t hi = static_cast<uint8_t>(idx_hi % 16);
            float exp_lo = d_lo * lo - m_lo;
            float exp_hi = d_hi * hi - m_hi;
            REQUIRE_NEAR(out[idx_lo], exp_lo, 1e-5);
            REQUIRE_NEAR(out[idx_hi], exp_hi, 1e-5);
        }
    }
}

TEST_CASE(q5_K_dequant_matches_formula) {
    // d = 0.5, dmin = 0.25 (powers of 2 so they round-trip exactly
    // through f16), simple scale/min values, qs/qh encoding values
    // that exercise the qh-bit-set path (lo > 15) and unset path
    // (lo <= 15).
    float d  = 0.5f;
    float dm = 0.25f;
    uint8_t Ls[8] = {4, 4, 4, 4, 4, 4, 4, 4};
    uint8_t Lm[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    uint8_t scales[12];
    pack_q4_K_scales(scales, Ls, Lm);

    std::vector<uint8_t> packed(kQ5_KBlockBytes);
    uint16_t d_h  = quantize_f32_to_f16(d);
    uint16_t dm_h = quantize_f32_to_f16(dm);
    std::memcpy(packed.data(),     &d_h,  sizeof(uint16_t));
    std::memcpy(packed.data() + 2, &dm_h, sizeof(uint16_t));
    std::memcpy(packed.data() + 4, scales, 12);

    // qs (128 bytes, packed 4-bit) and qh (32 bytes, 1 bit per element
    // arranged in the same interleaved pattern as the dequant uses:
    // u1/u2 advance every 64 elements).
    uint8_t* ql = packed.data() + 4 + 12;
    uint8_t* qh = ql + kQK_K / 2;
    std::memset(ql, 0, kQK_K / 2);
    std::memset(qh, 0, kQK_K / 8);

    // Strategy: every 64-element chunk, alternate setting the qh bits so
    // we exercise both "hi bit set" and "hi bit unset". qh is a 32-byte
    // table that all 4 iterations share: at iteration `chunk` the
    // dequant reads qh[l] with masks u1=(1<<(2*chunk)) (lo) and
    // u2=(2<<(2*chunk)) (hi). So each byte of qh holds 4 (lo, hi) bit
    // pairs for 4 (lo, hi) element pairs across the 4 iterations.
    for (int chunk = 0; chunk < 4; ++chunk) {
        bool set_hi_bit = (chunk % 2 == 0);
        for (int j = 0; j < 32; ++j) {
            int lo_idx = chunk * 64 + j;
            int hi_idx = chunk * 64 + j + 32;
            uint8_t lo_val = static_cast<uint8_t>(j % 16);
            uint8_t hi_val = static_cast<uint8_t>((j + 7) % 16);
            ql[chunk * 32 + j] = static_cast<uint8_t>(lo_val | (hi_val << 4));
            if (set_hi_bit) {
                // The dequant reads qh[l] with masks u1=(1<<(2*chunk))
                // for lo bit, u2=(2<<(2*chunk)) for hi bit. So bit
                // positions in qh[l] are 2*chunk and 2*chunk+1.
                qh[j] = static_cast<uint8_t>(
                    qh[j] | (1u << (2 * chunk)) | (2u << (2 * chunk)));
            }
        }
    }

    std::vector<float> out(kQK_K);
    dequantize_q5_K(packed.data(), kQK_K, out.data());

    uint8_t sc_l, m_l;
    for (int chunk = 0; chunk < 4; ++chunk) {
        int is_lo = chunk * 2;
        int is_hi = chunk * 2 + 1;
        get_scale_min_k4_ref(is_lo, scales, &sc_l, &m_l);
        float d_lo = d * sc_l;
        float m_lo = dm * m_l;
        get_scale_min_k4_ref(is_hi, scales, &sc_l, &m_l);
        float d_hi = d * sc_l;
        float m_hi = dm * m_l;
        bool set_hi_bit = (chunk % 2 == 0);
        for (int j = 0; j < 32; ++j) {
            int idx_lo = chunk * 64 + j;
            int idx_hi = chunk * 64 + j + 32;
            uint8_t lo_val = static_cast<uint8_t>(j % 16);
            uint8_t hi_val = static_cast<uint8_t>((j + 7) % 16);
            int q_lo = lo_val + (set_hi_bit ? 16 : 0);
            int q_hi = hi_val + (set_hi_bit ? 16 : 0);
            float exp_lo = d_lo * q_lo - m_lo;
            float exp_hi = d_hi * q_hi - m_hi;
            REQUIRE_NEAR(out[idx_lo], exp_lo, 1e-5);
            REQUIRE_NEAR(out[idx_hi], exp_hi, 1e-5);
        }
    }
}

TEST_CASE(q6_K_dequant_matches_formula) {
    // d = 0.5, all 16 scales = 2, qs/qh encoding 6-bit values 0..63
    // centered at 32. The dequant subtracts 32 from the reconstructed
    // 6-bit value, so a value of 30 → -2, 32 → 0, 35 → +3.
    float d = 0.5f;
    int8_t sc[16];
    for (int i = 0; i < 16; ++i) sc[i] = 2;

    std::vector<uint8_t> packed(kQ6_KBlockBytes);
    uint8_t* ql = packed.data();
    uint8_t* qh = ql + kQK_K / 2;
    int8_t*  sccol = reinterpret_cast<int8_t*>(qh + kQK_K / 4);
    uint16_t d_h = quantize_f32_to_f16(d);
    std::memset(ql, 0, kQK_K / 2);
    std::memset(qh, 0, kQK_K / 4);
    std::memcpy(sccol, sc, 16);
    std::memcpy(sccol + 16, &d_h, sizeof(uint16_t));

    // Fill ql and qh so that for each of 16 sub-blocks (16 elements
    // each), the reconstructed value is `block_idx * 2`. So sub-block
    // 0 → 0, sub-block 1 → 2, ... sub-block 7 → 14, sub-block 8 → 16,
    // ..., sub-block 15 → 30.
    for (int n_off = 0; n_off < kQK_K; n_off += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;  // 0..1 within this 128-element strip
            // Four quads at l-th position: idx 0, +32, +64, +96.
            // Each occupies sub-block is+0, is+2, is+4, is+6 in the
            // 16-block sc[].
            int target_val[4] = {
                (is + 0) * 2, (is + 2) * 2, (is + 4) * 2, (is + 6) * 2
            };
            for (int q = 0; q < 4; ++q) {
                int v = target_val[q];            // 0..30
                int centered = v;                  // we already pass 0..30
                int lo4 = centered & 0xF;          // low 4 bits
                int hi2 = (centered >> 4) & 0x3;   // upper 2 bits
                if (q == 0) {
                    ql[l + 0] = static_cast<uint8_t>(
                        (ql[l + 0] & 0xF0) | lo4);
                    qh[l] = static_cast<uint8_t>(
                        (qh[l] & 0xFC) | hi2);
                } else if (q == 1) {
                    ql[l + 32] = static_cast<uint8_t>(
                        (ql[l + 32] & 0xF0) | lo4);
                    qh[l] = static_cast<uint8_t>(
                        (qh[l] & 0xF3) | (hi2 << 2));
                } else if (q == 2) {
                    ql[l + 0] = static_cast<uint8_t>(
                        (ql[l + 0] & 0x0F) | (lo4 << 4));
                    qh[l] = static_cast<uint8_t>(
                        (qh[l] & 0xCF) | (hi2 << 4));
                } else {
                    ql[l + 32] = static_cast<uint8_t>(
                        (ql[l + 32] & 0x0F) | (lo4 << 4));
                    qh[l] = static_cast<uint8_t>(
                        (qh[l] & 0x3F) | (hi2 << 6));
                }
            }
        }
        ql += 64;
        qh += 32;
    }

    std::vector<float> out(kQK_K);
    dequantize_q6_K(packed.data(), kQK_K, out.data());

    // Expected: y[i] = d * sc[is] * (q6_value - 32) where
    // is = element_index / 16. The actual encoded value depends on
    // the (l, q) position: 0..30 with the layout from the encoding
    // pass above. We rebuild it exactly here.
    for (int n_off = 0; n_off < kQK_K; n_off += 128) {
        for (int l = 0; l < 32; ++l) {
            int is_in_strip = l / 16;
            int q_offset[4] = {0, 32, 64, 96};
            int q_subblk[4] = {is_in_strip + 0, is_in_strip + 2,
                               is_in_strip + 4, is_in_strip + 6};
            for (int q = 0; q < 4; ++q) {
                int i = n_off + l + q_offset[q];
                int sub = q_subblk[q];
                int v = sub * 2;          // value 0..30 we encoded
                float expected = d * sc[sub] * static_cast<float>(v - 32);
                REQUIRE_NEAR(out[i], expected, 1e-5);
            }
        }
    }
}

TEST_CASE(q4_K_zero_block_dequantizes_to_zero) {
    // d=0, dmin=0, all scales=0, all qs=0 → all output zero.
    std::vector<uint8_t> packed(kQ4_KBlockBytes, 0);
    std::vector<float> out(kQK_K, 99.0f);  // sentinel
    dequantize_q4_K(packed.data(), kQK_K, out.data());
    for (float v : out) REQUIRE(v == 0.0f);
}

TEST_CASE(q5_K_zero_block_dequantizes_to_zero) {
    std::vector<uint8_t> packed(kQ5_KBlockBytes, 0);
    std::vector<float> out(kQK_K, 99.0f);
    dequantize_q5_K(packed.data(), kQK_K, out.data());
    for (float v : out) REQUIRE(v == 0.0f);
}

TEST_CASE(q6_K_zero_block_dequantizes_to_zero) {
    std::vector<uint8_t> packed(kQ6_KBlockBytes, 0);
    std::vector<float> out(kQK_K, 99.0f);
    dequantize_q6_K(packed.data(), kQK_K, out.data());
    for (float v : out) REQUIRE(v == 0.0f);
}

TEST_CASE(gguf_loads_q4_K_tensor_all_zero) {
    // Write a synthetic GGUF v3 file with a single Q4_K tensor whose
    // data is all zeros. The load path should dequantize to all-zero F32.
    std::vector<uint8_t> file;
    file.insert(file.end(), {'G', 'G', 'U', 'F'});
    write_u32(file, 3);
    write_u64(file, 1);
    write_u64(file, 0);
    write_tensor_info(file, "q4k_zero", {kQK_K},
                      GgufTensorType::Q4_K, /*data_offset=*/0);
    write_u64(file, 32);
    align_data_section(file, 32);
    std::vector<uint8_t> block(kQ4_KBlockBytes, 0);
    file.insert(file.end(), block.begin(), block.end());

    const std::string path = "/tmp/tinyllm_q4_K_zero.gguf";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(file.data()),
                static_cast<std::streamsize>(file.size()));
    }
    GgufFile gf = GgufFile::open(path);
    REQUIRE(gf.n_tensors() == 1);
    Tensor t = gf.load_tensor(0);
    REQUIRE(t.dtype() == DType::Float32);
    REQUIRE(t.numel() == kQK_K);
    for (int64_t i = 0; i < t.numel(); ++i) {
        REQUIRE_NEAR(t.at_flat(i), 0.0f, 1e-6);
    }
}

TEST_CASE(gguf_loads_q5_K_tensor_all_zero) {
    std::vector<uint8_t> file;
    file.insert(file.end(), {'G', 'G', 'U', 'F'});
    write_u32(file, 3);
    write_u64(file, 1);
    write_u64(file, 0);
    write_tensor_info(file, "q5k_zero", {kQK_K},
                      GgufTensorType::Q5_K, /*data_offset=*/0);
    write_u64(file, 32);
    align_data_section(file, 32);
    std::vector<uint8_t> block(kQ5_KBlockBytes, 0);
    file.insert(file.end(), block.begin(), block.end());

    const std::string path = "/tmp/tinyllm_q5_K_zero.gguf";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(file.data()),
                static_cast<std::streamsize>(file.size()));
    }
    GgufFile gf = GgufFile::open(path);
    Tensor t = gf.load_tensor(0);
    REQUIRE(t.dtype() == DType::Float32);
    REQUIRE(t.numel() == kQK_K);
    for (int64_t i = 0; i < t.numel(); ++i) {
        REQUIRE_NEAR(t.at_flat(i), 0.0f, 1e-6);
    }
}

TEST_CASE(gguf_loads_q6_K_tensor_all_zero) {
    std::vector<uint8_t> file;
    file.insert(file.end(), {'G', 'G', 'U', 'F'});
    write_u32(file, 3);
    write_u64(file, 1);
    write_u64(file, 0);
    write_tensor_info(file, "q6k_zero", {kQK_K},
                      GgufTensorType::Q6_K, /*data_offset=*/0);
    write_u64(file, 32);
    align_data_section(file, 32);
    std::vector<uint8_t> block(kQ6_KBlockBytes, 0);
    file.insert(file.end(), block.begin(), block.end());

    const std::string path = "/tmp/tinyllm_q6_K_zero.gguf";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(file.data()),
                static_cast<std::streamsize>(file.size()));
    }
    GgufFile gf = GgufFile::open(path);
    Tensor t = gf.load_tensor(0);
    REQUIRE(t.dtype() == DType::Float32);
    REQUIRE(t.numel() == kQK_K);
    for (int64_t i = 0; i < t.numel(); ++i) {
        REQUIRE_NEAR(t.at_flat(i), 0.0f, 1e-6);
    }
}

TEST_CASE(matmul_q4_K_matches_f32) {
    // Dequantize a Q4_K matrix, do F32 matmul on the dequantized
    // result, and check that matmul_q4_K_f32 gives the same answer.
    int64_t M = 4;
    int64_t K = 2 * kQK_K;  // 2 super-blocks per row
    std::vector<float> w(static_cast<std::size_t>(M * K));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& v : w) v = dist(rng);

    // Build a packed Q4_K matrix where each super-block has d = block
    // amax / 15 (so dequantize recovers close to the original). This is
    // a cheap "fake quantize" — not bit-exact to llama.cpp but close
    // enough to verify the matmul path. We use d=0.1, dmin=0, sc=15
    // (max scale value), and qs encoding the quant-rounded values.
    std::size_t bytes_per_row = (K / kQK_K) * kQ4_KBlockBytes;
    std::vector<uint8_t> qmat(static_cast<std::size_t>(M) * bytes_per_row);
    float d_global = 0.05f;        // pick a scale that maps into 0..15
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t sb = 0; sb < K / kQK_K; ++sb) {
            uint8_t* p = qmat.data() + m * bytes_per_row + sb * kQ4_KBlockBytes;
            uint16_t d_h = quantize_f32_to_f16(d_global);
            uint16_t dm_h = quantize_f32_to_f16(0.0f);
            std::memcpy(p, &d_h, sizeof(uint16_t));
            std::memcpy(p + 2, &dm_h, sizeof(uint16_t));
            // Scales: sc_l = 15, m_l = 0 for all 8 sub-blocks.
            uint8_t scales[12];
            uint8_t Ls[8] = {15, 15, 15, 15, 15, 15, 15, 15};
            uint8_t Lm[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            pack_q4_K_scales(scales, Ls, Lm);
            std::memcpy(p + 4, scales, 12);
// qs: each 4-bit nibble = round(w / (d * sc))
            int64_t base = m * K + sb * kQK_K;
            uint8_t* qs = p + 4 + 12;
            for (int chunk = 0; chunk < 4; ++chunk) {
                for (int j = 0; j < 32; ++j) {
                    int64_t idx_lo = chunk * 64 + j;
                    int64_t idx_hi = chunk * 64 + j + 32;
                    uint8_t lo = static_cast<uint8_t>(
                        std::clamp(static_cast<int>(
                            std::round(w[base + idx_lo] / (d_global * 15.0f))),
                            0, 15));
                    uint8_t hi = static_cast<uint8_t>(
                        std::clamp(static_cast<int>(
                            std::round(w[base + idx_hi] / (d_global * 15.0f))),
                            0, 15));
                    qs[chunk * 32 + j] = static_cast<uint8_t>(lo | (hi << 4));
                }
            }
        }
    }

    std::vector<float> x(static_cast<std::size_t>(K));
    for (auto& v : x) v = dist(rng);
    std::vector<float> y_q(static_cast<std::size_t>(M));
    std::vector<float> y_f(static_cast<std::size_t>(M));
    matmul_q4_K_f32(qmat.data(), M, K, x.data(), y_q.data());

    // F32 reference: dequantize then matmul.
    std::vector<float> deq(static_cast<std::size_t>(M * K));
    for (int64_t m = 0; m < M; ++m) {
        dequantize_q4_K(qmat.data() + m * bytes_per_row, K,
                        deq.data() + m * K);
    }
    for (int64_t m = 0; m < M; ++m) {
        double acc = 0.0;
        for (int64_t k = 0; k < K; ++k) {
            acc += static_cast<double>(deq[m * K + k]) *
                   static_cast<double>(x[k]);
        }
        y_f[m] = static_cast<float>(acc);
    }
    for (int64_t m = 0; m < M; ++m) {
        REQUIRE_NEAR(y_q[m], y_f[m], 1e-3);
    }
}