//! SIMD-accelerated squared Euclidean distance.
//!
//! Two code paths:
//!   * `f32` — used by the builder's k-means (full precision, build time only).
//!   * `i16` — used at query time. Reference vectors are quantized to 16-bit
//!     fixed point (×`SCALE`); the index is half the size (fits in the RAM
//!     budget) and one 256-bit register holds all 16 dims. Distances are
//!     accumulated in `i64` so the ×10000 scale is numerically exact for the
//!     4-decimal dataset and preserves the exact 5-NN ordering.
//!
//! AVX2 implementations are compiled when the `avx2` target feature is enabled
//! (`-C target-cpu=haswell`); otherwise a scalar fallback with identical
//! semantics is used (e.g. local development on ARM).
//!
//! We never take a square root — squared distance preserves ordering.

/// Fixed-point scale for i16 quantization. Values lie in [-1, 1]; ×10000 maps
/// them to [-10000, 10000], exactly representing the 4-decimal source data.
pub const SCALE: f32 = 10_000.0;

/// A 16-float vector aligned to 32 bytes (used by the builder's f32 path).
#[repr(align(32))]
#[derive(Clone, Copy, Debug)]
pub struct AlignedVec(pub [f32; 16]);

impl AlignedVec {
    #[inline]
    pub const fn zeroed() -> Self {
        AlignedVec([0.0; 16])
    }
}

impl Default for AlignedVec {
    #[inline]
    fn default() -> Self {
        Self::zeroed()
    }
}

/// Quantizes a 16-float vector to 16-bit fixed point (×`SCALE`, rounded).
#[inline]
pub fn quantize_i16(v: &[f32; 16]) -> [i16; 16] {
    let mut q = [0i16; 16];
    for i in 0..16 {
        let s = (v[i] * SCALE).round();
        q[i] = s.clamp(-32767.0, 32767.0) as i16;
    }
    q
}

// ---------------------------------------------------------------------------
// AVX2 implementation (x86_64 + avx2).
// ---------------------------------------------------------------------------
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod imp {
    use std::arch::x86_64::*;

    // ---- f32 (builder / k-means) ----
    #[target_feature(enable = "avx2")]
    pub unsafe fn squared_euclidean(a: &[f32; 16], b: &[f32; 16]) -> f32 {
        let a0 = _mm256_loadu_ps(a.as_ptr());
        let a1 = _mm256_loadu_ps(a.as_ptr().add(8));
        let b0 = _mm256_loadu_ps(b.as_ptr());
        let b1 = _mm256_loadu_ps(b.as_ptr().add(8));
        let d0 = _mm256_sub_ps(a0, b0);
        let d1 = _mm256_sub_ps(a1, b1);
        let acc = _mm256_fmadd_ps(d1, d1, _mm256_mul_ps(d0, d0));
        hsum256_ps(acc)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn hsum256_ps(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        let sums = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(sums)
    }

    // ---- i16 (query time) ----
    #[target_feature(enable = "avx2")]
    pub unsafe fn squared_euclidean_i16(a: &[i16; 16], b: &[i16; 16]) -> i64 {
        let av = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
        let bv = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
        let d = _mm256_sub_epi16(av, bv); // 16×i16, |d| ≤ 20000 fits i16
        hsum_madd_i64(d)
    }

    /// `madd(d,d)` → 8×i32 (each ≤ 8e8), widened to i64 and horizontally summed.
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_madd_i64(d: __m256i) -> i64 {
        let madd = _mm256_madd_epi16(d, d); // 8×i32
        let lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(madd)); // 4×i64
        let hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(madd, 1));
        let s = _mm256_add_epi64(lo, hi); // 4×i64
        let s_lo = _mm256_castsi256_si128(s);
        let s_hi = _mm256_extracti128_si256(s, 1);
        let p = _mm_add_epi64(s_lo, s_hi); // 2×i64
        let p_hi = _mm_unpackhi_epi64(p, p);
        let r = _mm_add_epi64(p, p_hi);
        _mm_cvtsi128_si64(r)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn distances_to_slice_i16(
        query: &[i16; 16],
        vectors: &[[i16; 16]],
        out: &mut [i64],
    ) {
        const PREFETCH_AHEAD: usize = 4;
        let qv = _mm256_loadu_si256(query.as_ptr() as *const __m256i);
        for i in 0..vectors.len() {
            if i + PREFETCH_AHEAD < vectors.len() {
                _mm_prefetch(
                    vectors.as_ptr().add(i + PREFETCH_AHEAD) as *const i8,
                    _MM_HINT_T0,
                );
            }
            let bv = _mm256_loadu_si256(vectors[i].as_ptr() as *const __m256i);
            let d = _mm256_sub_epi16(qv, bv);
            out[i] = hsum_madd_i64(d);
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback (everything else, e.g. ARM dev machines).
// ---------------------------------------------------------------------------
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
mod imp {
    #[inline]
    pub unsafe fn squared_euclidean(a: &[f32; 16], b: &[f32; 16]) -> f32 {
        let mut acc = 0.0f32;
        for i in 0..16 {
            let d = a[i] - b[i];
            acc += d * d;
        }
        acc
    }

    #[inline]
    pub unsafe fn squared_euclidean_i16(a: &[i16; 16], b: &[i16; 16]) -> i64 {
        let mut acc = 0i64;
        for i in 0..16 {
            let d = a[i] as i64 - b[i] as i64;
            acc += d * d;
        }
        acc
    }

    #[inline]
    pub unsafe fn distances_to_slice_i16(
        query: &[i16; 16],
        vectors: &[[i16; 16]],
        out: &mut [i64],
    ) {
        for (i, v) in vectors.iter().enumerate() {
            out[i] = squared_euclidean_i16(query, v);
        }
    }
}

// ---------------------------------------------------------------------------
// Public wrappers.
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two padded 16-float vectors (f32 path).
#[inline]
pub fn squared_euclidean(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    // SAFETY: the AVX2 impl is only compiled when avx2 is statically enabled.
    unsafe { imp::squared_euclidean(a, b) }
}

/// Squared Euclidean distance between two quantized 16-int16 vectors (i64 acc).
#[inline]
pub fn squared_euclidean_i16(a: &[i16; 16], b: &[i16; 16]) -> i64 {
    unsafe { imp::squared_euclidean_i16(a, b) }
}

/// Distances from one i16 query to a contiguous slice of i16 vectors.
/// `out.len()` must be `>= vectors.len()`.
#[inline]
pub fn distances_to_slice_i16(query: &[i16; 16], vectors: &[[i16; 16]], out: &mut [i64]) {
    debug_assert!(out.len() >= vectors.len());
    unsafe { imp::distances_to_slice_i16(query, vectors, out) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_f32(a: &[f32; 16], b: &[f32; 16]) -> f32 {
        (0..16).map(|i| (a[i] - b[i]).powi(2)).sum()
    }

    fn naive_i16(a: &[i16; 16], b: &[i16; 16]) -> i64 {
        (0..16).map(|i| (a[i] as i64 - b[i] as i64).pow(2)).sum()
    }

    #[test]
    fn f32_matches_naive() {
        let mut a = [0.0f32; 16];
        let mut b = [0.0f32; 16];
        for i in 0..14 {
            a[i] = (i as f32) * 0.1;
            b[i] = (i as f32) * 0.05 - 0.3;
        }
        assert!((squared_euclidean(&a, &b) - naive_f32(&a, &b)).abs() < 1e-3);
    }

    #[test]
    fn i16_matches_naive() {
        let mut a = [0i16; 16];
        let mut b = [0i16; 16];
        for i in 0..16 {
            a[i] = (i as i16 - 8) * 1000;
            b[i] = (i as i16) * 700 - 3000;
        }
        assert_eq!(squared_euclidean_i16(&a, &b), naive_i16(&a, &b));
    }

    #[test]
    fn i16_quantize_roundtrip_and_distance() {
        let mut a = [0.0f32; 16];
        let mut b = [0.0f32; 16];
        a[0] = 0.5;
        a[5] = -1.0;
        b[0] = 0.5;
        b[5] = -1.0;
        let qa = quantize_i16(&a);
        let qb = quantize_i16(&b);
        // identical vectors → zero distance, sentinel preserved.
        assert_eq!(qa[5], -10000);
        assert_eq!(squared_euclidean_i16(&qa, &qb), 0);
        // Now b[5] = 0.5 (5000) vs a[5] = -1.0 (-10000): diff 15000.
        // dim0 matches, so distance = 15000^2 = 225_000_000.
        b[5] = 0.5;
        let qb = quantize_i16(&b);
        assert_eq!(squared_euclidean_i16(&qa, &qb), 225_000_000);
    }

    #[test]
    fn i16_no_overflow_extremes() {
        // Worst case: all dims at opposite extremes.
        let a = [10000i16; 16];
        let b = [-10000i16; 16];
        // 16 × 20000^2 = 6.4e9, must be exact in i64.
        assert_eq!(squared_euclidean_i16(&a, &b), 16 * 20000i64 * 20000);
    }

    #[test]
    fn batch_matches_single_i16() {
        let q = [123i16; 16];
        let vecs = [[100i16; 16], [-50i16; 16], [10000i16; 16]];
        let mut out = [0i64; 3];
        distances_to_slice_i16(&q, &vecs, &mut out);
        for i in 0..3 {
            assert_eq!(out[i], squared_euclidean_i16(&q, &vecs[i]));
        }
    }
}
