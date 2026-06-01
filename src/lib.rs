//! Shared library root for the Rinha de Backend 2026 fraud detection API.
//!
//! Re-exports every module so both the `api` and `builder` binaries can use
//! them, and defines the constants that are fixed by the challenge spec.

pub mod distance;
pub mod vectorizer;
pub mod index;
pub mod parser;
pub mod responses;

// Low-level networking (epoll, SCM_RIGHTS) is Linux-only.
#[cfg(target_os = "linux")]
pub mod net;

// ---------------------------------------------------------------------------
// Normalization constants (from resources/normalization.json). These never
// change, so they are hardcoded rather than read at runtime.
// ---------------------------------------------------------------------------
pub const MAX_AMOUNT: f32 = 10_000.0;
pub const MAX_INSTALLMENTS: f32 = 12.0;
pub const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
pub const MAX_MINUTES: f32 = 1_440.0;
pub const MAX_KM: f32 = 1_000.0;
pub const MAX_TX_COUNT_24H: f32 = 20.0;
pub const MAX_MERCHANT_AVG_AMOUNT: f32 = 10_000.0;

// ---------------------------------------------------------------------------
// Vector / KNN constants.
// ---------------------------------------------------------------------------
/// Number of real feature dimensions.
pub const DIMS: usize = 14;
/// Padded dimensionality for AVX2 alignment (last 2 floats are always 0.0).
pub const DIMS_PADDED: usize = 16;
/// Number of nearest neighbours to consider.
pub const K: usize = 5;
/// Fraud threshold: a transaction is denied when fraud_score >= THRESHOLD.
pub const THRESHOLD: f32 = 0.6;

/// Returns the risk weight for a merchant category code (MCC).
///
/// There are only 10 known MCCs; everything else defaults to 0.5. Implemented
/// as a `match` over the 4 ASCII digit bytes so there is no runtime HashMap.
#[inline]
pub fn mcc_risk(mcc: &[u8; 4]) -> f32 {
    match mcc {
        b"5411" => 0.15,
        b"5812" => 0.30,
        b"5912" => 0.20,
        b"5944" => 0.45,
        b"7801" => 0.80,
        b"7802" => 0.75,
        b"7995" => 0.85,
        b"4511" => 0.35,
        b"5311" => 0.25,
        b"5999" => 0.50,
        _ => 0.5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_mccs_have_expected_risk() {
        assert_eq!(mcc_risk(b"5411"), 0.15);
        assert_eq!(mcc_risk(b"7995"), 0.85);
        assert_eq!(mcc_risk(b"5999"), 0.50);
    }

    #[test]
    fn unknown_mcc_defaults_to_half() {
        assert_eq!(mcc_risk(b"0000"), 0.5);
        assert_eq!(mcc_risk(b"9999"), 0.5);
    }
}
