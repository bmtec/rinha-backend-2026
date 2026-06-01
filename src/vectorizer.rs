//! Transforms a parsed [`TransactionPayload`] into a 16-float padded vector
//! (14 real dimensions + 2 zero padding for AVX2 alignment).

use crate::parser::TransactionPayload;
use crate::{
    mcc_risk, AMOUNT_VS_AVG_RATIO, MAX_AMOUNT, MAX_INSTALLMENTS, MAX_KM,
    MAX_MERCHANT_AVG_AMOUNT, MAX_MINUTES, MAX_TX_COUNT_24H,
};

#[inline]
fn clamp(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

/// Parses two ASCII digits at `b[i..i+2]` into an integer.
#[inline]
fn d2(b: &[u8], i: usize) -> i64 {
    ((b[i] - b'0') as i64) * 10 + (b[i + 1] - b'0') as i64
}

/// Parses four ASCII digits at `b[i..i+4]` into an integer.
#[inline]
fn d4(b: &[u8], i: usize) -> i64 {
    ((b[i] - b'0') as i64) * 1000
        + ((b[i + 1] - b'0') as i64) * 100
        + ((b[i + 2] - b'0') as i64) * 10
        + (b[i + 3] - b'0') as i64
}

/// Day-of-week using Tomohiko Sakamoto's algorithm. Returns 0=Sunday..6=Saturday.
#[inline]
fn sakamoto(y: i64, m: i64, d: i64) -> i64 {
    const T: [i64; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if m < 3 { y - 1 } else { y };
    (y + y / 4 - y / 100 + y / 400 + T[(m - 1) as usize] + d).rem_euclid(7)
}

/// Days since 1970-01-01 (Howard Hinnant's days_from_civil).
#[inline]
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Total minutes since the epoch for an ISO-8601 timestamp `2026-03-11T18:45:53Z`.
#[inline]
fn total_minutes(ts: &[u8; 20]) -> i64 {
    let year = d4(ts, 0);
    let month = d2(ts, 5);
    let day = d2(ts, 8);
    let hour = d2(ts, 11);
    let minute = d2(ts, 14);
    days_from_civil(year, month, day) * 1440 + hour * 60 + minute
}

/// Hour of day (UTC) from the timestamp, as a float.
#[inline]
fn hour_of(ts: &[u8; 20]) -> f32 {
    d2(ts, 11) as f32
}

/// Weekday remapped to Monday=0 .. Sunday=6.
#[inline]
fn weekday_mon0(ts: &[u8; 20]) -> f32 {
    let year = d4(ts, 0);
    let month = d2(ts, 5);
    let day = d2(ts, 8);
    // Sakamoto: 0=Sunday. Remap so Monday=0 .. Sunday=6.
    ((sakamoto(year, month, day) + 6) % 7) as f32
}

/// Computes the 14-dimensional feature vector (padded to 16).
pub fn vectorize(p: &TransactionPayload) -> [f32; 16] {
    let mut v = [0.0f32; 16];

    v[0] = clamp(p.amount as f32 / MAX_AMOUNT);
    v[1] = clamp(p.installments as f32 / MAX_INSTALLMENTS);

    // amount vs customer average; guard against a zero average.
    v[2] = if p.avg_amount > 0.0 {
        clamp((p.amount as f32 / p.avg_amount as f32) / AMOUNT_VS_AVG_RATIO)
    } else {
        0.0
    };

    v[3] = hour_of(&p.requested_at) / 23.0;
    v[4] = weekday_mon0(&p.requested_at) / 6.0;

    // Dimensions 5 & 6 use the -1 sentinel when there is no last transaction.
    if p.has_last_transaction {
        if let Some(last_ts) = p.last_tx_timestamp.as_ref() {
            let diff = (total_minutes(&p.requested_at) - total_minutes(last_ts)) as f32;
            v[5] = clamp(diff / MAX_MINUTES);
        } else {
            v[5] = -1.0;
        }
        v[6] = match p.km_from_current {
            Some(km) => clamp(km as f32 / MAX_KM),
            None => -1.0,
        };
    } else {
        v[5] = -1.0;
        v[6] = -1.0;
    }

    v[7] = clamp(p.km_from_home as f32 / MAX_KM);
    v[8] = clamp(p.tx_count_24h as f32 / MAX_TX_COUNT_24H);
    v[9] = if p.is_online { 1.0 } else { 0.0 };
    v[10] = if p.card_present { 1.0 } else { 0.0 };
    v[11] = if p.is_unknown_merchant() { 1.0 } else { 0.0 };
    v[12] = mcc_risk(&p.mcc);
    v[13] = clamp(p.merchant_avg_amount as f32 / MAX_MERCHANT_AVG_AMOUNT);

    // v[14], v[15] remain 0.0 (padding).
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn day_of_week_known_dates() {
        // 2000-01-01 was a Saturday → Monday=0 mapping gives 5.
        assert_eq!(weekday_mon0(b"2000-01-01T00:00:00Z"), 5.0);
        // 2026-03-11 is a Wednesday → 2.
        assert_eq!(weekday_mon0(b"2026-03-11T00:00:00Z"), 2.0);
    }

    #[test]
    fn hour_extraction() {
        assert_eq!(hour_of(b"2026-03-11T18:45:53Z"), 18.0);
        assert_eq!(hour_of(b"2026-03-11T03:00:00Z"), 3.0);
    }

    #[test]
    fn minutes_diff_same_day() {
        // 18:45 - 14:58 = 3h47m = 227 minutes.
        let a = *b"2026-03-11T18:45:53Z";
        let b = *b"2026-03-11T14:58:35Z";
        assert_eq!(total_minutes(&a) - total_minutes(&b), 227);
    }

    #[test]
    fn null_last_transaction_yields_sentinels() {
        const NULL_LAST: &[u8] = br#"{"id":"t","transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},"last_transaction":null}"#;
        let p = parse(NULL_LAST).unwrap();
        let v = vectorize(&p);
        assert_eq!(v[5], -1.0);
        assert_eq!(v[6], -1.0);
        assert_eq!(v[9], 0.0); // is_online false
        assert_eq!(v[10], 1.0); // card_present true
        assert_eq!(v[11], 0.0); // known merchant
        assert!((v[12] - 0.15).abs() < 1e-6); // mcc 5411
        assert_eq!(v[14], 0.0);
        assert_eq!(v[15], 0.0);
    }

    #[test]
    fn dimensions_are_clamped() {
        const BIG: &[u8] = br#"{"id":"t","transaction":{"amount":999999.0,"installments":99,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":1.0,"tx_count_24h":999,"known_merchants":["MERC-001"]},"merchant":{"id":"MERC-016","mcc":"7995","avg_amount":999999.0},"terminal":{"is_online":true,"card_present":false,"km_from_home":99999.0},"last_transaction":null}"#;
        let p = parse(BIG).unwrap();
        let v = vectorize(&p);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 1.0);
        assert_eq!(v[2], 1.0);
        assert_eq!(v[7], 1.0);
        assert_eq!(v[8], 1.0);
        assert_eq!(v[11], 1.0); // MERC-016 not in [MERC-001]
        assert!((v[12] - 0.85).abs() < 1e-6); // mcc 7995
    }
}
