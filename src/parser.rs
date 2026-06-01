//! Hand-rolled, zero-allocation JSON field extractor for the fraud-score
//! request payload.
//!
//! The payload has a fixed structure and nesting:
//!
//! ```json
//! {
//!   "id": "tx-...",
//!   "transaction": { "amount", "installments", "requested_at" },
//!   "customer":    { "avg_amount", "tx_count_24h", "known_merchants" },
//!   "merchant":    { "id", "mcc", "avg_amount" },
//!   "terminal":    { "is_online", "card_present", "km_from_home" },
//!   "last_transaction": null | { "timestamp", "km_from_current" }
//! }
//! ```
//!
//! Two keys (`avg_amount` and `id`) appear in more than one object, so we first
//! locate each object's anchor with `memchr::memmem::find` and then resolve
//! every field relative to that anchor. No heap allocation: all borrowed data
//! is a slice into the original request bytes.

use memchr::memmem;

/// Maximum number of known merchants we will track for the membership check.
pub const MAX_MERCHANTS: usize = 16;

/// A parsed fraud-score request. All slices borrow from the input buffer.
#[derive(Debug)]
pub struct TransactionPayload<'a> {
    pub amount: f32,
    pub installments: u32,
    pub requested_at: [u8; 20],
    pub avg_amount: f32,
    pub tx_count_24h: u32,
    pub known_merchants: [&'a [u8]; MAX_MERCHANTS],
    pub known_merchants_len: usize,
    pub merchant_id: &'a [u8],
    pub mcc: [u8; 4],
    pub merchant_avg_amount: f32,
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
    pub has_last_transaction: bool,
    pub last_tx_timestamp: Option<[u8; 20]>,
    pub km_from_current: Option<f32>,
}

impl<'a> TransactionPayload<'a> {
    /// True when `merchant_id` is not among the known merchants. Duplicates in
    /// the known-merchants list do not affect the result.
    #[inline]
    pub fn is_unknown_merchant(&self) -> bool {
        for i in 0..self.known_merchants_len {
            if self.known_merchants[i] == self.merchant_id {
                return false;
            }
        }
        true
    }
}

#[inline]
fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() {
        match b[i] {
            b' ' | b'\t' | b'\n' | b'\r' => i += 1,
            _ => break,
        }
    }
    i
}

/// Returns the index immediately after `key` (i.e. the start of the value,
/// before whitespace is skipped). `key` must include the trailing colon, e.g.
/// `b"\"amount\":"`.
#[inline]
fn value_start(b: &[u8], key: &[u8]) -> Option<usize> {
    memmem::find(b, key).map(|p| p + key.len())
}

/// Parses a JSON number (optionally signed, no exponent) starting at `i`.
/// Returns the parsed value. Whitespace before the number is skipped.
#[inline]
fn parse_f32(b: &[u8], i: usize) -> Option<f32> {
    let mut i = skip_ws(b, i);
    let start = i;
    let mut neg = false;
    if i < b.len() && b[i] == b'-' {
        neg = true;
        i += 1;
    }
    let mut int_part: f32 = 0.0;
    let mut saw_digit = false;
    while i < b.len() && b[i].is_ascii_digit() {
        int_part = int_part * 10.0 + (b[i] - b'0') as f32;
        i += 1;
        saw_digit = true;
    }
    let mut value = int_part;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let mut frac: f32 = 0.0;
        let mut scale: f32 = 1.0;
        while i < b.len() && b[i].is_ascii_digit() {
            frac = frac * 10.0 + (b[i] - b'0') as f32;
            scale *= 10.0;
            i += 1;
            saw_digit = true;
        }
        value += frac / scale;
    }
    if !saw_digit {
        let _ = start;
        return None;
    }
    Some(if neg { -value } else { value })
}

/// Parses an unsigned integer starting at `i` (whitespace skipped first).
#[inline]
fn parse_u32(b: &[u8], i: usize) -> Option<u32> {
    let mut i = skip_ws(b, i);
    let mut v: u32 = 0;
    let mut saw = false;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v.wrapping_mul(10).wrapping_add((b[i] - b'0') as u32);
        i += 1;
        saw = true;
    }
    if saw {
        Some(v)
    } else {
        None
    }
}

/// Parses a JSON boolean starting at `i` (whitespace skipped first).
#[inline]
fn parse_bool(b: &[u8], i: usize) -> Option<bool> {
    let i = skip_ws(b, i);
    if b[i..].starts_with(b"true") {
        Some(true)
    } else if b[i..].starts_with(b"false") {
        Some(false)
    } else {
        None
    }
}

/// Returns the contents of the JSON string whose opening quote is the first
/// non-whitespace byte at/after `i`, as a slice, plus the index just past the
/// closing quote.
#[inline]
fn parse_string<'a>(b: &'a [u8], i: usize) -> Option<(&'a [u8], usize)> {
    let i = skip_ws(b, i);
    if i >= b.len() || b[i] != b'"' {
        return None;
    }
    let start = i + 1;
    let end = start + memchr::memchr(b'"', &b[start..])?;
    Some((&b[start..end], end + 1))
}

/// Parse the request body into a [`TransactionPayload`]. Returns `None` on any
/// malformed input; the caller falls back to the safe default response.
pub fn parse(buf: &[u8]) -> Option<TransactionPayload<'_>> {
    // Locate each object anchor so duplicated keys resolve unambiguously.
    let tx = memmem::find(buf, b"\"transaction\":")?;
    let cust = memmem::find(buf, b"\"customer\":")?;
    let merch = memmem::find(buf, b"\"merchant\":")?;
    let term = memmem::find(buf, b"\"terminal\":")?;
    let last = memmem::find(buf, b"\"last_transaction\":")?;

    // --- transaction ---
    let amount = parse_f32(buf, value_start(&buf[tx..], b"\"amount\":")? + tx)?;
    let installments = parse_u32(buf, value_start(&buf[tx..], b"\"installments\":")? + tx)?;
    let requested_at = {
        let p = value_start(&buf[tx..], b"\"requested_at\":")? + tx;
        let (s, _) = parse_string(buf, p)?;
        copy20(s)?
    };

    // --- customer ---
    let avg_amount = parse_f32(buf, value_start(&buf[cust..], b"\"avg_amount\":")? + cust)?;
    let tx_count_24h = parse_u32(buf, value_start(&buf[cust..], b"\"tx_count_24h\":")? + cust)?;

    let (known_merchants, known_merchants_len) = {
        let p = value_start(&buf[cust..], b"\"known_merchants\":")? + cust;
        parse_string_array(buf, p)?
    };

    // --- merchant ---
    let merchant_id = {
        let p = value_start(&buf[merch..], b"\"id\":")? + merch;
        parse_string(buf, p)?.0
    };
    let mcc = {
        let p = value_start(&buf[merch..], b"\"mcc\":")? + merch;
        let (s, _) = parse_string(buf, p)?;
        copy4(s)?
    };
    let merchant_avg_amount =
        parse_f32(buf, value_start(&buf[merch..], b"\"avg_amount\":")? + merch)?;

    // --- terminal ---
    let is_online = parse_bool(buf, value_start(&buf[term..], b"\"is_online\":")? + term)?;
    let card_present = parse_bool(buf, value_start(&buf[term..], b"\"card_present\":")? + term)?;
    let km_from_home = parse_f32(buf, value_start(&buf[term..], b"\"km_from_home\":")? + term)?;

    // --- last_transaction (may be null) ---
    let last_val = skip_ws(buf, last + "\"last_transaction\":".len());
    let (has_last_transaction, last_tx_timestamp, km_from_current) =
        if buf[last_val..].starts_with(b"null") {
            (false, None, None)
        } else {
            let ts = {
                let p = value_start(&buf[last..], b"\"timestamp\":")? + last;
                let (s, _) = parse_string(buf, p)?;
                copy20(s)?
            };
            let km = parse_f32(
                buf,
                value_start(&buf[last..], b"\"km_from_current\":")? + last,
            )?;
            (true, Some(ts), Some(km))
        };

    Some(TransactionPayload {
        amount,
        installments,
        requested_at,
        avg_amount,
        tx_count_24h,
        known_merchants,
        known_merchants_len,
        merchant_id,
        mcc,
        merchant_avg_amount,
        is_online,
        card_present,
        km_from_home,
        has_last_transaction,
        last_tx_timestamp,
        km_from_current,
    })
}

/// Scans a JSON array of strings starting at the `[`, collecting up to
/// `MAX_MERCHANTS` entries as slices.
#[inline]
fn parse_string_array<'a>(b: &'a [u8], i: usize) -> Option<([&'a [u8]; MAX_MERCHANTS], usize)> {
    let mut i = skip_ws(b, i);
    if i >= b.len() || b[i] != b'[' {
        return None;
    }
    i += 1;
    let mut out: [&[u8]; MAX_MERCHANTS] = [b""; MAX_MERCHANTS];
    let mut n = 0;
    loop {
        i = skip_ws(b, i);
        if i >= b.len() {
            return None;
        }
        if b[i] == b']' {
            break;
        }
        if b[i] == b',' {
            i += 1;
            continue;
        }
        if b[i] == b'"' {
            let (s, next) = parse_string(b, i)?;
            if n < MAX_MERCHANTS {
                out[n] = s;
                n += 1;
            }
            i = next;
        } else {
            return None;
        }
    }
    Some((out, n))
}

#[inline]
fn copy20(s: &[u8]) -> Option<[u8; 20]> {
    if s.len() < 20 {
        return None;
    }
    let mut out = [0u8; 20];
    out.copy_from_slice(&s[..20]);
    Some(out)
}

#[inline]
fn copy4(s: &[u8]) -> Option<[u8; 4]> {
    if s.len() < 4 {
        return None;
    }
    let mut out = [0u8; 4];
    out.copy_from_slice(&s[..4]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NULL_LAST: &[u8] = br#"{
        "id": "tx-1329056812",
        "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
        "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003","MERC-016"] },
        "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
        "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.2331036248 },
        "last_transaction": null
    }"#;

    const WITH_LAST: &[u8] = br#"{"id":"tx-3576980410","transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-009","MERC-001","MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7090520965},"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8626479774}}"#;

    #[test]
    fn parses_null_last_transaction() {
        let p = parse(NULL_LAST).unwrap();
        assert!((p.amount - 41.12).abs() < 1e-9);
        assert_eq!(p.installments, 2);
        assert_eq!(&p.requested_at, b"2026-03-11T18:45:53Z");
        assert!((p.avg_amount - 82.24).abs() < 1e-9);
        assert_eq!(p.tx_count_24h, 3);
        assert_eq!(p.known_merchants_len, 2);
        assert_eq!(p.merchant_id, b"MERC-016");
        assert_eq!(&p.mcc, b"5411");
        assert!((p.merchant_avg_amount - 60.25).abs() < 1e-9);
        assert!(!p.is_online);
        assert!(p.card_present);
        assert!((p.km_from_home - 29.2331036248).abs() < 1e-9);
        assert!(!p.has_last_transaction);
        assert!(p.last_tx_timestamp.is_none());
        assert!(p.km_from_current.is_none());
        // MERC-016 is in the known list → known merchant.
        assert!(!p.is_unknown_merchant());
    }

    #[test]
    fn parses_compact_with_last_transaction() {
        let p = parse(WITH_LAST).unwrap();
        assert!((p.amount - 384.88).abs() < 1e-9);
        assert_eq!(p.merchant_id, b"MERC-001");
        assert_eq!(&p.mcc, b"5912");
        assert!((p.merchant_avg_amount - 298.95).abs() < 1e-9);
        assert!(p.has_last_transaction);
        assert_eq!(&p.last_tx_timestamp.unwrap(), b"2026-03-11T14:58:35Z");
        assert!((p.km_from_current.unwrap() - 18.8626479774).abs() < 1e-9);
        // Duplicates in the list, MERC-001 present → known.
        assert_eq!(p.known_merchants_len, 4);
        assert!(!p.is_unknown_merchant());
    }

    #[test]
    fn detects_unknown_merchant() {
        // merchant.id (MERC-999) is absent from known_merchants → unknown.
        const UNKNOWN: &[u8] = br#"{"id":"tx-1","transaction":{"amount":10.0,"installments":1,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":10.0,"tx_count_24h":1,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-999","mcc":"5411","avg_amount":10.0},"terminal":{"is_online":true,"card_present":false,"km_from_home":1.0},"last_transaction":null}"#;
        let p = parse(UNKNOWN).unwrap();
        assert_eq!(p.merchant_id, b"MERC-999");
        assert!(p.is_unknown_merchant());
    }

    #[test]
    fn malformed_returns_none() {
        assert!(parse(b"not json").is_none());
        assert!(parse(b"{}").is_none());
    }
}
