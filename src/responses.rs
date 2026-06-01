//! Pre-formatted HTTP responses. `fraud_score` can only be 0.0/0.2/0.4/0.6/
//! 0.8/1.0 and `approved` is `fraud_score < 0.6`, so there are exactly six
//! possible response bodies. Everything here is `'static` — no runtime
//! formatting or allocation.

/// The six response bodies, indexed by fraud_count (0..=5).
pub const BODIES: [&[u8]; 6] = [
    br#"{"approved":true,"fraud_score":0.0}"#,
    br#"{"approved":true,"fraud_score":0.2}"#,
    br#"{"approved":true,"fraud_score":0.4}"#,
    br#"{"approved":false,"fraud_score":0.6}"#,
    br#"{"approved":false,"fraud_score":0.8}"#,
    br#"{"approved":false,"fraud_score":1.0}"#,
];

/// Safe fallback body used on any error: approve with score 0.0.
pub const FALLBACK_BODY: &[u8] = br#"{"approved":true,"fraud_score":0.0}"#;

/// Complete HTTP/1.1 responses (status line + headers + body) for each
/// fraud_count, with a pre-computed Content-Length and keep-alive.
pub const RESPONSES: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

/// The complete fallback HTTP response (== `RESPONSES[0]`).
pub const FALLBACK_RESPONSE: &[u8] = RESPONSES[0];

/// `GET /ready` → 200 with empty body, keep-alive.
pub const READY_RESPONSE: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";

/// Unknown route → 404, keep-alive.
pub const NOTFOUND_RESPONSE: &[u8] =
    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";

/// Returns the JSON body for a given fraud_count, falling back for any value
/// outside 0..=5.
#[inline]
pub fn response_body(fraud_count: usize) -> &'static [u8] {
    if fraud_count < BODIES.len() {
        BODIES[fraud_count]
    } else {
        FALLBACK_BODY
    }
}

/// Returns the complete HTTP response bytes for a given fraud_count.
#[inline]
pub fn full_response(fraud_count: usize) -> &'static [u8] {
    if fraud_count < RESPONSES.len() {
        RESPONSES[fraud_count]
    } else {
        FALLBACK_RESPONSE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bodies_have_expected_approval() {
        for c in 0..=2 {
            assert!(response_body(c).windows(4).any(|w| w == b"true"));
        }
        for c in 3..=5 {
            assert!(response_body(c).windows(5).any(|w| w == b"false"));
        }
    }

    #[test]
    fn content_length_matches_body() {
        for c in 0..6 {
            let body = BODIES[c];
            let resp = RESPONSES[c];
            // The response must end with exactly its body.
            assert!(resp.ends_with(body), "response {c} body mismatch");
            // And the declared Content-Length must equal the body length.
            let needle = format!("Content-Length: {}\r\n", body.len());
            let hay = std::str::from_utf8(resp).unwrap();
            assert!(hay.contains(&needle), "response {c} wrong content-length");
        }
    }

    #[test]
    fn fallback_is_approve_zero() {
        assert_eq!(FALLBACK_BODY, BODIES[0]);
        assert_eq!(FALLBACK_RESPONSE, RESPONSES[0]);
    }

    #[test]
    fn out_of_range_uses_fallback() {
        assert_eq!(response_body(99), FALLBACK_BODY);
        assert_eq!(full_response(99), FALLBACK_RESPONSE);
    }
}
