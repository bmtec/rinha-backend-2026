//! IVF (Inverted File) index, queried over a memory-mapped binary file.
//!
//! Vectors and centroids are stored as 16-bit fixed point (×`SCALE`) so the
//! whole index fits in the per-instance RAM budget (~96 MB vs ~192 MB for f32)
//! and one 256-bit register holds all 16 dims. Distances accumulate in i64.
//!
//! File layout (little-endian, produced by `builder.rs`):
//!
//! ```text
//! [Header]   magic:u32 version:u32 num_vectors:u32 num_centroids:u32 dims_padded:u32
//! [Centroids]    num_centroids × 16 × i16
//! [Cell offsets] num_centroids × (start:u32, count:u32, block_start:u32, block_count:u32)
//! [Block bounds] sum(ceil(cell.count / BLOCK_SIZE)) × (min:16×i16, max:16×i16)
//! [Vectors]      num_vectors  × 16 × i16  (sorted by cell assignment)
//! [Labels]       num_vectors  × u8        (0 = legit, 1 = fraud)
//! ```

use crate::distance::{distances_to_slice_i16, quantize_i16, SCALE};
use crate::K;

pub const MAGIC: u32 = 0x52494E48; // "RINH"
pub const VERSION: u32 = 3; // v3: i16 quantized vectors + block bounds
pub const HEADER_BYTES: usize = 20;
pub const BLOCK_SIZE: usize = 128;

/// Largest nprobe we will honour (bounds a stack buffer).
const MAX_NPROBE: usize = 256;
/// Largest centroid count we will honour (bounds a stack buffer; 4096 × i64
/// = 32 KiB). The current index uses 2048 centroids; keeping this tight avoids
/// clearing a cold 128 KiB stack buffer on every query.
const MAX_CENTROIDS: usize = 4096;
/// How many cell vectors to score per batch (bounds a stack scratch buffer).
const SCAN_CHUNK: usize = 1024;
/// Candidate buffer kept from the int16 scan, then re-ranked in exact f32.
/// Must be ≥ K with enough margin to contain the true 5-NN despite int16
/// ordering noise on near ties.
const K_RERANK: usize = 8;

#[repr(C)]
#[derive(Clone, Copy)]
struct Cell {
    start: u32,
    count: u32,
    block_start: u32,
    block_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockBounds {
    min: [i16; 16],
    max: [i16; 16],
}

const _: () = assert!(std::mem::size_of::<Cell>() == 16);
const _: () = assert!(std::mem::size_of::<BlockBounds>() == 64);

/// A read-only IVF index backed by a memory-mapped slice.
pub struct IvfIndex<'a> {
    pub num_vectors: usize,
    pub num_centroids: usize,
    centroids: &'a [[i16; 16]],
    cells: &'a [Cell],
    blocks: &'a [BlockBounds],
    vectors: &'a [[i16; 16]],
    labels: &'a [u8],
}

impl<'a> IvfIndex<'a> {
    /// Builds an index view over the raw mmap bytes, validating the header and
    /// that every section fits within `data`.
    pub fn from_bytes(data: &'a [u8]) -> Option<Self> {
        if data.len() < HEADER_BYTES {
            return None;
        }
        let magic = read_u32(data, 0);
        let version = read_u32(data, 4);
        let num_vectors = read_u32(data, 8) as usize;
        let num_centroids = read_u32(data, 12) as usize;
        let dims_padded = read_u32(data, 16) as usize;

        if magic != MAGIC || version != VERSION || dims_padded != 16 {
            return None;
        }

        let centroids_bytes = num_centroids * 16 * 2;
        let cells_bytes = num_centroids * std::mem::size_of::<Cell>();
        let vectors_bytes = num_vectors * 16 * 2;
        let labels_bytes = num_vectors;

        let centroids_off = HEADER_BYTES;
        let cells_off = centroids_off + centroids_bytes;
        let cells_end = cells_off + cells_bytes;
        if data.len() < cells_end {
            return None;
        }
        let cells = unsafe {
            std::slice::from_raw_parts(data.as_ptr().add(cells_off) as *const Cell, num_centroids)
        };
        let mut num_blocks = 0usize;
        for cell in cells {
            let start = cell.start as usize;
            let count = cell.count as usize;
            if start + count > num_vectors {
                return None;
            }
            let end = cell.block_start as usize + cell.block_count as usize;
            num_blocks = num_blocks.max(end);
        }

        let blocks_bytes = num_blocks * std::mem::size_of::<BlockBounds>();
        let blocks_off = cells_end;
        let vectors_off = blocks_off + blocks_bytes;
        let labels_off = vectors_off + vectors_bytes;
        let total = labels_off + labels_bytes;
        if data.len() < total {
            return None;
        }

        // SAFETY: offsets are 2-byte aligned by construction (HEADER_BYTES=20
        // and every preceding section is a multiple of 2 bytes), and the bounds
        // above guarantee each section fits within `data`. We use unaligned
        // SIMD loads, so 2-byte alignment for i16 is sufficient.
        let centroids = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr().add(centroids_off) as *const [i16; 16],
                num_centroids,
            )
        };
        let blocks = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr().add(blocks_off) as *const BlockBounds,
                num_blocks,
            )
        };
        let vectors = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr().add(vectors_off) as *const [i16; 16],
                num_vectors,
            )
        };
        let labels = &data[labels_off..labels_off + labels_bytes];

        Some(IvfIndex {
            num_vectors,
            num_centroids,
            centroids,
            cells,
            blocks,
            vectors,
            labels,
        })
    }

    /// Returns `(fraud_count_in_5, fraud_score)` for an f32 query vector.
    pub fn query(&self, vector: &[f32; 16], nprobe: usize) -> (u8, f32) {
        let q = quantize_i16(vector);
        let nc = self.num_centroids.min(MAX_CENTROIDS);
        let probe = nprobe.clamp(1, MAX_NPROBE).min(nc);
        let adaptive_probe = if probe == 10 || probe == 12 {
            48.min(nc)
        } else {
            probe
        };

        // 1. Distance to every centroid.
        let mut cdist = [0i64; MAX_CENTROIDS];
        distances_to_slice_i16(&q, &self.centroids[..nc], &mut cdist[..nc]);

        // 2. The nearest centroids (ascending by distance). For tuned adaptive
        // modes, keep 48 centroids but scan only the cheap prefix unless the
        // first pass lands on a known risky boundary pattern.
        let mut best: [(i64, u32); MAX_NPROBE] = [(i64::MAX, 0); MAX_NPROBE];
        let mut blen = 0usize;
        for i in 0..nc {
            insert_sorted(&mut best, &mut blen, adaptive_probe, cdist[i], i as u32);
        }

        // 3. Scan the chosen cells in fast int16, keeping the K_RERANK nearest
        //    candidates (int16 distance, global vector index).
        let mut cand: [(i64, u32); K_RERANK] = [(i64::MAX, 0); K_RERANK];
        let mut filled = 0usize;
        let mut scratch = [0i64; SCAN_CHUNK];

        self.scan_centroid_range(&q, &best, 0, probe, &mut cand, &mut filled, &mut scratch);
        let (mut fraud, bits) = self.rerank_candidates(vector, &cand, filled);

        if adaptive_probe > probe && is_risky_pattern(probe, bits, best[probe - 1].0, best[probe].0)
        {
            self.scan_centroid_range(
                &q,
                &best,
                probe,
                adaptive_probe,
                &mut cand,
                &mut filled,
                &mut scratch,
            );
            fraud = self.rerank_candidates(vector, &cand, filled).0;
        }

        (fraud, fraud as f32 / K as f32)
    }

    #[inline]
    fn scan_centroid_range(
        &self,
        q: &[i16; 16],
        best: &[(i64, u32); MAX_NPROBE],
        from: usize,
        to: usize,
        cand: &mut [(i64, u32); K_RERANK],
        filled: &mut usize,
        scratch: &mut [i64; SCAN_CHUNK],
    ) {
        for b in from..to {
            let cell = self.cells[best[b].1 as usize];
            let start = cell.start as usize;
            let count = cell.count as usize;
            if start + count > self.num_vectors {
                continue;
            }
            let block_start = cell.block_start as usize;
            let block_count = cell.block_count as usize;
            if block_start + block_count > self.blocks.len() {
                continue;
            }

            for rel_block in 0..block_count {
                let local_off = rel_block * BLOCK_SIZE;
                if local_off >= count {
                    break;
                }
                let n = (count - local_off).min(BLOCK_SIZE).min(SCAN_CHUNK);
                if *filled == K_RERANK {
                    let bound = lower_bound_block(
                        q,
                        &self.blocks[block_start + rel_block],
                        cand[K_RERANK - 1].0,
                    );
                    if bound >= cand[K_RERANK - 1].0 {
                        continue;
                    }
                }

                let off = start + local_off;
                let cell_vecs = &self.vectors[off..off + n];
                distances_to_slice_i16(q, cell_vecs, &mut scratch[..n]);
                for j in 0..n {
                    let d = scratch[j];
                    if *filled < K_RERANK || d < cand[K_RERANK - 1].0 {
                        insert_cand(cand, filled, d, (off + j) as u32);
                    }
                }
            }
        }
    }

    #[inline]
    fn rerank_candidates(
        &self,
        vector: &[f32; 16],
        cand: &[(i64, u32); K_RERANK],
        filled: usize,
    ) -> (u8, u8) {
        let mut top: [(f32, u8); K] = [(f32::INFINITY, 0); K];
        let mut tfill = 0usize;
        for c in 0..filled {
            let idx = cand[c].1 as usize;
            let v = &self.vectors[idx];
            let mut d = 0.0f32;
            for i in 0..16 {
                let x = v[i] as f32 / SCALE;
                let e = vector[i] - x;
                d += e * e;
            }
            if tfill < K || d < top[K - 1].0 {
                insert_topk_f32(&mut top, &mut tfill, d, self.labels[idx]);
            }
        }

        let mut fraud = 0u8;
        let mut bits = 0u8;
        for i in 0..tfill {
            if top[i].1 == 1 {
                fraud += 1;
                bits |= 1 << i;
            }
        }
        (fraud, bits)
    }
}

#[inline]
fn lower_bound_block(q: &[i16; 16], block: &BlockBounds, limit: i64) -> i64 {
    let mut acc = 0i64;
    for d in 0..16 {
        let x = q[d] as i64;
        let lo = block.min[d] as i64;
        let hi = block.max[d] as i64;
        let diff = if x < lo {
            lo - x
        } else if x > hi {
            x - hi
        } else {
            0
        };
        acc += diff * diff;
        if acc >= limit {
            return acc;
        }
    }
    acc
}

#[inline]
fn is_risky_pattern(probe: usize, bits: u8, centroid_probe: i64, centroid_next: i64) -> bool {
    let centroid_gap = centroid_next - centroid_probe;
    if probe == 10 {
        return match bits {
            0b00110 => centroid_gap <= 500_000,
            0b01010 => centroid_gap <= 500_000,
            0b01100 => centroid_gap <= 600_000,
            0b10010 => centroid_gap <= 1_200_000,
            0b10011 => centroid_gap <= 500_000,
            0b10110 => centroid_gap <= 700_000,
            0b11100 => centroid_gap <= 150_000,
            _ => false,
        };
    }

    if probe == 12 {
        return match bits {
            0b00110 => centroid_gap <= 1_600_000,
            0b01010 => centroid_gap <= 3_800_000,
            0b01100 => centroid_gap <= 1_000_000,
            0b10010 => centroid_gap <= 1_800_000,
            0b10011 => centroid_gap <= 500_000,
            0b11100 => centroid_gap <= 150_000,
            _ => false,
        };
    }

    false
}

#[inline]
fn read_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Inserts `(dist, payload)` into an ascending-by-distance array capped at
/// `cap`, keeping the smallest entries.
#[inline]
fn insert_sorted(arr: &mut [(i64, u32)], len: &mut usize, cap: usize, dist: i64, payload: u32) {
    if *len < cap {
        let mut i = *len;
        while i > 0 && arr[i - 1].0 > dist {
            arr[i] = arr[i - 1];
            i -= 1;
        }
        arr[i] = (dist, payload);
        *len += 1;
    } else if dist < arr[cap - 1].0 {
        let mut i = cap - 1;
        while i > 0 && arr[i - 1].0 > dist {
            arr[i] = arr[i - 1];
            i -= 1;
        }
        arr[i] = (dist, payload);
    }
}

/// Inserts a candidate `(int16 dist, global index)` into the ascending
/// candidate buffer of capacity `K_RERANK`.
#[inline]
fn insert_cand(arr: &mut [(i64, u32); K_RERANK], len: &mut usize, dist: i64, idx: u32) {
    if *len < K_RERANK {
        let mut i = *len;
        while i > 0 && arr[i - 1].0 > dist {
            arr[i] = arr[i - 1];
            i -= 1;
        }
        arr[i] = (dist, idx);
        *len += 1;
    } else if dist < arr[K_RERANK - 1].0 {
        let mut i = K_RERANK - 1;
        while i > 0 && arr[i - 1].0 > dist {
            arr[i] = arr[i - 1];
            i -= 1;
        }
        arr[i] = (dist, idx);
    }
}

/// Inserts `(f32 dist, label)` into the ascending top-K array (final ranking).
#[inline]
fn insert_topk_f32(arr: &mut [(f32, u8); K], len: &mut usize, dist: f32, label: u8) {
    if *len < K {
        let mut i = *len;
        while i > 0 && arr[i - 1].0 > dist {
            arr[i] = arr[i - 1];
            i -= 1;
        }
        arr[i] = (dist, label);
        *len += 1;
    } else if dist < arr[K - 1].0 {
        let mut i = K - 1;
        while i > 0 && arr[i - 1].0 > dist {
            arr[i] = arr[i - 1];
            i -= 1;
        }
        arr[i] = (dist, label);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::quantize_i16;

    /// Build a tiny valid index in memory: 2 centroids, a few vectors.
    fn build_tiny() -> Vec<u8> {
        let num_centroids = 2usize;
        let vecs = [
            mk(0.0), // cell 0
            mk(0.1),
            mk(0.2),
            mk(0.9), // cell 1
            mk(1.0),
            mk(0.95),
        ];
        let labels: Vec<u8> = vec![0, 1, 0, 1, 1, 1];
        let centroids = [mk(0.1), mk(0.95)];
        let cells = [(0u32, 3u32, 0u32, 1u32), (3u32, 3u32, 1u32, 1u32)];
        let blocks = [bounds(&vecs[0..3]), bounds(&vecs[3..6])];

        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(vecs.len() as u32).to_le_bytes());
        out.extend_from_slice(&(num_centroids as u32).to_le_bytes());
        out.extend_from_slice(&16u32.to_le_bytes());
        for c in &centroids {
            for x in c {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        for (s, c, bs, bc) in &cells {
            out.extend_from_slice(&s.to_le_bytes());
            out.extend_from_slice(&c.to_le_bytes());
            out.extend_from_slice(&bs.to_le_bytes());
            out.extend_from_slice(&bc.to_le_bytes());
        }
        for b in &blocks {
            for x in &b.min {
                out.extend_from_slice(&x.to_le_bytes());
            }
            for x in &b.max {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        for v in &vecs {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out.extend_from_slice(&labels);
        out
    }

    fn bounds(vecs: &[[i16; 16]]) -> BlockBounds {
        let mut min = [i16::MAX; 16];
        let mut max = [i16::MIN; 16];
        for v in vecs {
            for d in 0..16 {
                min[d] = min[d].min(v[d]);
                max[d] = max[d].max(v[d]);
            }
        }
        BlockBounds { min, max }
    }

    /// Build a quantized vector with a single non-zero leading dim.
    fn mk(x: f32) -> [i16; 16] {
        let mut v = [0.0f32; 16];
        v[0] = x;
        quantize_i16(&v)
    }

    fn qf(x: f32) -> [f32; 16] {
        let mut v = [0.0f32; 16];
        v[0] = x;
        v
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = build_tiny();
        bytes[0] = 0xFF;
        assert!(IvfIndex::from_bytes(&bytes).is_none());
    }

    #[test]
    fn rejects_old_version() {
        let mut bytes = build_tiny();
        bytes[4] = 1; // version 1
        assert!(IvfIndex::from_bytes(&bytes).is_none());
    }

    #[test]
    fn query_finds_legit_cluster() {
        let bytes = build_tiny();
        let idx = IvfIndex::from_bytes(&bytes).unwrap();
        // Query near the legit cluster (cell 0), labels 0,1,0; then fraud 1,1.
        let (fraud, score) = idx.query(&qf(0.05), 2);
        assert_eq!(fraud, 3);
        assert!((score - 0.6).abs() < 1e-6);
    }

    #[test]
    fn query_pure_fraud_cluster() {
        let bytes = build_tiny();
        let idx = IvfIndex::from_bytes(&bytes).unwrap();
        let (fraud, _) = idx.query(&qf(0.97), 1);
        assert_eq!(fraud, 3);
    }
}
