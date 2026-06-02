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
//! [Cell bounds]  num_centroids × (min:16×i16, max:16×i16)
//! [Block bounds] sum(ceil(cell.count / BLOCK_SIZE)) × (min:16×i16, max:16×i16)
//! [Vectors]      num_vectors  × 16 × i16  (sorted by cell assignment)
//! [Labels]       num_vectors  × u8        (0 = legit, 1 = fraud)
//! ```

use crate::distance::{distances_to_slice_i16, quantize_i16, SCALE};
use crate::K;

pub const MAGIC: u32 = 0x52494E48; // "RINH"
pub const VERSION: u32 = 5; // v5: v4 + per-cell bounds for repair
pub const MIN_SUPPORTED_VERSION: u32 = 3;
pub const HEADER_BYTES: usize = 20;
pub const BLOCK_SIZE: usize = 128;
const SECTION_ALIGN: usize = 32;

/// Largest nprobe we will honour (bounds a stack buffer). The adaptive path
/// expands to at most 48 cells, so 64 leaves room for manual tuning without
/// paying for a cold 256-entry stack buffer on every request.
const MAX_NPROBE: usize = 64;
/// Largest centroid count we will honour. The baked index is built with 2048
/// centroids.
const MAX_CENTROIDS: usize = 2048;
/// How many cell vectors to score per batch. Scans are block-oriented, so this
/// only needs to match `BLOCK_SIZE`; a larger scratch array is just stack churn.
const SCAN_CHUNK: usize = BLOCK_SIZE;
/// Candidate buffer kept from the int16 scan, then re-ranked in exact f32.
/// Kept at K after Xeon validation showed the quantized top-5 preserved zero
/// detection errors while tightening block pruning.
const K_RERANK: usize = 5;
const CENTROID_WORDS: usize = MAX_CENTROIDS.div_ceil(64);
const MAX_REPAIR_CANDIDATES: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepairMode {
    /// Current production-safe hand-tuned risky-pattern expansion.
    Pattern,
    /// Expand by centroid distance whenever the top-5 fraud count is ambiguous.
    AmbiguousCentroid,
    /// Expand ambiguous cases by cell bounding-box lower bound.
    Bbox,
}

#[derive(Clone, Copy, Debug)]
pub struct QueryOptions {
    pub nprobe: usize,
    pub repair_mode: RepairMode,
    pub repair_min: u8,
    pub repair_max: u8,
    pub repair_candidates: usize,
}

impl QueryOptions {
    #[inline]
    pub const fn new(nprobe: usize) -> Self {
        QueryOptions {
            nprobe,
            repair_mode: RepairMode::Pattern,
            repair_min: 1,
            repair_max: 4,
            repair_candidates: 64,
        }
    }
}

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
    cell_bounds: Option<&'a [BlockBounds]>,
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

        if magic != MAGIC
            || !(MIN_SUPPORTED_VERSION..=VERSION).contains(&version)
            || dims_padded != 16
        {
            return None;
        }
        let aligned_layout = version >= 4;

        let centroids_bytes = num_centroids * 16 * 2;
        let cells_bytes = num_centroids * std::mem::size_of::<Cell>();
        let vectors_bytes = num_vectors * 16 * 2;
        let labels_bytes = num_vectors;

        let centroids_off = section_offset(HEADER_BYTES, aligned_layout);
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

        let cell_bounds_bytes = num_centroids * std::mem::size_of::<BlockBounds>();
        let (cell_bounds, blocks_base_off) = if version >= 5 {
            let cell_bounds_off = section_offset(cells_end, aligned_layout);
            let cell_bounds_end = cell_bounds_off + cell_bounds_bytes;
            if data.len() < cell_bounds_end {
                return None;
            }
            let cell_bounds = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr().add(cell_bounds_off) as *const BlockBounds,
                    num_centroids,
                )
            };
            (Some(cell_bounds), cell_bounds_end)
        } else {
            (None, cells_end)
        };

        let blocks_bytes = num_blocks * std::mem::size_of::<BlockBounds>();
        let blocks_off = section_offset(blocks_base_off, aligned_layout);
        let vectors_off = section_offset(blocks_off + blocks_bytes, aligned_layout);
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
            cell_bounds,
            blocks,
            vectors,
            labels,
        })
    }

    /// Returns `(fraud_count_in_5, fraud_score)` for an f32 query vector.
    pub fn query(&self, vector: &[f32; 16], nprobe: usize) -> (u8, f32) {
        let q = quantize_i16(vector);
        self.query_quantized(vector, &q, nprobe)
    }

    #[inline]
    pub fn query_with_options(&self, vector: &[f32; 16], options: QueryOptions) -> (u8, f32) {
        let q = quantize_i16(vector);
        self.query_quantized_with_options(vector, &q, options)
    }

    /// Same as [`Self::query`], but accepts the already-quantized query vector.
    pub fn query_quantized(&self, vector: &[f32; 16], q: &[i16; 16], nprobe: usize) -> (u8, f32) {
        self.query_quantized_with_options(vector, q, QueryOptions::new(nprobe))
    }

    /// Same as [`Self::query_quantized`], with experimental repair policy knobs.
    pub fn query_quantized_with_options(
        &self,
        vector: &[f32; 16],
        q: &[i16; 16],
        options: QueryOptions,
    ) -> (u8, f32) {
        let nc = self.num_centroids.min(MAX_CENTROIDS);
        let probe = options.nprobe.clamp(1, MAX_NPROBE).min(nc);
        let repair_min = options.repair_min.min(K as u8);
        let repair_max = options.repair_max.min(K as u8);
        let adaptive_probe = match options.repair_mode {
            RepairMode::Pattern if probe == 10 || probe == 12 => 48.min(nc),
            RepairMode::AmbiguousCentroid => 48.min(nc),
            RepairMode::Pattern | RepairMode::Bbox => probe,
        };

        // 1. Distance to every centroid.
        let mut cdist = [0i64; MAX_CENTROIDS];
        distances_to_slice_i16(&q, &self.centroids[..nc], &mut cdist[..nc]);

        // 2. The nearest centroids (ascending by distance). Adaptive modes
        // need one extra centroid for the confidence gap, but the full 48-way
        // selection is paid only when the risky boundary path actually fires.
        let mut best: [(i64, u32); MAX_NPROBE] = [(i64::MAX, 0); MAX_NPROBE];
        let initial_probe = if adaptive_probe > probe {
            (probe + 1).min(adaptive_probe)
        } else {
            probe
        };
        fill_best_centroids(&cdist[..nc], initial_probe, &mut best);

        // 3. Scan the chosen cells in fast int16, keeping the K_RERANK nearest
        //    candidates (int16 distance, global vector index).
        let mut cand: [(i64, u32); K_RERANK] = [(i64::MAX, 0); K_RERANK];
        let mut filled = 0usize;
        let mut scratch = [0i64; SCAN_CHUNK];

        self.scan_centroid_range(&q, &best, 0, probe, &mut cand, &mut filled, &mut scratch);
        let (mut fraud, bits) = self.rerank_candidates(vector, &cand, filled);

        match options.repair_mode {
            RepairMode::Pattern => {
                if adaptive_probe > probe
                    && initial_probe > probe
                    && is_risky_pattern(probe, bits, best[probe - 1].0, best[probe].0)
                {
                    fill_best_centroids(&cdist[..nc], adaptive_probe, &mut best);
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
            }
            RepairMode::AmbiguousCentroid => {
                if adaptive_probe > probe && is_ambiguous(fraud, repair_min, repair_max) {
                    fill_best_centroids(&cdist[..nc], adaptive_probe, &mut best);
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
            }
            RepairMode::Bbox => {
                if is_ambiguous(fraud, repair_min, repair_max) {
                    fraud = self.repair_by_cell_bounds(
                        q,
                        vector,
                        nc,
                        &best,
                        probe,
                        &mut cand,
                        &mut filled,
                        &mut scratch,
                        repair_min,
                        repair_max,
                        options.repair_candidates,
                    );
                }
            }
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
            self.scan_centroid(q, best[b].1 as usize, cand, filled, scratch);
        }
    }

    #[inline]
    fn scan_centroid(
        &self,
        q: &[i16; 16],
        centroid: usize,
        cand: &mut [(i64, u32); K_RERANK],
        filled: &mut usize,
        scratch: &mut [i64; SCAN_CHUNK],
    ) {
        let cell = self.cells[centroid];
        let start = cell.start as usize;
        let count = cell.count as usize;
        if start + count > self.num_vectors {
            return;
        }
        let block_start = cell.block_start as usize;
        let block_count = cell.block_count as usize;
        if block_start + block_count > self.blocks.len() {
            return;
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

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn repair_by_cell_bounds(
        &self,
        q: &[i16; 16],
        vector: &[f32; 16],
        nc: usize,
        scanned: &[(i64, u32); MAX_NPROBE],
        scanned_len: usize,
        cand: &mut [(i64, u32); K_RERANK],
        filled: &mut usize,
        scratch: &mut [i64; SCAN_CHUNK],
        repair_min: u8,
        repair_max: u8,
        repair_candidates: usize,
    ) -> u8 {
        let Some(cell_bounds) = self.cell_bounds else {
            return self.rerank_candidates(vector, cand, *filled).0;
        };

        let mut skip = [0u64; CENTROID_WORDS];
        for &(_, c) in scanned.iter().take(scanned_len) {
            let c = c as usize;
            skip[c >> 6] |= 1u64 << (c & 63);
        }

        let cap = repair_candidates.clamp(1, MAX_REPAIR_CANDIDATES);
        let mut repair: [(i64, u32); MAX_REPAIR_CANDIDATES] =
            [(i64::MAX, 0); MAX_REPAIR_CANDIDATES];
        let mut rlen = 0usize;

        for c in 0..nc {
            if (skip[c >> 6] & (1u64 << (c & 63))) != 0 || self.cells[c].count == 0 {
                continue;
            }
            if *filled == K_RERANK {
                let lb = lower_bound_block(q, &cell_bounds[c], cand[K_RERANK - 1].0);
                if lb >= cand[K_RERANK - 1].0 {
                    continue;
                }
                insert_sorted(&mut repair, &mut rlen, cap, lb, c as u32);
            } else {
                let lb = lower_bound_block(q, &cell_bounds[c], i64::MAX);
                insert_sorted(&mut repair, &mut rlen, cap, lb, c as u32);
            }
        }

        let mut fraud = self.rerank_candidates(vector, cand, *filled).0;
        for &(lb, c) in repair.iter().take(rlen) {
            if *filled == K_RERANK && lb >= cand[K_RERANK - 1].0 {
                break;
            }
            self.scan_centroid(q, c as usize, cand, filled, scratch);
            fraud = self.rerank_candidates(vector, cand, *filled).0;
            if !is_ambiguous(fraud, repair_min, repair_max) {
                break;
            }
        }
        fraud
    }
}

#[inline]
fn lower_bound_block(q: &[i16; 16], block: &BlockBounds, limit: i64) -> i64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let _ = limit;
        return unsafe { lower_bound_block_avx2(q, block) };
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        lower_bound_block_scalar(q, block, limit)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn lower_bound_block_avx2(q: &[i16; 16], block: &BlockBounds) -> i64 {
    use std::arch::x86_64::*;

    let qv = _mm256_loadu_si256(q.as_ptr() as *const __m256i);
    let lo = _mm256_loadu_si256(block.min.as_ptr() as *const __m256i);
    let hi = _mm256_loadu_si256(block.max.as_ptr() as *const __m256i);

    let below = _mm256_cmpgt_epi16(lo, qv);
    let above = _mm256_cmpgt_epi16(qv, hi);
    let below_diff = _mm256_sub_epi16(lo, qv);
    let above_diff = _mm256_sub_epi16(qv, hi);
    let diff = _mm256_or_si256(
        _mm256_and_si256(below, below_diff),
        _mm256_and_si256(above, above_diff),
    );

    let madd = _mm256_madd_epi16(diff, diff);
    let lo64 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(madd));
    let hi64 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(madd, 1));
    let sum = _mm256_add_epi64(lo64, hi64);
    let sum_lo = _mm256_castsi256_si128(sum);
    let sum_hi = _mm256_extracti128_si256(sum, 1);
    let pair = _mm_add_epi64(sum_lo, sum_hi);
    let pair_hi = _mm_unpackhi_epi64(pair, pair);
    _mm_cvtsi128_si64(_mm_add_epi64(pair, pair_hi))
}

#[cfg_attr(all(target_arch = "x86_64", target_feature = "avx2"), allow(dead_code))]
#[inline]
fn lower_bound_block_scalar(q: &[i16; 16], block: &BlockBounds, limit: i64) -> i64 {
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
fn is_ambiguous(fraud: u8, repair_min: u8, repair_max: u8) -> bool {
    repair_min <= repair_max && fraud >= repair_min && fraud <= repair_max
}

#[inline]
fn read_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

#[inline]
fn section_offset(off: usize, aligned_layout: bool) -> usize {
    if aligned_layout {
        align_up(off, SECTION_ALIGN)
    } else {
        off
    }
}

#[inline]
pub fn align_up(off: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (off + align - 1) & !(align - 1)
}

#[inline]
fn fill_best_centroids(cdist: &[i64], cap: usize, best: &mut [(i64, u32); MAX_NPROBE]) {
    let mut len = 0usize;
    for (i, &dist) in cdist.iter().enumerate() {
        insert_sorted(best, &mut len, cap, dist, i as u32);
    }
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
        let cell_bounds = [bounds(&vecs[0..3]), bounds(&vecs[3..6])];
        let blocks = [bounds(&vecs[0..3]), bounds(&vecs[3..6])];

        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(vecs.len() as u32).to_le_bytes());
        out.extend_from_slice(&(num_centroids as u32).to_le_bytes());
        out.extend_from_slice(&16u32.to_le_bytes());
        out.resize(align_up(out.len(), SECTION_ALIGN), 0);
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
        out.resize(align_up(out.len(), SECTION_ALIGN), 0);
        for b in &cell_bounds {
            for x in &b.min {
                out.extend_from_slice(&x.to_le_bytes());
            }
            for x in &b.max {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out.resize(align_up(out.len(), SECTION_ALIGN), 0);
        for b in &blocks {
            for x in &b.min {
                out.extend_from_slice(&x.to_le_bytes());
            }
            for x in &b.max {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out.resize(align_up(out.len(), SECTION_ALIGN), 0);
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
