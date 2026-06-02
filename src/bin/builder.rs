//! Offline IVF index builder. Runs once at Docker image build time.
//!
//! Usage: `builder <references.json.gz> <output_path>`
//!
//! Reads the 3M reference vectors, runs K-means to produce 2048 centroids,
//! assigns + sorts vectors by cell, and writes the binary index described in
//! `src/index.rs`. Optimized for correctness, not speed.

use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use flate2::read::GzDecoder;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};
use serde::Deserialize;

use rinha::distance::quantize_i16;
use rinha::index::{align_up, BLOCK_SIZE, MAGIC, VERSION};

const DEFAULT_NUM_CENTROIDS: usize = 2048;
const KMEANS_ITERS: usize = 15;

/// Scalar f32 squared Euclidean distance for the builder's k-means.
///
/// Deliberately scalar f32 with no fused multiply-add: IEEE f32 mul/add are
/// correctly rounded and identical on every platform, so the index is
/// bit-identical whether built on ARM or x86. (The runtime SIMD path uses
/// `fmadd`, whose different rounding shifts boundary assignments; and f64
/// over-precises relative to the f32 ground-truth labels, which slightly
/// degrades agreement.) Build time is irrelevant — this runs once.
#[inline]
fn sqdist(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..16 {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Centroid count, overridable via the `CENTROIDS` env var (for tuning).
fn num_centroids() -> usize {
    std::env::var("CENTROIDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_CENTROIDS)
}

#[derive(Deserialize)]
struct Entry {
    vector: Vec<f32>,
    label: String,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: builder <references.json.gz> <output_path>");
        std::process::exit(2);
    }
    let input = &args[1];
    let output = &args[2];

    eprintln!("[builder] reading {input}");
    let file = File::open(input).expect("open input");
    let gz = GzDecoder::new(BufReader::new(file));
    let entries: Vec<Entry> = serde_json::from_reader(BufReader::new(gz)).expect("parse json");
    let n = entries.len();
    eprintln!("[builder] parsed {n} reference vectors");

    // Convert to padded 16-float vectors + u8 labels.
    let mut vectors: Vec<[f32; 16]> = Vec::with_capacity(n);
    let mut labels: Vec<u8> = Vec::with_capacity(n);
    for e in &entries {
        let mut v = [0.0f32; 16];
        let len = e.vector.len().min(14);
        v[..len].copy_from_slice(&e.vector[..len]);
        vectors.push(v);
        labels.push(if e.label == "fraud" { 1 } else { 0 });
    }
    drop(entries);

    // K-means clustering.
    let ncentroids = num_centroids();
    eprintln!("[builder] running k-means ({ncentroids} centroids, {KMEANS_ITERS} iters)");
    let centroids = kmeans(&vectors, ncentroids, KMEANS_ITERS);

    // Final assignment.
    let assign = parallel_assign(&vectors, &centroids);

    // Counting sort by cell.
    let mut counts = vec![0u32; ncentroids];
    for &a in &assign {
        counts[a as usize] += 1;
    }
    let mut starts = vec![0u32; ncentroids];
    let mut acc = 0u32;
    for c in 0..ncentroids {
        starts[c] = acc;
        acc += counts[c];
    }

    let mut sorted_vecs = vec![[0.0f32; 16]; n];
    let mut sorted_labels = vec![0u8; n];
    let mut cursor = starts.clone();
    for i in 0..n {
        let c = assign[i] as usize;
        let pos = cursor[c] as usize;
        sorted_vecs[pos] = vectors[i];
        sorted_labels[pos] = labels[i];
        cursor[c] += 1;
    }
    drop(vectors);
    drop(labels);

    // Make each cell's fixed-size blocks spatially tighter so the runtime
    // block lower-bound can prune whole chunks after the first candidates fill.
    sort_cells_by_centroid_distance(
        &mut sorted_vecs,
        &mut sorted_labels,
        &starts,
        &counts,
        &centroids,
    );

    eprintln!("[builder] writing index to {output}");
    write_index(
        output,
        &centroids,
        &starts,
        &counts,
        &sorted_vecs,
        &sorted_labels,
    );
    eprintln!("[builder] done");
}

/// Lloyd's algorithm. Empty clusters are reseeded to a random point.
fn kmeans(vectors: &[[f32; 16]], k: usize, iters: usize) -> Vec<[f32; 16]> {
    let mut rng = StdRng::seed_from_u64(MAGIC as u64);

    let mut idx: Vec<usize> = (0..vectors.len()).collect();
    idx.shuffle(&mut rng);
    let mut centroids: Vec<[f32; 16]> = idx[..k].iter().map(|&i| vectors[i]).collect();

    for it in 0..iters {
        let assign = parallel_assign(vectors, &centroids);

        let mut sums = vec![[0.0f64; 16]; k];
        let mut cnts = vec![0u64; k];
        for (i, &a) in assign.iter().enumerate() {
            let a = a as usize;
            for d in 0..16 {
                sums[a][d] += vectors[i][d] as f64;
            }
            cnts[a] += 1;
        }
        for c in 0..k {
            if cnts[c] > 0 {
                for d in 0..16 {
                    centroids[c][d] = (sums[c][d] / cnts[c] as f64) as f32;
                }
            } else {
                let r = (rng.next_u64() as usize) % vectors.len();
                centroids[c] = vectors[r];
            }
        }
        eprintln!("[builder]   k-means iter {}/{iters}", it + 1);
    }
    centroids
}

/// Assigns every vector to its nearest centroid, parallelized across cores.
fn parallel_assign(vectors: &[[f32; 16]], centroids: &[[f32; 16]]) -> Vec<u32> {
    let n = vectors.len();
    let mut assign = vec![0u32; n];
    let threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    let chunk = (n + threads - 1) / threads;

    std::thread::scope(|s| {
        for (ci, out_chunk) in assign.chunks_mut(chunk).enumerate() {
            let base = ci * chunk;
            s.spawn(move || {
                for (j, slot) in out_chunk.iter_mut().enumerate() {
                    let v = &vectors[base + j];
                    let mut best = 0u32;
                    let mut best_d = f32::INFINITY;
                    for (c, cen) in centroids.iter().enumerate() {
                        let d = sqdist(v, cen);
                        if d < best_d {
                            best_d = d;
                            best = c as u32;
                        }
                    }
                    *slot = best;
                }
            });
        }
    });
    assign
}

fn write_index(
    path: &str,
    centroids: &[[f32; 16]],
    starts: &[u32],
    counts: &[u32],
    vectors: &[[f32; 16]],
    labels: &[u8],
) {
    let file = File::create(path).expect("create output");
    let mut w = BufWriter::with_capacity(1 << 20, file);
    let qvectors: Vec<[i16; 16]> = vectors.iter().map(quantize_i16).collect();
    let qcentroids: Vec<[i16; 16]> = centroids.iter().map(quantize_i16).collect();

    let mut block_starts = vec![0u32; centroids.len()];
    let mut block_counts = vec![0u32; centroids.len()];
    let mut cell_bounds: Vec<([i16; 16], [i16; 16])> = Vec::with_capacity(centroids.len());
    let mut block_bounds: Vec<([i16; 16], [i16; 16])> = Vec::new();
    for c in 0..centroids.len() {
        let start = starts[c] as usize;
        let count = counts[c] as usize;
        cell_bounds.push(if count == 0 {
            ([0i16; 16], [0i16; 16])
        } else {
            bounds_i16(&qvectors[start..start + count])
        });
        block_starts[c] = block_bounds.len() as u32;
        let blocks = count.div_ceil(BLOCK_SIZE);
        block_counts[c] = blocks as u32;
        for b in 0..blocks {
            let off = start + b * BLOCK_SIZE;
            let n = (count - b * BLOCK_SIZE).min(BLOCK_SIZE);
            block_bounds.push(bounds_i16(&qvectors[off..off + n]));
        }
    }

    // Header.
    let mut pos = 0usize;
    w.write_all(&MAGIC.to_le_bytes()).unwrap();
    w.write_all(&VERSION.to_le_bytes()).unwrap();
    w.write_all(&(vectors.len() as u32).to_le_bytes()).unwrap();
    w.write_all(&(centroids.len() as u32).to_le_bytes())
        .unwrap();
    w.write_all(&16u32.to_le_bytes()).unwrap();
    pos += 20;
    pad_to_align(&mut w, &mut pos, 32);

    // Centroids (quantized to i16).
    for c in &qcentroids {
        write_vec_i16(&mut w, c);
    }
    pos += qcentroids.len() * 32;
    // Cell offsets + block ranges.
    for i in 0..centroids.len() {
        w.write_all(&starts[i].to_le_bytes()).unwrap();
        w.write_all(&counts[i].to_le_bytes()).unwrap();
        w.write_all(&block_starts[i].to_le_bytes()).unwrap();
        w.write_all(&block_counts[i].to_le_bytes()).unwrap();
    }
    pos += centroids.len() * 16;
    pad_to_align(&mut w, &mut pos, 32);
    // Cell lower-bound boxes.
    for (min, max) in &cell_bounds {
        write_vec_i16(&mut w, min);
        write_vec_i16(&mut w, max);
    }
    pos += cell_bounds.len() * 64;
    pad_to_align(&mut w, &mut pos, 32);
    // Block lower-bound boxes.
    for (min, max) in &block_bounds {
        write_vec_i16(&mut w, min);
        write_vec_i16(&mut w, max);
    }
    pos += block_bounds.len() * 64;
    pad_to_align(&mut w, &mut pos, 32);
    // Vectors (quantized to i16).
    for v in &qvectors {
        write_vec_i16(&mut w, v);
    }
    // Labels.
    w.write_all(labels).unwrap();
    w.flush().unwrap();
}

fn sort_cells_by_centroid_distance(
    vectors: &mut [[f32; 16]],
    labels: &mut [u8],
    starts: &[u32],
    counts: &[u32],
    centroids: &[[f32; 16]],
) {
    for c in 0..centroids.len() {
        let start = starts[c] as usize;
        let count = counts[c] as usize;
        if count <= 1 {
            continue;
        }
        let end = start + count;
        let mut tmp: Vec<(f32, [f32; 16], u8)> = Vec::with_capacity(count);
        for i in start..end {
            tmp.push((sqdist(&vectors[i], &centroids[c]), vectors[i], labels[i]));
        }
        tmp.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        for (i, (_, v, label)) in tmp.into_iter().enumerate() {
            vectors[start + i] = v;
            labels[start + i] = label;
        }
    }
}

fn bounds_i16(vecs: &[[i16; 16]]) -> ([i16; 16], [i16; 16]) {
    let mut min = [i16::MAX; 16];
    let mut max = [i16::MIN; 16];
    for v in vecs {
        for d in 0..16 {
            min[d] = min[d].min(v[d]);
            max[d] = max[d].max(v[d]);
        }
    }
    (min, max)
}

#[inline]
fn write_vec_i16<W: Write>(w: &mut W, v: &[i16; 16]) {
    let mut buf = [0u8; 32];
    for (i, x) in v.iter().enumerate() {
        buf[i * 2..i * 2 + 2].copy_from_slice(&x.to_le_bytes());
    }
    w.write_all(&buf).unwrap();
}

fn pad_to_align<W: Write>(w: &mut W, pos: &mut usize, align: usize) {
    let next = align_up(*pos, align);
    if next == *pos {
        return;
    }
    let pad = next - *pos;
    const ZEROES: [u8; 32] = [0; 32];
    w.write_all(&ZEROES[..pad]).unwrap();
    *pos = next;
}
