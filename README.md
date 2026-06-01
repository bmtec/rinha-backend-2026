# Rinha de Backend 2026 — Fraud Detection API

A low-latency fraud detection API built for the Rinha de Backend 2026 challenge.
It receives card transaction payloads, transforms them into 14-dimensional
vectors, finds the 5 nearest neighbors in a 3,000,000-vector reference dataset
using an IVF (Inverted File) index with AVX2 SIMD, and returns an approval
decision.

The system runs under extreme constraints: **1 CPU and 350 MB RAM total**.

## Architecture

```
        client
          │
          ▼
   Rust LB container (:9999)
   TCP accept + SCM_RIGHTS fd hand-off
          │
          ├───────────────┐
          ▼               ▼
   api1 epoll HTTP   api2 epoll HTTP
          │               │
          ▼               ▼
 mmap'd IVF index   mmap'd IVF index
 (≈96 MiB/image)    (≈96 MiB/image)
```

- The **LB** receives the public `9999` traffic and only passes accepted TCP
  sockets to API instances over Unix sockets; it does not parse HTTP or inspect
  payloads.
- Each **API** uses a low-level epoll HTTP/1.1 loop and serves keep-alive
  directly on the received client socket, keeping the LB outside the per-request
  hot path.
- The hot transport path uses fixed per-fd connection slots, fixed request
  buffers, short epoll spin/micro-timeout waits, and separate cpuset placement
  for the LB and each API instance.
- Request handling uses a hand-rolled JSON parser, zero heap allocation in the
  scoring path, and AVX2-accelerated distance computation.
- The **IVF index** is built once at image build time and memory-mapped at
  runtime.

## Tech stack

- **Rust** — no GC, deterministic latency.
- **libc epoll** — direct HTTP/1.1 socket handling.
- **SCM_RIGHTS fd passing** — documented LB topology without per-request proxying.
- **IVF index** (int16-quantized) with **AVX2 SIMD** squared-Euclidean distance.

## Performance targets

The official score is `p99_score + detection_score`, each capped at 3000:

```
p99_score       = 1000 · log10(1000 / max(p99_ms, 1))   → 3000 iff p99 ≤ 1 ms
detection_score = 1000·log10(1/max(ε,0.001)) − 300·log10(1+E)
                                                        → 3000 iff E = 0
                  E = 1·FP + 3·FN + 5·errors
```

So the maximum 6000 needs **both** `E = 0` (zero detection errors) **and**
`p99 ≤ 1 ms`. Runtime uses an adaptive IVF probe: scan 10 cells for the common
path and expand to 48 cells only for tuned boundary patterns.

On the Xeon benchmark host, the current documented stack measured
`0.57ms` p99 with `0 FP / 0 FN / 0 HTTP errors`. Note `p99_score`
saturates at p99 ≤ 1 ms.

## Index quantization and pruning

Reference vectors and centroids are stored as 16-bit fixed point (×10000),
with distances accumulated in i64. This is numerically exact for the 4-decimal
dataset (detection identical to f32, E = 0) while:

- shrinking the vectors from ~195 MB to a ~96 MiB mmap'd index so it **fits the 167 MB
  per-instance RAM limit** — the f32 index would be evicted/re-faulted under
  the cgroup limit, wrecking p99 tail latency, and
- packing all 16 dims into one 256-bit AVX2 register (2× the f32 throughput).

The current index also stores per-block min/max bounds inside each IVF cell.
After the first candidates fill, whole blocks are skipped when their lower-bound
distance cannot enter the top candidates.

## Build

```sh
docker compose build
```

This compiles the Rust binaries and runs the index builder to produce the
~96 MiB int16/block-pruned `index.bin`, which is baked into the runtime image.

> **Platform note:** the target deployment is an **Intel Mac mini (linux/amd64)**.
> The images are built for `linux/amd64` with `-C target-cpu=haswell` to enable
> AVX2. When building on an ARM machine, Docker uses emulation for the amd64
> target.

## Run

```sh
docker compose up
```

Wait for `/ready` to return `200`, then send requests to `http://localhost:9999`.

## Test locally

```sh
cd test
docker compose --profile smoke up
```

Then inspect `results.json` for the score.

## License

MIT — see [LICENSE](LICENSE).
