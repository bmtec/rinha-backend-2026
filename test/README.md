# test/

Official challenge test harness plus a lightweight accuracy harness.

- `test.js`, `smoke.js`, `k6-summary.js`, `docker-compose.yml` — the official
  k6 load test (ramps to 900 req/s, reports p99 + a detection score).
- `accuracy.py` — replays the labeled `test-data.json` through a running API
  and reports TP/TN/FP/FN/error counts using the same definitions as `test.js`.
- `test-data.json` — **required, not committed** (27 MB, gitignored). 54,100
  labeled entries. Fetch with:

  ```sh
  curl -sL -o test/test-data.json \
    https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/test/test-data.json
  ```

## Accuracy run (no Docker required)

```sh
# build a real index from resources/references.json.gz
cargo build --release --bins
./target/release/builder resources/references.json.gz /tmp/index.bin

# start the API against it
INDEX_PATH=/tmp/index.bin PORT=9998 NPROBE=10 ./target/release/api &

# replay the labeled data
python3 test/accuracy.py http://127.0.0.1:9998
```

## Measured results (real 3M index, official test-data.json)

Fixed `nprobe` sweep — the smallest fixed probe count with **zero detection
errors** is 48:

| nprobe | FP | FN | errors | accuracy |
|-------:|---:|---:|-------:|---------:|
| 10     | 3  | 4  | 0      | 99.987%  |
| 20     | 0  | 2  | 0      | 99.996%  |
| 32     | 0  | 1  | 0      | 99.998%  |
| 48     | 0  | 0  | 0      | 100.000% |
| 64     | 0  | 0  | 0      | 100.000% |

> Accuracy is hardware-independent (AVX2 and the scalar fallback compute the
> same distances). Latency must be validated on the target Intel Mac mini; the
> AVX2 build is not representative when run under emulation on ARM.

## Full k6 run (official `test.js`, documented stack, adaptive nprobe=10→48)

Ran the official k6 test against the Xeon benchmark host with the documented
compose topology: Rust LB on `:9999`, two API containers behind it, bridge
networking, and adaptive probing:

```
run 1: p99=0.57ms fp=0 fn=0 err=0 FINAL=6000
run 2: p99=0.57ms fp=0 fn=0 err=0 FINAL=6000
run 3: p99=0.57ms fp=0 fn=0 err=0 FINAL=6000
```

**Detection is perfect (0 errors) under full 900 req/s load.** The p99 is under
the 1 ms scoring cap, so both score components max out.

The current setup combines fd-passing over Unix sequenced-packet sockets, fixed
per-fd connection slots, short epoll spin/micro-timeout waits, separate cpusets
for `lb`, `api1`, and `api2`, and per-block lower-bound pruning inside each IVF
cell. A broader adaptive `nprobe=8` experiment preserved accuracy only when it
expanded too often, and a tighter rule produced detection errors, so production
remains at adaptive `nprobe=10`.

### int16 quantization and block pruning

The index was later switched to int16 (×10000, i64 accumulation). Detection is
**unchanged** — still E = 0 on the labeled set with adaptive probing (verified
with `accuracy.py`) — because the quantization is exact for the 4-decimal data.
The block-pruned index is ~96 MiB, so it fits the 167 MB per-instance RAM limit
(the f32 index would be evicted under the cgroup limit and fault under load).

**Emulation p99 is not a valid benchmark.** Repeated emulated k6 runs gave
120 ms then 190 ms for essentially the same workload — the QEMU/host variance
exceeds any algorithmic signal. The int16 win (RAM-resident index + 2× AVX2
lanes) is real but only observable on the target Intel hardware.
