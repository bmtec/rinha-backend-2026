# resources/

Reference data and config consumed at **build time** by `builder`.

- `mcc_risk.json` — the 10-entry MCC risk table (also hardcoded in `src/lib.rs`).
- `normalization.json` — feature normalization constants (also hardcoded in `src/lib.rs`).
- `references.json.gz` — required for the Docker build to create `/data/index.bin`.
  Array of 3,000,000 `{"vector":[14 floats],"label":"fraud"|"legit"}` records.

The two JSON files are kept for repo completeness; the runtime API does not
read them (their values are compiled into the binary).

Do not store request payload samples in this directory. Runtime behavior must
come from vectorization and nearest-neighbor search, never from payload lookup.
