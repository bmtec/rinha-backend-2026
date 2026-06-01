# resources/

Reference data and config consumed at **build time** by `builder`.

- `mcc_risk.json` — the 10-entry MCC risk table (also hardcoded in `src/lib.rs`).
- `normalization.json` — feature normalization constants (also hardcoded in `src/lib.rs`).
- `references.json.gz` — **required, not committed here** (50 MB, gitignored).
  Array of 3,000,000 `{"vector":[14 floats],"label":"fraud"|"legit"}` records.
  Fetch before `docker compose build`:

  ```sh
  curl -sL -o resources/references.json.gz \
    https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz
  ```

The two JSON files are kept for repo completeness; the runtime API does not
read them (their values are compiled into the binary).
