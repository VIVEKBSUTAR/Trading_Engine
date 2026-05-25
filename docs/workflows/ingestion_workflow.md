# Ingestion Workflow

1. Load feed asynchronously from CSV/API.
2. Validate schema and corrupted rows.
3. Normalize timestamps to Asia/Kolkata.
4. Remove duplicates and fill missing bars by frequency.
5. Persist to partitioned parquet.
6. Refresh DuckDB metadata catalog.

## Example CLI

```bash
te-ingest \
  --feed nifty50_spot \
  --source csv \
  --path data/raw/nifty50_spot.csv \
  --dataset market_data \
  --timestamp-col timestamp \
  --frequency 1min
```
