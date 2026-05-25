# Market Data Schema

## Common Columns

- `timestamp` (`datetime64[ns, Asia/Kolkata]`) : canonical event timestamp
- `feed` (`string`) : feed source identifier
- `date` (`YYYY-MM-DD`) : partition key for parquet layouts

## Nifty 50 Spot

- `open` (`float64`)
- `high` (`float64`)
- `low` (`float64`)
- `close` (`float64`)
- `volume` (`float64`, optional)

## GIFT Nifty Futures

- `open` (`float64`)
- `high` (`float64`)
- `low` (`float64`)
- `close` (`float64`)
- `volume` (`float64`, optional)
- `open_interest` (`float64`, optional)

## India VIX

- `close` (`float64`)

## NSE Option Chain Snapshot

- `expiry` (`date`)
- `strike` (`float64`)
- `option_type` (`string`, CE/PE)
- `ltp` (`float64`)
- `iv` (`float64`)
- `oi` (`float64`)
- `bid` (`float64`, optional)
- `ask` (`float64`, optional)
