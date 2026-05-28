# Trading Engine

Production-grade algorithmic trading framework for Indian markets, focused on NIFTY index analytics, options intelligence, backtesting, and live broker connectivity.

## What It Does

The project combines four layers:

- Historical ingestion and storage for NSE datasets
- Feature engineering and walk-forward evaluation for research
- Live Kite Connect connectivity for market snapshots and websocket streaming
- A probabilistic NIFTY options intelligence layer with a Streamlit dashboard

The live intelligence stack is designed to answer a narrow question: when the short-horizon market structure supports a directional options trade, which side, strike, size, and risk plan should be used.

## Key Capabilities

- Historical data ingestion and validation
- Partitioned Parquet and DuckDB storage
- Options analytics and regime detection
- Walk-forward model training and inference
- Backtest trade accounting and cost modeling
- Kite Connect auth, snapshots, and streaming
- NSE option-chain fetching and parsing
- Live market-state aggregation and probabilistic signal generation
- Automatic strike selection and live trade monitoring
- Streamlit dashboard for live monitoring

## Requirements

- Python 3.12 or newer
- A working Kite Connect account for live broker features
- Network access for NSE and Kite Connect endpoints

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Populate `.env` with your local values for Kite Connect and runtime tuning.

## Environment Variables

Important variables include:

- `KITE_API_KEY`
- `KITE_API_SECRET`
- `KITE_ACCESS_TOKEN`
- `KITE_REQUEST_TOKEN`
- `KITE_STREAM_TOKENS`
- `TE_FETCH_INTERVAL_SECONDS`
- `TE_DASHBOARD_REFRESH_SECONDS`
- `TE_BULLISH_PROB_THRESHOLD`
- `TE_BEARISH_PROB_THRESHOLD`
- `TE_TRAP_PROB_THRESHOLD`

The sample `.env.example` documents the full set of supported runtime settings.

## Commands

After installation, the main entrypoints are:

```bash
te-ingest
te-train
te-infer
te-eval
te-nse-ingest
te-kite-live
te-intelligence
te-dashboard
```

### Live Intelligence Service

```bash
te-intelligence
```

Runs the continuous live intelligence loop. It polls market data, refreshes the market state, generates a signal report, and logs updates.

### Dashboard

```bash
te-dashboard
```

Launches the Streamlit dashboard for the live NIFTY intelligence stack.

### Kite Live Runtime

```bash
te-kite-live
```

Starts the authenticated Kite Connect runtime and websocket stream.

## Live Flow

1. Authenticate with Kite Connect.
2. Fetch a live snapshot for NIFTY spot, India VIX, and option-chain data.
3. Normalize live ticks into a rolling state buffer.
4. Build short-horizon features from candles, VIX, and option-chain signals.
5. Classify the market regime.
6. Estimate directional probabilities.
7. Filter traps and fake breakouts.
8. Select the most suitable strike and risk plan.
9. Monitor open trades and render the result in the dashboard.

## Project Layout

```text
src/trading_engine/
  bench/         evaluation harness and backtest reports
  backtest/      execution, costs, and trade accounting
  cli/           command-line entrypoints
  dashboard/     Streamlit live dashboard
  data/          historical ingest and cleanup utilities
  features/      research feature engineering
  intelligence/  live market state, signal, strike, risk, and monitoring
  ml/            walk-forward model training and inference
```

## Notes

- Live NSE endpoints can be sensitive to headers and session bootstrap behavior.
- Kite access tokens should remain local and untracked.
- The intelligence layer is intentionally conservative and is not a guarantee of trade quality or profitability.

## Validation

The repository currently compiles cleanly with:

```bash
PYTHONPATH=src:. .venv/bin/python -m compileall -q src broker config main.py
```
