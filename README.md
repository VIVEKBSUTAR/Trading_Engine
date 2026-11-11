# NIFTY Analytical Signal Engine & Terminal

A pure analytical signal engine powered by Fyers API V3 data, with a focused
Streamlit terminal for NIFTY option alerts and position monitoring. Signals,
risk plans, positions, and PnL are analytical outputs; they do not establish
broker orders or fills.

## Two-page terminal

**Execution Desk** is the default page (`1_Execution_Desk.py`). Its visual order is:

1. One compact row: Fyers WebSocket status, NIFTY spot, and India VIX.
2. A fresh passed BUY_CE/BUY_PE alert with action, strike, and option entry premium.
3. Active positions with green profit/red loss and bold trailing-stop values.

Equity charts, performance metrics, and historical logs do not appear on this
page. New alerts toast once per signal ID in a browser session; the current alert
remains visible only within its 60-second window. When a batch arrives, the newest
qualifying signal is highlighted and older members of that batch are marked seen.
Stale publications suppress live alerts and mark cached positions as stale.

**Analytics & Logs** (`pages/2_Analytics_And_Logs.py`) contains Total Profit,
Closed Trades, Win Rate, and Profit Factor at the top, followed by a Plotly area
chart with restrained grid lines and hover details. The last 50 signal evaluations
and a Rejection Reason expander sit below the chart. Deployment performance covers
the most recent 1,000 analytical outcomes. Use the sidebar to switch pages.

The root `.streamlit/config.toml` sets a dark theme with green accents, background
`#0E1117`, and secondary background `#262730`. Both pages refresh every five
seconds without starting another broker session.

UI previews were captured with **isolated synthetic fixtures**, not live market
or account data: [Execution Desk](docs/ui/execution-preview.jpg) and
[Analytics & Logs](docs/ui/analytics-preview.jpg). These show the alert/position
hierarchy and the separate performance/chart/audit layout; “Connected” in the
preview is a fixture value, not evidence of market connectivity.

## Deployment

Use Docker with Compose, then run from the repository root:

```bash
cp .env.example .env
chmod 600 .env
# Fill in the Fyers credentials and registered redirect URI.
docker compose up --build
```

Open [the terminal](http://localhost:8501). The default view is Execution Desk;
Analytics & Logs is the secondary sidebar page.

Required headless settings: `FYERS_CLIENT_ID`, `FYERS_SECRET_KEY`,
`FYERS_REDIRECT_URI`, `FYERS_USER_ID`, `FYERS_PIN`, and `FYERS_TOTP_SECRET`.
Keep the PIN as a string to preserve leading zeros. Optional
`FYERS_STREAM_SYMBOLS` can specify contracts; otherwise nearby contracts are
discovered at startup. Synchronize the host clock for TOTP.

Engine startup generates today's token, overwrites `FYERS_ACCESS_TOKEN` in the
bind-mounted local `.env`, verifies its generation timestamp, and explicitly loads
the new value before execing `te-intelligence`. A failed login stops startup.
Tokens refresh on startup only: restart engine after the provider's daily expiry.
The SSO endpoints are undocumented and still require account-level validation.

The Python 3.12 slim image installs non-editably. `engine` owns the DuckDB writer
and publishes `./data/dashboard/latest.json` atomically. `dashboard` reads that
publication without receiving account credentials, opening the writer's database,
or starting a second analytical runtime. Both services mount data/log directories;
dashboard data access is read-only. Its healthcheck verifies the HTTP service,
not market connectivity.

The engine queries DuckDB separately for passed directional signals timestamped
within the last 60 seconds. This alert window is independent of the 50-row audit
log. Signals now persist `entry_price` from the selected option ask; old records
have NULL rather than substituting index spot. Rebuild both services together to
use the expanded publication fields for health, option premiums, and positions.

## Local installation and checks

Use Python 3.12 or newer. Run from the repository root so the Streamlit theme is
loaded:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
PYTHONPATH=src:. te-dashboard
```

The launcher now opens `src/trading_engine/dashboard/1_Execution_Desk.py`.
Set `TE_DASHBOARD_SNAPSHOT_DIR` to the engine publication directory. Without an
explicit setting, local mode first looks in `data/dashboard`; otherwise it can
read historical DuckDB data. A database-only view cannot establish websocket
health or active positions. It shows unavailable fields explicitly.

```bash
PYTHONPATH=src:. te-intelligence
PYTHONPATH=src:. python -m pytest tests -q
```

Offline verification: **108 tests and 5 subtests pass**. Changed-code lint and the
non-editable wheel build pass; both page scripts are present in the wheel. Browser
review used synthetic fixtures. Docker and live-account validation remain pending;
see `project_status.md` for exact operational status. The requested
`docker compose up --build` attempt exited 127 (`docker: command not found`).
The Docker launcher symlink points to an absent Docker Desktop binary. The
terminal is **not live**; restore/install Docker Desktop, then rerun the command.

## Isolated live websocket validation

Export `FYERS_CLIENT_ID` and `FYERS_ACCESS_TOKEN` securely, then run during an
active market session:

```bash
python live_fire_test.py --timeout 120
```

The script reads environment variables directly, discovers current-expiry ATM
CE/PE symbols, prints exactly 50 combined option updates, and closes cleanly.
It does not initialize the engine or load `.env` itself. Connection/subscription
acknowledgements do not count as ticks. Success proves data connectivity, not
signal accuracy or fills.

## Analytical logic and limitations

- Index spot supplies direction/strike context. Long-option valuation enters at
  ask and marks/exits at bid for the selected token.
- Spread rejection defaults to `(ask - bid) / bid > 0.05`; IV rejection uses the
  rolling 90th percentile. Prior missing-quote/IV and warm-up bypasses remain.
- DTE uses the snapshot's Asia/Kolkata date and selected contract expiry. Expiry
  day carries a 15-minute holding cap; later days carry 45 minutes. Invalid,
  missing, or expired contracts fail closed.
- The same-strike/same-expiry PE/CE OI ratio rejects BUY_CE below 0.8 and BUY_PE
  above 1.2. Missing or invalid OI fails closed.
- ATR uses 14 completed contiguous one-minute midpoint bars with true-range gaps.
  Initial stop is ask minus 1.5 ATR; at 1R profit trailing follows bid minus one
  ATR without widening. The premium target is ask plus 3 ATR. Insufficient ATR
  history suppresses admission. Signal TTL and holding lifetime are separate.
- Runtime, engine, and monitor share tick history. Stops/time limits require fresh
  option updates; missing/stale data freezes valuation and cannot establish an
  executable exit. A stale terminal never claims a healthy websocket.
- DuckDB stores decisions and analytical outcomes. Raw ticks flush to Parquet
  above 5,000 rows per token. Existing failed-writer buffer clearing, reused part
  filenames, shutdown flushing, and lot-size/sizing assumptions need hardening.

Historical research, backtesting, and model utilities remain available. The full
runtime combines Fyers websocket/REST with supplementary NSE option-chain data.
The Fyers 3.2.1 SDK pins requests 2.31.0; the project range accommodates it.
Plotly was already declared and remains required; Streamlit now requires at least
1.51 for the terminal's table font-weight support.

## Documentation protocol

Every completed development phase must update `decision.md`, this `README.md`,
and `project_status.md` together. Record architecture, actual system state,
validation evidence, pending work, and live-market consequences. `AGENTS.md`
persists the rule. Offline tests and UI previews do not establish live readiness.

Next pending task: **Live-fire deployment and real-time AWS EC2/VPS observation.**

Reference: [Streamlit multipage directory architecture](https://docs.streamlit.io/develop/concepts/multipage-apps/overview)
and [styled dataframe support](https://docs.streamlit.io/1.51.0/develop/api-reference/data/st.dataframe).

The deployment files are directly in the repository root: `docker-compose.yml`,
`Dockerfile`, and executable `entrypoint.sh`. Their paths, service definitions,
port/mount configuration, and auth-before-daemon startup order have been verified.
Compose loads `.env` for engine; dashboard consumes engine publications without
account secrets. Start Compose from this same root directory.


The canonical local checkout on this laptop is
`/Volumes/APFS/projects/trading_engine`. Source/configuration/documentation changes
from this chat are synchronized there before publication to
[VIVEKBSUTAR/Trading_Engine](https://github.com/VIVEKBSUTAR/Trading_Engine). Local
credentials, datasets, logs, and databases remain outside Git.

The combined local/root changes pass 108 tests and 5 subtests. ATR lifecycle tests
use a deterministic intraday clock so running the suite around midnight in India
does not mistake the expected session warm-up for a regression. The canonical
checkout retains local broker/storage fixes alongside this chat's upgrades.
