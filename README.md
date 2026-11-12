# NIFTY Analytical Signal Engine & Terminal

A pure analytical signal engine powered by Fyers API V3 data, with a focused
Streamlit terminal for NIFTY option alerts and position monitoring. Signals,
risk plans, positions, and PnL are analytical outputs; they do not establish
broker orders or fills.

## Current local runtime

The Greeks/lifecycle upgrade is implemented and tested in source. The previously
observed Docker deployment used the earlier parser release; this phase has not
restarted the running services or validated the new lifecycle rules against live
markets. Rebuild with `docker compose up --build -d` from the canonical checkout
when deploying this release. Analytical positions and IV warm-up history are
in memory and restart empty; historical decisions, alerts, and outcomes remain
in DuckDB. AWS/VPS observation remains pending.

## Two-page terminal

**Execution Desk** is the default page (`1_Execution_Desk.py`). Its visual order is:

1. One compact row: Fyers WebSocket status, NIFTY spot, and India VIX.
2. Fresh `ENTRY_BUY` and exact `EXIT_*` lifecycle alerts with direction, strike, and premium.
3. Active positions with Live Premium, bold Trailing Stop, Time in Trade, Live IV, and green/red analytical PnL.

Equity charts, performance metrics, and historical logs do not appear on this
page. Lifecycle alerts toast once per event ID in a browser session and appear
for 60 seconds, with exits ahead of entries. An `ENTRY_BUY` event means an
analytical position actually opened after risk and data checks; a passed
BUY_CE/BUY_PE candidate in the audit log alone does not produce an entry toast.
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

Offline verification: **149 tests and 5 subtests pass**. Changed-code lint and the
non-editable wheel build pass; both page scripts are present in the wheel. The saved UI preview images use synthetic fixtures. The actual local Docker stack
has now also been started and checked against live Fyers data. Docker Compose
configuration passes, daily token generation succeeded, and the dashboard returned
HTTP 200. The authentication payload uses `pyotp.TOTP(totp_secret).now()`; secrets
remain in ignored local `.env`, not in source or documentation. See
`project_status.md` for observed evidence and pending validation.

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
- Entries select the closest signed Delta to +0.70 (CE) or -0.70 (PE), within
  an absolute-Delta band of 0.60–0.80. Missing eligible strikes suppress entry.
  Fyers requests include `greeks="1"`; nested or flat Delta/IV are supported.
  IV uses provider percentage units (20 means 20%). If Delta alone is missing,
  a European Black–Scholes estimate uses supplied IV, expiry at 15:30 IST, and
  zero interest/dividend carry. This approximation is not a promise of lower Theta.
- Entry gates are Trap Score <0.5, same-expiry aggregate Put OI / Call OI >1.5,
  and IV percentile strictly below 50, alongside existing probability, quality,
  spread, and risk checks. The old directional strike OI thresholds are superseded.
  PCR and Max Pain use the returned chain's strikes, not unobserved market-wide OI.
- Spread rejection defaults to `(ask - bid) / bid >0.05`. IV percentile is the
  fraction of up to 100 same-contract observations <= current IV; at least 20
  distinct timestamped chain snapshots are required. All returned contracts are
  observed independently of entry gates; repeated socket evaluations do not add
  observations. Missing quotes, IV, PCR, Delta, or ATR suppress entry.
- NIFTY ATR is the mean of 14 completed contiguous one-minute true ranges,
  including close gaps and a preceding bar, built exclusively from index LTP
  history. The forming minute, prior sessions, and option prices are excluded.
  Initial stop = entry ask − NIFTY ATR × |Delta|. Each fresh evaluation ratchets
  stop = max(previous stop, current bid − live NIFTY ATR × |live Delta|).
  There is no percentage stop, fixed premium target, or 1R activation delay.
- Exit precedence: `EXIT_STOP_LOSS`; `EXIT_VOLATILITY_CRUSH` (IV falls >10%
  relative to entry IV); `EXIT_TIME_DECAY` (>20 minutes since the last new
  premium high and bid >2% below entry ask); `EXIT_TAKE_PROFIT` (fresh NIFTY
  touches/crosses live Max Pain); `EXIT_PCR_DROP` (PCR falls at least 20% from
  entry PCR); then `EXIT_TIME_LIMIT`. One exit closes each analytical position.
  Time decay is a price/time heuristic, not causal Theta attribution. The Max
  Pain reason does not guarantee positive PnL.
- DTE retains the 15-minute expiry-day and 45-minute later-expiry caps, measured
  independently of signal admission TTL. The 15-minute cap can preempt the
  20-minute decay rule on expiry day.
- Fyers snapshots refresh every 15 seconds during streaming. Evaluation is
  serialized; REST fetching happens outside the evaluation lock. IV/PCR/Delta
  context older than 45 seconds cannot trigger related exits or adjust stops.
  Fresh bids can still trigger an existing stop; missing Greeks never erase it.
  Missing quotes, >2-second-old bids, or future timestamps freeze evaluation.
  Entries require a real <=2-second quote and <=45-second contract context.
  Max Pain crossing requires fresh index data. Old REST quotes retain their
  original timestamps and cannot overwrite newer websocket marks.
- `TE_PCR_DROP_FRACTION=0.20` configures the relative PCR exit drop (0–1,
  exclusive). `TE_MAX_IMPLIED_VOLATILITY_PERCENTILE` defaults to 0.50;
  larger legacy values cannot relax the strict 50th-percentile gate.
- Runtime sizing now uses premium stop distance and the same assumed 65-unit
  NIFTY lot multiplier as analytical PnL. Verify the current contract specification
  before interpreting size or PnL; no orders are submitted.
- DuckDB stores candidate decisions, `lifecycle_alerts` (parameterized,
  idempotent event IDs), and analytical outcomes with exact exit reasons. The
  engine publishes these events to both terminal pages without a second broker
  runtime. Raw option ticks still flush to Parquet above 5,000 rows per token;
  pre-existing writer-failure, file-reuse, and shutdown-flush limitations remain.

Historical research, backtesting, and model utilities remain available. The full
live runtime uses Fyers websocket/REST exclusively for spot, VIX, and option-chain
data. Direct public NSE scraping and its scheduler have been removed permanently.
Generic API ingestion also blocks public NSE hosts and redirects to them; saved
legacy snapshots may still be parsed offline.
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


## Fyers-only runtime fix — 2026-11-12

`refresh_once()` calls the existing FyersMarketClient for a snapshot and passes
`snapshot.option_instruments` directly into MarketStateAggregator alongside
spot/VIX. There is no NSE HTTP enrichment, cookie bootstrap, or website fallback.
The scraper module and its unused scheduler were deleted; public NSE URL/header
settings were removed. Network-denial regression tests cover generic ingestion
and redirects, while runtime tests assert the exact Fyers chain reaches state.

This fix is synchronized to the canonical laptop checkout. Existing Docker
containers must be rebuilt to load the changed source:

```bash
docker compose up --build -d engine dashboard
```

The offline suite passes 114 tests and 5 subtests. Live Docker/Fyers execution
was not rerun as part of this source fix.


## Fault-tolerant websocket packets — 2026-11-12

The entire Fyers message callback now catches Exception. Malformed/incomplete
option packets and normalization/integrity failures log a redacted WARNING and
return immediately; they never call stop/close or escape into the SDK manager.
Validation precedes all state publication. Bid/ask/volume aliases and zero volume
are supported; missing bid/ask is never synthesized from LTP. Native index
messages are accepted without option-only fields, and control acknowledgements
update heartbeat health without creating market ticks.

Stale/future/duplicate/out-of-order packets are rejected locally, allowing the
next valid packet through without latching global safe mode. Independent security
safe mode and freshness/frozen-position protections remain intact. Exchange epoch
strings are parsed correctly; old last-trade time is not treated as quote time.
Consumer exceptions are isolated. Packet drop/error counters are on stream.state.

All 149 tests and 5 subtests pass in an isolated Python 3.12 validation container
with networking disabled and no account credentials or live data mounted. Thirty-
five new cases prove packet rejection/recovery and that no socket-close/stop call
occurs. Production rebuild/observation is recorded in project_status.md.

Schemas: [Fyers symbol updates](https://fyers.in/api-docs/v3/api/websockets/data-socket/data-socket-symbol-update/)
and [index updates](https://fyers.in/api-docs/v3/api/websockets/data-socket/data-socket-indices-update/).

Parser fix deployed locally at 2026-11-12 10:27 IST: engine rebuilt/restarted, dashboard
kept running. The inspected live interval recorded one skipped-packet warning,
zero socket-close events and no fatal parser/engine error, with connected fresh
Fyers data and safe_mode false. [The terminal](http://localhost:8501) remains
available. Long-duration feed/lifecycle observation remains pending.

## Lifecycle upgrade validation — 2026-11-12

The isolated Python 3.12 regression suite passes 167 tests plus 5 subtests.
Coverage includes reference Delta values, strict entry boundaries, complete-bar
index ATR, stop ratcheting, each exit route, event/outcome persistence, stale or
missing context, and exact dashboard labels. These are offline tests with
networking disabled and no account credentials mounted. Live lifecycle behavior
and AWS EC2/VPS observation remain unverified for this release.

## Source publication

Project source, tests, deployment configuration, and the three phase documents
are committed from the canonical laptop checkout. Credentials (`.env` and its
local variants), generated dashboard JSON, market data, and logs are excluded;
`.env.example` remains a safe configuration template. A Git push publishes code;
use the documented Docker rebuild separately to activate a release.

## Dashboard clock — Indian Standard Time

Signal audit rows, rejection details, entry/exit alerts, Live Engine Telemetry,
position Entry Time and Signal Time, and equity-chart labels use Asia/Kolkata.
Display format: `12 Nov 2026 11:30:07 AM IST` (date, twelve-hour clock, exact
seconds, AM/PM, literal IST; no ISO offsets). Time in Trade remains a duration.
Naive timestamps are interpreted as UTC; aware timestamps retain their actual
instant before conversion. Missing/invalid timestamps display `—`.

DuckDB, engine publications, sorting, and freshness/TTL calculations retain UTC.
New publications include position opening and signal timestamps; older snapshots
show `—` for unavailable position times until the updated engine publishes them.
This presentation change passed all 35 dashboard tests and changed-file lint;
live deployment/market validation is still pending. Rebuild the Docker services
to activate the updated dashboard and publication fields.

## Fyers packet clock tolerance

Fyers websocket ingress accepts exchange timestamps up to **60 seconds old**
without warning logs, including exactly 60 seconds. Any older row rejects its
packet with a warning; the socket keeps listening. `_on_message` passes this
explicit tolerance to the integrity guard, so an existing generic
`TE_STALE_STREAM_SECONDS=20` does not override it. Other timestamp-integrity
checks remain active.

This only changes packet admission. Stored exchange times, dashboard clocks,
and the monitor's two-second quote-freshness requirement are unchanged. A real
20-second hardware skew can still freeze analytical valuation; accepting packets
does not prove live execution readiness. Rebuild the services to activate the
change. Offline validation: 62 broker tests and 5 subtests pass; changed-file lint
passes. Live observation of the reported drift remains pending.
