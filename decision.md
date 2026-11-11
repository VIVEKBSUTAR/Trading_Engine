# Architectural decisions

## Permanent phase-completion protocol — 2026-11-11

Every completed development phase must update `decision.md` with architectural
reasoning, `README.md` with actual system state, and `project_status.md` with
exactly what was built, validation evidence, pending work, and consequences for
live market execution. The existing uppercase README is the canonical readme;
there must not be a competing lowercase copy. Root `AGENTS.md` persists this
protocol for future development. If architecture is unchanged, say so explicitly.
Implementation completion and real-market validation must remain distinct.

## Kite API to Fyers API V3

The active runtime now instantiates `FyersAuthManager`, `FyersMarketClient`, and
`FyersStream`. This consolidates live input around Fyers REST snapshots/option
chains and symbol-based DataSocket subscriptions, including option bid/ask data
needed for premium valuation. Symbol/token registration routes ticks into the
existing typed state engine without coupling signal logic to websocket payloads.

Legacy compatibility artifacts remain in the checkout, but they do not define
the active architecture. This phase removes outdated execution-oriented README
guidance; it does not remove unrelated historical modules. The runtime still
has supplementary NSE option-chain ingestion and requires end-to-end validation.
The system's supported role is analytical signals and lifecycle tracking, not
broker order execution.

## Index spot to option premium valuation

Index spot remains context for regimes, directional probability, and strike
selection. It cannot represent the realizable value of an option: option value
also responds to volatility, time decay, and spread. Long CE/PE analytical
positions therefore enter at the option ask and mark/exit at the option bid,
using the selected contract's token rather than NIFTY spot as a P&L proxy.

The monitor calculates `(current_bid - entry_ask) * lot_size * quantity` and
records exits to DuckDB. The current lot-size constant is 65; this is a code
assumption, not a verified current contract specification. Analytical P&L is not
a broker fill or net realized account P&L.

Liquidity rejection uses `(ask - bid) / bid > 0.05` by default. Volatility
rejection uses the default 90th percentile against rolling IV history (up to
100 observations per strike/side). Existing missing-data and warm-up bypasses
must be assessed during live validation; implemented filters alone do not prove
live readiness.

## DuckDB relational persistence and Parquet tick flushing

`storage/duckdb_manager.py` bootstraps `ingestion_snapshots`, `signals`, and
`trade_outcomes` in `data/cache/metadata.duckdb` by default. Prepared parameterized
inserts store accepted/rejected signal decisions, spread/IV diagnostics, and
closed analytical positions with ask entry, bid exit, duration, reason, and P&L.
Signal and outcome primary keys use INSERT OR REPLACE for repeated IDs. Timestamp
indexes support recent-event analytics; DuckDB can also query Parquet history.

`MarketStateAggregator` maintains per-token raw bid/ask buffers. Above 5,000 rows
(strictly greater, not greater-or-equal), it copies a buffer to
`ParquetWriter.write_option_ticks()` and clears that token's memory buffer.
Explicit one-token and all-token flush methods exist. PyArrow writes beneath
`processed_dir/option_ticks`, partitioned by UTC date and instrument token.
This separates compact relational decision history from bulk tick data.

Durability limitations remain: the current flush clears the buffer even if the
writer is missing or fails; repeated writes reuse `part-{i}.parquet` with
`overwrite_or_ignore`, risking replacement within a partition; runtime shutdown
does not currently invoke the all-buffer flush. These are pending hardening
items, not guarantees supplied by the existing persistence implementation.

## Isolated live-fire validation phase — 2026-11-11

Create `live_fire_test.py` without importing the analytical engine. Read Fyers
credentials only from environment variables; discover the nearest available
expiry dated today or later and obtain the exact CE/PE symbols from that expiry's
option chain. Select the nearest listed strike to the returned index price,
without guessing expiry calendars or constructing symbol strings.

Use full SymbolUpdate frames and count exactly 50 subscribed option updates
combined across both contracts, excluding acknowledgements. Use a lock and event
to cap output and ignore late callbacks. Disable reconnect and bound the websocket
wait (120 seconds by default); a short stream must fail. Close on the controlling
thread using the SDK's `close_connection()` because SDK shutdown joins its worker
threads. No extra keep_running thread is needed. Add the missing Fyers SDK to
project dependencies and Fyers variables to `.env.example`.

This probe separates connectivity evidence from analytics, persistence, and
signal behavior. It sends no order requests. Offline checks cannot complete the
pending real-world websocket milestone.

## Headless deployment architecture — 2026-11-11

Headless startup uses requests, account PIN, and a current pyotp TOTP to obtain
an SSO session, verify the returned redirect URI/state, exchange its auth code,
and write the daily API token into the mounted local .env under a file lock.
No response body or credential is logged. The shell verifies a fresh generation
timestamp and loads the new token explicitly before execing the daemon, because
Compose env_file values and parent process environments do not update themselves.
Headless-only settings validation preserves credential-free research workflows.

A Python 3.12 slim image installs the project non-editably. Engine and dashboard
share data/log volumes; only engine receives account secrets and the writable
.env. Dashboard reads atomic JSON publications from the engine rather than
opening a DuckDB file owned by another process or starting another live runtime.
Fyers symbol subscriptions now activate streaming in the daemon, and the wrapper
uses the SDK's close_connection method. Authentication failure stops startup;
there is no automatic restart loop that repeatedly submits account logins.
The image refreshes tokens on startup, not across subsequent daily expiry.

The SSO endpoints are undocumented and must be validated with the account.
The documented code exchange is described in the
[Fyers authentication flow](https://fyers.in/api-docs/v3/api/authentication/individual-app/authentication-flow/).

## Contract-specific mathematics and deployment verification — 2026-11-11

Directional generation calculates calendar DTE from the snapshot's Asia/Kolkata
date and the exact selected contract expiry. Expiry day carries a 15-minute cap;
later days carry 45 minutes. Missing, invalid, or expired contracts fail closed.
Signal admission TTL remains distinct from position lifetime; the monitor starts
the holding clock at entry and enforces the signal's cap instead of inheriting
a short admission TTL. Evaluation still requires fresh quotes; no stale bid is
presented as an executable deadline exit.

extract_strike_quote now returns bid, ask, IV, CE OI, and PE OI for one strike and
expiry. Put/call OI below 0.8 rejects BUY_CE; above 1.2 rejects BUY_PE. Boundaries
are inclusive for admission. Missing/nonfinite OI, negative put OI, or nonpositive
call OI fails closed instead of dividing by zero or mixing unrelated expiries.
OI is a heuristic exposure filter, not proof of the intent behind open positions.
Expiry dates are normalized from Fyers DD-MM-YYYY/epoch forms and nearest expiry
selection excludes expired contracts relative to the snapshot date.

The monitor derives option ATR from bid/ask midpoint ticks resampled into
completed, contiguous one-minute OHLC bars from the current market date. It uses
the arithmetic mean of 14 true ranges: max(high-low, abs(high-previous close),
abs(low-previous close)). The forming minute is excluded. Insufficient/invalid
history suppresses entries; no index-ATR or fixed-percentage fallback is used.
Initial risk is fixed at 1.5 entry ATR, with stop_loss_price = entry ask - risk.
At bid-entry >= initial risk (1R), trailing activates and ratchets upward at one
current ATR below bid. Missing later ATR keeps the last valid ATR. The initial
premium target is 3 ATR above ask (2R), replacing index-scaled target levels.

Runtime, websocket, engine, and monitor now share the engine's aggregator, so ATR
uses actual streamed option history rather than a disconnected snapshot buffer.
ATR can also read retained general tick history after an option archive flush.
Retention limits can still cause re-warm-up; the original Parquet durability
limitations are not changed in this phase.

DuckDB performs idempotent additive migrations for max_holding_minutes,
days_to_expiry, and strike_oi_ratio. Existing signal records remain readable with
NULL for previously absent diagnostics. New accepted/rejected OI decisions retain
the ratio and contract holding limits in the relational audit trail.

Deployment checks found Fyers 3.2.1 pins requests==2.31.0; the former project
requests>=2.32 requirement prevented resolution. The compatible project range is
now >=2.31,<3. The wheel builds non-editably. Offline auth, state, persistence,
math, and dashboard checks are distinct from unperformed Docker/account tests.
The deployed dashboard uses timed Streamlit fragments to refresh publications
without a blocking rerun loop or a second live runtime. Required documentation
was updated for both deployment and mathematics phases.

Final verification: 100 repository tests and 5 subtests pass. Changed-code lint,
Bash syntax, Compose YAML parsing, and non-editable wheel construction pass.
Docker execution and real account/market validation remain pending.

## Multi-page trading terminal — 2026-11-11

Rename the former dashboard app.py to 1_Execution_Desk.py and add
pages/2_Analytics_And_Logs.py, using Streamlit's native pages-directory navigation.
Shared read-only access, calculations, and components live in dashboard/shared.py;
the old large monolithic page and its unused panel hierarchy are removed.
The CLI launcher and Compose command point to the new default script, and the
wheel includes both page scripts. Docker copies the root .streamlit configuration.

Separate decision-time information from retrospective investigation. Execution
Desk contains only a three-metric health row, a current qualifying signal alert,
and active positions. Performance metrics, the equity curve, historical signals,
and rejection explanations belong to Analytics & Logs. Numeric position PnL uses
cell-specific green/red styling and trailing stops use bold font weight, preserving
numeric values rather than embedding presentation markup into data strings.
Streamlit's minimum version is raised to 1.51 to support table font-weight styling.
The configured dark theme uses the requested green accent and two dark backgrounds.

Live alerts must originate in persisted passed BUY_CE/BUY_PE decisions timestamped
between now minus 60 seconds and now. The engine queries that window separately
from the 50-row audit limit so frequent rejections cannot displace eligible alerts.
An additive entry_price DOUBLE migration stores the selected option ask. Older
records keep NULL; the UI does not substitute index spot for option premium.

Deployment continues to use one DuckDB writer and an atomic JSON publication.
The expanded payload carries separately queried live signals, actual websocket
connected/last-tick/safe-mode fields, spot/VIX, position quantities/strikes/sides,
and premium PnL. The dashboard never creates a broker runtime. Local offline mode
can query a read-only database, but reports unavailable live health/positions.

Recheck the 60-second signal window at render time. Show the newest qualifying
signal in st.success and emit st.toast once per ID/batch per browser session.
Mark older members of the current batch seen to avoid resurfacing them later.
Suppress alerts once a publication is over 30 seconds old, and mark its cached
positions stale. A connected socket without a recent tick is displayed as stale,
not as healthy market data. These are analytical notifications, not order actions.

Analytics uses a Plotly filled area chart with unified hover, restrained grid
lines, no unnecessary axis titles, and chronological cumulative realized PnL.
Plotly was already a dependency; no duplicate declaration is necessary. The four
performance metrics lead this page; its audit table and Rejection Reason expander
come after the chart. Timed Streamlit fragments refresh both pages without a
blocking rerun loop.

Validation: 107 tests and 5 subtests pass, including actual DuckDB alert-window
queries, uppercase/lowercase actions, time boundaries, alert deduplication, numeric
profit/loss styles, stale publication suppression, page navigation/hierarchy, and
Plotly area behavior. Browser review with isolated synthetic fixtures confirmed
both layouts and table styling. Saved previews are explicitly labeled as fixture
images in README. Non-editable wheel construction and changed-code lint pass.
Docker startup and real feed readiness must still be established separately.

### Deployment attempt after UI completion

No architectural changes were needed for this verification step. Theme values,
Compose YAML, and the renamed default entrypoint were checked. Running the exact
requested command, docker compose up --build, exited 127: docker: command not
found. The existing launcher symlink targets an absent Docker Desktop binary.
No container, account login, or live market feed was started. The isolated local
fixture server used for visual review was stopped. The terminal is not live.
Terminal tables use the supported width API to avoid deprecated container-width
warnings; Plotly uses its default responsive width.

## Root Docker configuration verification — 2026-11-11

No architectural change. The requested docker-compose.yml, Dockerfile, and
entrypoint.sh already exist directly in the repository root and meet the required
service/image/startup layout. Preserve the existing token freshness verification
and single-writer dashboard design instead of replacing them with a simpler
startup that could consume a stale token. Compose loads .env into engine; dashboard
uses the published analytical data without receiving account secrets. Both services
mount persistent data/log directories. Reapplied executable permission to the root
entrypoint and checked Bash syntax, YAML structure, required copies, non-editable
installation, and authentication-before-daemon ordering. Runtime deployment remains
unverified because the previously documented Docker tooling blocker persists.

## Canonical local checkout and publication

Use the user-confirmed /Volumes/APFS/projects/trading_engine checkout for local
source synchronization and Git publication. Its main branch already contains
three unpublished predecessor commits and descends from origin/main, so a normal
fast-forward push can publish the complete history without force. Preserve the
root checkout's seven existing local edits while integrating this chat's work.
Three-way merging combines the Fyers date/shutdown changes and DuckDB migration
with those edits. The old in-process dashboard has been replaced by the two-page
publication reader; retain temporary-copy refresh for the historical database
reader, while deployment continues to avoid database locking through JSON.
Keep a local backup Git ref for the original root changes before replacing files.
Never copy or stage .env, credentials, datasets, logs, or runtime databases.
AGENTS.md records the user's ongoing local-checkout synchronization instruction.

Combined-tree validation passes 108 tests and 5 subtests. Crossing midnight during
verification exposed wall-clock-dependent ATR fixtures; their clock is now fixed
to an intraday instant. Production session boundaries and ATR warm-up rules are
unchanged. The temporary-copy historical reader also has a refresh regression
test. Original local edits are preserved in the combined tree and a local backup
Git reference. Publication targets main through a normal fast-forward push.
