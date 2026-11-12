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

## Local headless environment validation — 2026-11-12

No architectural change. Populate only the canonical checkout's ignored local
.env with the requested TOTP secret and placeholder account ID/PIN; do not mirror
the credential file into the chat workspace. Keep restrictive file permissions
and preserve unrelated settings. The authentication script loads FYERS_TOTP_SECRET
from environment/.env, checks the TOTP time boundary, then calls exactly
pyotp.TOTP(totp_secret).now() for the OTP payload. Local six-digit generation and
five offline authentication tests pass without contacting Fyers. Docker Compose
configuration validation now succeeds; capture/suppress its rendered environment
because ordinary config output includes secrets. Placeholder account ID/PIN satisfy
interpolation but do not establish login readiness or live-market connectivity.

## Local account configuration completion — 2026-11-12

No architectural change. Replace the account ID/PIN placeholders only in the
canonical ignored .env, retain 0600 permissions, and validate Compose without
printing its rendered credentials. The values are not copied into source,
documentation, Git, or the chat workspace. Compose validation passes; this
configuration check does not contact SSO or establish account/login readiness.

## Permanent removal of public NSE scraping — 2026-11-12

The production error path was a second option-chain fetch inside refresh_once:
NSEOptionChainFetcher attempted public NSE HTTP scraping even though FyersMarketClient
already supplied normalized options in its snapshot. Remove that dependency,
constructor state, parser/validator enrichment pass, and its HTTP error/fallback
handling. Route snapshot.option_instruments directly through update_from_snapshot's
option_chain argument so spot/VIX and option premiums/OI share the Fyers source.
No additional option-chain REST request or website fallback is introduced.

Delete ingestion/nse_fetcher.py and the unused legacy ingestion/scheduler.py, which
otherwise retains an import of the removed scraper. Remove public NSE endpoint,
cookie-bootstrap, and browser-header defaults from runtime settings. Retain offline
parsers for previously saved data. Generic historical API ingestion rejects public
NSE hosts before opening a connection and in the redirect handler, preventing a
configured URL from restoring the removed path. This uses urllib's supported
[HTTP redirect hook](https://docs.python.org/3/library/urllib.request.html#urllib.request.HTTPRedirectHandler.redirect_request).
AGENTS.md now makes the Fyers-only live-source policy permanent.

Validation: 114 tests and 5 subtests pass. Runtime refresh uses one mocked Fyers
snapshot, preserves its exact option-chain DataFrame, and makes no auxiliary
HTTP request. The removed scraper and scheduler modules cannot be imported.
Dedicated regression cases deny public NSE URLs (including subdomains/case/trailing
dots) and redirects without network access. No live auth/session or container
restart was performed; rebuild running images to load the fix.

## Local live startup verification — 2026-11-12 10:01 IST

No architectural change. Start Docker Desktop through its CLI, then rebuild and
launch engine/dashboard from the canonical checkout with docker compose up --build
-d engine dashboard. Both services run on the existing Fyers-only deployment.
Headless daily token generation succeeded, the dashboard health endpoint returned
HTTP 200, and the Execution Desk visibly showed a connected websocket.
A real engine publication was 0.34 seconds old and its last Fyers tick was 0.37
seconds old at inspection, with safe_mode false. Python 3.12.16 is running in the
engine image and the retired scraper module is absent. Service logs contained no
legacy scraper error or fatal engine error during the inspected startup interval.
No credentials were displayed or mirrored. This establishes local startup/feed
connectivity, not AWS deployment, guaranteed strategy accuracy, or broker fills.

## Fault-tolerant Fyers parser upgrade — 2026-11-12

The old callback wrote option ticks before validating integrity and called stop()
on a validation error. A single malformed/stale packet could therefore contaminate
state and close the socket. The rewrite places decoding, extraction, normalization,
integrity validation, and routing inside a broad try/except Exception. An invalid
packet logs WARNING and returns without stop/close or an escaping exception.
Validate every row before aggregator/strategy publication. Isolate raw/tick
consumer errors so one consumer cannot crash the SDK or block the others.

Recognize dictionaries, JSON strings/bytes, and tick lists. Classify known control
frames as heartbeats; native sf packets and untyped complete quote dictionaries
remain market data. Required option fields are finite positive LTP/bid/ask, a
noncrossed book, nonnegative volume, and a valid token. Zero volume and native or
shorthand field names are supported. Index if packets legitimately omit bid/ask
and volume, so they remain valid without adding them to option quote buffers.

Remove fabricated bid/ask from LTP. Preserve zero-valued volume aliases and parse
numeric-string feed epochs. Feed/receive time describes quote freshness; last
executed trade time can legitimately be old and is not a substitute. Rejected
packets never update last_tick_at. Warnings show stage/reason without raw packets
or arbitrary exception text; stream state records drop count and last error.

Fyers uses APISecurityGuard packet_only validation to reject stale/future/duplicate/
out-of-order data without permanently latching global safe mode from that packet.
Independent security safe mode is never cleared and legacy callers retain their
default policy. Freshness and frozen-position protections still apply. Keeping
the transport alive does not mean accepting invalid analytical data.

Verification: 149 tests and 5 subtests pass in a network-disabled Python 3.12
container without credentials/live data. Thirty-five new cases cover malformed
messages, required fields, control/index packets, aliases/zero volume, timestamp
forms, consumer/normalizer/validator/aggregator failures, data-integrity rejection,
valid next-packet recovery, and assertions that no stop/close occurs. Local source
and documentation are synchronized before deployment observation.

### Deployed parser verification — 2026-11-12 10:27 IST

No additional architectural change during deployment. Rebuilt/restarted only the
engine with docker compose up --build -d engine, leaving dashboard available.
Inspected the running callback to confirm broad exception containment and absence
of stop/close calls, plus the packet_only guard API. At the short live check the
new container logged one skipped-packet WARNING, zero old fatal integrity messages,
zero SDK close events, and no fatal engine error. Websocket remained connected
with safe_mode false; publication age was 1.95 seconds and published tick age 1.99
seconds. Dashboard health returned HTTP 200. This is bounded local observation,
not a guarantee of long-duration stability or executable fills.

## Delta/IV/positioning lifecycle architecture — 2026-11-12

This phase changes analytical entry selection, premium risk sizing, lifecycle
monitoring, event persistence, and the Execution Desk. It supersedes the earlier
ATM/confidence strike heuristic, 90th-percentile warm-up bypass, directional
strike OI gates, and option-midpoint ATR / 1R trailing / fixed premium target.
Fyers remains the exclusive live data source; no order operations are introduced.

Select the nearest eligible |Delta| to 0.70 within 0.60–0.80 using the correct CE
positive / PE negative sign. Request optional Fyers Greeks with `greeks="1"`
and retain IV in percentage units. Provider Delta is preferred; a validated
European Black–Scholes zero-carry approximation is available when only Delta is
missing. Expiry time is 15:30 IST and the year basis is 365 days. Missing IV is
never synthesized. Higher Delta reduces relative extrinsic exposure in some
contracts, but does not itself prove minimal Theta exposure. Mathematical
reference: [OIC option price behavior](https://www.optionseducation.org/referencelibrary/faq/option-price-behavior).
Provider contract: [Fyers option-chain documentation](https://fyers.in/api-docs/ai/skills/option-chain/)
and [Fyers market-data reference](https://github.com/FyersDev/fyers-skills/blob/master/skills/fyers-trading/references/market-data.md).

Entries require Trap Score <0.5, same-expiry aggregate PCR >1.5, and empirical
IV percentile <50, plus existing quality/probability/spread/risk controls. A PE
entry is subject to the same requested support gate; the old PE ratio >1.2
rejection would contradict that gate and is removed. Use at most 100 distinct
same-contract chain timestamps and require 20 observations. Observe every
returned contract independently of entry gates to avoid sampling only passed
candidates. Deduplicate historical chain rows before computing OI or Max Pain.
This is positioning inferred from the returned strike range, not proof of
institutional ownership or the entire market's positions.

The monitor uses 14 completed contiguous one-minute index-LTP true ranges plus
a preceding close. It deliberately avoids mixed option candles and daily OHLC.
Initial premium risk is NIFTY ATR × |Delta|; the stop ratchets upward from bid
minus that live distance and can never widen. Entries and sizing use the same
premium risk, with the monitor's assumed 65-unit lot multiplier. No percentage
stop, fixed premium target, or profit-activation delay remains in the live path.
Warm-up and stale data fail closed rather than manufacturing prices.

Exit precedence is stop, IV crush (>10% relative entry-IV fall), decay heuristic
(>20 minutes without a premium high and >2% below entry ask), Max Pain touch or
crossing, PCR drop (>=20% relative entry PCR), then retained DTE holding limits.
The user accepted the 20% PCR default and Max Pain crossing interpretation;
`TE_PCR_DROP_FRACTION` makes the PCR default configurable. The 15-minute expiry-day
limit intentionally preempts a 20-minute decay exit. Time/price alone cannot
attribute losses causally to Theta, and Max Pain exits can close at a loss.

Periodic 15-second Fyers REST refreshes keep IV/OI context current alongside
socket prices. REST fetches run outside the evaluation lock; state mutation,
monitoring, persistence, and report publication are serialized. Contract context
expires after 45 seconds, bids after two seconds, and Max Pain uses fresh index
observations. Stale REST quotes retain their observation timestamps and cannot
overwrite newer marks. Existing protective stops survive missing new Greeks.

Candidate direction remains BUY_CE/BUY_PE for compatibility with historical
analysis. Separate `ENTRY_BUY` and exact `EXIT_*` events describe actual
analytical lifecycle transitions. Parameterized DuckDB `lifecycle_alerts` inserts
use unique event IDs, preserving exits beyond the monitor's current report.
The dashboard prioritizes exits, deduplicates toasts, and shows live bid premium,
bold stop, elapsed minutes, and IV. Neither events nor outcomes imply broker fills.

Validation: 167 offline Python 3.12 tests and 5 subtests pass, including numerical
Delta references, strict boundaries, underlying ATR, all exit routes, idempotent
persistence, and Streamlit rendering. Changed-file Ruff checks pass (the existing
DuckDB close-method SIM105 warning is excluded). Networking was disabled and no
credentials were mounted. New lifecycle deployment/observation, model calibration,
restart recovery for active analytical positions, and durable IV history remain
pending. This phase does not restart the previously running Docker services.

## Git publication and runtime exclusions — 2026-11-12

Architecture is unchanged in this publication phase. The pending Fyers-only
pipeline removal, resilient packet parser, Greeks/lifecycle mathematics, tests,
configuration, and documentation are grouped into one reviewed Git change from
the canonical laptop checkout. Runtime dashboard JSON is excluded alongside
credentials, logs, and market data. Environment variants are ignored while
`.env.example` remains versioned. Publishing source does not rebuild containers,
recover in-memory analytical positions, or validate live market behavior.

Validation evidence remains 167 passing Python 3.12 tests plus 5 subtests,
changed-file lint, and a clean whitespace check. Canonical and workspace source
files match. Remote publication is verified against the pushed commit; live-fire
lifecycle deployment and AWS EC2/VPS observation remain pending.

## Dashboard IST presentation — 2026-11-12

Architecture is unchanged. Both Streamlit pages delegate to shared renderers;
apply one display formatter there rather than converting the DuckDB query output
or changing engine clocks. `pd.to_datetime(..., utc=True, format="mixed")`
localizes naive inputs to UTC and normalizes aware inputs without double
localization. Convert with `.dt.tz_convert("Asia/Kolkata")`, then format a dated
12-hour clock with seconds, AM/PM, and literal IST. Invalid/missing values show
an em dash rather than an invented time. Dates preserve midnight rollovers.

Format audit rows, rejection text, lifecycle banners/toasts, telemetry update/tick
times, and position entry/signal timestamps only at presentation boundaries.
Publish the latter fields as UTC ISO values for the renderer. Time in Trade is
still elapsed minutes, and UTC freshness checks precede display formatting.
Plotly receives IST wall-clock x coordinates with explicit IST tick/hover labels
so the browser cannot silently substitute another timezone.

Validation: all 35 dashboard tests pass in isolated Python 3.12 with networking
disabled; changed-file Ruff passes. Tests cover mixed naive/aware timestamps,
UTC/IST offsets, noon/midnight/day rollover, missing/invalid values, audit/rejection
rendering, active positions, telemetry, alerts, chart hover, and unchanged UTC
freshness calculations. No live deployment or market-readiness claim is made.
