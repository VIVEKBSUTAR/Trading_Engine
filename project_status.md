# Project status

Last updated: 2026-11-12 (Asia/Kolkata).

## Current phase: Fyers packet clock tolerance — 2026-11-12

Built: `_on_message` explicitly applies a 60-second stale-age tolerance through
the shared integrity guard. Ages <=60 seconds are accepted without warnings;
any row older than 60 seconds rejects its packet without closing the connection
or latching global safe mode. Generic callers retain their configured tolerance;
other integrity checks and original exchange timestamps are preserved.

Validation: **62 broker tests passed plus 5 subtests**, with a deterministic
clock testing 20/59.999/60-second acceptance, 60.001/61-second rejection, no
accepted-drift warnings, old rows in mixed batches, unchanged generic policies,
and socket continuity. Python 3.12 tests ran with networking disabled. Changed-file
Ruff passes excluding existing SIM103.

Pending: rebuild/deploy the services and observe real packets under the reported
clock skew. The separate two-second trade-quote freshness check remains strict,
so this admission change alone cannot establish fresh valuation or live readiness.
Live-fire lifecycle deployment and real-time AWS EC2/VPS observation remain
pending. No broker orders or fills are introduced.

## Previous phase: dashboard IST timestamps — 2026-11-12

Built: one shared pandas UTC-to-Asia/Kolkata display formatter; exact seconds in
a dated twelve-hour AM/PM clock ending in IST; converted signal audit/rejection
rows, lifecycle alerts/toasts, telemetry timestamps, position Entry Time/Signal
Time columns, and equity hover/axis labels. Engine snapshots now publish UTC
position opening and signal times. Missing/invalid/older missing fields show `—`.
Time in Trade stays in minutes. Storage, UTC freshness windows, and trading
mathematics are unchanged.

Validation: **35 dashboard tests passed**, including timezone-aware/naive and
mixed-format inputs, noon/midnight/date rollover, invalid data, all affected
renderers, and UTC freshness preservation. Changed-file Ruff passes. Tests ran
in an isolated Python 3.12 container with networking disabled.

Pending: rebuild the engine/dashboard containers and observe the rendered times
from real Fyers publications; live-fire lifecycle deployment and real-time AWS
EC2/VPS observation remain pending. This phase changes analytical presentation
only, with no broker orders or changes to signal timing or exit rules.

## Previous phase: Greeks and lifecycle upgrade — 2026-11-12

Built: signed Delta-near-0.70 strike selection; strict Trap <0.5, aggregate PCR
>1.5, IV percentile <50 gates; 20-snapshot IV warm-up; explicit Fyers Greeks
requests/normalization; zero-carry Black–Scholes Delta fallback when IV exists;
index-only 14-period completed-minute ATR; Delta-adjusted initial/ratcheting
premium stops; consistent premium risk sizing; separate persisted `ENTRY_BUY`
and five requested exit alerts, plus retained DTE `EXIT_TIME_LIMIT`.

Built: 15-second Fyers snapshot refresh during streaming; serialized evaluations;
45-second IV/positioning context limits; fresh real entry quotes; preservation of
websocket timestamps/marks; Max Pain/PCR from deduplicated same-expiry rows;
Execution Desk Live Premium, bold Trailing Stop, Time in Trade, Live IV columns,
exact reason banners/toasts, and exit priority. Candidate signal logs remain
separate from actual analytical entry events. The 20% PCR fall default and Max
Pain touch/crossing rule were explicitly accepted by the user.

Validation: **167 tests passed and 5 subtests passed**, in an isolated Python 3.12
container with networking disabled and no credentials/runtime data mounted.
Coverage includes known Delta values/signs, missing Greeks, strict entry gates,
index ATR warm-up/gaps, stop non-widening, every exit route and precedence,
idempotent DuckDB events/outcomes, stale-context stop protection, and Streamlit
labels/columns. Changed-file Ruff passes excluding pre-existing SIM105 in the
DuckDB close method. This evidence is offline, not proof of live readiness.

Pending: **Live-fire deployment and real-time AWS EC2/VPS observation.** This
release has not restarted the existing local Docker services. Deploy from the
canonical checkout using `docker compose up --build -d`, then observe real IV,
Delta, PCR, index ATR warm-up, entries/exits, timing, and dashboard delivery.
Active analytical positions and IV observations restart empty; durable state
recovery, longer IV history/calibration, contract lot-size verification, and
pre-existing Parquet durability work remain pending.

Execution impact: alerts and positions are analytical only. The new gates reduce
admission and require warm-up; missing data suppresses new entries. Fresh quotes
can close at the retained protective stop while missing/stale Greeks suppress
only dependent exit checks. The 15-minute expiry-day cap preempts the 20-minute
decay heuristic. Max Pain/PCR reflect returned strikes; time/price decline is not
causal evidence of Theta loss. No broker orders, fills, or account PnL are claimed.

## Earlier completed implementation (superseded math noted above)

| Area | Exactly what was built | Impact on live market analysis/execution |
| --- | --- | --- |
| Fyers integration | Active auth/REST/option-chain/websocket adapters, bid/ask normalization, symbol/token routing; daemon now starts symbol/discovered subscriptions; SDK shutdown uses close_connection | Supplies analytical market data; actual account/feed behavior still needs live observation |
| Premium valuation | Option ask entry, bid marking/exit, token-specific lifecycle and analytical P&L | Removes index spot as the P&L proxy; recorded outcomes do not establish broker fills |
| Spread/IV filters | Default 5% spread cap, strict IV percentile <50, 20 distinct observations required | Missing quotes/IV/history suppress admission |
| DuckDB/Parquet | Decision/rejection/outcome tables and per-token archival flush above 5,000 rows | Provides analytical audit/replay data; pre-existing durability gaps remain below |
| Headless authentication | requests SSO login, current pyotp TOTP, encoded PIN, redirect URI/state validation, SHA-256 auth-code exchange, fail-closed redacted errors | Removes browser steps from container startup; provider SSO is undocumented and not live-validated |
| Credential/config support | FYERS_USER_ID, FYERS_PIN, FYERS_TOTP_SECRET fields, .env example, and strict validation for headless mode | Headless startup refuses missing credentials; research and token-only probes stay usable |
| Token injection | Locked in-place .env update, duplicate token removal, restrictive permissions, fsync, generation timestamp, persisted-value verification | Preserves the file bind mount and unrelated settings; avoids logging tokens; successful startup explicitly loads the new token |
| Container architecture | Python 3.12 slim Dockerfile with non-editable installation; engine and dashboard Compose services; port 8501, persistent data/log volumes, .dockerignore, dashboard healthcheck | Makes deployment repeatable; engine is sole analytical writer; no image has been built/run here |
| Entrypoint | Authenticates, verifies a fresh generated token, then execs te-intelligence | Authentication failure prevents stale-token startup; no login restart storm or daily scheduler is introduced |
| Independent dashboard | Engine atomically publishes latest.json; dashboard reads it with timed fragments, publication age, signals, positions, and recent outcomes | Avoids DuckDB multi-process locking and duplicate broker runtimes; dashboard receives no account credentials |
| DTE/theta limits | Snapshot-local calendar DTE; 15-minute expiry-day cap, 45-minute later-expiry cap embedded in SignalResult; missing/expired expiry rejection | Limits analytical exposure to time decay; fresh quotes are still required to value an exit |
| Positioning | Deduplicated same-expiry aggregate PCR >1.5; live Max Pain/PCR exit context | Supersedes old directional strike thresholds; returned-strike proxy only |
| ATR stops/trailing | Fourteen completed index-minute true ranges; initial/trailing distance = NIFTY ATR × absolute Delta; stop never widens | No percentage stop, fixed target, or activation delay; entry waits for valid history |
| Shared tick state | Runtime/stream/engine/monitor use one aggregator; ATR reads retained raw stream history across archival flushes | Ensures monitor mathematics sees streamed ticks rather than a separate snapshot-only buffer |
| Schema migration | Additive DuckDB signals columns max_holding_minutes, days_to_expiry, strike_oi_ratio and parameterized insert support | Preserves existing records and records new holding/OI diagnostics |
| Documentation | Permanent AGENTS.md protocol, decision.md architectural reasoning, live README, and this status ledger | Every completed phase updates all three required documents |
| Isolated live probe | live_fire_test.py discovers current ATM CE/PE, prints exactly 50 combined updates and closes | Reproducible network validation without importing the core engine; real run remains pending |

Requests was already declared. Its range now permits 2.31.0 because the installed
Fyers 3.2.1 SDK pins that exact release; pyotp is newly declared. Existing filter
and monitor tests now provide current paired contracts/OI and warmed option
history instead of relying on expired contracts or missing ATR bypasses.

## Validation evidence

- Full repository suite: **149 tests passed**, plus **5 unittest subtests**,
  including two-page navigation/hierarchy, live alert windows/deduplication,
  position styling, Plotly area behavior, and exact-expiry/market-date checks.
  Command: `PYTHONPATH=src:. python -m pytest tests -q`.
- Changed-code Ruff checks pass; shell entrypoint passes bash -n; Compose YAML
  parses and contains both services and the requested port mapping.
- Non-editable Python wheel builds successfully using pip wheel --no-deps
  --no-build-isolation. This checks packaging, not the full Docker dependency build.
- Auth tests cover TOTP flow, PIN encoding, redirect state mismatch, HTTP failure
  redaction, missing credentials, and .env overwrite/permissions/inode preservation.
- Math tests cover DTE caps, OI boundaries, missing inputs, completed-minute ATR,
  true-range gaps, warm-up admission, 1R trailing/non-widening stops, time stops,
  and migration of an older DuckDB schema.
- Dashboard checks render both waiting-for-engine and published snapshot states
  without credentials or a DuckDB connection.
- Tests used an isolated temporary Python 3.14 environment with installed test
  dependencies. Python 3.12.16 container execution is now verified by the live startup below.
- Existing warnings concern SDK pkg_resources deprecation, research config field
  naming, timestamp inference, and sparse-feature means; they are not test failures.

## Immediate pending task

**Live-fire deployment and real-time AWS EC2/VPS observation.**

With real credentials and an active market session, build/run the Compose stack
and observe headless login, .env injection, current-expiry subscriptions, fifty
real option ticks, bid/ask freshness, OI/DTE rejections, ATR warm-up/stops/trailing,
DuckDB migration/writes, dashboard publication age, and SIGTERM cleanup. Record
timestamps, symbols, counts, and exit statuses without credentials.

The local Docker stack now runs with successful daily authentication and real
Fyers websocket updates; see the live startup record below. EC2/VPS deployment,
longer strategy/lifecycle observation, the standalone fifty-tick probe, and broker
fill validation remain unperformed. Analytical positions are not broker orders.

## Remaining operational work and limitations

- The SSO endpoints are undocumented and may change; confirm they work for the
  account. Tokens are refreshed at process startup only. Restart after the
  provider's daily token expiry; continuous daily renewal remains future work.
- .env rewriting preserves the bind-mounted inode under a lock but is not a
  crash-atomic rename. Startup fails closed on incomplete token verification.
- Time limits and stops are analytical rules evaluated on fresh option updates.
  Missing/stale data freezes valuation; a deadline cannot guarantee an actual
  exit, particularly during disconnection. No broker-side orders are sent.
- ATR needs current-session contiguous history; missing minutes, insufficient
  retained ticks, or zero ATR suppress admission. Stream retention and archive
  behavior still require observation at market traffic rates.
- Current Parquet flush code clears buffers after a missing/failed writer, can
  replace repeated part filenames, and is not invoked for remaining rows at
  runtime shutdown. Durable replay still needs these earlier hardening items.
- Verify lot-size and risk-sizing assumptions, quote/IV availability and bypasses,
  and callback-triggered shutdown under real SDK traffic. Live market data now
  comes exclusively from Fyers REST/websocket; public NSE enrichment is removed.
- Dashboard performance is based on at most the most recent 1,000 outcomes;
  local database mode remains available without claiming live health or positions.
- The existing checkout .git file points to an invalid repository location, so
  Git status/diff is unavailable. Files were inspected and validated directly.


## Professional terminal UI/UX overhaul — completed implementation

- Renamed dashboard/app.py to dashboard/1_Execution_Desk.py and created
  dashboard/pages/2_Analytics_And_Logs.py. Shared components now live in shared.py.
  Updated the CLI launcher, Compose entrypoint, wheel checks, and Docker theme copy.
- Added the requested root dark-mode theme. Streamlit minimum is 1.51 for bold
  table styling; Plotly was already a declared dependency.
- Execution Desk contains only websocket/spot/VIX health, fresh signal alerts, and
  active positions. No historical chart, performance banner, or signal log appears
  on this page. PnL is green/red and trailing stops are bold.
- Added an independent DuckDB last-60-second passed BUY_CE/BUY_PE query and nullable
  option entry_price persistence. New signal notifications use st.success and
  deduplicated st.toast. Old records do not invent an entry premium.
- Expanded the engine publication with actual stream health, spot/VIX, live alerts,
  and full position/PnL fields, preserving sole-writer DuckDB ownership. Stale
  publications pause alerts and mark cached positions explicitly.
- Analytics & Logs leads with the four performance metrics, then a Plotly equity
  area chart, then the 50-setup audit and Rejection Reason expander.
- Verified both pages and navigation in Streamlit tests and a real browser. Saved
  docs/ui previews use isolated synthetic fixtures, including DEMO positions;
  their displayed connection state is not live-account evidence.
- Validation: **107 tests and 5 subtests passed**; changed-code Ruff checks pass;
  wheel build passes and includes both page scripts. Theme and Compose configuration
  remain subject to the requested actual Docker startup attempt below.

Live-market impact: the desk emphasizes current analytical decisions and positions,
while retrospective charts and rejections are isolated. Alerts remain informational;
no order placement, changes to filter thresholds, or broker-fill guarantees are
introduced by this UI phase. Previously documented live deployment and durability
hardening tasks remain pending.


## Requested deployment attempt — blocked by host tooling

After implementation and documentation updates, ran the exact requested command:

```bash
docker compose up --build
```

Result: **exit 127 — docker: command not found**. The existing
/usr/local/bin/docker symlink points to
/Applications/Docker.app/Contents/Resources/bin/docker, which is absent.
No Docker container was built or started. **The terminal is not live.** Restore
or install Docker Desktop and rerun the command with configured credentials.

Theme values and Compose entrypoint were verified independently. The temporary
localhost UI review server used only isolated synthetic data and has been stopped.
The saved previews demonstrate layout, not market connectivity. Next pending task
remains **Live-fire deployment and real-time AWS EC2/VPS observation.**

## Root Docker files — verified

Confirmed docker-compose.yml, Dockerfile, and entrypoint.sh are regular files
immediately inside the current repository root, not a subdirectory. The existing
files already satisfied the requested architecture. Reapplied chmod +x to
entrypoint.sh and verified executable access and Bash syntax. Parsed Compose YAML
and checked engine/dashboard services, engine .env loading, dashboard port 8501,
and data/log mounts for both services. Verified Python 3.12 slim, required source
copies, non-editable dependency installation, shell entrypoint, and authentication
before daemon startup. Updated all three protocol documents. No live deployment
or market connection was performed during this verification; Docker tooling and
live-fire EC2/VPS observation remain pending as previously recorded.


## Canonical laptop checkout synchronization and publication

The user-confirmed local root is /Volumes/APFS/projects/trading_engine. Prepared
a combined source tree preserving the seven existing local edits, including
authentication import paths, market-closed handling, and read-only storage access.
The old dashboard's read-copy refresh behavior is retained in the shared reader;
its in-process runtime path is superseded by the two-page publication design.
The complete original local changes will also have a local backup Git reference.
The root's three previously unpublished commits are descendants of GitHub main;
publication therefore uses a normal fast-forward push, without rewriting history.
Credentials and runtime data/logs are excluded. Final validation and push result
are recorded after the synchronization checks.

Combined source verification: **108 tests and 5 subtests passed** in the prepared
merged tree, with credentials/runtime data excluded. Added deterministic intraday
ATR test clocks and a regression test for historical read-copy refresh. Production
session/ATR logic is unchanged. Source/configuration/tests/documentation are
maintained in the canonical laptop checkout and published to GitHub main; live
account validation and live-fire EC2/VPS observation remain pending.


## Local environment and headless auth verification — 2026-11-12

Updated the canonical root .env with the user-supplied TOTP secret and placeholder
FYERS_USER_ID/FYERS_PIN values, preserving other configuration and 0600 permissions.
Credentials were not copied to the chat workspace or recorded in documentation.
Verified .env is explicitly listed in .gitignore, ignored by Git, and untracked.
Auth source reads FYERS_TOTP_SECRET and uses pyotp.TOTP(totp_secret).now() directly
for the SSO OTP payload. Local generation produces six numeric digits; neither
the code nor the secret was displayed or transmitted. Five offline auth tests pass.

Executed docker compose config in /Volumes/APFS/projects/trading_engine: exit 0;
all three required environment values interpolate correctly. Rendered output was
captured and suppressed. This supersedes the earlier missing-Docker tooling
blocker for configuration validation. No container startup, SSO login, or live
websocket connection was attempted. Replace the placeholder user ID and PIN before
account authentication. Live-fire deployment and real-time AWS EC2/VPS observation
remain pending. Source and all three protocol documents are synchronized locally.


## Account placeholders replaced — 2026-11-12

The user supplied account ID/PIN; both placeholders were replaced only in the
canonical local .env. Docker Compose configuration/interpolation passes again.
Verified .env remains ignored, untracked, and mode 0600. This supersedes the
earlier pending placeholder replacement. No account values appear in tracked
files, and no SSO request, container startup, or live-feed test was performed.
Live authentication and market validation remain pending.


## Unified Fyers data pipeline — completed 2026-11-12

Removed every live runtime scraper import/instance and the NSE HTTP enrichment
block. refresh_once now uses the existing FyersMarketClient snapshot and routes
its normalized option_instruments DataFrame directly into MarketStateAggregator.
Deleted ingestion/nse_fetcher.py and its unused legacy scheduler to leave no
dangling imports. Removed the public NSE endpoint/bootstrap/header configuration.
Historical parsing remains offline; generic API ingestion blocks public NSE hosts
and redirects before making requests. Added a permanent Fyers-only source policy
to AGENTS.md and updated README deployment/source descriptions.

Validation: **114 tests and 5 subtests passed**; changed-code Ruff checks pass.
Runtime regression verifies exactly one Fyers snapshot fetch, unchanged chain
routing, no auxiliary HTTP request, and absence of the scraper/parser/validator
instances. Additional checks verify deleted modules stay unavailable and generic
ingestion cannot request or redirect to the public NSE website.

Live-market impact: the public-website 404/bootstrap/retry path is removed from
analytical refreshes. All live spot, VIX, option premiums/OI, and ticks now originate
in Fyers. Data quality, availability, and freshness still require broker/live-feed
observation. Source/config/tests/docs are synchronized to the canonical laptop
root; running containers must be rebuilt to receive the fix. No live SSO call,
container restart, or market validation was performed for this source change.
Next pending task remains Live-fire deployment and real-time AWS EC2/VPS observation.


## Local project started with real Fyers data — 2026-11-12 10:01 IST

Executed docker desktop start, then docker compose up --build -d engine dashboard
from /Volumes/APFS/projects/trading_engine. Both images built and both containers
started successfully. Dashboard container is healthy, engine remains running, and
http://localhost:8501/_stcore/health returns HTTP 200 with body ok. Opened the real
Execution Desk in the browser and confirmed its connected websocket status.

Observed daily token generation success, websocket_connected true, safe_mode false,
publication age 0.34 seconds and last-tick age 0.37 seconds. Live NIFTY/VIX values
were present; the inspected signal was no_trade and there were no active analytical
positions. Engine Python is 3.12.16. Confirmed the deleted NSE scraper is absent
from the running image. Inspected startup logs contained no fatal engine error or
legacy scraper error. All log inspection suppressed local credential values.

This supersedes earlier missing-Docker/unperformed-startup entries for the local
machine. Services are intentionally left running. No code architecture changed
and no additional unit tests were needed for this operational phase; the previous
114-test result remains the offline baseline. Long-duration live observation, the
standalone fifty-tick probe, ATR/filter/exit behavior, persistence durability, and
AWS EC2/VPS deployment remain pending. No broker orders or fills are claimed.


## Fault-tolerant websocket parser — implementation verified 2026-11-12

Rewrote _on_message with a broad callback exception boundary. Invalid messages
WARN and return without calling stop/close or escaping into the SDK. Validate
all rows before publishing to state; isolate raw/tick consumer exceptions. Accept
native complete option fields and aliases including zero volume; preserve valid
index messages without option-only fields. Heartbeat/control packets never become
market ticks. Added dropped-packet count and last-error diagnostics on stream.state.

Removed synthetic bid/ask fallback, fixed numeric-string exchange epochs and
zero volume, and stopped treating old last-trade time as current quote time.
Fyers stream validation now rejects individual stale/future/duplicate/out-of-order
packets without latching packet-induced global safe mode. Independent safe mode
is retained and data freshness/frozen-position rules are unchanged.

Validation: **149 tests and 5 subtests passed**, including 35 new parser/recovery
cases, inside a network-disabled Python 3.12 container with no credentials/live
data mounted. Changed-file lint passes with pre-existing unrelated SIM103 excluded.
Verified that every tested failed packet warns, produces no invalid state/callback
update, leaves the transport open, and permits the next valid packet. Deployment
and short live observation are recorded below; offline tests are not a guarantee
of long-duration market stability.


## Parser fix deployed and observed — 2026-11-12 10:27 IST

Rebuilt and restarted the engine service with the tested parser; dashboard remained
healthy. Confirmed the actual running image contains the broad Exception handler,
no stop/close call in _on_message, and the packet_only validation API. During the
short live check: one packet WARNING, zero Stream integrity validation failed
messages, zero Fyers DataSocket closed events, and no fatal engine error.
Websocket_connected was true, safe_mode false, publication age 1.95 seconds and
published last-tick age 1.99 seconds. Dashboard HTTP health returned 200.

Services are left running. All required documents and source/tests are synchronized
to the canonical local checkout. The 149-test/5-subtest result is the isolated
regression evidence; this short live window additionally verifies active packet
processing without observed parser-driven disconnect. Longer traffic/strategy/
lifecycle observation and AWS EC2/VPS validation remain pending. No credentials
or raw live packets were added to tracked files.

## Git publication phase — 2026-11-12

Prepared all pending source/configuration/test/documentation changes for a single
canonical-checkout commit: unified Fyers data, scraper removal, resilient stream
parsing, Greeks/lifecycle alerts, and dashboard updates. Added explicit ignore
rules for runtime dashboard publication and local environment variants, retaining
`.env.example`. Existing local changes are included; credentials and runtime
artifacts are excluded. Architecture is unchanged by publication.

Validation: canonical/workspace files match, 167 offline tests plus 5 subtests
passed for the combined implementation, changed-file lint passed, and Git's
whitespace check is clean. Source publication does not deploy the running Docker
services. Live-fire lifecycle deployment and real-time AWS EC2/VPS observation
remain pending; no broker orders or fills are implied.
