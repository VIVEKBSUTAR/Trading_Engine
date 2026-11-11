# Project status

Last updated: 2026-11-12 (Asia/Kolkata).

## Completed implementation

| Area | Exactly what was built | Impact on live market analysis/execution |
| --- | --- | --- |
| Fyers integration | Active auth/REST/option-chain/websocket adapters, bid/ask normalization, symbol/token routing; daemon now starts symbol/discovered subscriptions; SDK shutdown uses close_connection | Supplies analytical market data; actual account/feed behavior still needs live observation |
| Premium valuation | Option ask entry, bid marking/exit, token-specific lifecycle and analytical P&L | Removes index spot as the P&L proxy; recorded outcomes do not establish broker fills |
| Spread/IV filters | Default 5% spread cap and rolling 90th percentile IV rejection | Screens liquidity and volatility when the required inputs exist; prior missing-data/IV warm-up bypasses remain |
| DuckDB/Parquet | Decision/rejection/outcome tables and per-token archival flush above 5,000 rows | Provides analytical audit/replay data; pre-existing durability gaps remain below |
| Headless authentication | requests SSO login, current pyotp TOTP, encoded PIN, redirect URI/state validation, SHA-256 auth-code exchange, fail-closed redacted errors | Removes browser steps from container startup; provider SSO is undocumented and not live-validated |
| Credential/config support | FYERS_USER_ID, FYERS_PIN, FYERS_TOTP_SECRET fields, .env example, and strict validation for headless mode | Headless startup refuses missing credentials; research and token-only probes stay usable |
| Token injection | Locked in-place .env update, duplicate token removal, restrictive permissions, fsync, generation timestamp, persisted-value verification | Preserves the file bind mount and unrelated settings; avoids logging tokens; successful startup explicitly loads the new token |
| Container architecture | Python 3.12 slim Dockerfile with non-editable installation; engine and dashboard Compose services; port 8501, persistent data/log volumes, .dockerignore, dashboard healthcheck | Makes deployment repeatable; engine is sole analytical writer; no image has been built/run here |
| Entrypoint | Authenticates, verifies a fresh generated token, then execs te-intelligence | Authentication failure prevents stale-token startup; no login restart storm or daily scheduler is introduced |
| Independent dashboard | Engine atomically publishes latest.json; dashboard reads it with timed fragments, publication age, signals, positions, and recent outcomes | Avoids DuckDB multi-process locking and duplicate broker runtimes; dashboard receives no account credentials |
| DTE/theta limits | Snapshot-local calendar DTE; 15-minute expiry-day cap, 45-minute later-expiry cap embedded in SignalResult; missing/expired expiry rejection | Limits analytical exposure to time decay; fresh quotes are still required to value an exit |
| OI imbalance | CE/PE open interest for the exact selected strike/expiry; PE/CE ratio; BUY_CE rejection below 0.8, BUY_PE rejection above 1.2; missing/invalid OI rejection | Filters opposing strike exposure without combining contracts from different expiries |
| ATR stops/trailing | Fourteen completed contiguous one-minute midpoint bars; mean true range with gaps; stop at ask minus 1.5 ATR; trailing begins at 1R and follows bid minus 1 ATR without widening; premium target at ask plus 3 ATR | Replaces percentage/index-scaled monitor stops with premium volatility; new positions wait for valid history |
| Shared tick state | Runtime/stream/engine/monitor use one aggregator; ATR reads retained raw stream history across archival flushes | Ensures monitor mathematics sees streamed ticks rather than a separate snapshot-only buffer |
| Schema migration | Additive DuckDB signals columns max_holding_minutes, days_to_expiry, strike_oi_ratio and parameterized insert support | Preserves existing records and records new holding/OI diagnostics |
| Documentation | Permanent AGENTS.md protocol, decision.md architectural reasoning, live README, and this status ledger | Every completed phase updates all three required documents |
| Isolated live probe | live_fire_test.py discovers current ATM CE/PE, prints exactly 50 combined updates and closes | Reproducible network validation without importing the core engine; real run remains pending |

Requests was already declared. Its range now permits 2.31.0 because the installed
Fyers 3.2.1 SDK pins that exact release; pyotp is newly declared. Existing filter
and monitor tests now provide current paired contracts/OI and warmed option
history instead of relying on expired contracts or missing ATR bypasses.

## Validation evidence

- Full repository suite: **107 tests passed**, plus **5 unittest subtests**,
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
  dependencies. Python 3.12 container execution has not been verified here.
- Existing warnings concern SDK pkg_resources deprecation, research config field
  naming, timestamp inference, and sparse-feature means; they are not test failures.

## Immediate pending task

**Live-fire deployment and real-time AWS EC2/VPS observation.**

With real credentials and an active market session, build/run the Compose stack
and observe headless login, .env injection, current-expiry subscriptions, fifty
real option ticks, bid/ask freshness, OI/DTE rejections, ATR warm-up/stops/trailing,
DuckDB migration/writes, dashboard publication age, and SIGTERM cleanup. Record
timestamps, symbols, counts, and exit statuses without credentials.

Docker is unavailable on this host. No real Fyers SSO request, token exchange,
market websocket run, EC2/VPS deployment, or broker order was performed. Offline
completion does not establish live readiness.

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
  supplementary NSE chain behavior, and callback-triggered shutdown under real
  SDK traffic. The runtime still combines REST/NSE enrichment with websocket data.
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
