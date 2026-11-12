"""Centralized configuration for the Fyers analytical signal runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class APISettings:
    """Shared HTTP timeout/retry defaults; no public-exchange scraper endpoints."""

    timeout_seconds: int = 15
    max_retries: int = 4
    backoff_seconds: float = 1.5


@dataclass(slots=True)
class StorageSettings:
    """Filesystem and DuckDB paths for raw, processed, and metadata datasets."""

    project_root: Path
    raw_dir: Path
    processed_dir: Path
    metadata_dir: Path
    duckdb_path: Path
    processed_dataset_name: str = "nse_option_chain"


@dataclass(slots=True)
class SchedulerSettings:
    """Collection loop behavior and runtime controls."""

    interval_seconds: int = 60
    symbol: str = "NIFTY"
    timezone: str = "Asia/Kolkata"


@dataclass(slots=True)
class LoggingSettings:
    """Log sink settings for loguru output."""

    log_dir: Path
    file_name: str = "ingestion.log"
    broker_file_name: str = "broker.log"
    level: str = "INFO"
    rotation: str = "20 MB"
    retention: str = "14 days"


@dataclass(slots=True)
class FyersSettings:
    """Fyers API V3 authentication and streaming configuration."""

    client_id: str | None = None
    secret_key: str | None = None
    redirect_uri: str = "https://127.0.0.1"
    access_token: str | None = None
    auth_code: str | None = None
    user_id: str | None = field(default=None, repr=False)
    pin: str | None = field(default=None, repr=False)
    totp_secret: str | None = field(default=None, repr=False)
    token_store_path: Path | None = None
    reconnect: bool = True
    reconnect_max_tries: int = 50
    reconnect_max_delay: int = 60
    connect_timeout: int = 30
    stream_tokens: tuple[int, ...] = ()
    stream_symbols: tuple[str, ...] = ()
    stream_mode: str = "full"
    litemode: bool = False

    def validate_headless_auth(self) -> None:
        """Headless deployment requires account credentials; research does not."""
        required = {
            "FYERS_CLIENT_ID": self.client_id,
            "FYERS_SECRET_KEY": self.secret_key,
            "FYERS_REDIRECT_URI": self.redirect_uri,
            "FYERS_USER_ID": self.user_id,
            "FYERS_PIN": self.pin,
            "FYERS_TOTP_SECRET": self.totp_secret,
        }
        missing = [key for key, value in required.items() if not value or not value.strip()]
        if missing:
            raise ValueError("Headless authentication requires: " + ", ".join(missing))

    @property
    def api_key(self) -> str | None:
        return self.client_id

    @property
    def api_secret(self) -> str | None:
        return self.secret_key

    @property
    def request_token(self) -> str | None:
        return self.auth_code


# Backward compatibility alias
KiteSettings = FyersSettings


@dataclass(slots=True)
class SecuritySettings:
    """API integrity, rate-limit, and environment-separation controls."""

    environment: str = "development"
    allow_live_execution: bool = False
    require_manual_execution_approval: bool = True
    api_requests_per_minute: int = 120
    request_timeout_seconds: int = 15
    stale_snapshot_seconds: int = 20
    stale_option_chain_seconds: int = 120
    stale_stream_seconds: int = 20
    stream_heartbeat_timeout_seconds: int = 15
    reconnect_cooldown_seconds: int = 10
    max_reconnect_attempts: int = 5
    max_duplicate_ticks: int = 3
    max_out_of_order_ticks: int = 3
    max_payload_age_seconds: int = 30


@dataclass(slots=True)
class IntelligenceSettings:
    """Thresholds and runtime controls for the live market intelligence engine."""

    state_buffer_rows: int = 4000
    bullish_probability_threshold: float = 0.65
    bearish_probability_threshold: float = 0.65
    trap_probability_threshold: float = 0.60
    confidence_floor: float = 0.60
    trend_strength_threshold: float = 0.0025
    compression_threshold: float = 0.60
    volatility_high_threshold: float = 18.0
    trade_quality_threshold: float = 0.55
    breakout_lookback_bars: int = 20
    structure_lookback_bars: int = 50
    max_trades_per_day: int = 3
    max_daily_loss_pct: float = 0.02
    dashboard_refresh_seconds: int = 5
    default_expiry_days: int = 7
    partial_exit_fraction: float = 0.50
    time_stop_minutes: int = 12
    signal_ttl_min_candles: int = 2
    signal_ttl_max_candles: int = 4
    max_spread_percentage: float = 0.05
    max_implied_volatility_percentile: float = 0.50
    pcr_drop_fraction: float = 0.20


@dataclass(slots=True)
class RiskSettings:
    """Capital and exposure controls for signal sizing."""

    capital: float = 10_000_000.0
    risk_per_trade: float = 0.005
    max_capital_exposure: float = 0.10
    contract_multiplier: float = 1.0
    min_units: int = 1


@dataclass(slots=True)
class AppSettings:
    """Aggregate runtime settings used by pipeline components."""

    api: APISettings
    storage: StorageSettings
    scheduler: SchedulerSettings
    logging: LoggingSettings
    fyers: FyersSettings
    security: SecuritySettings
    intelligence: IntelligenceSettings
    risk: RiskSettings

    @property
    def kite(self) -> FyersSettings:
        """Backward compatibility alias for fyers settings."""
        return self.fyers

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> AppSettings:
        """Build settings from environment variables with safe defaults."""
        load_dotenv()
        root = project_root or Path(__file__).resolve().parents[1]

        raw_dir = Path(os.getenv("TE_RAW_DIR", str(root / "data" / "raw"))).resolve()
        processed_dir = Path(os.getenv("TE_PROCESSED_DIR", str(root / "data" / "processed"))).resolve()
        metadata_dir = Path(os.getenv("TE_METADATA_DIR", str(root / "data" / "metadata"))).resolve()
        duckdb_path = Path(os.getenv("TE_DUCKDB_PATH", str(metadata_dir / "nse_ingestion.duckdb"))).resolve()
        log_dir = Path(os.getenv("TE_LOG_DIR", str(root / "logs"))).resolve()

        interval_seconds = int(os.getenv("TE_FETCH_INTERVAL_SECONDS", "60"))
        timeout_seconds = int(os.getenv("TE_API_TIMEOUT_SECONDS", "15"))
        max_retries = int(os.getenv("TE_API_MAX_RETRIES", "4"))

        token_store_path = Path(
            os.getenv(
                "FYERS_TOKEN_STORE_PATH",
                os.getenv("KITE_TOKEN_STORE_PATH", str(metadata_dir / "fyers_access_token.txt")),
            )
        ).resolve()

        stream_tokens_raw = (os.getenv("FYERS_STREAM_TOKENS") or os.getenv("KITE_STREAM_TOKENS") or "").strip()
        stream_tokens = tuple(
            int(token.strip())
            for token in stream_tokens_raw.split(",")
            if token.strip() and token.strip().isdigit()
        )

        stream_symbols_raw = os.getenv("FYERS_STREAM_SYMBOLS", "").strip()
        stream_symbols = tuple(
            sym.strip()
            for sym in stream_symbols_raw.split(",")
            if sym.strip()
        )

        environment = os.getenv("TE_ENVIRONMENT", "development").strip().lower()
        allow_live_execution = os.getenv("TE_ALLOW_LIVE_EXECUTION", "false").lower() in {"1", "true", "yes"}
        require_manual_execution_approval = os.getenv("TE_REQUIRE_MANUAL_EXECUTION_APPROVAL", "true").lower() in {"1", "true", "yes"}

        fyers_client_id = os.getenv("FYERS_CLIENT_ID") or os.getenv("KITE_API_KEY") or None
        fyers_secret_key = os.getenv("FYERS_SECRET_KEY") or os.getenv("KITE_API_SECRET") or None
        fyers_redirect_uri = os.getenv("FYERS_REDIRECT_URI", "https://127.0.0.1").strip()
        fyers_access_token = os.getenv("FYERS_ACCESS_TOKEN") or os.getenv("KITE_ACCESS_TOKEN") or None
        fyers_auth_code = os.getenv("FYERS_AUTH_CODE") or os.getenv("KITE_REQUEST_TOKEN") or None

        settings = cls(
            api=APISettings(timeout_seconds=timeout_seconds, max_retries=max_retries),
            storage=StorageSettings(
                project_root=root.resolve(),
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                metadata_dir=metadata_dir,
                duckdb_path=duckdb_path,
            ),
            scheduler=SchedulerSettings(
                interval_seconds=interval_seconds,
                symbol=os.getenv("TE_SYMBOL", "NIFTY"),
                timezone=os.getenv("TE_TIMEZONE", "Asia/Kolkata"),
            ),
            logging=LoggingSettings(
                log_dir=log_dir,
                level=os.getenv("TE_LOG_LEVEL", "INFO"),
            ),
            fyers=FyersSettings(
                client_id=fyers_client_id,
                secret_key=fyers_secret_key,
                redirect_uri=fyers_redirect_uri,
                access_token=fyers_access_token,
                auth_code=fyers_auth_code,
                user_id=os.getenv("FYERS_USER_ID"),
                pin=os.getenv("FYERS_PIN"),
                totp_secret=os.getenv("FYERS_TOTP_SECRET"),
                token_store_path=token_store_path,
                reconnect=os.getenv("FYERS_RECONNECT", os.getenv("KITE_RECONNECT", "true")).lower() in {"1", "true", "yes"},
                reconnect_max_tries=int(os.getenv("FYERS_RECONNECT_MAX_TRIES", os.getenv("KITE_RECONNECT_MAX_TRIES", "50"))),
                reconnect_max_delay=int(os.getenv("FYERS_RECONNECT_MAX_DELAY", os.getenv("KITE_RECONNECT_MAX_DELAY", "60"))),
                connect_timeout=int(os.getenv("FYERS_CONNECT_TIMEOUT", os.getenv("KITE_CONNECT_TIMEOUT", "30"))),
                stream_tokens=stream_tokens,
                stream_symbols=stream_symbols,
                stream_mode=os.getenv("FYERS_STREAM_MODE", os.getenv("KITE_STREAM_MODE", "full")).lower(),
                litemode=os.getenv("FYERS_LITEMODE", "false").lower() in {"1", "true", "yes"},
            ),
            security=SecuritySettings(
                environment=environment,
                allow_live_execution=allow_live_execution,
                require_manual_execution_approval=require_manual_execution_approval,
                api_requests_per_minute=int(os.getenv("TE_API_REQUESTS_PER_MINUTE", "120")),
                request_timeout_seconds=int(os.getenv("TE_SECURITY_REQUEST_TIMEOUT_SECONDS", str(timeout_seconds))),
                stale_snapshot_seconds=int(os.getenv("TE_STALE_SNAPSHOT_SECONDS", "20")),
                stale_option_chain_seconds=int(os.getenv("TE_STALE_OPTION_CHAIN_SECONDS", "120")),
                stale_stream_seconds=int(os.getenv("TE_STALE_STREAM_SECONDS", "20")),
                stream_heartbeat_timeout_seconds=int(os.getenv("TE_STREAM_HEARTBEAT_TIMEOUT_SECONDS", "15")),
                reconnect_cooldown_seconds=int(os.getenv("TE_RECONNECT_COOLDOWN_SECONDS", "10")),
                max_reconnect_attempts=int(os.getenv("TE_MAX_RECONNECT_ATTEMPTS", "5")),
                max_duplicate_ticks=int(os.getenv("TE_MAX_DUPLICATE_TICKS", "3")),
                max_out_of_order_ticks=int(os.getenv("TE_MAX_OUT_OF_ORDER_TICKS", "3")),
                max_payload_age_seconds=int(os.getenv("TE_MAX_PAYLOAD_AGE_SECONDS", "30")),
            ),
            intelligence=IntelligenceSettings(
                state_buffer_rows=int(os.getenv("TE_STATE_BUFFER_ROWS", "4000")),
                bullish_probability_threshold=float(os.getenv("TE_BULLISH_PROB_THRESHOLD", "0.65")),
                bearish_probability_threshold=float(os.getenv("TE_BEARISH_PROB_THRESHOLD", "0.65")),
                trap_probability_threshold=float(os.getenv("TE_TRAP_PROB_THRESHOLD", "0.60")),
                confidence_floor=float(os.getenv("TE_CONFIDENCE_FLOOR", "0.60")),
                trend_strength_threshold=float(os.getenv("TE_TREND_STRENGTH_THRESHOLD", "0.0025")),
                compression_threshold=float(os.getenv("TE_COMPRESSION_THRESHOLD", "0.60")),
                volatility_high_threshold=float(os.getenv("TE_VOLATILITY_HIGH_THRESHOLD", "18.0")),
                trade_quality_threshold=float(os.getenv("TE_TRADE_QUALITY_THRESHOLD", "0.55")),
                breakout_lookback_bars=int(os.getenv("TE_BREAKOUT_LOOKBACK_BARS", "20")),
                structure_lookback_bars=int(os.getenv("TE_STRUCTURE_LOOKBACK_BARS", "50")),
                max_trades_per_day=int(os.getenv("TE_MAX_TRADES_PER_DAY", "3")),
                max_daily_loss_pct=float(os.getenv("TE_MAX_DAILY_LOSS_PCT", "0.02")),
                dashboard_refresh_seconds=int(os.getenv("TE_DASHBOARD_REFRESH_SECONDS", "5")),
                default_expiry_days=int(os.getenv("TE_DEFAULT_EXPIRY_DAYS", "7")),
                partial_exit_fraction=float(os.getenv("TE_PARTIAL_EXIT_FRACTION", "0.50")),
                time_stop_minutes=int(os.getenv("TE_TIME_STOP_MINUTES", "12")),
                signal_ttl_min_candles=int(os.getenv("TE_SIGNAL_TTL_MIN_CANDLES", "2")),
                signal_ttl_max_candles=int(os.getenv("TE_SIGNAL_TTL_MAX_CANDLES", "4")),
                max_spread_percentage=float(os.getenv("TE_MAX_SPREAD_PERCENTAGE", "0.05")),
                pcr_drop_fraction=float(os.getenv("TE_PCR_DROP_FRACTION", "0.20")),
                max_implied_volatility_percentile=float(os.getenv("TE_MAX_IMPLIED_VOLATILITY_PERCENTILE", "0.50")),
            ),
            risk=RiskSettings(
                capital=float(os.getenv("TE_CAPITAL", "10000000")),
                risk_per_trade=float(os.getenv("TE_RISK_PER_TRADE", "0.005")),
                max_capital_exposure=float(os.getenv("TE_MAX_CAPITAL_EXPOSURE", "0.10")),
                contract_multiplier=float(os.getenv("TE_CONTRACT_MULTIPLIER", "1.0")),
                min_units=int(os.getenv("TE_MIN_UNITS", "1")),
            ),
        )

        settings.ensure_directories()
        return settings

    def ensure_directories(self) -> None:
        """Create all required runtime directories."""
        self.storage.raw_dir.mkdir(parents=True, exist_ok=True)
        self.storage.processed_dir.mkdir(parents=True, exist_ok=True)
        self.storage.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.storage.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.logging.log_dir.mkdir(parents=True, exist_ok=True)
        if self.fyers.token_store_path is not None:
            self.fyers.token_store_path.parent.mkdir(parents=True, exist_ok=True)
