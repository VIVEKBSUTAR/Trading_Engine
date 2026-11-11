"""Unit and integration tests for Fyers API V3 broker integration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from broker.fyers_auth import FyersAuthError, FyersAuthManager
from broker.fyers_client import FyersMarketClient, LiveMarketSnapshot, MarketQuote
from broker.fyers_stream import FyersStream
from config.settings import AppSettings, FyersSettings, KiteSettings
from trading_engine.intelligence.runtime import LiveIntelligenceRuntime
from trading_engine.intelligence.state import MarketStateAggregator


class TestStep1FyersSettings:
    """Step 1: Test settings.py in config directory with FyersSettings."""

    def test_fyers_settings_fields_and_defaults(self, tmp_path: Path) -> None:
        token_path = tmp_path / "token.txt"
        settings = FyersSettings(
            client_id="TEST_CLIENT_ID",
            secret_key="TEST_SECRET_KEY",
            redirect_uri="https://127.0.0.1:8000/callback",
            access_token="TEST_ACCESS_TOKEN",
            token_store_path=token_path,
        )

        assert settings.client_id == "TEST_CLIENT_ID"
        assert settings.secret_key == "TEST_SECRET_KEY"
        assert settings.redirect_uri == "https://127.0.0.1:8000/callback"
        assert settings.access_token == "TEST_ACCESS_TOKEN"
        # Backward compatibility properties
        assert settings.api_key == "TEST_CLIENT_ID"
        assert settings.api_secret == "TEST_SECRET_KEY"

    def test_app_settings_from_env_loads_fyers(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("FYERS_CLIENT_ID", "FYERS_APP_123")
        monkeypatch.setenv("FYERS_SECRET_KEY", "FYERS_SEC_XYZ")
        monkeypatch.setenv("FYERS_REDIRECT_URI", "https://localhost:5000")
        monkeypatch.setenv("FYERS_ACCESS_TOKEN", "DAILY_ACCESS_TOKEN")

        app_settings = AppSettings.from_env(project_root=tmp_path)
        assert isinstance(app_settings.fyers, FyersSettings)
        assert app_settings.fyers.client_id == "FYERS_APP_123"
        assert app_settings.fyers.secret_key == "FYERS_SEC_XYZ"
        assert app_settings.fyers.redirect_uri == "https://localhost:5000"
        assert app_settings.fyers.access_token == "DAILY_ACCESS_TOKEN"

        # Check backward compatibility alias
        assert app_settings.kite.client_id == "FYERS_APP_123"
        assert issubclass(KiteSettings, FyersSettings) or KiteSettings is FyersSettings


class TestStep2FyersAuth:
    """Step 2: Test broker/fyers_auth.py script and manager."""

    def test_generate_login_url(self) -> None:
        settings = FyersSettings(
            client_id="MY_APP_ID-100",
            secret_key="MY_SECRET",
            redirect_uri="https://trade.myapp.com/callback",
        )
        auth = FyersAuthManager(settings)
        login_url = auth.generate_login_url()

        assert "https://api-t1.fyers.in/api/v3/generate-authcode" in login_url
        assert "client_id=MY_APP_ID-100" in login_url
        assert "redirect_uri=https%3A%2F%2Ftrade.myapp.com%2Fcallback" in login_url

    def test_exchange_auth_code_success_and_persists(self, tmp_path: Path) -> None:
        token_file = tmp_path / "fyers_token.txt"
        settings = FyersSettings(
            client_id="APP_ID-100",
            secret_key="SECRET_KEY",
            redirect_uri="https://localhost",
            token_store_path=token_file,
        )
        auth = FyersAuthManager(settings)

        # Mock SessionModel.generate_token
        mock_response = {
            "s": "ok",
            "code": 200,
            "message": "Token generated successfully",
            "access_token": "EXCHANGED_DAILY_ACCESS_TOKEN_XYZ",
        }
        with patch.object(auth._get_session_model(), "generate_token", return_value=mock_response):
            token = auth.exchange_auth_code("BROWSER_AUTH_CODE_123")

        assert token == "EXCHANGED_DAILY_ACCESS_TOKEN_XYZ"
        assert settings.access_token == "EXCHANGED_DAILY_ACCESS_TOKEN_XYZ"
        assert token_file.exists()
        assert token_file.read_text().strip() == "EXCHANGED_DAILY_ACCESS_TOKEN_XYZ"

    def test_exchange_auth_code_handles_failure(self) -> None:
        settings = FyersSettings(
            client_id="APP_ID-100",
            secret_key="SECRET_KEY",
            redirect_uri="https://localhost",
        )
        auth = FyersAuthManager(settings)

        mock_err_response = {
            "s": "error",
            "code": -300,
            "message": "Invalid authorization code",
        }
        with (
            patch.object(auth._get_session_model(), "generate_token", return_value=mock_err_response),
            pytest.raises(FyersAuthError, match="Fyers token exchange failed"),
        ):
            auth.exchange_auth_code("INVALID_CODE")


class TestStep3FyersClient:
    """Step 3: Test broker/fyers_client.py REST client and option chain normalization."""

    def test_normalize_option_chain_exact_schema(self) -> None:
        settings = FyersSettings(client_id="APP-100", access_token="TOKEN_123")
        client = FyersMarketClient(settings)

        sample_fyers_json = {
            "s": "ok",
            "code": 200,
            "data": {
                "optionsChain": [
                    {
                        "strike_price": 24500.0,
                        "symbol": "NSE:NIFTY24OCT24500CE",
                        "option_type": "CE",
                        "ltp": 165.50,
                        "bid": 165.20,
                        "ask": 165.80,
                        "expiry": "2024-10-24",
                        "fyToken": "1011124102424500",
                        "oi": 150000,
                    },
                    {
                        "strike_price": 24500.0,
                        "symbol": "NSE:NIFTY24OCT24500PE",
                        "option_type": "PE",
                        "ltp": 98.75,
                        "bid": 98.50,
                        "ask": 99.00,
                        "expiry": "2024-10-24",
                        "fyToken": "1011124102424501",
                        "oi": 210000,
                    },
                ]
            },
        }

        df = client.normalize_option_chain(sample_fyers_json)

        # Verify exact required columns exist
        expected_cols = ["strike", "expiry", "option_type", "last_price", "bid_price", "ask_price"]
        for col in expected_cols:
            assert col in df.columns, f"Missing required column: {col}"

        assert len(df) == 2
        ce_row = df.loc[df["option_type"] == "CE"].iloc[0]
        assert ce_row["strike"] == 24500.0
        assert ce_row["expiry"] == "2024-10-24"
        assert ce_row["option_type"] == "CE"
        assert ce_row["last_price"] == 165.50
        assert ce_row["bid_price"] == 165.20
        assert ce_row["ask_price"] == 165.80

        pe_row = df.loc[df["option_type"] == "PE"].iloc[0]
        assert pe_row["strike"] == 24500.0
        assert pe_row["expiry"] == "2024-10-24"
        assert pe_row["option_type"] == "PE"
        assert pe_row["last_price"] == 98.75
        assert pe_row["bid_price"] == 98.50
        assert pe_row["ask_price"] == 99.00

    def test_get_live_snapshot(self) -> None:
        settings = FyersSettings(client_id="APP-100", access_token="TOKEN_123")
        client = FyersMarketClient(settings)

        mock_nifty_quote = MarketQuote(
            instrument="NSE:NIFTY50-INDEX",
            last_price=24500.0,
            change=50.0,
            ohlc={"open": 24450.0, "high": 24550.0, "low": 24400.0, "close": 24450.0},
            received_at=datetime.now(UTC),
            exchange_timestamp=datetime.now(UTC),
            last_trade_time=datetime.now(UTC),
            raw={},
        )
        mock_vix_quote = MarketQuote(
            instrument="NSE:INDIAVIX-INDEX",
            last_price=13.5,
            change=-0.2,
            ohlc={"open": 13.7, "high": 13.8, "low": 13.4, "close": 13.7},
            received_at=datetime.now(UTC),
            exchange_timestamp=datetime.now(UTC),
            last_trade_time=datetime.now(UTC),
            raw={},
        )
        mock_chain_df = pd.DataFrame(
            [
                {
                    "strike": 24500.0,
                    "expiry": "2024-10-24",
                    "option_type": "CE",
                    "last_price": 150.0,
                    "bid_price": 149.8,
                    "ask_price": 150.2,
                    "instrument_token": 245001,
                }
            ]
        )

        with patch.object(client, "get_nifty_spot", return_value=mock_nifty_quote), \
             patch.object(client, "get_india_vix", return_value=mock_vix_quote), \
             patch.object(client, "get_option_instruments", return_value=mock_chain_df):
            snapshot = client.get_live_snapshot()

        assert isinstance(snapshot, LiveMarketSnapshot)
        assert snapshot.nifty_spot.last_price == 24500.0
        assert snapshot.india_vix.last_price == 13.5
        assert len(snapshot.option_instruments) == 1


class TestStep4FyersStream:
    """Step 4: Test broker/fyers_stream.py threaded websocket client and aggregator routing."""

    def test_stream_subscribes_and_pushes_ticks_to_aggregator_option_token_buffer(self) -> None:
        aggregator = MarketStateAggregator()
        settings = FyersSettings(client_id="APP-100", access_token="TOKEN_123")

        stream = FyersStream(settings, aggregator=aggregator)
        token_id = 245001
        symbol_name = "NSE:NIFTY24OCT24500CE"
        stream.register_symbol_token(symbol_name, token_id)

        # Verify subscription adds to tracked sets
        stream.subscribe([token_id])
        assert token_id in stream.state.subscribed_tokens
        assert symbol_name in stream.state.subscribed_symbols

        # Mock incoming websocket tick
        sample_tick = {
            "symbol": symbol_name,
            "ltp": 160.0,
            "bid_price": 159.5,
            "ask_price": 160.5,
            "vol_traded_today": 12000,
            "exch_feed_time": 1729000000,
        }

        stream._on_message(sample_tick)

        # Check aggregator option token buffer received the tick
        assert token_id in aggregator._option_token_ticks
        token_buffer = aggregator._option_token_ticks[token_id]
        assert len(token_buffer) == 1
        row = token_buffer.iloc[0]
        assert row["instrument_token"] == token_id
        assert row["bid_price"] == 159.5
        assert row["ask_price"] == 160.5
        assert row["last_price"] == 160.0


class TestStep5RuntimeAndMain:
    """Step 5: Test LiveIntelligenceRuntime wiring with Fyers client and stream."""

    def test_runtime_initializes_with_fyers_and_refreshes_snapshot(self, tmp_path: Path) -> None:
        app_settings = AppSettings.from_env(project_root=tmp_path)
        app_settings.fyers.access_token = "MOCK_TOKEN"
        app_settings.fyers.client_id = "APP-100"

        mock_spot = MarketQuote(
            instrument="NSE:NIFTY50-INDEX",
            last_price=24500.0,
            change=15.0,
            ohlc={"open": 24450.0, "high": 24520.0, "low": 24400.0, "close": 24450.0},
            received_at=datetime.now(UTC),
            exchange_timestamp=datetime.now(UTC),
            last_trade_time=datetime.now(UTC),
            raw={},
        )
        mock_vix = MarketQuote(
            instrument="NSE:INDIAVIX-INDEX",
            last_price=14.0,
            change=0.1,
            ohlc={"open": 13.9, "high": 14.1, "low": 13.8, "close": 13.9},
            received_at=datetime.now(UTC),
            exchange_timestamp=datetime.now(UTC),
            last_trade_time=datetime.now(UTC),
            raw={},
        )
        mock_chain = pd.DataFrame(
            [
                {
                    "strike": 24500.0,
                    "expiry": "2024-10-24",
                    "option_type": "CE",
                    "last_price": 150.0,
                    "bid_price": 149.8,
                    "ask_price": 150.2,
                    "instrument_token": 245001,
                    "symbol": "NSE:NIFTY24OCT24500CE",
                },
                {
                    "strike": 24500.0,
                    "expiry": "2024-10-24",
                    "option_type": "PE",
                    "last_price": 120.0,
                    "bid_price": 119.5,
                    "ask_price": 120.5,
                    "instrument_token": 245002,
                    "symbol": "NSE:NIFTY24OCT24500PE",
                },
            ]
        )
        snapshot = LiveMarketSnapshot(
            nifty_spot=mock_spot,
            india_vix=mock_vix,
            option_instruments=mock_chain,
        )

        with patch("broker.fyers_client.FyersMarketClient.get_live_snapshot", return_value=snapshot), \
             patch("ingestion.nse_fetcher.NSEOptionChainFetcher.fetch_option_chain") as mock_fetch:
            mock_fetch.side_effect = RuntimeError("Mock fetch disable")
            runtime = LiveIntelligenceRuntime(app_settings)
            assert isinstance(runtime.stream, FyersStream)

            report = runtime.refresh_once()
            assert report is not None
            assert runtime.latest_state is not None
            # Check aggregator's snapshot bundle has the spot price
            bundle = runtime._aggregator.build_snapshot()
            assert bundle.spot_price == 24500.0
