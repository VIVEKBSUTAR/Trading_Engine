"""Backtrader-based engine for directional-first execution simulation."""

from __future__ import annotations

from dataclasses import dataclass

import backtrader as bt
import pandas as pd

from trading_engine.backtest.metrics import BacktestMetrics, compute_backtest_metrics


class SignalPandasData(bt.feeds.PandasData):
	"""Pandas feed exposing model signal/confidence lines."""

	lines = ("signal_long", "signal_short", "confidence")
	params = (
		("datetime", None),
		("open", "open"),
		("high", "high"),
		("low", "low"),
		("close", "close"),
		("volume", "volume"),
		("openinterest", -1),
		("signal_long", "signal_long"),
		("signal_short", "signal_short"),
		("confidence", "confidence"),
	)


class DirectionalSignalStrategy(bt.Strategy):
	"""Long/short directional strategy from precomputed model signals."""

	params = (
		("confidence_threshold", 0.60),
		("stop_loss_pct", 0.005),
		("max_holding_bars", 20),
		("commission", 0.0002),
		("slippage_perc", 0.0005),
	)

	def __init__(self) -> None:
		self._entry_bar = None
		self._entry_price = None
		self._entry_size = 0
		self._pending_order = None
		# ledger of executed trades with detailed accounting
		self.trades: list[dict] = []

	def next(self) -> None:
		confidence = float(self.data.confidence[0])
		signal_long = int(self.data.signal_long[0])
		signal_short = int(self.data.signal_short[0])
		close_price = float(self.data.close[0])

		if not self.position:
			if confidence < self.params.confidence_threshold:
				return

			if signal_long == 1:
				# place market order and store tentative entry info; actual price
				# is recorded at the time of close (we assume immediate fill at close)
				self._entry_bar = len(self)
				self._entry_price = close_price
				self._entry_size = 1
				self.buy()
				return

			if signal_short == 1:
				self._entry_bar = len(self)
				self._entry_price = close_price
				self._entry_size = 1
				self.sell()
				return

		if self.position and self._entry_bar is not None and self._entry_price is not None:
			holding_bars = len(self) - self._entry_bar
			if holding_bars >= self.params.max_holding_bars:
				# compute exit accounting before closing
				self._record_and_close(exit_price=close_price)
				return

			if self.position.size > 0:
				stop_price = self._entry_price * (1.0 - self.params.stop_loss_pct)
				if close_price <= stop_price:
					self._record_and_close(exit_price=close_price)
			else:
				stop_price = self._entry_price * (1.0 + self.params.stop_loss_pct)
				if close_price >= stop_price:
					self._record_and_close(exit_price=close_price)

	def _record_and_close(self, *, exit_price: float) -> None:
		"""Record trade details applying conservative slippage and commission, then close position."""
		if not self.position:
			return

		direction = "long" if self.position.size > 0 else "short"
		size = abs(self.position.size) if hasattr(self.position, "size") else getattr(self, "_entry_size", 1)
		entry_price = float(self._entry_price) if self._entry_price is not None else exit_price
		exit_price = float(exit_price)

		# gross pnl before costs
		if direction == "long":
			gross_pnl = (exit_price - entry_price) * size
		else:
			gross_pnl = (entry_price - exit_price) * size

		trade_value = entry_price * size

		# conservative: apply commission and slippage on entry and exit (both sides)
		total_commission = 2.0 * self.params.commission * trade_value
		total_slippage = 2.0 * self.params.slippage_perc * trade_value

		net_pnl = gross_pnl - total_commission - total_slippage

		holding_bars = len(self) - int(self._entry_bar) if self._entry_bar is not None else 0

		self.trades.append(
			{
				"entry_bar": int(self._entry_bar) if self._entry_bar is not None else None,
				"exit_bar": len(self),
				"entry_price": float(entry_price),
				"exit_price": float(exit_price),
				"size": float(size),
				"direction": direction,
				"gross_pnl": float(gross_pnl),
				"net_pnl": float(net_pnl),
				"total_commission": float(total_commission),
				"total_slippage": float(total_slippage),
				"holding_bars": int(holding_bars),
			}
		)

		# reset entry markers then issue a market close
		self._entry_bar = None
		self._entry_price = None
		self._entry_size = 0
		self.close()


@dataclass(slots=True)
class BacktestConfig:
	"""Run configuration for the Backtrader engine wrapper."""

	initial_cash: float = 10_000_000.0
	commission: float = 0.0002
	slippage_perc: float = 0.0005
	confidence_threshold: float = 0.60
	stop_loss_pct: float = 0.005
	max_holding_bars: int = 20


@dataclass(slots=True)
class BacktestRunResult:
	"""Output payload for one backtest run."""

	equity_curve: pd.Series
	trade_returns: pd.Series
	metrics: BacktestMetrics
	trade_ledger: pd.DataFrame


def run_directional_backtest(
	frame: pd.DataFrame,
	*,
	config: BacktestConfig,
	datetime_col: str = "timestamp",
) -> BacktestRunResult:
	"""Run directional signal backtest and return standardized diagnostics."""
	required = {
		datetime_col,
		"open",
		"high",
		"low",
		"close",
		"volume",
		"signal_long",
		"signal_short",
		"confidence",
	}
	missing = required - set(frame.columns)
	if missing:
		raise ValueError(f"Missing columns for backtest: {sorted(missing)}")

	market = frame.copy().sort_values(datetime_col).set_index(datetime_col)

	cerebro = bt.Cerebro(stdstats=False)
	data_feed = SignalPandasData(dataname=market)
	cerebro.adddata(data_feed)
	cerebro.addstrategy(
		DirectionalSignalStrategy,
		confidence_threshold=config.confidence_threshold,
		stop_loss_pct=config.stop_loss_pct,
		max_holding_bars=config.max_holding_bars,
		commission=config.commission,
		slippage_perc=config.slippage_perc,
	)

	cerebro.broker.setcash(config.initial_cash)
	cerebro.broker.setcommission(commission=config.commission)
	cerebro.broker.set_slippage_perc(config.slippage_perc)

	cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
	cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")

	result = cerebro.run()[0]

	time_return = result.analyzers.time_return.get_analysis()
	ret_series = pd.Series(time_return).sort_index()
	equity_curve = (1.0 + ret_series.fillna(0.0)).cumprod() * config.initial_cash

	# Prefer detailed ledger from strategy if available
	if hasattr(result, "trades") and isinstance(result.trades, list):
		ledgers = result.trades
		trade_returns = pd.Series([float(t.get("net_pnl", 0.0)) for t in ledgers], dtype=float)
		trade_ledger = pd.DataFrame(ledgers)
	else:
		# Fallback to TradeAnalyzer summary if ledger not present
		trade_analyzer = result.analyzers.trade_analyzer.get_analysis()
		won_total = float(getattr(getattr(trade_analyzer, "won", {}), "pnl", {}).get("total", 0.0) if isinstance(trade_analyzer, dict) else 0.0)
		lost_total = float(getattr(getattr(trade_analyzer, "lost", {}), "pnl", {}).get("total", 0.0) if isinstance(trade_analyzer, dict) else 0.0)
		trade_returns = pd.Series([won_total, lost_total], dtype=float)
		trade_ledger = pd.DataFrame([])

	metrics = compute_backtest_metrics(equity_curve=equity_curve, trade_returns=trade_returns)
	return BacktestRunResult(equity_curve=equity_curve, trade_returns=trade_returns, metrics=metrics, trade_ledger=trade_ledger)
