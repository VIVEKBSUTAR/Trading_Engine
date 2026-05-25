# Regime-Aware Probabilistic Engine

## Objective

Build a regime-aware probabilistic engine for Nifty directional and options-structure decisions.

## Decision Sequence

1. Detect regime
2. Estimate conditional probability
3. Filter by execution suitability
4. Apply risk controls
5. Trigger execution workflow

## Regimes

- trending_up
- trending_down
- mean_reverting
- vol_expansion
- vol_compression
- expiry_distortion
- risk_off_expansion

## Directional-First Rollout

1. Directional target only (next-bar up/down)
2. Confidence-filtered entries
3. Add options structures after stable out-of-sample behavior

## Core Safeguards

- No random train-test splits
- Walk-forward validation only
- Feature shifting to prevent lookahead
- Confidence threshold for participation
- Costs/slippage modeled at backtest stage

## Feature Priorities

- realized_vol
- iv_realized_spread
- put_call_ratio
- oi_imbalance
- nifty_vix_corr
- overnight_gap
- gift_lead
- intraday_range_frac
- regime label
