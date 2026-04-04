#!/usr/bin/env python3
"""
Backtesting framework for BTC/JPY momentum strategies.

Compares v1 (original) vs v2 (optimized) using bitbank hourly data.
Fetches data via bitbank public API, then simulates both strategies.

Usage:
    python backtest.py                  # Run with default settings
    python backtest.py --days 365       # Custom period
    python backtest.py --v1-only        # Only run v1 strategy
    python backtest.py --v2-only        # Only run v2 strategy
"""

import sys
import math
import json
import argparse
import numpy as np
from datetime import datetime, timezone, timedelta
from urllib.request import Request, urlopen
from dataclasses import dataclass, field

from indicators import atr, rsi, adx, momentum, realized_volatility, bollinger_bands
from risk_manager import (
    VolatilityRegime, PositionSizer, DynamicStopLoss,
    ScaleOutManager, EntryFilter, AdaptiveThreshold,
    MRStopLoss, MREntryFilter, MRTakeProfit,
)

try:
    from ml_model import (
        FeatureBuilder, SignalPredictor, features_to_matrix,
        MRFeatureBuilder, MRSignalPredictor,
    )
    HAS_ML = True
except ImportError:
    HAS_ML = False

BB_BASE = "https://public.bitbank.cc"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_hourly_data(days=365 * 3):
    """Fetch hourly BTC/JPY OHLCV from bitbank.

    Returns dict with numpy arrays: open, high, low, close, volume, timestamps
    """
    all_candles = []
    today = datetime.now(timezone.utc)

    print(f"Fetching {days} days of hourly data from bitbank...")

    for d in range(days, -1, -1):
        date = today - timedelta(days=d)
        date_str = date.strftime("%Y%m%d")
        url = f"{BB_BASE}/btc_jpy/candlestick/1hour/{date_str}"
        try:
            req = Request(url)
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            ohlcv = data["data"]["candlestick"][0]["ohlcv"]
            for row in ohlcv:
                all_candles.append({
                    "open": float(row[0]),
                    "high": float(row[1]),
                    "low": float(row[2]),
                    "close": float(row[3]),
                    "volume": float(row[4]),
                    "timestamp": int(row[5]),
                })
        except Exception as e:
            if d % 30 == 0:
                print(f"  Warning: failed {date_str}: {e}")
            continue

        if d % 90 == 0:
            print(f"  Fetched up to {date_str} ({len(all_candles)} candles)")

    all_candles.sort(key=lambda c: c["timestamp"])
    n = len(all_candles)
    print(f"Total: {n} hourly candles ({n/24:.0f} days)")

    return {
        "open": np.array([c["open"] for c in all_candles]),
        "high": np.array([c["high"] for c in all_candles]),
        "low": np.array([c["low"] for c in all_candles]),
        "close": np.array([c["close"] for c in all_candles]),
        "volume": np.array([c["volume"] for c in all_candles]),
        "timestamps": np.array([c["timestamp"] for c in all_candles]),
    }


# ---------------------------------------------------------------------------
# Trade result
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    side: str          # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    entry_bar: int
    exit_bar: int
    size: float
    pnl_pct: float
    pnl_jpy: float
    reason: str


@dataclass
class BacktestResult:
    name: str
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    initial_capital: float = 1_000_000.0

    @property
    def total_return(self):
        if not self.equity_curve:
            return 0.0
        return (self.equity_curve[-1] / self.initial_capital - 1) * 100

    @property
    def num_trades(self):
        return len(self.trades)

    @property
    def win_rate(self):
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl_pct > 0)
        return wins / len(self.trades) * 100

    @property
    def max_drawdown(self):
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            dd = (eq - peak) / peak
            max_dd = min(max_dd, dd)
        return max_dd * 100

    @property
    def sharpe_ratio(self):
        """Approximate Sharpe from equity curve (annualized, hourly data)."""
        if len(self.equity_curve) < 2:
            return 0.0
        eq = np.array(self.equity_curve)
        returns = np.diff(eq) / eq[:-1]
        if np.std(returns) == 0:
            return 0.0
        # Annualize: sqrt(24*365) for hourly
        return (np.mean(returns) / np.std(returns)) * math.sqrt(24 * 365)

    @property
    def avg_win(self):
        wins = [t.pnl_pct for t in self.trades if t.pnl_pct > 0]
        return np.mean(wins) * 100 if wins else 0.0

    @property
    def avg_loss(self):
        losses = [t.pnl_pct for t in self.trades if t.pnl_pct <= 0]
        return np.mean(losses) * 100 if losses else 0.0

    def summary(self):
        years = len(self.equity_curve) / (24 * 365) if self.equity_curve else 0
        annual = ((self.equity_curve[-1] / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 and self.equity_curve else 0
        return (
            f"\n{'='*60}\n"
            f"  {self.name}\n"
            f"{'='*60}\n"
            f"  Total Return:     {self.total_return:>10.1f}%\n"
            f"  Annual Return:    {annual:>10.1f}%\n"
            f"  Trades:           {self.num_trades:>10d}\n"
            f"  Win Rate:         {self.win_rate:>10.1f}%\n"
            f"  Avg Win:          {self.avg_win:>10.2f}%\n"
            f"  Avg Loss:         {self.avg_loss:>10.2f}%\n"
            f"  Max Drawdown:     {self.max_drawdown:>10.1f}%\n"
            f"  Sharpe Ratio:     {self.sharpe_ratio:>10.2f}\n"
            f"  Final Equity:     {self.equity_curve[-1]:>10,.0f} JPY\n"
            f"{'='*60}"
        )


# ---------------------------------------------------------------------------
# V1 Strategy (Original)
# ---------------------------------------------------------------------------

def backtest_v1(data, initial_capital=1_000_000):
    """Original strategy: LB=72h, TH=1%, SL=2.5%, 90% capital."""

    result = BacktestResult(name="V1 Original (LB72h TH1% SL2.5% Cap90%)",
                            initial_capital=initial_capital)
    closes = data["close"]
    n = len(closes)

    LB = 72  # hours
    TH = 0.01
    SL = 0.025
    CAP_RATIO = 0.90

    equity = initial_capital
    position = None  # {'side', 'size', 'entry', 'peak'/'trough', 'bar'}

    for i in range(LB, n):
        result.equity_curve.append(equity)

        price = closes[i]
        mom = (price - closes[i - LB]) / closes[i - LB]

        if position is None:
            # Entry check
            if mom > TH:
                size_btc = (equity * CAP_RATIO) / price
                position = {
                    "side": "LONG", "size": size_btc,
                    "entry": price, "peak": price, "bar": i,
                }
            elif mom < -TH:
                size_btc = (equity * CAP_RATIO) / price
                position = {
                    "side": "SHORT", "size": size_btc,
                    "entry": price, "trough": price, "bar": i,
                }
        else:
            # Position management
            if position["side"] == "LONG":
                position["peak"] = max(position["peak"], price)
                dd = (price - position["peak"]) / position["peak"]
                if dd <= -SL:
                    pnl_pct = (price - position["entry"]) / position["entry"]
                    pnl_jpy = pnl_pct * position["entry"] * position["size"]
                    equity += pnl_jpy
                    result.trades.append(Trade(
                        "LONG", position["entry"], price, position["bar"], i,
                        position["size"], pnl_pct, pnl_jpy, "SL",
                    ))
                    position = None
            else:
                position["trough"] = min(position["trough"], price)
                ru = (price - position["trough"]) / position["trough"]
                if ru >= SL:
                    pnl_pct = (position["entry"] - price) / position["entry"]
                    pnl_jpy = pnl_pct * position["entry"] * position["size"]
                    equity += pnl_jpy
                    result.trades.append(Trade(
                        "SHORT", position["entry"], price, position["bar"], i,
                        position["size"], pnl_pct, pnl_jpy, "SL",
                    ))
                    position = None

    # Close any remaining position at last price
    if position:
        price = closes[-1]
        if position["side"] == "LONG":
            pnl_pct = (price - position["entry"]) / position["entry"]
        else:
            pnl_pct = (position["entry"] - price) / position["entry"]
        pnl_jpy = pnl_pct * position["entry"] * position["size"]
        equity += pnl_jpy
        result.trades.append(Trade(
            position["side"], position["entry"], price, position["bar"],
            n - 1, position["size"], pnl_pct, pnl_jpy, "END",
        ))
    result.equity_curve.append(equity)

    return result


# ---------------------------------------------------------------------------
# V2 Strategy (Optimized)
# ---------------------------------------------------------------------------

def backtest_v2(data, initial_capital=1_000_000, capital_ratio=0.20, label=None):
    """Optimized strategy with all Phase 1-3 improvements."""

    name = label or f"V2 Optimized (Cap{int(capital_ratio*100)}%)"
    result = BacktestResult(name=name, initial_capital=initial_capital)
    closes = data["close"]
    highs = data["high"]
    lows = data["low"]
    volumes = data["volume"]
    n = len(closes)

    LB_L = 72   # 72h long lookback
    LB_S = 24   # 24h short lookback (confirmation)

    # Risk modules
    vol_regime = VolatilityRegime()
    pos_sizer = PositionSizer(capital_ratio=capital_ratio)
    dyn_sl = DynamicStopLoss()
    scale_mgr = ScaleOutManager()
    entry_flt = EntryFilter()
    adapt_th = AdaptiveThreshold()

    # Pre-compute batch indicators
    print("  Computing indicators...")
    atr_arr = atr(highs, lows, closes, period=14)
    rsi_arr = rsi(closes, period=14)
    adx_arr, _, _ = adx(highs, lows, closes, period=14)
    rvol_arr = realized_volatility(closes, period=30)
    # Annualize hourly vol: sqrt(24*365)
    rvol_annual = rvol_arr * math.sqrt(24 * 365)

    equity = initial_capital
    peak_equity = initial_capital
    position = None

    for i in range(LB_L, n):
        result.equity_curve.append(equity)

        price = closes[i]

        # Momentum
        mom_l = (price - closes[i - LB_L]) / closes[i - LB_L]
        mom_s = (price - closes[i - LB_S]) / closes[i - LB_S] if i >= LB_S else None

        # Current indicator values
        cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else None
        cur_rsi = rsi_arr[i] if not np.isnan(rsi_arr[i]) else None
        cur_adx = adx_arr[i] if not np.isnan(adx_arr[i]) else None
        cur_rvol = rvol_annual[i] if not np.isnan(rvol_annual[i]) else None

        regime = vol_regime.classify(cur_rvol)
        threshold = adapt_th.get_threshold(regime)

        if position is None:
            # === Entry ===

            # LONG
            if mom_l > threshold:
                # Multi-timeframe confirmation
                if mom_s is not None and mom_s <= 0:
                    continue

                ok, _ = entry_flt.check_long(adx_value=cur_adx, rsi_value=cur_rsi)
                if not ok:
                    continue

                peak_equity = max(peak_equity, equity)
                size = pos_sizer.calculate(
                    equity, price, regime, peak_equity, equity, min_size=0.001,
                )
                if size <= 0:
                    continue

                position = {
                    "side": "LONG", "size": size, "original_size": size,
                    "entry": price, "peak": price, "bar": i,
                }
                scale_mgr.reset()

            # SHORT
            elif mom_l < -threshold:
                if mom_s is not None and mom_s >= 0:
                    continue

                ok, _ = entry_flt.check_short(adx_value=cur_adx, rsi_value=cur_rsi)
                if not ok:
                    continue

                peak_equity = max(peak_equity, equity)
                size = pos_sizer.calculate(
                    equity, price, regime, peak_equity, equity, min_size=0.001,
                )
                if size <= 0:
                    continue

                position = {
                    "side": "SHORT", "size": size, "original_size": size,
                    "entry": price, "trough": price, "bar": i,
                }
                scale_mgr.reset()

        else:
            # === Position management ===

            # Dynamic stop distance
            stop_pct = dyn_sl.stop_distance_pct(cur_atr, price, regime)

            # Scale-out
            partial = scale_mgr.check(
                position["entry"], price, position["side"],
                position["original_size"],
            )
            if partial > 0 and partial < position["size"]:
                # Realize partial profit
                if position["side"] == "LONG":
                    pnl_pct = (price - position["entry"]) / position["entry"]
                else:
                    pnl_pct = (position["entry"] - price) / position["entry"]
                partial_pnl = pnl_pct * position["entry"] * partial
                equity += partial_pnl
                position["size"] -= partial
                position["size"] = round(position["size"], 8)

            # Trailing stop
            should_close = False
            if position["side"] == "LONG":
                position["peak"] = max(position["peak"], price)
                dd = (price - position["peak"]) / position["peak"]
                if dd <= -stop_pct:
                    should_close = True
            else:
                position["trough"] = min(position["trough"], price)
                ru = (price - position["trough"]) / position["trough"]
                if ru >= stop_pct:
                    should_close = True

            if should_close:
                if position["side"] == "LONG":
                    pnl_pct = (price - position["entry"]) / position["entry"]
                else:
                    pnl_pct = (position["entry"] - price) / position["entry"]
                pnl_jpy = pnl_pct * position["entry"] * position["size"]
                equity += pnl_jpy
                result.trades.append(Trade(
                    position["side"], position["entry"], price,
                    position["bar"], i, position["size"],
                    pnl_pct, pnl_jpy, "SL",
                ))
                position = None

    # Close remaining
    if position:
        price = closes[-1]
        if position["side"] == "LONG":
            pnl_pct = (price - position["entry"]) / position["entry"]
        else:
            pnl_pct = (position["entry"] - price) / position["entry"]
        pnl_jpy = pnl_pct * position["entry"] * position["size"]
        equity += pnl_jpy
        result.trades.append(Trade(
            position["side"], position["entry"], price,
            position["bar"], n - 1, position["size"],
            pnl_pct, pnl_jpy, "END",
        ))
    result.equity_curve.append(equity)

    return result


# ---------------------------------------------------------------------------
# V3 Strategy (Aggressive Optimized)
# ---------------------------------------------------------------------------

def backtest_v3(data, initial_capital=1_000_000, capital_ratio=0.90,
                adx_th=20.0, use_mtf=False, use_scaleout=False,
                sl_mult=2.5, eq_levels=None, label=None):
    """V3: Keep ATR dynamic SL + adaptive threshold, relax filters for more return.

    vs V2: higher cap ratio, lower/optional ADX, optional MTF, optional scale-out.
    eq_levels: custom equity drawdown levels for PositionSizer, or None for default.
    """
    name = label or f"V3 Cap{int(capital_ratio*100)}% ADX{int(adx_th)} MTF={'Y' if use_mtf else 'N'} SL×{sl_mult}"
    result = BacktestResult(name=name, initial_capital=initial_capital)
    closes = data["close"]
    highs = data["high"]
    lows = data["low"]
    n = len(closes)

    LB_L = 72
    LB_S = 24

    vol_regime_cls = VolatilityRegime()
    pos_sizer = PositionSizer(capital_ratio=capital_ratio,
                              equity_drawdown_levels=eq_levels)
    dyn_sl = DynamicStopLoss(atr_multiplier=sl_mult)
    scale_mgr = ScaleOutManager() if use_scaleout else None
    entry_flt = EntryFilter(adx_threshold=adx_th, enable_rsi=True)
    adapt_th = AdaptiveThreshold()

    atr_arr = atr(highs, lows, closes, period=14)
    rsi_arr = rsi(closes, period=14)
    adx_arr, _, _ = adx(highs, lows, closes, period=14)
    rvol_arr = realized_volatility(closes, period=30)
    rvol_annual = rvol_arr * math.sqrt(24 * 365)

    equity = initial_capital
    peak_equity = initial_capital
    position = None

    for i in range(LB_L, n):
        result.equity_curve.append(equity)
        price = closes[i]

        mom_l = (price - closes[i - LB_L]) / closes[i - LB_L]
        mom_s = (price - closes[i - LB_S]) / closes[i - LB_S] if i >= LB_S else None

        cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else None
        cur_rsi = rsi_arr[i] if not np.isnan(rsi_arr[i]) else None
        cur_adx = adx_arr[i] if not np.isnan(adx_arr[i]) else None
        cur_rvol = rvol_annual[i] if not np.isnan(rvol_annual[i]) else None

        regime = vol_regime_cls.classify(cur_rvol)
        threshold = adapt_th.get_threshold(regime)

        if position is None:
            if mom_l > threshold:
                if use_mtf and mom_s is not None and mom_s <= 0:
                    continue
                ok, _ = entry_flt.check_long(adx_value=cur_adx, rsi_value=cur_rsi)
                if not ok:
                    continue
                peak_equity = max(peak_equity, equity)
                size = pos_sizer.calculate(equity, price, regime, peak_equity, equity, min_size=0.001)
                if size <= 0:
                    continue
                position = {"side": "LONG", "size": size, "original_size": size,
                            "entry": price, "peak": price, "bar": i}
                if scale_mgr:
                    scale_mgr.reset()

            elif mom_l < -threshold:
                if use_mtf and mom_s is not None and mom_s >= 0:
                    continue
                ok, _ = entry_flt.check_short(adx_value=cur_adx, rsi_value=cur_rsi)
                if not ok:
                    continue
                peak_equity = max(peak_equity, equity)
                size = pos_sizer.calculate(equity, price, regime, peak_equity, equity, min_size=0.001)
                if size <= 0:
                    continue
                position = {"side": "SHORT", "size": size, "original_size": size,
                            "entry": price, "trough": price, "bar": i}
                if scale_mgr:
                    scale_mgr.reset()
        else:
            stop_pct = dyn_sl.stop_distance_pct(cur_atr, price, regime)

            if scale_mgr:
                partial = scale_mgr.check(position["entry"], price, position["side"], position["original_size"])
                if partial > 0 and partial < position["size"]:
                    pnl_pct = ((price - position["entry"]) / position["entry"]
                               if position["side"] == "LONG"
                               else (position["entry"] - price) / position["entry"])
                    equity += pnl_pct * position["entry"] * partial
                    position["size"] = round(position["size"] - partial, 8)

            should_close = False
            if position["side"] == "LONG":
                position["peak"] = max(position["peak"], price)
                if (price - position["peak"]) / position["peak"] <= -stop_pct:
                    should_close = True
            else:
                position["trough"] = min(position["trough"], price)
                if (price - position["trough"]) / position["trough"] >= stop_pct:
                    should_close = True

            if should_close:
                pnl_pct = ((price - position["entry"]) / position["entry"]
                           if position["side"] == "LONG"
                           else (position["entry"] - price) / position["entry"])
                pnl_jpy = pnl_pct * position["entry"] * position["size"]
                equity += pnl_jpy
                result.trades.append(Trade(
                    position["side"], position["entry"], price,
                    position["bar"], i, position["size"], pnl_pct, pnl_jpy, "SL"))
                position = None

    if position:
        price = closes[-1]
        pnl_pct = ((price - position["entry"]) / position["entry"]
                   if position["side"] == "LONG"
                   else (position["entry"] - price) / position["entry"])
        pnl_jpy = pnl_pct * position["entry"] * position["size"]
        equity += pnl_jpy
        result.trades.append(Trade(
            position["side"], position["entry"], price,
            position["bar"], n - 1, position["size"], pnl_pct, pnl_jpy, "END"))
    result.equity_curve.append(equity)
    return result


# ---------------------------------------------------------------------------
# V4 Hybrid Strategy (V3 + XGBoost confidence filter)
# ---------------------------------------------------------------------------

def backtest_v4_hybrid(data, initial_capital=1_000_000, capital_ratio=0.90,
                       adx_th=25.0, sl_mult=2.5,
                       train_hours=24*180, embargo_hours=24*7,
                       prob_full=0.6, prob_half=0.5,
                       label=None):
    """V4: V3 rules + XGBoost confidence-based position sizing.

    Walk-forward: train on past data, predict on current bar.
    When V3 generates a signal, XGBoost scores confidence:
      prob > prob_full  → full size (capital_ratio)
      prob > prob_half  → half size (capital_ratio * 0.5)
      prob <= prob_half → skip signal
    """
    if not HAS_ML:
        print("ERROR: ml_model not available. Install xgboost: pip install xgboost scikit-learn")
        sys.exit(1)

    name = label or f"V4 Hybrid (XGB prob>{prob_full:.1f}/{prob_half:.1f})"
    result = BacktestResult(name=name, initial_capital=initial_capital)
    closes = data["close"]
    highs = data["high"]
    lows = data["low"]
    volumes = data["volume"]
    timestamps = data["timestamps"]
    n = len(closes)

    LB_L = 72
    LB_S = 24

    # V3 risk modules
    vol_regime_cls = VolatilityRegime()
    pos_sizer = PositionSizer(capital_ratio=capital_ratio,
                              equity_drawdown_levels=[(1.0, 1.0)])  # no eq scaling
    dyn_sl = DynamicStopLoss(atr_multiplier=sl_mult)
    entry_flt = EntryFilter(adx_threshold=adx_th, enable_rsi=True)
    adapt_th = AdaptiveThreshold()

    # Pre-compute V3 indicators
    print("  Computing V3 indicators...")
    atr_arr = atr(highs, lows, closes, period=14)
    rsi_arr = rsi(closes, period=14)
    adx_arr, _, _ = adx(highs, lows, closes, period=14)
    rvol_arr = realized_volatility(closes, period=30)
    rvol_annual = rvol_arr * math.sqrt(24 * 365)

    # Build ML features for entire dataset
    print("  Building ML features...")
    fb = FeatureBuilder()
    features = fb.build(closes, highs, lows, volumes, timestamps)
    labels = fb.build_labels(closes, horizon=5, threshold=0.01)
    X, valid_mask = features_to_matrix(features, fb.feature_names)

    label_valid = np.isfinite(labels)
    full_mask = valid_mask & label_valid

    equity = initial_capital
    peak_equity = initial_capital
    position = None

    # ML model state
    predictor = None
    last_train_bar = 0
    retrain_interval = 24 * 30  # retrain monthly
    skipped_by_ml = 0
    halved_by_ml = 0
    full_by_ml = 0

    print("  Running V4 hybrid backtest...")

    for i in range(LB_L, n):
        result.equity_curve.append(equity)
        price = closes[i]

        # === Retrain ML model periodically (walk-forward) ===
        if (predictor is None or i - last_train_bar >= retrain_interval) and i >= train_hours:
            train_start = max(0, i - train_hours)
            train_end = i - embargo_hours  # embargo gap

            if train_end > train_start + 100:
                train_idx = np.arange(train_start, train_end)
                tmask = full_mask[train_idx]
                X_t = X[train_idx[tmask]]
                y_t = labels[train_idx[tmask]]

                if len(X_t) >= 100:
                    predictor = SignalPredictor()
                    predictor.train(X_t, y_t)
                    last_train_bar = i

        # V3 indicators
        mom_l = (price - closes[i - LB_L]) / closes[i - LB_L]
        mom_s = (price - closes[i - LB_S]) / closes[i - LB_S] if i >= LB_S else None

        cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else None
        cur_rsi = rsi_arr[i] if not np.isnan(rsi_arr[i]) else None
        cur_adx = adx_arr[i] if not np.isnan(adx_arr[i]) else None
        cur_rvol = rvol_annual[i] if not np.isnan(rvol_annual[i]) else None

        regime = vol_regime_cls.classify(cur_rvol)
        threshold = adapt_th.get_threshold(regime)

        if position is None:
            direction = None
            if mom_l > threshold:
                # MTF check
                if mom_s is not None and mom_s <= 0:
                    continue
                ok, _ = entry_flt.check_long(adx_value=cur_adx, rsi_value=cur_rsi)
                if not ok:
                    continue
                direction = "LONG"
            elif mom_l < -threshold:
                if mom_s is not None and mom_s >= 0:
                    continue
                ok, _ = entry_flt.check_short(adx_value=cur_adx, rsi_value=cur_rsi)
                if not ok:
                    continue
                direction = "SHORT"

            if direction is not None:
                # === XGBoost confidence filter ===
                size_mult = 1.0  # default full size
                if predictor is not None and predictor.is_ready and valid_mask[i]:
                    x_i = X[i:i+1]
                    prob = predictor.predict_proba(x_i, direction=direction)
                    prob_val = float(prob[0]) if hasattr(prob, '__len__') else float(prob)

                    if prob_val > prob_full:
                        size_mult = 1.2  # ML boost: 20% extra size
                        full_by_ml += 1
                    elif prob_val > prob_half:
                        size_mult = 1.0  # Normal V3 size
                        halved_by_ml += 1
                    else:
                        size_mult = 0.0
                        skipped_by_ml += 1
                        continue  # ML says skip

                peak_equity = max(peak_equity, equity)
                base_size = pos_sizer.calculate(
                    equity, price, regime, peak_equity, equity, min_size=0.001)
                size = base_size * size_mult
                if size <= 0.001:
                    continue

                if direction == "LONG":
                    position = {"side": "LONG", "size": size,
                                "entry": price, "peak": price, "bar": i}
                else:
                    position = {"side": "SHORT", "size": size,
                                "entry": price, "trough": price, "bar": i}

        else:
            # === Position management (same as V3) ===
            stop_pct = dyn_sl.stop_distance_pct(cur_atr, price, regime)

            should_close = False
            if position["side"] == "LONG":
                position["peak"] = max(position["peak"], price)
                if (price - position["peak"]) / position["peak"] <= -stop_pct:
                    should_close = True
            else:
                position["trough"] = min(position["trough"], price)
                if (price - position["trough"]) / position["trough"] >= stop_pct:
                    should_close = True

            if should_close:
                pnl_pct = ((price - position["entry"]) / position["entry"]
                           if position["side"] == "LONG"
                           else (position["entry"] - price) / position["entry"])
                pnl_jpy = pnl_pct * position["entry"] * position["size"]
                equity += pnl_jpy
                result.trades.append(Trade(
                    position["side"], position["entry"], price,
                    position["bar"], i, position["size"], pnl_pct, pnl_jpy, "SL"))
                position = None

    # Close remaining
    if position:
        price = closes[-1]
        pnl_pct = ((price - position["entry"]) / position["entry"]
                   if position["side"] == "LONG"
                   else (position["entry"] - price) / position["entry"])
        pnl_jpy = pnl_pct * position["entry"] * position["size"]
        equity += pnl_jpy
        result.trades.append(Trade(
            position["side"], position["entry"], price,
            position["bar"], n - 1, position["size"], pnl_pct, pnl_jpy, "END"))
    result.equity_curve.append(equity)

    # ML stats
    total_signals = full_by_ml + halved_by_ml + skipped_by_ml
    if total_signals > 0:
        print(f"\n  ML Filter Stats:")
        print(f"    Full size (prob>{prob_full}):  {full_by_ml:>5d} ({full_by_ml/total_signals*100:.1f}%)")
        print(f"    Half size (prob>{prob_half}):  {halved_by_ml:>5d} ({halved_by_ml/total_signals*100:.1f}%)")
        print(f"    Skipped   (prob<{prob_half}):  {skipped_by_ml:>5d} ({skipped_by_ml/total_signals*100:.1f}%)")
        print(f"    Total V3 signals:       {total_signals:>5d}")

    return result


# ---------------------------------------------------------------------------
# V5 Dual Strategy (Momentum + Mean Reversion)
# ---------------------------------------------------------------------------

def backtest_v5_multi(data, initial_capital=1_000_000, label=None):
    """V5: Dual strategy — Momentum (ADX>=20) + Mean Reversion (ADX<20).

    Momentum priority: if MR position open and momentum signal fires,
    close MR and open momentum.
    """
    if not HAS_ML:
        print("ERROR: ml_model not available. pip install xgboost")
        sys.exit(1)

    name = label or "V5 Dual (MOM+MR)"
    result = BacktestResult(name=name, initial_capital=initial_capital)
    closes = data["close"]
    highs = data["high"]
    lows = data["low"]
    volumes = data["volume"]
    timestamps = data["timestamps"]
    n = len(closes)

    # === Momentum params (improved V4) ===
    MOM_CAP = 0.90
    MOM_ADX_TH = 20.0  # lowered from 25
    MOM_ML_SKIP = 0.35
    MOM_ML_BOOST = 0.55
    LB_L = 72
    LB_S = 24
    LB_12 = 12  # new 12h lookback

    # === MR params ===
    MR_CAP = 0.45
    MR_TIME_STOP = 8  # hours
    MR_ADX_EXIT = 25.0  # exit MR if ADX rises above this

    # Risk modules
    vol_regime_cls = VolatilityRegime()
    mom_sizer = PositionSizer(capital_ratio=MOM_CAP,
                               equity_drawdown_levels=[(1.0, 1.0)])
    mr_sizer = PositionSizer(capital_ratio=MR_CAP,
                              equity_drawdown_levels=[(1.0, 1.0)])
    mom_sl = DynamicStopLoss(atr_multiplier=2.0)
    mr_sl_mgr = MRStopLoss()
    mom_entry = EntryFilter(adx_threshold=MOM_ADX_TH, enable_rsi=True)
    mr_entry = MREntryFilter(adx_max=MOM_ADX_TH)
    mr_tp = MRTakeProfit()
    adapt_th = AdaptiveThreshold()

    # Pre-compute indicators
    print("  Computing indicators...")
    atr_arr = atr(highs, lows, closes, period=14)
    rsi_arr = rsi(closes, period=14)
    adx_arr, _, _ = adx(highs, lows, closes, period=14)
    rvol_arr = realized_volatility(closes, period=30)
    rvol_annual = rvol_arr * math.sqrt(24 * 365)
    bb_upper, bb_middle, bb_lower = bollinger_bands(closes, period=20)

    # Build ML features (momentum model)
    print("  Building ML features (momentum)...")
    fb_mom = FeatureBuilder()
    feat_mom = fb_mom.build(closes, highs, lows, volumes, timestamps)
    labels_mom = fb_mom.build_labels(closes, horizon=12, threshold=0.01)
    X_mom, valid_mom = features_to_matrix(feat_mom, fb_mom.feature_names)
    label_valid_mom = np.isfinite(labels_mom)
    full_mask_mom = valid_mom & label_valid_mom

    # Build ML features (MR model)
    print("  Building ML features (MR)...")
    fb_mr = MRFeatureBuilder()
    feat_mr = fb_mr.build(closes, highs, lows, volumes, timestamps)
    labels_mr = fb_mr.build_labels(closes, horizon=3, threshold=0.005)
    X_mr, valid_mr = features_to_matrix(feat_mr, fb_mr.feature_names)
    label_valid_mr = np.isfinite(labels_mr)
    full_mask_mr = valid_mr & label_valid_mr

    # Walk-forward ML
    train_hours = 24 * 180
    embargo = 24 * 7
    retrain_interval = 24 * 30
    pred_mom = None
    pred_mr = None
    last_train_mom = 0
    last_train_mr = 0

    equity = initial_capital
    peak_equity = initial_capital
    position = None  # {side, size, entry, peak/trough, bar, strategy: 'MOM'|'MR'}

    # Stats
    mom_trades = 0
    mr_trades = 0
    mom_skipped = 0
    mr_skipped = 0
    mr_tp_count = 0
    mr_time_stop = 0
    mr_regime_exit = 0

    print("  Running V5 dual backtest...")

    for i in range(LB_L, n):
        result.equity_curve.append(equity)
        price = closes[i]

        # Retrain momentum model
        if (pred_mom is None or i - last_train_mom >= retrain_interval) and i >= train_hours:
            ts, te = max(0, i - train_hours), i - embargo
            if te > ts + 100:
                idx = np.arange(ts, te)
                m = full_mask_mom[idx]
                Xt, yt = X_mom[idx[m]], labels_mom[idx[m]]
                if len(Xt) >= 100:
                    pred_mom = SignalPredictor()
                    pred_mom.train(Xt, yt)
                    last_train_mom = i

        # Retrain MR model
        if (pred_mr is None or i - last_train_mr >= retrain_interval) and i >= train_hours:
            ts, te = max(0, i - train_hours), i - embargo
            if te > ts + 100:
                idx = np.arange(ts, te)
                m = full_mask_mr[idx]
                Xt, yt = X_mr[idx[m]], labels_mr[idx[m]]
                if len(Xt) >= 100:
                    pred_mr = MRSignalPredictor()
                    pred_mr.train(Xt, yt)
                    last_train_mr = i

        # Current indicators
        cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else None
        cur_rsi = rsi_arr[i] if not np.isnan(rsi_arr[i]) else None
        cur_adx = adx_arr[i] if not np.isnan(adx_arr[i]) else None
        cur_rvol = rvol_annual[i] if not np.isnan(rvol_annual[i]) else None
        cur_bb_upper = bb_upper[i] if not np.isnan(bb_upper[i]) else None
        cur_bb_middle = bb_middle[i] if not np.isnan(bb_middle[i]) else None
        cur_bb_lower = bb_lower[i] if not np.isnan(bb_lower[i]) else None

        regime = vol_regime_cls.classify(cur_rvol)
        threshold = adapt_th.get_threshold(regime)

        # Momentum signals
        mom_l = (price - closes[i - LB_L]) / closes[i - LB_L]
        mom_s = (price - closes[i - LB_S]) / closes[i - LB_S] if i >= LB_S else None
        mom_12 = (price - closes[i - LB_12]) / closes[i - LB_12] if i >= LB_12 else None

        # Dual-timeframe momentum: (72h>TH AND 24h>0) OR (24h>TH AND 12h>0)
        def mom_long_signal():
            cond_a = mom_l > threshold and (mom_s is not None and mom_s > 0)
            cond_b = (mom_s is not None and mom_s > threshold and
                      mom_12 is not None and mom_12 > 0)
            return cond_a or cond_b

        def mom_short_signal():
            cond_a = mom_l < -threshold and (mom_s is not None and mom_s < 0)
            cond_b = (mom_s is not None and mom_s < -threshold and
                      mom_12 is not None and mom_12 < 0)
            return cond_a or cond_b

        # === Check for momentum signal (can override MR position) ===
        mom_direction = None
        if mom_long_signal():
            ok, _ = mom_entry.check_long(adx_value=cur_adx, rsi_value=cur_rsi)
            if ok:
                mom_direction = "LONG"
        elif mom_short_signal():
            ok, _ = mom_entry.check_short(adx_value=cur_adx, rsi_value=cur_rsi)
            if ok:
                mom_direction = "SHORT"

        # ML filter for momentum
        if mom_direction is not None and pred_mom is not None and pred_mom.is_ready and valid_mom[i]:
            prob = float(pred_mom.predict_proba(X_mom[i:i+1], direction=mom_direction)[0])
            if prob < MOM_ML_SKIP:
                mom_skipped += 1
                mom_direction = None

        # If we have MR position and momentum signal fires → close MR, open momentum
        if position is not None and position["strategy"] == "MR" and mom_direction is not None:
            # Close MR position
            if position["side"] == "LONG":
                pnl_pct = (price - position["entry"]) / position["entry"]
            else:
                pnl_pct = (position["entry"] - price) / position["entry"]
            pnl_jpy = pnl_pct * position["entry"] * position["size"]
            equity += pnl_jpy
            result.trades.append(Trade(
                position["side"], position["entry"], price,
                position["bar"], i, position["size"], pnl_pct, pnl_jpy, "MR→MOM"))
            mr_regime_exit += 1
            position = None

        if position is None:
            if mom_direction is not None:
                # === Momentum entry ===
                peak_equity = max(peak_equity, equity)
                size_mult = 1.0
                if pred_mom is not None and pred_mom.is_ready and valid_mom[i]:
                    prob = float(pred_mom.predict_proba(X_mom[i:i+1], direction=mom_direction)[0])
                    if prob > MOM_ML_BOOST:
                        size_mult = 1.2
                size = mom_sizer.calculate(
                    equity, price, regime, peak_equity, equity, min_size=0.001) * size_mult
                if size > 0.001:
                    if mom_direction == "LONG":
                        position = {"side": "LONG", "size": size, "entry": price,
                                    "peak": price, "bar": i, "strategy": "MOM"}
                    else:
                        position = {"side": "SHORT", "size": size, "entry": price,
                                    "trough": price, "bar": i, "strategy": "MOM"}
                    mom_trades += 1

            else:
                # === MR entry (only when no momentum signal) ===
                mr_direction = None
                if cur_bb_lower is not None and cur_bb_upper is not None:
                    ok_l, _ = mr_entry.check_long(cur_adx, cur_rsi, price, cur_bb_lower)
                    ok_s, _ = mr_entry.check_short(cur_adx, cur_rsi, price, cur_bb_upper)
                    if ok_l:
                        mr_direction = "LONG"
                    elif ok_s:
                        mr_direction = "SHORT"

                # ML filter for MR
                if mr_direction is not None and pred_mr is not None and pred_mr.is_ready and valid_mr[i]:
                    prob = float(pred_mr.predict_proba(X_mr[i:i+1], direction=mr_direction)[0])
                    if prob < MOM_ML_SKIP:
                        mr_skipped += 1
                        mr_direction = None

                if mr_direction is not None:
                    peak_equity = max(peak_equity, equity)
                    size = mr_sizer.calculate(
                        equity, price, regime, peak_equity, equity, min_size=0.001)
                    if size > 0.001:
                        if mr_direction == "LONG":
                            position = {"side": "LONG", "size": size, "entry": price,
                                        "peak": price, "bar": i, "strategy": "MR"}
                        else:
                            position = {"side": "SHORT", "size": size, "entry": price,
                                        "trough": price, "bar": i, "strategy": "MR"}
                        mr_trades += 1

        elif position["strategy"] == "MOM":
            # === Momentum position management ===
            stop_pct = mom_sl.stop_distance_pct(cur_atr, price, regime)
            should_close = False
            if position["side"] == "LONG":
                position["peak"] = max(position["peak"], price)
                if (price - position["peak"]) / position["peak"] <= -stop_pct:
                    should_close = True
            else:
                position["trough"] = min(position["trough"], price)
                if (price - position["trough"]) / position["trough"] >= stop_pct:
                    should_close = True

            if should_close:
                pnl_pct = ((price - position["entry"]) / position["entry"]
                           if position["side"] == "LONG"
                           else (position["entry"] - price) / position["entry"])
                pnl_jpy = pnl_pct * position["entry"] * position["size"]
                equity += pnl_jpy
                result.trades.append(Trade(
                    position["side"], position["entry"], price,
                    position["bar"], i, position["size"], pnl_pct, pnl_jpy, "MOM_SL"))
                position = None

        elif position["strategy"] == "MR":
            # === MR position management ===
            hours_held = i - position["bar"]

            # 1. Take profit at BB middle
            if mr_tp.should_take_profit(position["side"], price, cur_bb_middle):
                pnl_pct = ((price - position["entry"]) / position["entry"]
                           if position["side"] == "LONG"
                           else (position["entry"] - price) / position["entry"])
                pnl_jpy = pnl_pct * position["entry"] * position["size"]
                equity += pnl_jpy
                result.trades.append(Trade(
                    position["side"], position["entry"], price,
                    position["bar"], i, position["size"], pnl_pct, pnl_jpy, "MR_TP"))
                mr_tp_count += 1
                position = None
                continue

            # 2. Time stop
            if hours_held >= MR_TIME_STOP:
                pnl_pct = ((price - position["entry"]) / position["entry"]
                           if position["side"] == "LONG"
                           else (position["entry"] - price) / position["entry"])
                pnl_jpy = pnl_pct * position["entry"] * position["size"]
                equity += pnl_jpy
                result.trades.append(Trade(
                    position["side"], position["entry"], price,
                    position["bar"], i, position["size"], pnl_pct, pnl_jpy, "MR_TIME"))
                mr_time_stop += 1
                position = None
                continue

            # 3. Regime change exit
            if cur_adx is not None and cur_adx > MR_ADX_EXIT:
                pnl_pct = ((price - position["entry"]) / position["entry"]
                           if position["side"] == "LONG"
                           else (position["entry"] - price) / position["entry"])
                pnl_jpy = pnl_pct * position["entry"] * position["size"]
                equity += pnl_jpy
                result.trades.append(Trade(
                    position["side"], position["entry"], price,
                    position["bar"], i, position["size"], pnl_pct, pnl_jpy, "MR_REGIME"))
                mr_regime_exit += 1
                position = None
                continue

            # 4. Trailing stop
            stop_pct = mr_sl_mgr.stop_distance_pct(cur_atr, price)
            should_close = False
            if position["side"] == "LONG":
                position["peak"] = max(position["peak"], price)
                if (price - position["peak"]) / position["peak"] <= -stop_pct:
                    should_close = True
            else:
                position["trough"] = min(position["trough"], price)
                if (price - position["trough"]) / position["trough"] >= stop_pct:
                    should_close = True

            if should_close:
                pnl_pct = ((price - position["entry"]) / position["entry"]
                           if position["side"] == "LONG"
                           else (position["entry"] - price) / position["entry"])
                pnl_jpy = pnl_pct * position["entry"] * position["size"]
                equity += pnl_jpy
                result.trades.append(Trade(
                    position["side"], position["entry"], price,
                    position["bar"], i, position["size"], pnl_pct, pnl_jpy, "MR_SL"))
                position = None

    # Close remaining
    if position:
        price = closes[-1]
        pnl_pct = ((price - position["entry"]) / position["entry"]
                   if position["side"] == "LONG"
                   else (position["entry"] - price) / position["entry"])
        pnl_jpy = pnl_pct * position["entry"] * position["size"]
        equity += pnl_jpy
        result.trades.append(Trade(
            position["side"], position["entry"], price,
            position["bar"], n - 1, position["size"], pnl_pct, pnl_jpy, "END"))
    result.equity_curve.append(equity)

    # Stats
    total = mom_trades + mr_trades
    print(f"\n  V5 Strategy Stats:")
    print(f"    Momentum trades: {mom_trades} (skipped by ML: {mom_skipped})")
    print(f"    MR trades:       {mr_trades} (skipped by ML: {mr_skipped})")
    print(f"    Total trades:    {total}")
    print(f"    MR exits → TP: {mr_tp_count}, Time: {mr_time_stop}, Regime: {mr_regime_exit}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BTC/JPY Momentum Backtest")
    parser.add_argument("--days", type=int, default=365 * 3, help="Days of data")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital JPY")
    parser.add_argument("--v1-only", action="store_true")
    parser.add_argument("--v2-only", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep capital ratios: 20%,30%,40%,50%,60%,70%,80%,90%")
    parser.add_argument("--v3", action="store_true",
                        help="Run V3 aggressive sweep: vary ADX, MTF, SL multiplier")
    parser.add_argument("--v4", action="store_true",
                        help="Run V4 hybrid (V3 + XGBoost) vs V3 comparison")
    parser.add_argument("--v5", action="store_true",
                        help="Run V5 dual (MOM+MR) vs V4 comparison")
    args = parser.parse_args()

    data = fetch_hourly_data(args.days)

    if len(data["close"]) < 100:
        print("ERROR: Not enough data fetched.")
        sys.exit(1)

    # === V5 dual strategy comparison ===
    if args.v5:
        print("\nRunning V1 baseline...")
        r1 = backtest_v1(data, args.capital)

        EQ_NONE = [(1.0, 1.0)]
        print("\nRunning V4 Boost (best V4)...")
        r4 = backtest_v4_hybrid(data, args.capital, capital_ratio=0.90,
                                adx_th=25, sl_mult=2.5,
                                prob_full=0.55, prob_half=0.40,
                                label="V4 Boost (0.55/0.40)")

        print("\nRunning V5 Dual (MOM+MR)...")
        r5 = backtest_v5_multi(data, args.capital)

        all_results = [r1, r4, r5]

        print(f"\n{'='*100}")
        print(f"  V4 vs V5 DUAL STRATEGY COMPARISON")
        print(f"{'='*100}")
        hdr = f"{'Strategy':<42} {'Return%':>9} {'Annual%':>9} {'Trades':>7} {'WinR%':>7} {'MaxDD%':>8} {'Sharpe':>7}"
        print(hdr)
        print("-" * 100)

        for r in all_results:
            years = len(r.equity_curve) / (24 * 365) if r.equity_curve else 1
            ann = ((r.equity_curve[-1] / r.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
            print(f"{r.name:<42} {r.total_return:>8.1f}% {ann:>8.1f}% {r.num_trades:>7d} {r.win_rate:>6.1f}% {r.max_drawdown:>7.1f}% {r.sharpe_ratio:>7.2f}")

        print(f"{'='*100}")

        # MR breakdown
        mr_trades = [t for t in r5.trades if t.reason.startswith("MR")]
        mom_trades = [t for t in r5.trades if t.reason.startswith("MOM") or t.reason == "END" or t.reason == "MR→MOM"]
        if mr_trades:
            mr_wins = sum(1 for t in mr_trades if t.pnl_pct > 0)
            mr_pnl = sum(t.pnl_jpy for t in mr_trades)
            print(f"\n  MR Breakdown: {len(mr_trades)} trades, "
                  f"WR={mr_wins/len(mr_trades)*100:.1f}%, "
                  f"PnL={mr_pnl:+,.0f} JPY")
        if mom_trades:
            mom_wins = sum(1 for t in mom_trades if t.pnl_pct > 0)
            mom_pnl = sum(t.pnl_jpy for t in mom_trades)
            print(f"  MOM Breakdown: {len(mom_trades)} trades, "
                  f"WR={mom_wins/len(mom_trades)*100:.1f}%, "
                  f"PnL={mom_pnl:+,.0f} JPY")

        return

    # === V4 hybrid comparison ===
    if args.v4:
        print("\nRunning V1 baseline...")
        r1 = backtest_v1(data, args.capital)

        print("\nRunning V3 best (C90 ADX25 MTF noEQ)...")
        EQ_NONE = [(1.0, 1.0)]
        r3 = backtest_v3(data, args.capital, capital_ratio=0.90,
                         adx_th=25, use_mtf=True, use_scaleout=False,
                         sl_mult=2.5, eq_levels=EQ_NONE,
                         label="V3 C90 ADX25 MTF noEQ (best)")

        print("\nRunning V4 Hybrid (filter mode: 0.6/0.5)...")
        r4a = backtest_v4_hybrid(data, args.capital, capital_ratio=0.90,
                                 adx_th=25, sl_mult=2.5,
                                 prob_full=0.6, prob_half=0.5,
                                 label="V4 Filter (0.6/0.5)")

        print("\nRunning V4 Hybrid (boost mode: skip<0.40, boost>0.55)...")
        r4b = backtest_v4_hybrid(data, args.capital, capital_ratio=0.90,
                                 adx_th=25, sl_mult=2.5,
                                 prob_full=0.55, prob_half=0.40,
                                 label="V4 Boost (0.55/0.40)")

        print("\nRunning V4 Hybrid (light filter: skip<0.35 only)...")
        r4c = backtest_v4_hybrid(data, args.capital, capital_ratio=0.90,
                                 adx_th=25, sl_mult=2.5,
                                 prob_full=0.55, prob_half=0.35,
                                 label="V4 Light (0.55/0.35)")

        all_results = [r1, r3, r4a, r4b, r4c]

        print(f"\n{'='*100}")
        print(f"  V3 vs V4 HYBRID COMPARISON")
        print(f"{'='*100}")
        hdr = f"{'Strategy':<42} {'Return%':>9} {'Annual%':>9} {'Trades':>7} {'WinR%':>7} {'MaxDD%':>8} {'Sharpe':>7}"
        print(hdr)
        print("-" * 100)

        for r in all_results:
            years = len(r.equity_curve) / (24 * 365) if r.equity_curve else 1
            ann = ((r.equity_curve[-1] / r.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
            print(f"{r.name:<42} {r.total_return:>8.1f}% {ann:>8.1f}% {r.num_trades:>7d} {r.win_rate:>6.1f}% {r.max_drawdown:>7.1f}% {r.sharpe_ratio:>7.2f}")

        print(f"{'='*100}")
        return

    # === V3 aggressive sweep ===
    if args.v3:
        print("\nRunning V1 baseline...")
        r1 = backtest_v1(data, args.capital)

        # Equity level presets
        EQ_STRICT = [(0.05, 0.80), (0.10, 0.60), (0.15, 0.00)]      # Default: stop at -15%
        EQ_MODERATE = [(0.10, 0.70), (0.20, 0.40), (0.25, 0.00)]    # Stop at -25%
        EQ_RELAXED = [(0.15, 0.70), (0.25, 0.40), (0.30, 0.00)]     # Stop at -30%
        EQ_NONE = [(1.0, 1.0)]                                        # No scaling

        configs = [
            # (cap, adx, mtf, so, sl_mult, eq_levels, tag)
            (0.90, 25, False, False, 2.5, EQ_STRICT,   "strict"),
            (0.90, 25, False, False, 2.5, EQ_MODERATE,  "moderate"),
            (0.90, 25, False, False, 2.5, EQ_RELAXED,   "relaxed"),
            (0.90, 25, False, False, 2.5, EQ_NONE,      "no-eq"),
            (0.90, 25, True,  False, 2.5, EQ_MODERATE,  "MTF+mod"),
            (0.90, 25, True,  False, 2.5, EQ_RELAXED,   "MTF+rlx"),
            (0.90, 25, True,  False, 2.5, EQ_NONE,      "MTF+noEQ"),
            (0.90, 20, True,  False, 2.5, EQ_NONE,      "A20+MTF+noEQ"),
            (0.90, 25, False, False, 3.0, EQ_NONE,      "SL3+noEQ"),
            (0.90, 25, False, False, 2.0, EQ_NONE,      "SL2+noEQ"),
        ]

        v3_results = []
        for cap, adx_t, mtf, so, sl_m, eq_lvl, tag in configs:
            label = f"V3 C{int(cap*100)} ADX{adx_t} {'MTF' if mtf else '---'} SL×{sl_m} {tag}"
            print(f"  Running {label}...")
            r = backtest_v3(data, args.capital, capital_ratio=cap,
                           adx_th=adx_t, use_mtf=mtf, use_scaleout=so,
                           sl_mult=sl_m, eq_levels=eq_lvl, label=label)
            v3_results.append(r)

        print(f"\n{'='*95}")
        print(f"  V3 AGGRESSIVE OPTIMIZATION SWEEP")
        print(f"{'='*95}")
        hdr = f"{'Strategy':<40} {'Return%':>9} {'Annual%':>9} {'Trades':>7} {'WinR%':>7} {'MaxDD%':>8} {'Sharpe':>7}"
        print(hdr)
        print("-" * 95)

        years = len(r1.equity_curve) / (24 * 365) if r1.equity_curve else 1
        a1 = ((r1.equity_curve[-1] / r1.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        print(f"{'V1 Original (fixed SL2.5% Cap90%)':<40} {r1.total_return:>8.1f}% {a1:>8.1f}% {r1.num_trades:>7d} {r1.win_rate:>6.1f}% {r1.max_drawdown:>7.1f}% {r1.sharpe_ratio:>7.2f}")
        print("-" * 95)

        for r in v3_results:
            yrs = len(r.equity_curve) / (24 * 365) if r.equity_curve else 1
            ann = ((r.equity_curve[-1] / r.initial_capital) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
            print(f"{r.name:<40} {r.total_return:>8.1f}% {ann:>8.1f}% {r.num_trades:>7d} {r.win_rate:>6.1f}% {r.max_drawdown:>7.1f}% {r.sharpe_ratio:>7.2f}")

        print(f"{'='*95}")
        return

    # === Sweep mode: compare multiple capital ratios ===
    if args.sweep:
        ratios = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

        print("\nRunning V1 (Original 90%) backtest...")
        r1 = backtest_v1(data, args.capital)
        print(r1.summary())

        sweep_results = []
        for ratio in ratios:
            print(f"\nRunning V2 Cap{int(ratio*100)}%...")
            r = backtest_v2(data, args.capital, capital_ratio=ratio)
            sweep_results.append((ratio, r))

        # Summary table
        print(f"\n{'='*90}")
        print(f"  SWEEP COMPARISON: V1 (90% fixed SL) vs V2 variants (ATR-SL + filters)")
        print(f"{'='*90}")
        print(f"{'Strategy':<22} {'Return%':>9} {'Annual%':>9} {'Trades':>7} {'WinR%':>7} {'MaxDD%':>8} {'Sharpe':>7}")
        print("-" * 90)

        years = len(r1.equity_curve) / (24 * 365) if r1.equity_curve else 1
        annual1 = ((r1.equity_curve[-1] / r1.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        print(f"{'V1 Original 90%':<22} {r1.total_return:>8.1f}% {annual1:>8.1f}% {r1.num_trades:>7d} {r1.win_rate:>6.1f}% {r1.max_drawdown:>7.1f}% {r1.sharpe_ratio:>7.2f}")
        print("-" * 90)

        for ratio, r in sweep_results:
            years_r = len(r.equity_curve) / (24 * 365) if r.equity_curve else 1
            annual_r = ((r.equity_curve[-1] / r.initial_capital) ** (1 / years_r) - 1) * 100 if years_r > 0 else 0
            print(f"{'V2 Cap'+str(int(ratio*100))+'%':<22} {r.total_return:>8.1f}% {annual_r:>8.1f}% {r.num_trades:>7d} {r.win_rate:>6.1f}% {r.max_drawdown:>7.1f}% {r.sharpe_ratio:>7.2f}")

        print(f"{'='*90}")
        return

    # === Normal mode ===
    results = []

    if not args.v2_only:
        print("\nRunning V1 (Original) backtest...")
        r1 = backtest_v1(data, args.capital)
        results.append(r1)

    if not args.v1_only:
        print("\nRunning V2 (Optimized) backtest...")
        r2 = backtest_v2(data, args.capital)
        results.append(r2)

    for r in results:
        print(r.summary())

    if len(results) == 2:
        r1, r2 = results
        print(f"\n{'Metric':<20} {'V1':>12} {'V2':>12} {'Delta':>12}")
        print("-" * 58)
        print(f"{'Return %':<20} {r1.total_return:>11.1f}% {r2.total_return:>11.1f}% {r2.total_return-r1.total_return:>+11.1f}%")
        print(f"{'Trades':<20} {r1.num_trades:>12d} {r2.num_trades:>12d} {r2.num_trades-r1.num_trades:>+12d}")
        print(f"{'Win Rate %':<20} {r1.win_rate:>11.1f}% {r2.win_rate:>11.1f}% {r2.win_rate-r1.win_rate:>+11.1f}%")
        print(f"{'Max Drawdown %':<20} {r1.max_drawdown:>11.1f}% {r2.max_drawdown:>11.1f}% {r2.max_drawdown-r1.max_drawdown:>+11.1f}%")
        print(f"{'Sharpe Ratio':<20} {r1.sharpe_ratio:>12.2f} {r2.sharpe_ratio:>12.2f} {r2.sharpe_ratio-r1.sharpe_ratio:>+12.2f}")
        print(f"{'Avg Win %':<20} {r1.avg_win:>11.2f}% {r2.avg_win:>11.2f}% {r2.avg_win-r1.avg_win:>+11.2f}%")
        print(f"{'Avg Loss %':<20} {r1.avg_loss:>11.2f}% {r2.avg_loss:>11.2f}% {r2.avg_loss-r1.avg_loss:>+11.2f}%")


if __name__ == "__main__":
    main()
