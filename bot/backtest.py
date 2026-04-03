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

from indicators import atr, rsi, adx, momentum, realized_volatility
from risk_manager import (
    VolatilityRegime, PositionSizer, DynamicStopLoss,
    ScaleOutManager, EntryFilter, AdaptiveThreshold,
)

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

def backtest_v2(data, initial_capital=1_000_000):
    """Optimized strategy with all Phase 1-3 improvements."""

    result = BacktestResult(
        name="V2 Optimized (Adaptive TH, ATR-SL, Kelly20%, ADX+RSI filter)",
        initial_capital=initial_capital,
    )
    closes = data["close"]
    highs = data["high"]
    lows = data["low"]
    volumes = data["volume"]
    n = len(closes)

    LB_L = 72   # 72h long lookback
    LB_S = 24   # 24h short lookback (confirmation)

    # Risk modules
    vol_regime = VolatilityRegime()
    pos_sizer = PositionSizer(capital_ratio=0.20)
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BTC/JPY Momentum Backtest")
    parser.add_argument("--days", type=int, default=365 * 3, help="Days of data")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital JPY")
    parser.add_argument("--v1-only", action="store_true")
    parser.add_argument("--v2-only", action="store_true")
    args = parser.parse_args()

    data = fetch_hourly_data(args.days)

    if len(data["close"]) < 100:
        print("ERROR: Not enough data fetched.")
        sys.exit(1)

    results = []

    if not args.v2_only:
        print("\nRunning V1 (Original) backtest...")
        r1 = backtest_v1(data, args.capital)
        results.append(r1)

    if not args.v1_only:
        print("\nRunning V2 (Optimized) backtest...")
        r2 = backtest_v2(data, args.capital)
        results.append(r2)

    # Print results
    for r in results:
        print(r.summary())

    # Comparison table
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
