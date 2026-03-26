#!/usr/bin/env python3
"""
USD/JPY FX戦略バックテスト分析システム
取引期間: 2017/04/01 - 2026/03/25
通貨ペア: USD/JPY
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os

# ============================================================
# [A] Constants & Configuration
# ============================================================
TICKER = "JPY=X"
START_DATE = "2017-04-01"
END_DATE = "2026-03-26"
LEVERAGE = 25
SPREAD_PIPS = 0.3
SPREAD_JPY = SPREAD_PIPS * 0.01  # 0.003 JPY per unit
INITIAL_CAPITAL = 1_000_000  # 100万円 for main backtest
SIMULATION_CAPITAL = 100_000  # 10万円 for 1-month sim
RISK_PER_TRADE = 0.02  # 2% risk per trade

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# [B] Data Loading & Preprocessing
# ============================================================
def load_data(ticker=TICKER, start=START_DATE, end=END_DATE):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.dropna(inplace=True)
    return df


# ============================================================
# [C] Indicator Calculation Functions
# ============================================================
def calc_sma(series, period):
    return series.rolling(window=period).mean()


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calc_bollinger(series, period=20, std_mult=2.0):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_donchian(df, period=20):
    upper = df['High'].rolling(window=period).max()
    lower = df['Low'].rolling(window=period).min()
    return upper, lower


def calc_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


# ============================================================
# [D] Strategy Classes
# ============================================================
class BaseStrategy(ABC):
    name: str
    description_ja: str

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return Series of signals: +1 (long), -1 (short), 0 (flat)"""
        pass


class SMACross(BaseStrategy):
    def __init__(self, fast=5, slow=20):
        self.fast = fast
        self.slow = slow
        self.name = f"SMA Cross ({fast}/{slow})"
        self.description_ja = (
            f"■ 売買ロジック:\n"
            f"  - {fast}日単純移動平均線(SMA)と{slow}日SMAのクロスオーバー戦略\n"
            f"  - {fast}日SMAが{slow}日SMAを上回った翌日の始値で買い(ロング)\n"
            f"  - {fast}日SMAが{slow}日SMAを下回った翌日の始値で売り(ショート)\n"
            f"  - 常にポジションを保有するリバーサル型(ドテン売買)\n"
            f"  - トレンドフォロー系の基本戦略"
        )

    def generate_signals(self, df):
        sma_fast = calc_sma(df['Close'], self.fast)
        sma_slow = calc_sma(df['Close'], self.slow)
        signal = pd.Series(0, index=df.index, dtype=float)
        signal[sma_fast > sma_slow] = 1
        signal[sma_fast <= sma_slow] = -1
        return signal


class EMACross(BaseStrategy):
    def __init__(self, fast=12, slow=26):
        self.fast = fast
        self.slow = slow
        self.name = f"EMA Cross ({fast}/{slow})"
        self.description_ja = (
            f"■ 売買ロジック:\n"
            f"  - {fast}日指数移動平均線(EMA)と{slow}日EMAのクロスオーバー戦略\n"
            f"  - {fast}日EMAが{slow}日EMAを上回った翌日の始値で買い\n"
            f"  - {fast}日EMAが{slow}日EMAを下回った翌日の始値で売り\n"
            f"  - EMAは直近の価格に重みを置くため、SMAより反応が早い\n"
            f"  - リバーサル型(ドテン売買)"
        )

    def generate_signals(self, df):
        ema_fast = calc_ema(df['Close'], self.fast)
        ema_slow = calc_ema(df['Close'], self.slow)
        signal = pd.Series(0, index=df.index, dtype=float)
        signal[ema_fast > ema_slow] = 1
        signal[ema_fast <= ema_slow] = -1
        return signal


class RSIStrategy(BaseStrategy):
    def __init__(self, period=14, overbought=70, oversold=30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.name = f"RSI ({period})"
        self.description_ja = (
            f"■ 売買ロジック:\n"
            f"  - RSI({period})を使った逆張り(ミーンリバージョン)戦略\n"
            f"  - RSIが{oversold}以下に低下 → 翌日始値で買い(売られすぎ)\n"
            f"  - RSIが{overbought}以上に上昇 → 翌日始値で売り(買われすぎ)\n"
            f"  - RSIが50を通過した時点で決済\n"
            f"  - レンジ相場で有効、トレンド相場では損失リスクあり"
        )

    def generate_signals(self, df):
        rsi = calc_rsi(df['Close'], self.period)
        signal = pd.Series(0, index=df.index, dtype=float)
        position = 0
        for i in range(len(df)):
            if rsi.iloc[i] < self.oversold and position <= 0:
                position = 1
            elif rsi.iloc[i] > self.overbought and position >= 0:
                position = -1
            elif position == 1 and rsi.iloc[i] > 50:
                position = 0
            elif position == -1 and rsi.iloc[i] < 50:
                position = 0
            signal.iloc[i] = position
        return signal


class BollingerMeanReversion(BaseStrategy):
    def __init__(self, period=20, std_mult=2.0):
        self.period = period
        self.std_mult = std_mult
        self.name = f"Bollinger MR ({period})"
        self.description_ja = (
            f"■ 売買ロジック:\n"
            f"  - ボリンジャーバンド({period}日, {std_mult}σ)のミーンリバージョン戦略\n"
            f"  - 価格が下側バンドを下回った翌日始値で買い(平均回帰を期待)\n"
            f"  - 価格が上側バンドを上回った翌日始値で売り\n"
            f"  - 中央バンド(移動平均線)到達で決済\n"
            f"  - レンジ相場・ボラティリティ回帰を狙う戦略"
        )

    def generate_signals(self, df):
        upper, mid, lower = calc_bollinger(df['Close'], self.period, self.std_mult)
        signal = pd.Series(0, index=df.index, dtype=float)
        position = 0
        for i in range(self.period, len(df)):
            close = df['Close'].iloc[i]
            if close < lower.iloc[i] and position <= 0:
                position = 1
            elif close > upper.iloc[i] and position >= 0:
                position = -1
            elif position == 1 and close >= mid.iloc[i]:
                position = 0
            elif position == -1 and close <= mid.iloc[i]:
                position = 0
            signal.iloc[i] = position
        return signal


class BollingerBreakout(BaseStrategy):
    def __init__(self, period=20, std_mult=2.0):
        self.period = period
        self.std_mult = std_mult
        self.name = f"Bollinger BO ({period})"
        self.description_ja = (
            f"■ 売買ロジック:\n"
            f"  - ボリンジャーバンド({period}日, {std_mult}σ)のブレイクアウト戦略\n"
            f"  - 価格が上側バンドを上抜けた翌日始値で買い(トレンド発生を期待)\n"
            f"  - 価格が下側バンドを下抜けた翌日始値で売り\n"
            f"  - 中央バンドに戻った時点で決済\n"
            f"  - ボラティリティ拡大・トレンド発生を狙う戦略"
        )

    def generate_signals(self, df):
        upper, mid, lower = calc_bollinger(df['Close'], self.period, self.std_mult)
        signal = pd.Series(0, index=df.index, dtype=float)
        position = 0
        for i in range(self.period, len(df)):
            close = df['Close'].iloc[i]
            if close > upper.iloc[i] and position <= 0:
                position = 1
            elif close < lower.iloc[i] and position >= 0:
                position = -1
            elif position == 1 and close <= mid.iloc[i]:
                position = 0
            elif position == -1 and close >= mid.iloc[i]:
                position = 0
            signal.iloc[i] = position
        return signal


class MACDStrategy(BaseStrategy):
    def __init__(self, fast=12, slow=26, signal_period=9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period
        self.name = f"MACD ({fast}/{slow}/{signal_period})"
        self.description_ja = (
            f"■ 売買ロジック:\n"
            f"  - MACD({fast},{slow},{signal_period})のクロスオーバー戦略\n"
            f"  - MACDライン(EMA{fast}-EMA{slow})がシグナルライン(MACD9日EMA)を上抜け → 買い\n"
            f"  - MACDラインがシグナルラインを下抜け → 売り\n"
            f"  - リバーサル型(常時ポジション保有)\n"
            f"  - トレンドの転換点を捉えるモメンタム系戦略"
        )

    def generate_signals(self, df):
        macd_line, signal_line, _ = calc_macd(df['Close'], self.fast, self.slow, self.signal_period)
        signal = pd.Series(0, index=df.index, dtype=float)
        signal[macd_line > signal_line] = 1
        signal[macd_line <= signal_line] = -1
        return signal


class DonchianBreakout(BaseStrategy):
    def __init__(self, entry_period=20, exit_period=10):
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.name = f"Donchian ({entry_period}/{exit_period})"
        self.description_ja = (
            f"■ 売買ロジック:\n"
            f"  - ドンチャンチャネル({entry_period}日)ブレイクアウト戦略\n"
            f"  - 価格が過去{entry_period}日の最高値を上抜け → 買い\n"
            f"  - 価格が過去{entry_period}日の最安値を下抜け → 売り\n"
            f"  - 買いポジションは過去{exit_period}日の最安値割れで決済\n"
            f"  - 売りポジションは過去{exit_period}日の最高値超えで決済\n"
            f"  - タートルズに由来するトレンドフォロー戦略"
        )

    def generate_signals(self, df):
        entry_upper, entry_lower = calc_donchian(df, self.entry_period)
        exit_upper = df['High'].rolling(window=self.exit_period).max()
        exit_lower = df['Low'].rolling(window=self.exit_period).min()
        signal = pd.Series(0, index=df.index, dtype=float)
        position = 0
        for i in range(self.entry_period, len(df)):
            close = df['Close'].iloc[i]
            if close > entry_upper.iloc[i-1] and position <= 0:
                position = 1
            elif close < entry_lower.iloc[i-1] and position >= 0:
                position = -1
            elif position == 1 and close < exit_lower.iloc[i-1]:
                position = 0
            elif position == -1 and close > exit_upper.iloc[i-1]:
                position = 0
            signal.iloc[i] = position
        return signal


# ============================================================
# [E] Backtesting Engine
# ============================================================
@dataclass
class Trade:
    entry_date: object
    exit_date: object = None
    entry_price: float = 0
    exit_price: float = 0
    direction: int = 0  # +1 long, -1 short
    units: float = 0
    pnl: float = 0


def run_backtest(df, strategy, initial_capital=INITIAL_CAPITAL, leverage=LEVERAGE):
    signals = strategy.generate_signals(df)
    # Shift signals by 1 to avoid look-ahead bias: trade on next day's Open
    positions = signals.shift(1).fillna(0)

    atr = calc_atr(df)

    equity = initial_capital
    cash = initial_capital
    trades = []
    equity_curve = []
    current_trade = None

    for i in range(1, len(df)):
        date = df.index[i]
        open_price = df['Open'].iloc[i]
        close_price = df['Close'].iloc[i]
        current_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else 1.0
        desired_pos = int(positions.iloc[i])
        current_pos = current_trade.direction if current_trade else 0

        # Position change needed
        if desired_pos != current_pos:
            # Close existing position
            if current_trade is not None:
                current_trade.exit_date = date
                current_trade.exit_price = open_price
                if current_trade.direction == 1:
                    current_trade.pnl = current_trade.units * (open_price - current_trade.entry_price) - current_trade.units * SPREAD_JPY
                else:
                    current_trade.pnl = current_trade.units * (current_trade.entry_price - open_price) - current_trade.units * SPREAD_JPY
                cash += current_trade.pnl + current_trade.units * current_trade.entry_price / leverage
                trades.append(current_trade)
                current_trade = None

            # Open new position
            if desired_pos != 0:
                # Position sizing: risk 2% of equity, stop at 2x ATR
                stop_distance = max(current_atr * 2, 0.01)
                risk_amount = equity * RISK_PER_TRADE
                units_by_risk = risk_amount / stop_distance
                # Cap by leverage
                max_units = (equity * leverage) / open_price
                units = min(units_by_risk, max_units)
                units = max(units, 0)

                margin_required = units * open_price / leverage
                if margin_required <= cash:
                    current_trade = Trade(
                        entry_date=date,
                        entry_price=open_price,
                        direction=desired_pos,
                        units=units
                    )
                    cash -= margin_required

        # Mark to market
        if current_trade is not None:
            if current_trade.direction == 1:
                unrealized = current_trade.units * (close_price - current_trade.entry_price)
            else:
                unrealized = current_trade.units * (current_trade.entry_price - close_price)
            equity = cash + current_trade.units * current_trade.entry_price / leverage + unrealized
        else:
            equity = cash

        equity_curve.append({'date': date, 'equity': equity})

    # Close any remaining position at the end
    if current_trade is not None:
        last_close = df['Close'].iloc[-1]
        current_trade.exit_date = df.index[-1]
        current_trade.exit_price = last_close
        if current_trade.direction == 1:
            current_trade.pnl = current_trade.units * (last_close - current_trade.entry_price) - current_trade.units * SPREAD_JPY
        else:
            current_trade.pnl = current_trade.units * (current_trade.entry_price - last_close) - current_trade.units * SPREAD_JPY
        trades.append(current_trade)

    equity_df = pd.DataFrame(equity_curve).set_index('date')
    return trades, equity_df


# ============================================================
# [F] Metrics Calculator
# ============================================================
def calculate_metrics(trades, equity_df, initial_capital):
    if len(trades) == 0:
        return {
            'total_return': 0, 'annual_return': 0, 'win_rate': 0,
            'profit_factor': 0, 'max_drawdown': 0, 'sharpe': 0,
            'num_trades': 0, 'avg_pnl': 0, 'avg_win_loss_ratio': 0,
            'max_consecutive_losses': 0, 'final_equity': initial_capital
        }

    final_equity = equity_df['equity'].iloc[-1]
    total_return = (final_equity / initial_capital - 1) * 100
    trading_days = len(equity_df)
    years = trading_days / 252
    annual_return = ((final_equity / initial_capital) ** (1 / max(years, 0.01)) - 1) * 100

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(trades) * 100

    total_profit = sum(wins) if wins else 0
    total_loss = abs(sum(losses)) if losses else 0
    profit_factor = total_profit / max(total_loss, 1) if total_loss > 0 else float('inf')

    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    avg_win_loss_ratio = avg_win / max(avg_loss, 1) if avg_loss > 0 else float('inf')

    # Max drawdown
    peak = equity_df['equity'].expanding().max()
    drawdown = (equity_df['equity'] - peak) / peak * 100
    max_drawdown = drawdown.min()

    # Sharpe ratio
    daily_returns = equity_df['equity'].pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for p in pnls:
        if p <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    avg_pnl = np.mean(pnls)

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'num_trades': len(trades),
        'avg_pnl': avg_pnl,
        'avg_win_loss_ratio': avg_win_loss_ratio,
        'max_consecutive_losses': max_consec,
        'final_equity': final_equity
    }


# ============================================================
# [G] Report Generator
# ============================================================
def generate_report(all_results, df):
    lines = []
    lines.append("=" * 70)
    lines.append("  USD/JPY FX戦略バックテスト分析レポート")
    lines.append(f"  取引期間: {df.index[0].strftime('%Y/%m/%d')} - {df.index[-1].strftime('%Y/%m/%d')}")
    lines.append(f"  初期資金: ¥{INITIAL_CAPITAL:,.0f}  レバレッジ: {LEVERAGE}倍  スプレッド: {SPREAD_PIPS}pips")
    lines.append("=" * 70)
    lines.append("")

    # Sort by Sharpe ratio
    sorted_results = sorted(all_results, key=lambda x: x['metrics']['sharpe'], reverse=True)

    lines.append("■ 全戦略パフォーマンス比較 (シャープレシオ順):")
    lines.append("-" * 70)
    header = f"{'戦略名':<25} {'総リターン':>8} {'年率':>7} {'勝率':>6} {'PF':>6} {'最大DD':>8} {'シャープ':>7} {'取引数':>5}"
    lines.append(header)
    lines.append("-" * 70)

    for r in sorted_results:
        m = r['metrics']
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "∞"
        line = (
            f"{r['strategy'].name:<25} "
            f"{m['total_return']:>7.1f}% "
            f"{m['annual_return']:>6.1f}% "
            f"{m['win_rate']:>5.1f}% "
            f"{pf_str:>6} "
            f"{m['max_drawdown']:>7.1f}% "
            f"{m['sharpe']:>7.2f} "
            f"{m['num_trades']:>5}"
        )
        lines.append(line)

    lines.append("-" * 70)
    lines.append("")

    # Best strategy detail
    best = sorted_results[0]
    bm = best['metrics']
    lines.append("=" * 70)
    lines.append(f"  【最優秀戦略】 {best['strategy'].name}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(best['strategy'].description_ja)
    lines.append("")
    lines.append("■ 詳細パフォーマンス:")
    lines.append(f"  総リターン:           {bm['total_return']:.2f}%")
    lines.append(f"  年率リターン:         {bm['annual_return']:.2f}%")
    lines.append(f"  勝率:                 {bm['win_rate']:.1f}%")
    lines.append(f"  プロフィットファクター: {bm['profit_factor']:.2f}")
    lines.append(f"  損益比 (平均勝ち/負け): {bm['avg_win_loss_ratio']:.2f}")
    lines.append(f"  最大ドローダウン:      {bm['max_drawdown']:.2f}%")
    lines.append(f"  シャープレシオ:        {bm['sharpe']:.2f}")
    lines.append(f"  取引回数:             {bm['num_trades']}回")
    lines.append(f"  平均損益/トレード:     ¥{bm['avg_pnl']:,.0f}")
    lines.append(f"  最大連敗:             {bm['max_consecutive_losses']}回")
    lines.append(f"  最終資産:             ¥{bm['final_equity']:,.0f} (初期: ¥{INITIAL_CAPITAL:,.0f})")
    lines.append("")

    # Monthly return for best strategy
    equity_df = best['equity_df']
    monthly = equity_df['equity'].resample('ME').last()
    lines.append("■ 月次リターン (直近12ヶ月):")
    lines.append("-" * 40)
    prev = None
    for date, eq in monthly.tail(12).items():
        if prev is not None:
            ret = (eq / prev - 1) * 100
            lines.append(f"  {date.strftime('%Y/%m')}: {ret:>+7.2f}%  (¥{eq:,.0f})")
        prev = eq
    lines.append("")

    # 10万円 1ヶ月シミュレーション
    lines.append("=" * 70)
    lines.append("  【10万円 × 1ヶ月シミュレーション】")
    lines.append("=" * 70)

    # Use last ~22 trading days
    last_month_df = df.tail(22).copy()
    sim_trades, sim_equity = run_backtest(last_month_df, best['strategy'],
                                          initial_capital=SIMULATION_CAPITAL, leverage=LEVERAGE)
    if len(sim_equity) > 0:
        sim_final = sim_equity['equity'].iloc[-1]
        sim_return = (sim_final / SIMULATION_CAPITAL - 1) * 100
        sim_num_trades = len(sim_trades)
        sim_pnl = sim_final - SIMULATION_CAPITAL

        lines.append(f"  期間:     {last_month_df.index[0].strftime('%Y/%m/%d')} - {last_month_df.index[-1].strftime('%Y/%m/%d')}")
        lines.append(f"  開始資金: ¥{SIMULATION_CAPITAL:,.0f}")
        lines.append(f"  終了資金: ¥{sim_final:,.0f}")
        lines.append(f"  損益:     ¥{sim_pnl:>+,.0f}")
        lines.append(f"  リターン: {sim_return:>+.2f}%")
        lines.append(f"  取引回数: {sim_num_trades}回")
    else:
        lines.append("  シミュレーションデータ不足")
    lines.append("")

    # Average monthly return from full backtest
    monthly_returns = equity_df['equity'].resample('ME').last().pct_change().dropna()
    avg_monthly = monthly_returns.mean() * 100
    lines.append("■ 参考: 過去全期間の平均月次リターン:")
    lines.append(f"  平均月次リターン: {avg_monthly:>+.2f}%")
    if avg_monthly > 0:
        projected = SIMULATION_CAPITAL * (1 + avg_monthly / 100)
        lines.append(f"  10万円 × 平均月次リターン → ¥{projected:,.0f}")
    lines.append("")

    # Disclaimer
    lines.append("=" * 70)
    lines.append("【注意事項】")
    lines.append("  - 本分析は過去データに基づくバックテストであり、将来の利益を保証しません")
    lines.append("  - 最良戦略の選択自体にバイアス(過剰適合)のリスクがあります")
    lines.append("  - 実際の取引ではスリッページ、約定拒否、システム障害等のリスクがあります")
    lines.append("  - FX取引は元本を超える損失が発生する可能性があります")
    lines.append("=" * 70)

    report = "\n".join(lines)
    return report, sorted_results


# ============================================================
# [H] Chart Generator
# ============================================================
def generate_charts(all_results, df):
    # Try to set Japanese font
    try:
        plt.rcParams['font.family'] = 'sans-serif'
    except Exception:
        pass

    sorted_results = sorted(all_results, key=lambda x: x['metrics']['sharpe'], reverse=True)

    # Chart 1: Equity Curve Comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    for r in sorted_results:
        eq = r['equity_df']['equity']
        normalized = eq / eq.iloc[0] * 100
        label = f"{r['strategy'].name} (Sharpe: {r['metrics']['sharpe']:.2f})"
        ax.plot(normalized.index, normalized.values, label=label, linewidth=1.2)

    ax.set_title('USD/JPY Strategy Comparison - Equity Curves (2017-2026)', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Equity (Start=100)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.tight_layout()
    path1 = os.path.join(SCRIPT_DIR, 'fx_equity_curves.png')
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  Chart saved: {path1}")

    # Chart 2: Best Strategy Detail
    best = sorted_results[0]
    eq = best['equity_df']['equity']

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1.5, 1]})

    # Price chart
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], color='#333333', linewidth=0.8, label='USD/JPY Close')
    # Add trade markers
    for t in best['trades'][:200]:  # limit markers for readability
        if t.direction == 1:
            ax1.axvline(t.entry_date, color='green', alpha=0.1, linewidth=0.5)
        else:
            ax1.axvline(t.entry_date, color='red', alpha=0.1, linewidth=0.5)
    ax1.set_title(f'Best Strategy: {best["strategy"].name} - USD/JPY Price', fontsize=12)
    ax1.set_ylabel('USD/JPY')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Equity curve with drawdown
    ax2 = axes[1]
    ax2.plot(eq.index, eq.values, color='#1f77b4', linewidth=1)
    peak = eq.expanding().max()
    ax2.fill_between(eq.index, eq.values, peak.values, alpha=0.3, color='red', label='Drawdown')
    ax2.set_title('Equity Curve & Drawdown', fontsize=12)
    ax2.set_ylabel('Equity (JPY)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    # Drawdown percentage
    ax3 = axes[2]
    dd_pct = (eq - peak) / peak * 100
    ax3.fill_between(dd_pct.index, dd_pct.values, 0, alpha=0.5, color='red')
    ax3.set_title('Drawdown (%)', fontsize=12)
    ax3.set_ylabel('Drawdown %')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    path2 = os.path.join(SCRIPT_DIR, 'fx_best_strategy.png')
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Chart saved: {path2}")

    return path1, path2


# ============================================================
# [I] Main Execution
# ============================================================
def main():
    print("USD/JPY FX戦略バックテスト分析システム")
    print("データ読み込み中...")
    df = load_data()
    print(f"  データ取得完了: {len(df)}日分 ({df.index[0].strftime('%Y/%m/%d')} - {df.index[-1].strftime('%Y/%m/%d')})")

    strategies = [
        SMACross(5, 20),
        SMACross(20, 100),
        EMACross(12, 26),
        RSIStrategy(14, 70, 30),
        BollingerMeanReversion(20, 2.0),
        BollingerBreakout(20, 2.0),
        MACDStrategy(12, 26, 9),
        DonchianBreakout(20, 10),
    ]

    all_results = []
    print(f"\n{len(strategies)}戦略をバックテスト中...")
    for strat in strategies:
        print(f"  テスト中: {strat.name}...", end=" ")
        trades, equity_df = run_backtest(df, strat)
        metrics = calculate_metrics(trades, equity_df, INITIAL_CAPITAL)
        all_results.append({
            'strategy': strat,
            'trades': trades,
            'equity_df': equity_df,
            'metrics': metrics
        })
        print(f"完了 (取引数: {metrics['num_trades']}, リターン: {metrics['total_return']:.1f}%)")

    print("\nレポート生成中...")
    report, sorted_results = generate_report(all_results, df)

    print("\nチャート生成中...")
    chart1, chart2 = generate_charts(all_results, df)

    # Save report to file
    report_path = os.path.join(SCRIPT_DIR, 'fx_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    print("\n" + report)


if __name__ == "__main__":
    main()
