#!/usr/bin/env python3
"""
USD/JPY NYクローズ・ミーンリバージョン戦略 バックテスト
取引期間: 2011/01 - 2026/03 (約15年)
通貨ペア: USD/JPY

戦略概要:
  - NYクローズ(前日終値)を基準レートとする
  - 東京9時(当日始値)で基準から±30pips以上乖離していればエントリー
  - NYクローズ方向へ逆張り
  - TP1: 中間地点で半分決済 / TP2: NYクローズ到達で残り決済
  - SL: エントリーから30pips逆方向
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
from dataclasses import dataclass
from typing import List
import os

# ============================================================
# Constants
# ============================================================
TICKER = "JPY=X"
START_DATE = "2011-01-01"
END_DATE = "2026-03-26"
LEVERAGE = 25
SPREAD_PIPS = 0.3
SPREAD_JPY = SPREAD_PIPS * 0.01  # 0.003 JPY per unit
ENTRY_THRESHOLD_JPY = 0.30       # 30 pips
SL_JPY = 0.30                    # 30 pips stop loss
INITIAL_CAPITAL = 1_000_000
SIMULATION_CAPITAL = 100_000
RISK_PER_TRADE = 0.02

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Data Loading
# ============================================================
def load_data(start=START_DATE, end=END_DATE):
    df = yf.download(TICKER, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.dropna(inplace=True)
    df['NY_Close'] = df['Close'].shift(1)
    df.dropna(inplace=True)
    return df


# ============================================================
# Trade Data Structure
# ============================================================
@dataclass
class NYCloseTrade:
    date: object
    ny_close: float
    entry_price: float
    direction: int          # +1 LONG, -1 SHORT
    deviation_pips: float
    units: float
    tp1_level: float
    tp2_level: float
    sl_level: float
    outcome: str            # 'SL', 'TP1+TP2', 'TP1+SL', 'TP1_only', 'NO_HIT'
    pnl_tp1: float = 0.0
    pnl_tp2: float = 0.0
    pnl_total: float = 0.0


# ============================================================
# Outcome Resolution
# ============================================================
def resolve_outcome(direction, open_price, high, low, tp1_level, tp2_level, sl_level):
    if direction == 1:  # LONG
        sl_hit = low <= sl_level
        tp1_hit = high >= tp1_level
        tp2_hit = high >= tp2_level
        dist_to_sl = open_price - sl_level
        dist_to_tp1 = tp1_level - open_price
    else:  # SHORT
        sl_hit = high >= sl_level
        tp1_hit = low <= tp1_level
        tp2_hit = low <= tp2_level
        dist_to_sl = sl_level - open_price
        dist_to_tp1 = open_price - tp1_level

    if not sl_hit and not tp1_hit:
        return 'NO_HIT'

    if sl_hit and not tp1_hit:
        return 'SL'

    if tp1_hit and not sl_hit:
        if tp2_hit:
            return 'TP1+TP2'
        else:
            return 'TP1_only'

    # Both sl_hit and tp1_hit
    if dist_to_sl < dist_to_tp1:
        return 'SL'
    else:
        if tp2_hit:
            return 'TP1+TP2'
        else:
            return 'TP1+SL'


# ============================================================
# Backtest Engine
# ============================================================
def run_backtest(df, initial_capital=INITIAL_CAPITAL):
    equity = initial_capital
    trades: List[NYCloseTrade] = []
    equity_curve = []

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        ny_close = row['NY_Close']
        open_price = row['Open']
        high = row['High']
        low = row['Low']
        close = row['Close']

        deviation = open_price - ny_close
        deviation_abs = abs(deviation)

        # No trade if deviation < 30 pips
        if deviation_abs < ENTRY_THRESHOLD_JPY:
            equity_curve.append({'date': date, 'equity': equity})
            continue

        # Determine direction
        if deviation > 0:
            direction = -1  # SHORT: price above NY Close, expect reversion down
        else:
            direction = 1   # LONG: price below NY Close, expect reversion up

        # Calculate levels
        if direction == 1:  # LONG
            effective_entry = open_price + SPREAD_JPY / 2
            tp1_level = effective_entry + (ny_close - effective_entry) / 2
            tp2_level = ny_close
            sl_level = effective_entry - SL_JPY
        else:  # SHORT
            effective_entry = open_price - SPREAD_JPY / 2
            tp1_level = effective_entry - (effective_entry - ny_close) / 2
            tp2_level = ny_close
            sl_level = effective_entry + SL_JPY

        # Position sizing: 2% risk / 30 pip stop
        risk_amount = equity * RISK_PER_TRADE
        units = risk_amount / SL_JPY
        max_units = (equity * LEVERAGE) / open_price
        units = min(units, max_units)

        if units <= 0:
            equity_curve.append({'date': date, 'equity': equity})
            continue

        half_units = units / 2

        # Resolve outcome
        outcome = resolve_outcome(direction, open_price, high, low,
                                  tp1_level, tp2_level, sl_level)

        # Calculate PnL
        pnl_tp1 = 0.0
        pnl_tp2 = 0.0

        if outcome == 'SL':
            pnl_tp1 = 0
            pnl_tp2 = 0
            pnl_total = -units * SL_JPY

        elif outcome == 'TP1+TP2':
            tp1_profit = abs(tp1_level - effective_entry)
            tp2_profit = abs(tp2_level - effective_entry)
            pnl_tp1 = half_units * tp1_profit
            pnl_tp2 = half_units * tp2_profit
            pnl_total = pnl_tp1 + pnl_tp2

        elif outcome == 'TP1+SL':
            tp1_profit = abs(tp1_level - effective_entry)
            pnl_tp1 = half_units * tp1_profit
            pnl_tp2 = -half_units * SL_JPY
            pnl_total = pnl_tp1 + pnl_tp2

        elif outcome == 'TP1_only':
            tp1_profit = abs(tp1_level - effective_entry)
            pnl_tp1 = half_units * tp1_profit
            if direction == 1:
                pnl_tp2 = half_units * (close - effective_entry)
            else:
                pnl_tp2 = half_units * (effective_entry - close)
            pnl_total = pnl_tp1 + pnl_tp2

        elif outcome == 'NO_HIT':
            if direction == 1:
                pnl_total = units * (close - effective_entry)
            else:
                pnl_total = units * (effective_entry - close)
            pnl_tp1 = pnl_total / 2
            pnl_tp2 = pnl_total / 2

        # Deduct spread cost (round-trip)
        spread_cost = units * SPREAD_JPY
        pnl_total -= spread_cost

        trade = NYCloseTrade(
            date=date,
            ny_close=ny_close,
            entry_price=open_price,
            direction=direction,
            deviation_pips=deviation_abs / 0.01,
            units=units,
            tp1_level=tp1_level,
            tp2_level=tp2_level,
            sl_level=sl_level,
            outcome=outcome,
            pnl_tp1=pnl_tp1,
            pnl_tp2=pnl_tp2,
            pnl_total=pnl_total,
        )
        trades.append(trade)
        equity += pnl_total
        equity = max(equity, 0)
        equity_curve.append({'date': date, 'equity': equity})

    equity_df = pd.DataFrame(equity_curve).set_index('date')
    return trades, equity_df


# ============================================================
# Metrics Calculator
# ============================================================
def calculate_metrics(trades, equity_df, initial_capital):
    if not trades:
        return {k: 0 for k in [
            'total_return', 'annual_return', 'win_rate', 'profit_factor',
            'max_drawdown', 'sharpe', 'num_trades', 'avg_pnl',
            'avg_win_loss_ratio', 'max_consecutive_losses', 'final_equity'
        ]}

    final_equity = equity_df['equity'].iloc[-1]
    total_return = (final_equity / initial_capital - 1) * 100
    years = len(equity_df) / 252
    annual_return = ((final_equity / initial_capital) ** (1 / max(years, 0.01)) - 1) * 100

    pnls = [t.pnl_total for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(trades) * 100

    total_profit = sum(wins) if wins else 0
    total_loss = abs(sum(losses)) if losses else 0
    profit_factor = total_profit / max(total_loss, 1)

    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 1
    avg_win_loss_ratio = avg_win / max(avg_loss, 1)

    peak = equity_df['equity'].expanding().max()
    drawdown = (equity_df['equity'] - peak) / peak * 100
    max_drawdown = drawdown.min()

    daily_returns = equity_df['equity'].pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    max_consec = 0
    current_consec = 0
    for p in pnls:
        if p <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'num_trades': len(trades),
        'avg_pnl': np.mean(pnls),
        'avg_win_loss_ratio': avg_win_loss_ratio,
        'max_consecutive_losses': max_consec,
        'final_equity': final_equity,
    }


# ============================================================
# Report Generator
# ============================================================
def generate_report(trades, equity_df, metrics, sim_trades, sim_equity, df):
    m = metrics
    lines = []
    lines.append("=" * 70)
    lines.append("  USD/JPY NYクローズ・ミーンリバージョン戦略 バックテストレポート")
    lines.append(f"  取引期間: {df.index[0].strftime('%Y/%m/%d')} - {df.index[-1].strftime('%Y/%m/%d')}")
    lines.append(f"  初期資金: ¥{INITIAL_CAPITAL:,.0f}  レバレッジ: {LEVERAGE}倍  スプレッド: {SPREAD_PIPS}pips")
    lines.append("=" * 70)
    lines.append("")

    lines.append("■ 売買ロジック:")
    lines.append("  1. 毎日、NYクローズ(前日終値)を基準レートに設定")
    lines.append("  2. 日本時間9:00(当日始値)に基準レートからの乖離を判定")
    lines.append("  3. ±30pips以上の乖離があればエントリー")
    lines.append("     → 基準より上に乖離 → ショート(下落を期待)")
    lines.append("     → 基準より下に乖離 → ロング(上昇を期待)")
    lines.append("  4. 利確条件:")
    lines.append("     → TP1: エントリー〜NYクローズの中間到達で半分決済")
    lines.append("     → TP2: NYクローズ到達で残り半分決済")
    lines.append("  5. 損切り: エントリーから30pips逆方向で全決済")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  【パフォーマンス】")
    lines.append("=" * 70)
    lines.append(f"  総リターン:             {m['total_return']:>+.2f}%")
    lines.append(f"  年率リターン:           {m['annual_return']:>+.2f}%")
    lines.append(f"  勝率:                   {m['win_rate']:.1f}%")
    lines.append(f"  プロフィットファクター:  {m['profit_factor']:.2f}")
    lines.append(f"  損益比 (平均勝ち/負け):  {m['avg_win_loss_ratio']:.2f}")
    lines.append(f"  最大ドローダウン:        {m['max_drawdown']:.2f}%")
    lines.append(f"  シャープレシオ:          {m['sharpe']:.2f}")
    lines.append(f"  取引回数:               {m['num_trades']}回 (全{len(df)}営業日中)")
    lines.append(f"  取引頻度:               {m['num_trades']/len(df)*100:.1f}% の日にエントリー")
    lines.append(f"  平均損益/トレード:       ¥{m['avg_pnl']:>+,.0f}")
    lines.append(f"  最大連敗:               {m['max_consecutive_losses']}回")
    lines.append(f"  最終資産:               ¥{m['final_equity']:,.0f} (初期: ¥{INITIAL_CAPITAL:,.0f})")
    lines.append("")

    # Outcome distribution
    outcomes = {}
    for t in trades:
        outcomes[t.outcome] = outcomes.get(t.outcome, 0) + 1
    total_trades = len(trades)

    lines.append("■ トレード結果分布:")
    lines.append("-" * 50)
    outcome_names = {
        'TP1+TP2': 'TP1+TP2 (両利確)',
        'TP1_only': 'TP1のみ (残り終値決済)',
        'TP1+SL': 'TP1利確後SL (分割損益)',
        'SL': 'ストップロス (全損切り)',
        'NO_HIT': 'TP/SL未到達 (終値決済)',
    }
    for key in ['TP1+TP2', 'TP1_only', 'TP1+SL', 'SL', 'NO_HIT']:
        count = outcomes.get(key, 0)
        pct = count / total_trades * 100 if total_trades > 0 else 0
        label = outcome_names.get(key, key)
        # Avg PnL per outcome
        outcome_pnls = [t.pnl_total for t in trades if t.outcome == key]
        avg_pnl = np.mean(outcome_pnls) if outcome_pnls else 0
        lines.append(f"  {label:<30} {count:>4}回 ({pct:>5.1f}%)  平均損益: ¥{avg_pnl:>+,.0f}")
    lines.append("")

    # Direction breakdown
    long_trades = [t for t in trades if t.direction == 1]
    short_trades = [t for t in trades if t.direction == -1]
    long_pnl = sum(t.pnl_total for t in long_trades)
    short_pnl = sum(t.pnl_total for t in short_trades)
    lines.append("■ 方向別成績:")
    lines.append(f"  ロング: {len(long_trades)}回  合計損益: ¥{long_pnl:>+,.0f}")
    lines.append(f"  ショート: {len(short_trades)}回  合計損益: ¥{short_pnl:>+,.0f}")
    lines.append("")

    # Yearly breakdown
    lines.append("■ 年次別成績:")
    lines.append("-" * 60)
    lines.append(f"  {'年':<6} {'取引数':>6} {'勝率':>7} {'損益':>12} {'年末資産':>14}")
    lines.append("-" * 60)

    yearly_equity = equity_df['equity'].resample('YE').last()
    prev_eq = INITIAL_CAPITAL
    for year in sorted(set(t.date.year for t in trades)):
        year_trades = [t for t in trades if t.date.year == year]
        year_wins = [t for t in year_trades if t.pnl_total > 0]
        year_pnl = sum(t.pnl_total for t in year_trades)
        wr = len(year_wins) / len(year_trades) * 100 if year_trades else 0
        eq_end = yearly_equity.get(pd.Timestamp(f'{year}-12-31'), None)
        if eq_end is None:
            matching = [v for k, v in yearly_equity.items() if k.year == year]
            eq_end = matching[-1] if matching else prev_eq + year_pnl
        lines.append(f"  {year:<6} {len(year_trades):>6} {wr:>6.1f}% ¥{year_pnl:>+11,.0f} ¥{eq_end:>13,.0f}")
        prev_eq = eq_end
    lines.append("-" * 60)
    lines.append("")

    # Monthly returns (last 12 months)
    lines.append("■ 月次リターン (直近12ヶ月):")
    lines.append("-" * 40)
    monthly = equity_df['equity'].resample('ME').last()
    prev = None
    for date, eq in monthly.tail(13).items():
        if prev is not None:
            ret = (eq / prev - 1) * 100
            lines.append(f"  {date.strftime('%Y/%m')}: {ret:>+7.2f}%  (¥{eq:,.0f})")
        prev = eq
    lines.append("")

    # 10万円 1ヶ月シミュレーション
    lines.append("=" * 70)
    lines.append("  【10万円 × 1ヶ月シミュレーション】")
    lines.append("=" * 70)
    if len(sim_equity) > 0 and len(sim_trades) > 0:
        sim_final = sim_equity['equity'].iloc[-1]
        sim_return = (sim_final / SIMULATION_CAPITAL - 1) * 100
        sim_pnl = sim_final - SIMULATION_CAPITAL
        lines.append(f"  期間:     {sim_equity.index[0].strftime('%Y/%m/%d')} - {sim_equity.index[-1].strftime('%Y/%m/%d')}")
        lines.append(f"  開始資金: ¥{SIMULATION_CAPITAL:,.0f}")
        lines.append(f"  終了資金: ¥{sim_final:,.0f}")
        lines.append(f"  損益:     ¥{sim_pnl:>+,.0f}")
        lines.append(f"  リターン: {sim_return:>+.2f}%")
        lines.append(f"  取引回数: {len(sim_trades)}回")
        for t in sim_trades:
            dir_str = "ロング" if t.direction == 1 else "ショート"
            lines.append(f"    {t.date.strftime('%m/%d')} {dir_str} 乖離{t.deviation_pips:.0f}pips → {t.outcome} ¥{t.pnl_total:>+,.0f}")
    elif len(sim_equity) > 0:
        lines.append(f"  期間:     {sim_equity.index[0].strftime('%Y/%m/%d')} - {sim_equity.index[-1].strftime('%Y/%m/%d')}")
        lines.append(f"  開始資金: ¥{SIMULATION_CAPITAL:,.0f}")
        lines.append(f"  終了資金: ¥{SIMULATION_CAPITAL:,.0f}")
        lines.append(f"  エントリー条件を満たす日がありませんでした (乖離<30pips)")
    else:
        lines.append("  シミュレーションデータ不足")
    lines.append("")

    # Average monthly return projection
    monthly_returns = equity_df['equity'].resample('ME').last().pct_change().dropna()
    avg_monthly = monthly_returns.mean() * 100
    lines.append("■ 参考: 過去全期間の平均月次リターン:")
    lines.append(f"  平均月次リターン: {avg_monthly:>+.2f}%")
    if avg_monthly > 0:
        projected = SIMULATION_CAPITAL * (1 + avg_monthly / 100)
        lines.append(f"  10万円 × 平均月次リターン → ¥{projected:,.0f}")
    else:
        projected = SIMULATION_CAPITAL * (1 + avg_monthly / 100)
        lines.append(f"  10万円 × 平均月次リターン → ¥{projected:,.0f} (マイナス)")
    lines.append("")

    # Average deviation at entry
    avg_dev = np.mean([t.deviation_pips for t in trades])
    lines.append(f"■ エントリー時の平均乖離: {avg_dev:.1f} pips")
    lines.append("")

    lines.append("=" * 70)
    lines.append("【注意事項】")
    lines.append("  - 本分析は日足OHLC(始値/高値/安値/終値)での近似バックテストです")
    lines.append("  - 9:00 JST = 当日始値、NYクローズ = 前日終値として近似しています")
    lines.append("  - 同一日にTP/SLが両方到達可能な場合は保守的に判定しています")
    lines.append("  - 過去データに基づく結果であり、将来の利益を保証しません")
    lines.append("  - 実際にはスリッページ、約定遅延等で結果が異なります")
    lines.append("  - FX取引は元本を超える損失が発生する可能性があります")
    lines.append("=" * 70)

    return "\n".join(lines)


# ============================================================
# Chart Generator
# ============================================================
def generate_charts(trades, equity_df, df):
    # Chart 1: Equity Curve with Drawdown
    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             gridspec_kw={'height_ratios': [2, 1.5, 1]})

    # Price chart with trade markers
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], color='#333', linewidth=0.7, label='USD/JPY')
    for t in trades:
        color = 'green' if t.pnl_total > 0 else 'red'
        marker = '^' if t.direction == 1 else 'v'
        ax1.scatter(t.date, t.entry_price, color=color, marker=marker, s=8, alpha=0.6)
    ax1.set_title('USD/JPY - NY Close Mean Reversion Strategy (2011-2026)', fontsize=12)
    ax1.set_ylabel('USD/JPY')
    ax1.grid(True, alpha=0.3)

    # Equity curve
    ax2 = axes[1]
    eq = equity_df['equity']
    ax2.plot(eq.index, eq.values, color='#1f77b4', linewidth=1)
    peak = eq.expanding().max()
    ax2.fill_between(eq.index, eq.values, peak.values, alpha=0.3, color='red')
    ax2.set_title('Equity Curve & Drawdown', fontsize=12)
    ax2.set_ylabel('Equity (JPY)')
    ax2.grid(True, alpha=0.3)

    # Drawdown %
    ax3 = axes[2]
    dd_pct = (eq - peak) / peak * 100
    ax3.fill_between(dd_pct.index, dd_pct.values, 0, alpha=0.5, color='red')
    ax3.set_title('Drawdown (%)', fontsize=12)
    ax3.set_ylabel('DD %')
    ax3.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    path1 = os.path.join(SCRIPT_DIR, 'fx_ny_close_equity.png')
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  Chart saved: {path1}")

    # Chart 2: Trade outcome distribution & monthly PnL
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Outcome bar chart
    ax1 = axes[0]
    outcome_counts = {}
    for t in trades:
        outcome_counts[t.outcome] = outcome_counts.get(t.outcome, 0) + 1
    labels = list(outcome_counts.keys())
    counts = list(outcome_counts.values())
    colors = {'TP1+TP2': '#2ecc71', 'TP1_only': '#27ae60', 'TP1+SL': '#f39c12',
              'SL': '#e74c3c', 'NO_HIT': '#95a5a6'}
    bar_colors = [colors.get(k, '#999') for k in labels]
    ax1.bar(labels, counts, color=bar_colors)
    ax1.set_title('Trade Outcome Distribution', fontsize=12)
    ax1.set_ylabel('Count')
    for i, (l, c) in enumerate(zip(labels, counts)):
        ax1.text(i, c + 1, str(c), ha='center', fontsize=10)

    # Monthly PnL bar chart
    ax2 = axes[1]
    trade_df = pd.DataFrame([{'date': t.date, 'pnl': t.pnl_total} for t in trades])
    trade_df['date'] = pd.to_datetime(trade_df['date'])
    trade_df.set_index('date', inplace=True)
    monthly_pnl = trade_df['pnl'].resample('ME').sum()
    bar_colors2 = ['#2ecc71' if v >= 0 else '#e74c3c' for v in monthly_pnl.values]
    ax2.bar(monthly_pnl.index, monthly_pnl.values, width=25, color=bar_colors2, alpha=0.8)
    ax2.set_title('Monthly P&L', fontsize=12)
    ax2.set_ylabel('P&L (JPY)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    path2 = os.path.join(SCRIPT_DIR, 'fx_ny_close_trades.png')
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Chart saved: {path2}")

    return path1, path2


# ============================================================
# Main Execution
# ============================================================
def main():
    print("USD/JPY NYクローズ・ミーンリバージョン戦略 バックテスト")
    print("=" * 50)
    print("データ読み込み中...")
    df = load_data()
    print(f"  データ: {len(df)}日分 ({df.index[0].strftime('%Y/%m/%d')} - {df.index[-1].strftime('%Y/%m/%d')})")

    # Check how many days meet the entry condition
    deviations = abs(df['Open'] - df['NY_Close'])
    entry_days = (deviations >= ENTRY_THRESHOLD_JPY).sum()
    print(f"  エントリー条件(±30pips)を満たす日: {entry_days}日 ({entry_days/len(df)*100:.1f}%)")

    print("\nバックテスト実行中 (初期資金: ¥1,000,000)...")
    trades, equity_df = run_backtest(df, INITIAL_CAPITAL)
    metrics = calculate_metrics(trades, equity_df, INITIAL_CAPITAL)
    print(f"  完了: {len(trades)}回取引, 最終資産: ¥{metrics['final_equity']:,.0f}")

    print("\n10万円シミュレーション実行中 (直近1ヶ月)...")
    last_month_df = df.tail(22).copy()
    sim_trades, sim_equity = run_backtest(last_month_df, SIMULATION_CAPITAL)
    print(f"  完了: {len(sim_trades)}回取引")

    print("\nレポート生成中...")
    report = generate_report(trades, equity_df, metrics, sim_trades, sim_equity, df)

    print("チャート生成中...")
    generate_charts(trades, equity_df, df)

    # Save report
    report_path = os.path.join(SCRIPT_DIR, 'fx_ny_close_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    print("\n" + report)


if __name__ == "__main__":
    main()
