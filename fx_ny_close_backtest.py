#!/usr/bin/env python3
"""
USD/JPY 仲値ショート戦略 バックテスト
取引期間: 2011/01 - 2026/03 (約15年)
通貨ペア: USD/JPY

戦略概要:
  - NYクローズ(前日終値)を基準レートとする
  - 日本時間9:55(仲値)でNYクローズから+30pips以上上回っている場合のみ
  - NYクローズ方向へショートエントリー(ロングなし)
  - TP1: 中間地点で半分決済 / TP2: NYクローズ到達で残り決済
  - SL: エントリーから30pips逆方向
  ※ 9:55 JSTの仲値レートは日足始値(9:00 JST)で近似
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
TP_JPY = 0.10                    # 10 pips take profit
SL_JPY = 0.10                    # 10 pips stop loss
INITIAL_CAPITAL = 1_000_000
SIMULATION_CAPITAL = 100_000
RISK_PER_TRADE = 0.02

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Data Loading
# ============================================================
def load_data(start=START_DATE, end=END_DATE):
    # Daily data (full period)
    df = yf.download(TICKER, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.dropna(inplace=True)

    # Default: use Open (9:00 JST) as entry price proxy
    df['Entry_Price'] = df['Open']

    # Hourly data for recent period: get exact 10:00 JST (01:00 UTC)
    try:
        df_h = yf.download(TICKER, period='730d', interval='1h', progress=False)
        if isinstance(df_h.columns, pd.MultiIndex):
            df_h.columns = df_h.columns.droplevel(1)
        # Filter to 01:00 UTC = 10:00 JST candles
        jst_10 = df_h[df_h.index.hour == 1].copy()
        jst_10.index = jst_10.index.normalize().tz_localize(None)  # Convert to date-only for merge
        jst_10 = jst_10[~jst_10.index.duplicated(keep='first')]
        # Override Entry_Price where hourly data is available
        matched = df.index.isin(jst_10.index)
        df.loc[matched, 'Entry_Price'] = jst_10.loc[df.index[matched], 'Open'].values
        hourly_count = matched.sum()
        print(f"  10:00 JST正確レート: {hourly_count}日分 (時間足データ)")
        print(f"  9:00 JST近似レート: {(~matched).sum()}日分 (日足始値)")
    except Exception as e:
        print(f"  時間足データ取得失敗、全期間9:00 JST近似: {e}")

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
    direction: int          # -1 SHORT
    deviation_pips: float
    units: float
    tp_level: float
    sl_level: float
    outcome: str            # 'TP', 'SL', 'NO_HIT'
    pnl_total: float = 0.0


# ============================================================
# Outcome Resolution
# ============================================================
def resolve_outcome(direction, entry_price, high, low, tp_level, sl_level):
    # SHORT: TP is below entry, SL is above entry
    sl_hit = high >= sl_level
    tp_hit = low <= tp_level
    dist_to_sl = sl_level - entry_price
    dist_to_tp = entry_price - tp_level

    if not sl_hit and not tp_hit:
        return 'NO_HIT'
    if sl_hit and not tp_hit:
        return 'SL'
    if tp_hit and not sl_hit:
        return 'TP'
    # Both hit: closer one hit first
    if dist_to_sl <= dist_to_tp:
        return 'SL'
    else:
        return 'TP'


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
        entry_price = row['Entry_Price']  # 10:00 JST (or 9:00 JST approx)
        high = row['High']
        low = row['Low']
        close = row['Close']

        deviation = entry_price - ny_close  # positive = price above NY Close

        # SHORT ONLY: only enter when price is 30+ pips ABOVE NY Close
        if deviation < ENTRY_THRESHOLD_JPY:
            equity_curve.append({'date': date, 'equity': equity})
            continue

        direction = -1  # SHORT only

        # Calculate levels (SHORT): TP = 10pips down, SL = 10pips up
        effective_entry = entry_price - SPREAD_JPY / 2
        tp_level = effective_entry - TP_JPY
        sl_level = effective_entry + SL_JPY

        # Position sizing: 2% risk / SL distance
        risk_amount = equity * RISK_PER_TRADE
        units = risk_amount / SL_JPY
        max_units = (equity * LEVERAGE) / entry_price
        units = min(units, max_units)

        if units <= 0:
            equity_curve.append({'date': date, 'equity': equity})
            continue

        # Resolve outcome
        outcome = resolve_outcome(direction, entry_price, high, low,
                                  tp_level, sl_level)

        # Calculate PnL
        if outcome == 'TP':
            pnl_total = units * TP_JPY
        elif outcome == 'SL':
            pnl_total = -units * SL_JPY
        elif outcome == 'NO_HIT':
            pnl_total = units * (effective_entry - close)

        # Deduct spread cost (round-trip)
        spread_cost = units * SPREAD_JPY
        pnl_total -= spread_cost

        trade = NYCloseTrade(
            date=date,
            ny_close=ny_close,
            entry_price=entry_price,
            direction=direction,
            deviation_pips=deviation / 0.01,
            units=units,
            tp_level=tp_level,
            sl_level=sl_level,
            outcome=outcome,
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
    lines.append("  USD/JPY 仲値ショート戦略 バックテストレポート")
    lines.append(f"  取引期間: {df.index[0].strftime('%Y/%m/%d')} - {df.index[-1].strftime('%Y/%m/%d')}")
    lines.append(f"  初期資金: ¥{INITIAL_CAPITAL:,.0f}  レバレッジ: {LEVERAGE}倍  スプレッド: {SPREAD_PIPS}pips")
    lines.append("=" * 70)
    lines.append("")

    lines.append("■ 売買ロジック:")
    lines.append("  1. 毎日、NYクローズ(前日終値)を基準レートに設定")
    lines.append("  2. 日本時間10:00のレートでNYクローズからの乖離を判定")
    lines.append("     ※直近2.7年は時間足から10:00 JST正確値、それ以前は始値(9:00)近似")
    lines.append("  3. NYクローズから+30pips以上上回っている場合のみエントリー")
    lines.append("     → ショート(売り)のみ。ロングエントリーなし")
    lines.append(f"  4. 利確: NYクローズ方向(下方向)に{TP_JPY/0.01:.0f}pips動いたら全決済")
    lines.append(f"  5. 損切り: 逆方向(上方向)に{SL_JPY/0.01:.0f}pips動いたら全決済")
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
        'TP': '利確 (+10pips)',
        'SL': '損切り (-10pips)',
        'NO_HIT': 'TP/SL未到達 (終値決済)',
    }
    for key in ['TP', 'SL', 'NO_HIT']:
        count = outcomes.get(key, 0)
        pct = count / total_trades * 100 if total_trades > 0 else 0
        label = outcome_names.get(key, key)
        # Avg PnL per outcome
        outcome_pnls = [t.pnl_total for t in trades if t.outcome == key]
        avg_pnl = np.mean(outcome_pnls) if outcome_pnls else 0
        lines.append(f"  {label:<30} {count:>4}回 ({pct:>5.1f}%)  平均損益: ¥{avg_pnl:>+,.0f}")
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

    # Full monthly results table
    lines.append("■ 月次成績一覧 (全期間):")
    lines.append("-" * 80)
    lines.append(f"  {'年/月':<8} {'取引数':>5} {'勝数':>4} {'勝率':>6} {'月次損益':>12} {'月次リターン':>9} {'累計資産':>14}")
    lines.append("-" * 80)

    monthly_equity = equity_df['equity'].resample('ME').last()
    trade_df_monthly = pd.DataFrame([{
        'date': t.date, 'pnl': t.pnl_total, 'win': 1 if t.pnl_total > 0 else 0
    } for t in trades])
    trade_df_monthly['date'] = pd.to_datetime(trade_df_monthly['date'])
    trade_df_monthly.set_index('date', inplace=True)
    monthly_trades = trade_df_monthly.resample('ME').agg({'pnl': ['sum', 'count'], 'win': 'sum'})
    monthly_trades.columns = ['pnl_sum', 'trade_count', 'win_count']

    prev_eq = None
    for date, eq in monthly_equity.items():
        ym = date.strftime('%Y/%m')
        if date in monthly_trades.index:
            row = monthly_trades.loc[date]
            tc = int(row['trade_count'])
            wc = int(row['win_count'])
            pnl = row['pnl_sum']
            wr = wc / tc * 100 if tc > 0 else 0
        else:
            tc, wc, pnl, wr = 0, 0, 0, 0
        if prev_eq is not None and prev_eq > 0:
            ret = (eq / prev_eq - 1) * 100
        else:
            ret = 0
        lines.append(f"  {ym:<8} {tc:>5} {wc:>4} {wr:>5.1f}% ¥{pnl:>+11,.0f} {ret:>+8.2f}% ¥{eq:>13,.0f}")
        prev_eq = eq
    lines.append("-" * 80)
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
    lines.append("  - 直近2.7年は時間足から10:00 JST正確値、それ以前は日足始値(9:00)で近似")
    lines.append("  - NYクローズ = 前日終値として近似")
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
    ax1.set_title('USD/JPY - Nakaene Short Strategy (10:00 JST, 2011-2026)', fontsize=12)
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
    print("USD/JPY 仲値ショート戦略 バックテスト")
    print("=" * 50)
    print("データ読み込み中...")
    df = load_data()
    print(f"  データ: {len(df)}日分 ({df.index[0].strftime('%Y/%m/%d')} - {df.index[-1].strftime('%Y/%m/%d')})")

    # Check how many days meet the entry condition (SHORT only: price above NY Close + 30pips)
    deviations = df['Entry_Price'] - df['NY_Close']
    entry_days = (deviations >= ENTRY_THRESHOLD_JPY).sum()
    print(f"  エントリー条件(+30pips以上ショート)を満たす日: {entry_days}日 ({entry_days/len(df)*100:.1f}%)")

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
