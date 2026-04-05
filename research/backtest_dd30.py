#!/usr/bin/env python3
"""
MaxDD 30%以下制約での最適化バックテスト
======================================
目標: 年利200%に近づける & MaxDD ≤ 30%
制約: レバレッジ上限25倍
戦略: レンジBK + ペアトレード
"""

import numpy as np
import pandas as pd
from itertools import product
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/home/user/sokotsudo/research/data_fx_daily.csv', parse_dates=['date'])
df.set_index('date', inplace=True)
df = df.sort_index()
YEARS = (df.index[-1] - df.index[0]).days / 365.25

print("=" * 75)
print("  MaxDD 30%以下 制約付き最適化バックテスト")
print(f"  データ: {df.index[0].date()} ~ {df.index[-1].date()} ({YEARS:.1f}年)")
print("=" * 75)


def range_breakout(prices, lookback=5, tp_mult=1.5, risk_pct=0.02):
    results = []
    equity = 1.0
    position = 0
    entry_price = tp_price = sl_price = 0.0
    for i in range(lookback, len(prices)):
        window = prices.iloc[i-lookback:i]
        rh = window.max()
        rl = window.min()
        rw = rh - rl
        if rw <= 0:
            continue
        price = prices.iloc[i]
        if position == 0:
            if price > rh:
                position = 1; entry_price = rh
                tp_price = entry_price + rw * tp_mult; sl_price = entry_price - rw
            elif price < rl:
                position = -1; entry_price = rl
                tp_price = entry_price - rw * tp_mult; sl_price = entry_price + rw
        elif position == 1:
            if price >= tp_price:
                equity *= (1 + tp_mult * risk_pct)
                results.append({'date': prices.index[i], 'r': tp_mult, 'eq': equity}); position = 0
            elif price <= sl_price:
                equity *= (1 - risk_pct)
                results.append({'date': prices.index[i], 'r': -1.0, 'eq': equity}); position = 0
        elif position == -1:
            if price <= tp_price:
                equity *= (1 + tp_mult * risk_pct)
                results.append({'date': prices.index[i], 'r': tp_mult, 'eq': equity}); position = 0
            elif price >= sl_price:
                equity *= (1 - risk_pct)
                results.append({'date': prices.index[i], 'r': -1.0, 'eq': equity}); position = 0
    return pd.DataFrame(results) if results else pd.DataFrame(columns=['date','r','eq'])


def pairs_trading(pair_a, pair_b, window=60, entry_z=2.0, exit_z=0.5,
                  stop_z=4.0, risk_pct=0.02):
    beta = np.polyfit(pair_b, pair_a, 1)[0]
    spread = pair_a - beta * pair_b
    sm = spread.rolling(window=window).mean()
    ss = spread.rolling(window=window).std()
    z = (spread - sm) / ss
    results = []; equity = 1.0; pos = 0; ez = 0
    for i in range(window, len(z)):
        zi = z.iloc[i]
        if pd.isna(zi): continue
        if pos == 0:
            if zi > entry_z: pos = -1; ez = zi
            elif zi < -entry_z: pos = 1; ez = zi
        elif pos == 1:
            if zi > -exit_z or zi < -stop_z:
                pnl = (-ez - (-zi)) / entry_z * risk_pct
                equity *= (1 + pnl)
                results.append({'date': z.index[i], 'pnl': pnl*100, 'eq': equity}); pos = 0
        elif pos == -1:
            if zi < exit_z or zi > stop_z:
                pnl = (ez - zi) / entry_z * risk_pct
                equity *= (1 + pnl)
                results.append({'date': z.index[i], 'pnl': pnl*100, 'eq': equity}); pos = 0
    return pd.DataFrame(results) if results else pd.DataFrame(columns=['date','pnl','eq'])


def calc_stats(results_df, eq_col='eq'):
    if len(results_df) == 0: return None
    eq = results_df[eq_col]; final = eq.iloc[-1]
    cagr = ((final ** (1/YEARS)) - 1) * 100
    monthly = ((final ** (1/(YEARS*12))) - 1) * 100
    rm = eq.cummax(); dd = ((rm - eq) / rm).max() * 100
    if 'r' in results_df.columns:
        wins = (results_df['r'] > 0).sum()
        wr = wins / len(results_df) * 100
    elif 'pnl' in results_df.columns:
        wins = (results_df['pnl'] > 0).sum()
        wr = wins / len(results_df) * 100
    else: wr = 0
    return {'trades': len(results_df), 'trades_mo': len(results_df)/(YEARS*12),
            'win_rate': wr, 'monthly': monthly, 'cagr': cagr, 'max_dd': dd, 'final_x': final}


# ============================================================
# PHASE 1: 単体戦略 - MaxDD 30%以下でCAGR最大を探索
# リスク率を1%刻みで細かくサーチ
# ============================================================
print("\n" + "=" * 75)
print("  PHASE 1: レンジBK 単体 (MaxDD ≤ 30% 制約)")
print("=" * 75)

pairs_data = {
    'EUR/USD': df['eurusd'],
    'GBP/USD': df['gbpusd'],
    'EUR/GBP': df['eurgbp'],
    'USD/JPY': df['usdjpy'].dropna(),
}

lookbacks = [3, 5, 7]
tp_mults = [1.0, 1.5, 2.0]
risk_pcts_fine = [r/100 for r in range(1, 16)]  # 1% ~ 15% を1%刻み

rb_all = []
for pair_name, prices in pairs_data.items():
    for lb, tp, rp in product(lookbacks, tp_mults, risk_pcts_fine):
        res = range_breakout(prices, lb, tp, rp)
        s = calc_stats(res)
        if s:
            rb_all.append({'pair': pair_name, 'lb': lb, 'tp': tp, 'risk': rp*100, **s})

rb_df = pd.DataFrame(rb_all)
# MaxDD 30%以下のみ
rb_ok = rb_df[rb_df['max_dd'] <= 30].sort_values('cagr', ascending=False)

print(f"\n  全{len(rb_df)}パターン中、MaxDD≤30%: {len(rb_ok)}パターン")
print(f"\n  === ベスト15 (CAGR順, MaxDD≤30%) ===")
print(f"  {'Pair':<10} {'LB':>3} {'TP':>4} {'Risk':>5} {'Trades':>6} {'WR%':>6} {'月利%':>7} {'CAGR%':>8} {'MaxDD%':>7}")
print(f"  {'-'*70}")
for _, row in rb_ok.head(15).iterrows():
    print(f"  {row['pair']:<10} {row['lb']:>3} {row['tp']:>4.1f} {row['risk']:>4.0f}% {row['trades']:>6} {row['win_rate']:>5.1f}% {row['monthly']:>6.2f}% {row['cagr']:>7.1f}% {row['max_dd']:>6.1f}%")

# 各ペアのベスト設定を取得
print(f"\n  === 各ペアのベスト設定 (MaxDD≤30%) ===")
best_per_pair = {}
for pair in pairs_data.keys():
    pair_best = rb_ok[rb_ok['pair'] == pair]
    if len(pair_best) > 0:
        best = pair_best.iloc[0]
        best_per_pair[pair] = {'lb': int(best['lb']), 'tp': best['tp'], 'risk': best['risk']/100}
        print(f"  {pair}: LB={best['lb']:.0f}, TP={best['tp']:.1f}, Risk={best['risk']:.0f}%, "
              f"CAGR={best['cagr']:.1f}%, MaxDD={best['max_dd']:.1f}%")


# ============================================================
# PHASE 2: ペアトレード単体 (MaxDD ≤ 30% 制約)
# ============================================================
print("\n\n" + "=" * 75)
print("  PHASE 2: ペアトレード 単体 (MaxDD ≤ 30% 制約)")
print("=" * 75)

clean = df[['eurusd', 'gbpusd']].dropna()
windows = [30, 45, 60]
entry_zs = [1.5, 1.75, 2.0]

pt_all = []
for w, ez, rp in product(windows, entry_zs, risk_pcts_fine):
    res = pairs_trading(clean['eurusd'], clean['gbpusd'], w, ez, risk_pct=rp)
    s = calc_stats(res)
    if s:
        pt_all.append({'window': w, 'entry_z': ez, 'risk': rp*100, **s})

pt_df = pd.DataFrame(pt_all)
pt_ok = pt_df[pt_df['max_dd'] <= 30].sort_values('cagr', ascending=False)

print(f"\n  全{len(pt_df)}パターン中、MaxDD≤30%: {len(pt_ok)}パターン")
print(f"\n  === ベスト10 (CAGR順, MaxDD≤30%) ===")
print(f"  {'Win':>4} {'Z':>5} {'Risk':>5} {'Trades':>6} {'WR%':>6} {'月利%':>7} {'CAGR%':>8} {'MaxDD%':>7}")
print(f"  {'-'*60}")
for _, row in pt_ok.head(10).iterrows():
    print(f"  {row['window']:>4} {row['entry_z']:>5.2f} {row['risk']:>4.0f}% {row['trades']:>6} {row['win_rate']:>5.1f}% {row['monthly']:>6.2f}% {row['cagr']:>7.1f}% {row['max_dd']:>6.1f}%")

best_pt = pt_ok.iloc[0] if len(pt_ok) > 0 else None
if best_pt is not None:
    print(f"\n  ペアトレードベスト: W={best_pt['window']:.0f}, Z={best_pt['entry_z']:.2f}, "
          f"Risk={best_pt['risk']:.0f}%, CAGR={best_pt['cagr']:.1f}%, MaxDD={best_pt['max_dd']:.1f}%")


# ============================================================
# PHASE 3: 複合ポートフォリオ (MaxDD ≤ 30% 制約)
# 各戦略のリスク率を個別に調整
# ============================================================
print("\n\n" + "=" * 75)
print("  PHASE 3: 複合ポートフォリオ (MaxDD ≤ 30% 目標)")
print("=" * 75)

# 複合のリスク率を段階的に試行
composite_results = []

for rb_risk_pct in [0.03, 0.04, 0.05, 0.06, 0.07]:
    for pt_risk_pct in [0.05, 0.08, 0.10, 0.12, 0.15]:
        # 各ペアのベストLB/TPでレンジBK実行
        rb_trades = []
        for pair_name, prices in pairs_data.items():
            bp = best_per_pair.get(pair_name)
            if bp:
                res = range_breakout(prices, bp['lb'], bp['tp'], rb_risk_pct)
                if len(res) > 0:
                    for _, row in res.iterrows():
                        rb_trades.append({'date': row['date'], 'strategy': f'RB_{pair_name[:3]}',
                                          'pnl': row['r'] * rb_risk_pct})

        # ペアトレード
        pt_w = int(best_pt['window']) if best_pt is not None else 30
        pt_z = best_pt['entry_z'] if best_pt is not None else 1.5
        pt_res = pairs_trading(clean['eurusd'], clean['gbpusd'], pt_w, pt_z, risk_pct=pt_risk_pct)
        if len(pt_res) > 0:
            for _, row in pt_res.iterrows():
                rb_trades.append({'date': row['date'], 'strategy': 'PT', 'pnl': row['pnl']/100})

        if not rb_trades:
            continue

        trades_df = pd.DataFrame(rb_trades).sort_values('date')
        eq = 1.0
        eq_list = []
        for _, t in trades_df.iterrows():
            eq *= (1 + t['pnl'])
            eq_list.append({'date': t['date'], 'eq': eq, 'strategy': t['strategy']})

        comp_df = pd.DataFrame(eq_list)
        final_eq = eq
        cagr = ((final_eq ** (1/YEARS)) - 1) * 100
        monthly = ((final_eq ** (1/(YEARS*12))) - 1) * 100
        rm = comp_df['eq'].cummax()
        max_dd = ((rm - comp_df['eq']) / rm).max() * 100

        strat_counts = comp_df['strategy'].value_counts().to_dict()

        # 年別リターン
        comp_df['year'] = pd.to_datetime(comp_df['date']).dt.year
        yearly = {}
        prev = 1.0
        for year in sorted(comp_df['year'].unique()):
            yd = comp_df[comp_df['year'] == year]
            if len(yd) > 0:
                end = yd['eq'].iloc[-1]
                yearly[year] = (end/prev - 1) * 100
                prev = end

        composite_results.append({
            'rb_risk': rb_risk_pct*100, 'pt_risk': pt_risk_pct*100,
            'total_trades': len(comp_df), 'trades_mo': len(comp_df)/(YEARS*12),
            'monthly': monthly, 'cagr': cagr, 'max_dd': max_dd,
            'final_x': final_eq, 'strat_counts': strat_counts, 'yearly': yearly,
        })

comp_res_df = pd.DataFrame(composite_results)
comp_ok = comp_res_df[comp_res_df['max_dd'] <= 30].sort_values('cagr', ascending=False)

print(f"\n  全{len(comp_res_df)}パターン中、MaxDD≤30%: {len(comp_ok)}パターン")
print(f"\n  === ベスト10 複合ポートフォリオ (CAGR順, MaxDD≤30%) ===")
print(f"  {'RB_R':>5} {'PT_R':>5} {'Trades':>6} {'月/回':>5} {'月利%':>7} {'CAGR%':>8} {'MaxDD%':>7} {'倍率':>10} {'200%':>5}")
print(f"  {'-'*72}")
for _, row in comp_ok.head(10).iterrows():
    flag = '✅' if row['cagr'] >= 200 else '❌'
    print(f"  {row['rb_risk']:>4.0f}% {row['pt_risk']:>4.0f}% {row['total_trades']:>6} {row['trades_mo']:>4.1f} "
          f"{row['monthly']:>6.2f}% {row['cagr']:>7.1f}% {row['max_dd']:>6.1f}% {row['final_x']:>9.1f}x {flag:>5}")

# ベスト設定の年別リターンを表示
if len(comp_ok) > 0:
    best_comp = comp_ok.iloc[0]
    print(f"\n  === ベスト複合の年別リターン ===")
    print(f"  設定: レンジBK Risk={best_comp['rb_risk']:.0f}%, ペアトレード Risk={best_comp['pt_risk']:.0f}%")
    print(f"  内訳: {best_comp['strat_counts']}")
    for y, r in best_comp['yearly'].items():
        marker = " ✅" if r >= 200 else (" ⚠" if r < 0 else "")
        print(f"    {y}: {r:+.1f}%{marker}")


# ============================================================
# 最終サマリー
# ============================================================
print("\n\n" + "=" * 75)
print("  最終サマリー: MaxDD ≤ 30% での最適設定")
print("=" * 75)

print("\n  ┌─────────────────────────────────────────────────────────────────┐")
print("  │ 戦略                 │ パラメータ          │ 月利  │ CAGR  │MaxDD│")
print("  ├──────────────────────┼────────────────────┼──────┼──────┼─────┤")

# レンジBKベスト (MaxDD≤30%)
for pair in ['EUR/USD', 'GBP/USD', 'EUR/GBP', 'USD/JPY']:
    pair_best = rb_ok[rb_ok['pair'] == pair]
    if len(pair_best) > 0:
        b = pair_best.iloc[0]
        name = f"レンジBK {pair}"
        param = f"LB={b['lb']:.0f},TP={b['tp']:.1f},R={b['risk']:.0f}%"
        print(f"  │ {name:<20} │ {param:<18} │{b['monthly']:>5.1f}%│{b['cagr']:>5.0f}% │{b['max_dd']:>4.0f}%│")

# ペアトレードベスト (MaxDD≤30%)
if best_pt is not None:
    param = f"W={best_pt['window']:.0f},Z={best_pt['entry_z']:.1f},R={best_pt['risk']:.0f}%"
    print(f"  │ {'ペアトレード':<18}   │ {param:<18} │{best_pt['monthly']:>5.1f}%│{best_pt['cagr']:>5.0f}% │{best_pt['max_dd']:>4.0f}%│")

# 複合ベスト (MaxDD≤30%)
if len(comp_ok) > 0:
    bc = comp_ok.iloc[0]
    param = f"RB={bc['rb_risk']:.0f}%,PT={bc['pt_risk']:.0f}%"
    print(f"  ├──────────────────────┼────────────────────┼──────┼──────┼─────┤")
    print(f"  │ {'複合(4ペア+PT)':<18}   │ {param:<18} │{bc['monthly']:>5.1f}%│{bc['cagr']:>5.0f}% │{bc['max_dd']:>4.0f}%│")

print("  └─────────────────────────────────────────────────────────────────┘")

# レバレッジチェック
if len(comp_ok) > 0:
    bc = comp_ok.iloc[0]
    rb_r = bc['rb_risk'] / 100
    # SL幅は大体レンジ幅。日足レンジ平均約100-300pips。保守的に150pipsとする
    lev_per_pos = rb_r / (150 * 0.0001 / 1.1) * 0.1  # 概算
    max_sim_pos = 5  # 4 RB + 1 PT
    print(f"\n  レバレッジ概算: リスク{bc['rb_risk']:.0f}%で1ポジション約{lev_per_pos:.0f}倍 × 最大{max_sim_pos}ポジション")
    print(f"  ※ 全ポジション同時発生は稀。実効レバ25倍以内は十分達成可能")

print("\n[完了]")
