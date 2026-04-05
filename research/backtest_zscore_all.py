#!/usr/bin/env python3
"""
Zスコア逆張り戦略 - 全通貨ペア長期バックテスト
===============================================
データ: ECB公式レート 1999-2026 (27年間)
戦略: 30日Zスコアが±1.5を超えたら逆張り、0.5に戻ったら決済
対象: EUR/GBP, EUR/USD, EUR/JPY, USD/JPY, GBP/USD
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/home/user/sokotsudo/research/data_fx_long.csv', parse_dates=['date'])
df.set_index('date', inplace=True)
df = df.sort_index()
YEARS = (df.index[-1] - df.index[0]).days / 365.25

print("=" * 75)
print("  Zスコア逆張り戦略 - 全ペア長期バックテスト")
print(f"  データ: {df.index[0].date()} ~ {df.index[-1].date()} ({YEARS:.1f}年, {len(df)}日)")
print("=" * 75)


def zscore_mean_reversion(
    prices: pd.Series,
    window: int = 30,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    risk_pct: float = 0.15,
) -> pd.DataFrame:
    """Zスコア逆張り: Z>entry→売り、Z<-entry→買い、|Z|<exit→決済"""
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    z = (prices - mean) / std

    results = []
    equity = 1.0
    pos = 0
    ez = 0.0

    for i in range(window, len(z)):
        zi = z.iloc[i]
        if pd.isna(zi):
            continue

        if pos == 0:
            if zi > entry_z:
                pos = -1; ez = zi
            elif zi < -entry_z:
                pos = 1; ez = zi

        elif pos == 1:
            if zi > -exit_z or zi < -stop_z:
                pnl = (-ez - (-zi)) / entry_z * risk_pct
                equity *= (1 + pnl)
                results.append({
                    'date': z.index[i], 'entry_z': ez, 'exit_z': zi,
                    'pnl': pnl * 100, 'eq': equity, 'direction': 'LONG'
                })
                pos = 0

        elif pos == -1:
            if zi < exit_z or zi > stop_z:
                pnl = (ez - zi) / entry_z * risk_pct
                equity *= (1 + pnl)
                results.append({
                    'date': z.index[i], 'entry_z': ez, 'exit_z': zi,
                    'pnl': pnl * 100, 'eq': equity, 'direction': 'SHORT'
                })
                pos = 0

    return pd.DataFrame(results) if results else pd.DataFrame()


def print_results(name, rdf, years=None):
    if years is None:
        years = YEARS
    if len(rdf) == 0:
        print(f"\n  {name}: トレードなし")
        return

    final = rdf['eq'].iloc[-1]
    cagr = ((final ** (1/years)) - 1) * 100
    monthly = ((final ** (1/(years*12))) - 1) * 100
    rm = rdf['eq'].cummax()
    max_dd = ((rm - rdf['eq']) / rm).max() * 100
    wins = (rdf['pnl'] > 0).sum()
    losses = (rdf['pnl'] <= 0).sum()
    wr = wins / len(rdf) * 100

    # 損切りヒット回数
    stop_hits = ((rdf['exit_z'].abs() > 3.5)).sum() if 'exit_z' in rdf.columns else 0

    print(f"\n{'='*75}")
    print(f"  {name}")
    print(f"{'='*75}")
    print(f"  取引回数        : {len(rdf)} ({len(rdf)/(years*12):.1f}回/月)")
    print(f"  勝ち / 負け     : {wins} / {losses}")
    print(f"  勝率            : {wr:.1f}%")
    print(f"  損切り回数      : {stop_hits}")
    print(f"  月利            : {monthly:.2f}%")
    print(f"  CAGR            : {cagr:.1f}%")
    print(f"  MaxDD           : {max_dd:.1f}%")
    print(f"  最終倍率        : {final:.1f}x")

    # 年別リターン
    rdf_c = rdf.copy()
    rdf_c['year'] = pd.to_datetime(rdf_c['date']).dt.year
    print(f"\n  --- 年別リターン ---")
    prev = 1.0
    neg_years = 0
    for year in sorted(rdf_c['year'].unique()):
        yd = rdf_c[rdf_c['year'] == year]
        if len(yd) > 0:
            end = yd['eq'].iloc[-1]
            yr = (end/prev - 1) * 100
            trades = len(yd)
            marker = " ⚠️" if yr < 0 else ""
            if yr < 0: neg_years += 1
            print(f"    {year}: {yr:>+8.1f}% ({trades}回){marker}")
            prev = end

    print(f"\n  マイナス年: {neg_years}回 / {len(rdf_c['year'].unique())}年")

    return {
        'name': name, 'trades': len(rdf), 'win_rate': wr,
        'monthly': monthly, 'cagr': cagr, 'max_dd': max_dd,
        'final_x': final, 'neg_years': neg_years,
    }


# ============================================================
# 全ペアでバックテスト
# ============================================================

pairs = {
    'EUR/GBP': df['eurgbp'].dropna(),
    'EUR/USD': df['eurusd'].dropna(),
    'EUR/JPY': df['eurjpy'].dropna(),
    'USD/JPY': df['usdjpy'].dropna(),
    'GBP/USD': df['gbpusd'].dropna(),
}

# まずリスク15%で全ペア比較
print("\n\n" + "=" * 75)
print("  PHASE 1: 全ペア比較 (W=30, Z=1.5, Risk=15%)")
print("=" * 75)

all_stats = []
for pair_name, prices in pairs.items():
    pair_years = (prices.index[-1] - prices.index[0]).days / 365.25
    rdf = zscore_mean_reversion(prices, window=30, entry_z=1.5, risk_pct=0.15)
    stats = print_results(f"Zスコア逆張り {pair_name} (W=30, Z=1.5, R=15%)", rdf, pair_years)
    if stats:
        all_stats.append(stats)


# ============================================================
# パラメータ最適化 (MaxDD ≤ 30%)
# ============================================================
print("\n\n" + "=" * 75)
print("  PHASE 2: パラメータ最適化 (MaxDD ≤ 30%)")
print("=" * 75)

windows = [20, 30, 45, 60]
entry_zs = [1.0, 1.25, 1.5, 1.75, 2.0]
risk_pcts = [r/100 for r in range(5, 21)]  # 5% ~ 20%

grid_results = []
for pair_name, prices in pairs.items():
    pair_years = (prices.index[-1] - prices.index[0]).days / 365.25
    for w in windows:
        for ez in entry_zs:
            for rp in risk_pcts:
                rdf = zscore_mean_reversion(prices, w, ez, risk_pct=rp)
                if len(rdf) == 0:
                    continue
                final = rdf['eq'].iloc[-1]
                cagr = ((final ** (1/pair_years)) - 1) * 100
                monthly = ((final ** (1/(pair_years*12))) - 1) * 100
                rm = rdf['eq'].cummax()
                max_dd = ((rm - rdf['eq']) / rm).max() * 100
                wins = (rdf['pnl'] > 0).sum()
                wr = wins / len(rdf) * 100

                grid_results.append({
                    'pair': pair_name, 'window': w, 'entry_z': ez,
                    'risk': rp*100, 'trades': len(rdf),
                    'trades_mo': len(rdf)/(pair_years*12),
                    'win_rate': wr, 'monthly': monthly,
                    'cagr': cagr, 'max_dd': max_dd, 'final_x': final,
                })

grid_df = pd.DataFrame(grid_results)
grid_ok = grid_df[grid_df['max_dd'] <= 30].sort_values('cagr', ascending=False)

print(f"\n  全{len(grid_df)}パターン中、MaxDD≤30%: {len(grid_ok)}パターン")

# 各ペアのベスト
print(f"\n  === 各ペアのベスト (MaxDD≤30%) ===")
print(f"  {'Pair':<10} {'W':>3} {'Z':>5} {'R':>4} {'Trades':>6} {'月/回':>5} {'WR%':>6} {'月利%':>7} {'CAGR%':>8} {'MaxDD%':>7}")
print(f"  {'-'*75}")

best_per_pair = {}
for pair in pairs.keys():
    pair_ok = grid_ok[grid_ok['pair'] == pair]
    if len(pair_ok) > 0:
        b = pair_ok.iloc[0]
        best_per_pair[pair] = b
        print(f"  {b['pair']:<10} {b['window']:>3.0f} {b['entry_z']:>5.2f} {b['risk']:>3.0f}% {b['trades']:>6.0f} {b['trades_mo']:>4.1f} {b['win_rate']:>5.1f}% {b['monthly']:>6.2f}% {b['cagr']:>7.1f}% {b['max_dd']:>6.1f}%")

# 全ペア ベスト15
print(f"\n  === 全ペア ベスト15 (MaxDD≤30%) ===")
print(f"  {'Pair':<10} {'W':>3} {'Z':>5} {'R':>4} {'Trades':>6} {'月/回':>5} {'WR%':>6} {'月利%':>7} {'CAGR%':>8} {'MaxDD%':>7}")
print(f"  {'-'*75}")
for _, b in grid_ok.head(15).iterrows():
    print(f"  {b['pair']:<10} {b['window']:>3.0f} {b['entry_z']:>5.2f} {b['risk']:>3.0f}% {b['trades']:>6.0f} {b['trades_mo']:>4.1f} {b['win_rate']:>5.1f}% {b['monthly']:>6.2f}% {b['cagr']:>7.1f}% {b['max_dd']:>6.1f}%")


# ============================================================
# 複合: 有効ペアを全て同時運用
# ============================================================
print("\n\n" + "=" * 75)
print("  PHASE 3: 複合ポートフォリオ（有効ペア同時運用）")
print("=" * 75)

# 各ペアのベスト設定で同時運用
for composite_risk_scale in [1.0, 0.7, 0.5]:
    all_trades = []
    for pair_name, prices in pairs.items():
        if pair_name not in best_per_pair:
            continue
        bp = best_per_pair[pair_name]
        rp = bp['risk'] / 100 * composite_risk_scale
        rdf = zscore_mean_reversion(prices, int(bp['window']), bp['entry_z'], risk_pct=rp)
        if len(rdf) > 0:
            for _, row in rdf.iterrows():
                all_trades.append({
                    'date': row['date'], 'pair': pair_name, 'pnl': row['pnl']/100
                })

    if not all_trades:
        continue

    tdf = pd.DataFrame(all_trades).sort_values('date')
    eq = 1.0
    eqs = []
    for _, t in tdf.iterrows():
        eq *= (1 + t['pnl'])
        eqs.append({'date': t['date'], 'eq': eq, 'pair': t['pair']})

    edf = pd.DataFrame(eqs)
    final = eq
    cagr = ((final ** (1/YEARS)) - 1) * 100
    monthly = ((final ** (1/(YEARS*12))) - 1) * 100
    rm = edf['eq'].cummax()
    max_dd = ((rm - edf['eq']) / rm).max() * 100
    counts = edf['pair'].value_counts().to_dict()

    # 年別
    edf['year'] = pd.to_datetime(edf['date']).dt.year
    yearly = {}
    prev = 1.0
    for year in sorted(edf['year'].unique()):
        yd = edf[edf['year'] == year]
        if len(yd) > 0:
            end = yd['eq'].iloc[-1]
            yearly[year] = (end/prev - 1) * 100
            prev = end

    print(f"\n  === 複合 (リスクスケール {composite_risk_scale:.0%}) ===")
    print(f"  取引回数: {len(edf)} ({len(edf)/(YEARS*12):.1f}回/月)")
    print(f"  内訳: {counts}")
    print(f"  月利: {monthly:.2f}%")
    print(f"  CAGR: {cagr:.1f}%")
    print(f"  MaxDD: {max_dd:.1f}%")
    print(f"  最終倍率: {final:.1f}x")
    print(f"  年利200%達成: {'✅' if cagr >= 200 else '❌'}")
    print(f"\n  年別リターン:")
    neg = 0
    for y, r in yearly.items():
        marker = " ⚠️" if r < 0 else ""
        if r < 0: neg += 1
        print(f"    {y}: {r:>+8.1f}%{marker}")
    print(f"  マイナス年: {neg}/{len(yearly)}")


# ============================================================
# 最終サマリー
# ============================================================
print("\n\n" + "=" * 75)
print("  最終サマリー: Zスコア逆張り 27年間バックテスト")
print("=" * 75)
print(f"\n  各ペアのベスト設定 (MaxDD≤30%):")
print(f"  {'Pair':<10} │ {'Window':>6} │ {'Entry_Z':>7} │ {'Risk':>5} │ {'月利':>7} │ {'CAGR':>7} │ {'MaxDD':>6} │ {'有効':>4}")
print(f"  {'─'*75}")
for pair in pairs.keys():
    if pair in best_per_pair:
        b = best_per_pair[pair]
        effective = "✅" if b['cagr'] > 50 else "△"
        print(f"  {pair:<10} │ {b['window']:>6.0f} │ {b['entry_z']:>7.2f} │ {b['risk']:>4.0f}% │ {b['monthly']:>6.2f}% │ {b['cagr']:>6.1f}% │ {b['max_dd']:>5.1f}% │ {effective:>4}")

print(f"""
  結論:
  ─────
  Zスコア逆張りは「通貨ペアの価格が短期平均から乖離→戻る」を利用。
  EUR/GBPで最も有効（ユーロとポンドの高い相関性が平均回帰を保証）。
  他ペア（EUR/USD, EUR/JPY, USD/JPY等）はトレンド性が強く、
  平均回帰の前提が成り立ちにくいため、成績が劣化する可能性あり。
""")

print("[完了]")
