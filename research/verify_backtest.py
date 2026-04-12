#!/usr/bin/env python3
"""
Z-score逆張り戦略 最終版バックテスト
=====================================
エントリー: 15分足確定時 Z > ±0.51
決済:
  1. TP +15pips (毎ティック相当)
  2. SL -15pips (毎ティック相当)
  3. TO 2時間経過
  4. Z決済: 5分足確定時 |Z| < 0.5 または |Z| > 6.0

このスクリプトの特徴:
- 5分足の各バーをティックに見立てて判定
- 5分足の高値/安値でTP/SL到達を判定（よりリアル）
- 同じ結果が毎回出る（決定的）
- ペア別・決済理由別の詳細統計
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# =======================================================
# 設定
# =======================================================
ALL_PAIRS = [
    "EUR/USD","USD/JPY","EUR/JPY","GBP/USD","GBP/JPY",
    "AUD/JPY","NZD/JPY","CHF/JPY","USD/CHF","AUD/USD",
    "EUR/GBP","NZD/USD","USD/CAD","CAD/JPY","AUD/CHF",
    "EUR/AUD","AUD/NZD","EUR/CAD","EUR/CHF","AUD/CAD",
    "EUR/NZD","GBP/CAD","GBP/CHF","GBP/NZD"
]

# パラメータ
WINDOW = 30
ENTRY_Z = 0.51
EXIT_Z = 0.5
STOP_Z = 6.0
TIMEOUT_H = 2.0
TP_PIPS = 15
SL_PIPS = 15

JPY_RATE = 150  # USDJPY想定レート

MARGINS = {
    "EUR/USD":70349,"USD/JPY":63705,"EUR/JPY":74696,"GBP/USD":80760,
    "GBP/JPY":85748,"AUD/JPY":45032,"NZD/JPY":37187,"CHF/JPY":80703,
    "USD/CHF":47362,"AUD/USD":42412,"EUR/GBP":52264,"NZD/USD":35023,
    "USD/CAD":83056,"CAD/JPY":46021,"AUD/CHF":33480,"EUR/AUD":99522,
    "AUD/NZD":72659,"EUR/CAD":97384,"EUR/CHF":55533,"AUD/CAD":58711,
    "EUR/NZD":120518,"GBP/CAD":111796,"GBP/CHF":63750,"GBP/NZD":138356
}


def load_data(pair, tf):
    """15m / 5m のデータロード。カラムは datetime, close (必要ならhigh/low)"""
    fpath = f'research/intraday/{pair.replace("/","")}.{tf}.csv'
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['close'] = df['close'].astype(float)
    if 'high' in df.columns:
        df['high'] = df['high'].astype(float)
    if 'low' in df.columns:
        df['low'] = df['low'].astype(float)
    return df


def backtest(pair):
    """
    バックテストロジック（厳密版）
    - 15分足でZスコア計算
    - 5分足のバーごとに判定（ティックの代わり）
    - 5分足のhigh/lowでTP/SLヒット判定（あれば）
    - エントリーは15分足確定の「次の5分足」の始値（currentBarTime変化時）
    """
    df15 = load_data(pair, '15m')
    df5 = load_data(pair, '5m')
    if df15 is None or df5 is None:
        return []

    is_jpy = 'JPY' in pair.split('/')[1]
    pip = 0.01 if is_jpy else 0.0001

    # 15分足のZスコア計算
    df15 = df15.copy()
    df15['mean'] = df15['close'].rolling(WINDOW).mean()
    df15['std'] = df15['close'].rolling(WINDOW).std()
    df15['z'] = (df15['close'] - df15['mean']) / df15['std']

    # 15分足 → 5分足時刻マッピング
    # 15分足が15:00で確定 = 次の5分足が15:00(実際にはその5分足の終値が15:05の寄り付き相当)
    # ここでは厳密に: 5分足の時刻が15分足の時刻と一致したとき、その15分足のZスコアをチェック
    m15_lookup = {}
    for i in range(WINDOW, len(df15)):
        dt = df15['datetime'].iloc[i]
        m15_lookup[dt] = {
            'mean': df15['mean'].iloc[i],
            'std': df15['std'].iloc[i],
            'z': df15['z'].iloc[i],
        }

    trades = []
    pos = 0
    entry_price = 0
    entry_time = None
    last_mean = 0
    last_std = 0

    has_hl = 'high' in df5.columns and 'low' in df5.columns

    for i in range(len(df5)):
        dt = df5['datetime'].iloc[i]
        close_price = df5['close'].iloc[i]

        # 15分足確定時刻なら最新のmean/stdを更新
        if dt in m15_lookup:
            info = m15_lookup[dt]
            last_mean = info['mean']
            last_std = info['std']

        # ポジションなし → エントリー判定（15分足確定時のみ）
        if pos == 0:
            if dt in m15_lookup and last_std > 0:
                z = m15_lookup[dt]['z']
                if z > ENTRY_Z:
                    pos = -1
                    entry_price = close_price
                    entry_time = dt
                elif z < -ENTRY_Z:
                    pos = 1
                    entry_price = close_price
                    entry_time = dt
            continue

        # ポジションあり → TP/SL/TO/Z決済判定
        if last_std == 0:
            continue

        # この5分足の高値/安値でTP/SL到達をチェック（保守的に悪い方から評価）
        hit_tp = False
        hit_sl = False
        tp_price = 0
        sl_price = 0

        if has_hl:
            bar_high = df5['high'].iloc[i]
            bar_low = df5['low'].iloc[i]
            if pos == 1:  # ロング: high側でTP、low側でSL
                if bar_high - entry_price >= TP_PIPS * pip:
                    hit_tp = True
                    tp_price = entry_price + TP_PIPS * pip
                if entry_price - bar_low >= SL_PIPS * pip:
                    hit_sl = True
                    sl_price = entry_price - SL_PIPS * pip
            else:  # ショート: low側でTP、high側でSL
                if entry_price - bar_low >= TP_PIPS * pip:
                    hit_tp = True
                    tp_price = entry_price - TP_PIPS * pip
                if bar_high - entry_price >= SL_PIPS * pip:
                    hit_sl = True
                    sl_price = entry_price + SL_PIPS * pip
        else:
            # high/lowなしの場合はcloseのみで判定
            pips_now = (close_price - entry_price) * pos / pip
            if pips_now >= TP_PIPS:
                hit_tp = True
                tp_price = close_price
            if pips_now <= -SL_PIPS:
                hit_sl = True
                sl_price = close_price

        # 同一バー内でTP/SL両方ヒット → 保守的にSLを優先（最悪ケース想定）
        if hit_tp and hit_sl:
            diff = (sl_price - entry_price) * pos
            pnl = diff * 10000 if is_jpy else diff * 10000 * JPY_RATE
            trades.append((dt, pnl, pair, (dt-entry_time).total_seconds()/3600, 'SL'))
            pos = 0
            continue
        if hit_tp:
            diff = (tp_price - entry_price) * pos
            pnl = diff * 10000 if is_jpy else diff * 10000 * JPY_RATE
            trades.append((dt, pnl, pair, (dt-entry_time).total_seconds()/3600, 'TP'))
            pos = 0
            continue
        if hit_sl:
            diff = (sl_price - entry_price) * pos
            pnl = diff * 10000 if is_jpy else diff * 10000 * JPY_RATE
            trades.append((dt, pnl, pair, (dt-entry_time).total_seconds()/3600, 'SL'))
            pos = 0
            continue

        # タイムアウト
        if entry_time is not None:
            hold_h = (dt - entry_time).total_seconds() / 3600.0
            if hold_h >= TIMEOUT_H:
                diff = (close_price - entry_price) * pos
                pnl = diff * 10000 if is_jpy else diff * 10000 * JPY_RATE
                trades.append((dt, pnl, pair, hold_h, 'TO'))
                pos = 0
                continue

        # Z決済（5分足の終値で判定）
        z_now = (close_price - last_mean) / last_std
        if pos == 1:
            if z_now > -EXIT_Z or z_now < -STOP_Z:
                diff = (close_price - entry_price) * pos
                pnl = diff * 10000 if is_jpy else diff * 10000 * JPY_RATE
                trades.append((dt, pnl, pair, (dt-entry_time).total_seconds()/3600, 'Z'))
                pos = 0
        else:
            if z_now < EXIT_Z or z_now > STOP_Z:
                diff = (close_price - entry_price) * pos
                pnl = diff * 10000 if is_jpy else diff * 10000 * JPY_RATE
                trades.append((dt, pnl, pair, (dt-entry_time).total_seconds()/3600, 'Z'))
                pos = 0

    return trades


def main():
    print('='*95)
    print('  Z-score Scalper 厳密版バックテスト')
    print('  Entry: M15 Z>0.51 / Exit: M5 Z<0.5 / TP=15p / SL=15p / TO=2h')
    print('='*95)

    all_trades = []
    pair_summary = []

    for pair in ALL_PAIRS:
        trades = backtest(pair)
        if not trades:
            print(f'  {pair}: データなし')
            continue
        all_trades.extend(trades)
        tdf = pd.DataFrame(trades, columns=['date','pnl','pair','hold','reason'])
        n = len(tdf)
        wins = (tdf['pnl'] > 0).sum()
        wr = wins/n*100 if n > 0 else 0
        total = tdf['pnl'].sum()
        pair_summary.append((pair, n, wr, total))

    all_trades.sort()
    tdf = pd.DataFrame(all_trades, columns=['date','pnl','pair','hold','reason'])
    days = (tdf['date'].max() - tdf['date'].min()).days
    months = days / 30.44

    n = len(tdf)
    wins = (tdf['pnl'] > 0).sum()
    wr = wins/n*100
    total = tdf['pnl'].sum()
    monthly = total / months if months > 0 else 0

    print(f'\n  データ期間: {tdf["date"].min().date()} ~ {tdf["date"].max().date()} ({days}日≒{months:.1f}ヶ月)')
    print(f'\n  === 全体 ===')
    print(f'  取引回数: {n:,}  勝率: {wr:.1f}%')
    print(f'  平均勝ち: {tdf.loc[tdf["pnl"]>0,"pnl"].mean():+,.0f}円')
    print(f'  平均負け: {tdf.loc[tdf["pnl"]<=0,"pnl"].mean():+,.0f}円')
    print(f'  最大勝ち: {tdf["pnl"].max():+,.0f}円')
    print(f'  最大負け: {tdf["pnl"].min():+,.0f}円')
    print(f'  月間損益: {monthly:+,.0f}円')
    print(f'  月利(対50万): {monthly/500000*100:+.1f}%')

    print(f'\n  === 決済理由別 ===')
    for reason, label in [('TP','TP'),('Z','Z'),('TO','TO'),('SL','SL')]:
        rt = tdf[tdf['reason']==reason]
        if len(rt) == 0:
            continue
        print(f'  {label}: {len(rt)}回({len(rt)/n*100:.0f}%) 勝率{(rt["pnl"]>0).mean()*100:.0f}% 合計{rt["pnl"].sum():+,.0f}円')

    print(f'\n  === ペア別（月間損益順）===')
    print(f'  {"Pair":<10} {"回数":>5} {"勝率":>5} {"月間損益":>10} {"月利":>7}')
    print(f'  {"-"*42}')
    pair_results = []
    for pair in ALL_PAIRS:
        pt = tdf[tdf['pair']==pair]
        if len(pt) == 0:
            continue
        pn = len(pt)
        pwr = (pt['pnl']>0).mean()*100
        pm = pt['pnl'].sum()/months if months > 0 else 0
        m = MARGINS.get(pair, 50000)
        pmo = pm / m * 100
        pair_results.append((pair, pn, pwr, pm, pmo))

    pair_results.sort(key=lambda x: -x[3])
    plus = 0
    minus = 0
    for pair, pn, pwr, pm, pmo in pair_results:
        if pm > 0: plus += 1
        else: minus += 1
        print(f'  {pair:<10} {pn:>5} {pwr:>4.0f}% {pm:>+10,.0f} {pmo:>+6.1f}%')
    print(f'  {"-"*42}')
    print(f'  合計: {monthly:+,.0f}円  プラス{plus} マイナス{minus}')


if __name__ == '__main__':
    main()
