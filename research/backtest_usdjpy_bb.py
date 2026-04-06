#!/usr/bin/env python3
"""
USD/JPY BB逆張り＋押し目 × 3段マーチン バックテスト
実チャートデータ（yfinance）を使用
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from collections import defaultdict

# ============================================================
# パラメータ
# ============================================================
INITIAL_CAPITAL = 1_000_000  # 初期資金（円）
RISK_PER_TRADE = 0.008       # 1トレードリスク 0.8%
RR_RATIO = 2.0               # リスクリワード比（2.5→2.0に引き下げ）
ATR_FILTER_MULT = 2.2        # ATRフィルター倍率
SPREAD_PIPS = 0.3             # 想定スプレッド（pips）
SPREAD = SPREAD_PIPS * 0.01  # 価格ベース（USD/JPYなので0.01=1pip）

# シグナル別SL倍率（動的SL）
SL_ATR_MULT = {
    'BB_reversal': 2.0,       # 逆張りは余裕を持つ
    'Fast_BB': 1.8,
    'Pullback': 1.5,
}

# マーチンゲール倍率（緩和: 4.84→2.0）
MARTIN_MULTIPLIERS = [1.0, 1.5, 2.0]
MAX_MARTIN_STAGE = 3

# ブレイクイーブン＋トレーリング
BE_TRIGGER_RR = 1.0           # SL幅分の利益でBE発動
PARTIAL_CLOSE_RR = 1.5        # SL幅×1.5で50%利確
PARTIAL_CLOSE_PCT = 0.5       # 利確割合
TRAIL_ATR_MULT = 0.5          # トレーリングSLの余裕幅

# 最大保有期間
MAX_HOLDING_BARS = 20

# DD停止ライン
MONTHLY_DD_LIMIT = -0.15
ANNUAL_DD_LIMIT = -0.20

# 取引時間帯（UTC）
TRADING_HOUR_START = 7
TRADING_HOUR_END = 22


# ============================================================
# インジケーター計算
# ============================================================
def calc_indicators(df):
    """全テクニカル指標を計算"""
    c = df['Close'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)

    # SMA
    df['SMA200'] = pd.Series(c).rolling(200).mean().values
    df['SMA50'] = pd.Series(c).rolling(50).mean().values
    df['SMA20'] = pd.Series(c).rolling(20).mean().values

    # BB(20, 2.5σ) — メイン
    sma20 = pd.Series(c).rolling(20).mean()
    std20 = pd.Series(c).rolling(20).std()
    df['BB_upper'] = (sma20 + 2.5 * std20).values
    df['BB_lower'] = (sma20 - 2.5 * std20).values

    # BB(10, 2.0σ) — 高速
    sma10 = pd.Series(c).rolling(10).mean()
    std10 = pd.Series(c).rolling(10).std()
    df['FBB_upper'] = (sma10 + 2.0 * std10).values
    df['FBB_lower'] = (sma10 - 2.0 * std10).values

    # RSI(10)
    delta = pd.Series(c).diff()
    gain = delta.clip(lower=0).rolling(10).mean()
    loss = (-delta.clip(upper=0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = (100 - 100 / (1 + rs)).values

    # ATR(14)
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    df['ATR'] = pd.Series(tr).rolling(14).mean().values
    df['ATR_MA100'] = pd.Series(df['ATR']).rolling(100).mean().values

    # SMAトレンド方向（5期間前と比較）
    df['SMA200_up'] = df['SMA200'] > pd.Series(df['SMA200']).shift(5).values
    df['SMA50_up'] = df['SMA50'] > pd.Series(df['SMA50']).shift(5).values

    return df


# ============================================================
# シグナル判定
# ============================================================
def check_signals(row, prev_row):
    """3種のエントリーシグナルをチェック。戻り値: (direction, signal_type) or None"""
    close = row['Close']
    rsi = row['RSI']

    # --- フィルター ---
    if pd.isna(row['ATR']) or pd.isna(row['ATR_MA100']):
        return None
    if row['ATR'] >= row['ATR_MA100'] * ATR_FILTER_MULT:
        return None

    # 時間帯フィルター
    hour = row.name.hour if hasattr(row.name, 'hour') else 0
    if hour < TRADING_HOUR_START or hour >= TRADING_HOUR_END:
        return None

    # NaNチェック
    required = ['SMA200', 'SMA50', 'SMA20', 'BB_upper', 'BB_lower',
                'FBB_upper', 'FBB_lower', 'RSI', 'ATR']
    if any(pd.isna(row[k]) for k in required):
        return None
    if prev_row is not None and any(pd.isna(prev_row.get(k, np.nan)) for k in ['Close', 'BB_upper', 'BB_lower']):
        return None

    sma200_up = row['SMA200_up']

    # --- シグナル1: BB2.5σ逆張り（メイン）---
    if prev_row is not None:
        prev_close = prev_row['Close']
        # 買い: トレンド上向き + 前足がBB下限タッチ + 当足がBB下限を回復 + RSI < 38
        if sma200_up and prev_close <= prev_row['BB_lower'] and close > row['BB_lower'] and rsi < 38:
            return ('BUY', 'BB_reversal')
        # 売り: トレンド下向き + 前足がBB上限タッチ + 当足がBB上限を下抜け + RSI > 62
        if not sma200_up and prev_close >= prev_row['BB_upper'] and close < row['BB_upper'] and rsi > 62:
            return ('SELL', 'BB_reversal')

    # --- シグナル2: 高速BB逆張り（サブ）---
    sma50_up = row['SMA50_up']
    # 買い: SMA200・SMA50ともに上向き + BB下限タッチ + RSI < 42
    if sma200_up and sma50_up and close <= row['FBB_lower'] and rsi < 42:
        return ('BUY', 'Fast_BB')
    # 売り: SMA200・SMA50ともに下向き + BB上限タッチ + RSI > 58
    if not sma200_up and not sma50_up and close >= row['FBB_upper'] and rsi > 58:
        return ('SELL', 'Fast_BB')

    # --- シグナル3: 押し目・戻り売り（条件厳格化）---
    sma20 = row['SMA20']
    sma50 = row['SMA50']
    atr = row['ATR']
    sma_gap = abs(sma20 - sma50)
    # 追加条件: SMA20-SMA50の間隔がATR×2以上（十分な押し目幅）
    if sma_gap >= atr * 2:
        # 買い: SMA200上向き + 価格がSMA20とSMA50の間 + RSI 35〜45
        if sma200_up:
            lower_band = min(sma20, sma50)
            upper_band = max(sma20, sma50)
            if lower_band <= close <= upper_band and 35 <= rsi <= 45:
                return ('BUY', 'Pullback')
        # 売り: SMA200下向き + 価格がSMA20とSMA50の間 + RSI 55〜65
        if not sma200_up:
            lower_band = min(sma20, sma50)
            upper_band = max(sma20, sma50)
            if lower_band <= close <= upper_band and 55 <= rsi <= 65:
                return ('SELL', 'Pullback')

    return None


# ============================================================
# バックテストエンジン（改善版: BE/トレーリング/部分利確/時間切れ対応）
# ============================================================
class RowProxy:
    """dictデータにname属性を付与するプロキシ"""
    def __init__(self, data, name):
        self._data = data
        self.name = name
    def __getitem__(self, key):
        return self._data[key]
    def get(self, key, default=None):
        return self._data.get(key, default)


def run_backtest(df, label=""):
    """メインバックテストループ"""
    capital = float(INITIAL_CAPITAL)
    peak_capital = capital
    month_start_capital = capital
    year_start_capital = capital

    position = None
    martin_stage = 0
    consecutive_losses = 0
    system_stopped = False
    stop_reason = ""

    trades = []
    monthly_pnl = defaultdict(float)
    current_month = None
    current_year = None

    rows = df.to_dict('index')
    indices = list(rows.keys())

    for i in range(1, len(indices)):
        idx = indices[i]
        prev_idx = indices[i - 1]
        row = rows[idx]
        prev_row = rows[prev_idx]
        row_name = idx

        close = float(row['Close'])
        high = float(row['High'])
        low = float(row['Low'])

        # 月・年の切り替え検知
        ts = row_name
        month_key = f"{ts.year}-{ts.month:02d}"
        year_key = str(ts.year)

        if current_month is None:
            current_month = month_key
            month_start_capital = capital
        if current_year is None:
            current_year = year_key
            year_start_capital = capital

        if month_key != current_month:
            month_start_capital = capital
            current_month = month_key
        if year_key != current_year:
            year_start_capital = capital
            current_year = year_key

        if system_stopped:
            continue

        # --- ポジション管理 ---
        if position is not None:
            position['bars_held'] += 1
            closed = False
            pnl = 0.0
            result = ''
            sl = position['sl']
            tp = position['tp']
            entry = position['entry']
            lots = position['lots']
            direction = position['direction']
            sl_distance = position['sl_distance']

            # --- ブレイクイーブン＋トレーリング判定（SL/TP判定前に状態更新）---
            if direction == 'BUY':
                unrealized_move = high - entry
                # BE発動: 1:1 RR到達
                if not position['be_activated'] and unrealized_move >= sl_distance * BE_TRIGGER_RR:
                    position['be_activated'] = True
                    position['sl'] = entry + SPREAD  # BEに移動（スプレッド分のみ）
                    sl = position['sl']

                # 部分利確: 1:1.5 RR到達
                if not position['partial_closed'] and unrealized_move >= sl_distance * PARTIAL_CLOSE_RR:
                    partial_pnl = (sl_distance * PARTIAL_CLOSE_RR - SPREAD) * lots * PARTIAL_CLOSE_PCT
                    capital += partial_pnl
                    monthly_pnl[month_key] += partial_pnl
                    position['partial_closed'] = True
                    position['partial_pnl'] = partial_pnl
                    position['lots'] = lots * (1 - PARTIAL_CLOSE_PCT)
                    lots = position['lots']
                    # トレーリングSL開始
                    trail_sl = high - float(row['ATR']) * TRAIL_ATR_MULT
                    if trail_sl > sl:
                        position['sl'] = trail_sl
                        sl = trail_sl

                # トレーリング更新（部分利確後）
                if position['partial_closed']:
                    trail_sl = high - float(row['ATR']) * TRAIL_ATR_MULT
                    if trail_sl > position['sl']:
                        position['sl'] = trail_sl
                        sl = position['sl']

                # SL判定
                if low <= sl:
                    pnl = (sl - entry - SPREAD) * lots
                    closed = True
                    if position['be_activated'] and not position['partial_closed']:
                        result = 'BE'
                    elif position['partial_closed']:
                        result = 'TRAIL'
                    else:
                        result = 'SL'
                # TP判定
                elif high >= tp:
                    pnl = (tp - entry - SPREAD) * lots
                    closed = True
                    result = 'TP'

            else:  # SELL
                unrealized_move = entry - low
                if not position['be_activated'] and unrealized_move >= sl_distance * BE_TRIGGER_RR:
                    position['be_activated'] = True
                    position['sl'] = entry - SPREAD
                    sl = position['sl']

                if not position['partial_closed'] and unrealized_move >= sl_distance * PARTIAL_CLOSE_RR:
                    partial_pnl = (sl_distance * PARTIAL_CLOSE_RR - SPREAD) * lots * PARTIAL_CLOSE_PCT
                    capital += partial_pnl
                    monthly_pnl[month_key] += partial_pnl
                    position['partial_closed'] = True
                    position['partial_pnl'] = partial_pnl
                    position['lots'] = lots * (1 - PARTIAL_CLOSE_PCT)
                    lots = position['lots']
                    trail_sl = low + float(row['ATR']) * TRAIL_ATR_MULT
                    if trail_sl < sl:
                        position['sl'] = trail_sl
                        sl = trail_sl

                if position['partial_closed']:
                    trail_sl = low + float(row['ATR']) * TRAIL_ATR_MULT
                    if trail_sl < position['sl']:
                        position['sl'] = trail_sl
                        sl = position['sl']

                if high >= sl:
                    pnl = (entry - sl - SPREAD) * lots
                    closed = True
                    if position['be_activated'] and not position['partial_closed']:
                        result = 'BE'
                    elif position['partial_closed']:
                        result = 'TRAIL'
                    else:
                        result = 'SL'
                elif low <= tp:
                    pnl = (entry - tp - SPREAD) * lots
                    closed = True
                    result = 'TP'

            # 最大保有期間チェック
            if not closed and position['bars_held'] >= MAX_HOLDING_BARS:
                if direction == 'BUY':
                    pnl = (close - entry - SPREAD) * lots
                else:
                    pnl = (entry - close - SPREAD) * lots
                closed = True
                result = 'TIME'

            if closed:
                # 部分利確分を加算
                total_pnl = pnl + position.get('partial_pnl', 0)
                capital += pnl  # partial_pnlは既にcapitalに加算済み
                monthly_pnl[month_key] += pnl

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': row_name,
                    'direction': direction,
                    'signal': position['signal'],
                    'entry_price': entry,
                    'sl': position['original_sl'],
                    'tp': tp,
                    'lots': position['original_lots'],
                    'stage': position['stage'] + 1,
                    'result': result,
                    'pnl': total_pnl,
                    'capital_after': capital,
                    'bars_held': position['bars_held'],
                })

                # マーチン段階更新（SLのみ負けカウント、BE/TIME/TRAILは負けに含めない）
                if result == 'SL':
                    consecutive_losses += 1
                    if consecutive_losses >= MAX_MARTIN_STAGE:
                        martin_stage = 0
                        consecutive_losses = 0
                    else:
                        martin_stage = min(consecutive_losses, MAX_MARTIN_STAGE - 1)
                else:
                    consecutive_losses = 0
                    martin_stage = 0

                position = None

                if capital > peak_capital:
                    peak_capital = capital

                monthly_return = (capital - month_start_capital) / month_start_capital
                annual_return = (capital - year_start_capital) / year_start_capital
                if monthly_return <= MONTHLY_DD_LIMIT:
                    system_stopped = True
                    stop_reason = f"月次DD停止 ({monthly_return:.1%}) at {row_name}"
                if annual_return <= ANNUAL_DD_LIMIT:
                    system_stopped = True
                    stop_reason = f"年次DD停止 ({annual_return:.1%}) at {row_name}"

            if position is not None or system_stopped:
                continue

        # --- 新規エントリー判定 ---
        row_p = RowProxy(row, row_name)
        prev_p = RowProxy(prev_row, prev_idx)

        signal = check_signals(row_p, prev_p)
        if signal is None:
            continue

        direction, signal_type = signal
        atr = float(row['ATR'])

        # シグナル別SL倍率
        sl_mult = SL_ATR_MULT.get(signal_type, 1.5)
        sl_distance = atr * sl_mult
        tp_distance = sl_distance * RR_RATIO

        if direction == 'BUY':
            entry_price = close + SPREAD / 2
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            entry_price = close - SPREAD / 2
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        risk_amount = capital * RISK_PER_TRADE
        martin_mult = MARTIN_MULTIPLIERS[martin_stage]
        risk_amount_martin = risk_amount * martin_mult

        if sl_distance <= 0:
            continue
        lots = risk_amount_martin / sl_distance

        margin_required = (lots * entry_price) / 25
        if margin_required > capital * 0.9:
            continue

        position = {
            'direction': direction,
            'entry': entry_price,
            'sl': sl_price,
            'tp': tp_price,
            'original_sl': sl_price,
            'original_lots': lots,
            'sl_distance': sl_distance,
            'lots': lots,
            'signal': signal_type,
            'stage': martin_stage,
            'entry_time': row_name,
            'bars_held': 0,
            'be_activated': False,
            'partial_closed': False,
            'partial_pnl': 0,
        }

    return trades, capital, monthly_pnl, system_stopped, stop_reason


# ============================================================
# レポート出力
# ============================================================
def print_report(trades, final_capital, monthly_pnl, stopped, stop_reason, label, data_rows):
    """バックテスト結果を表示"""
    print(f"\n{'='*70}")
    print(f"  USD/JPY BB逆張り＋マーチン バックテスト結果 [{label}]")
    print(f"{'='*70}")

    if not trades:
        print("  トレードなし")
        return

    total_trades = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    breakevens = [t for t in trades if t['pnl'] == 0]
    win_rate = len(wins) / total_trades * 100

    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))
    net_profit = final_capital - INITIAL_CAPITAL
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # 最大ドローダウン計算
    peak = INITIAL_CAPITAL
    max_dd = 0
    running = INITIAL_CAPITAL
    for t in trades:
        running += t['pnl']
        if running > peak:
            peak = running
        dd = (peak - running) / peak
        if dd > max_dd:
            max_dd = dd

    # 期間
    first_trade = trades[0]['entry_time']
    last_trade = trades[-1]['exit_time']

    print(f"\n  期間: {first_trade} 〜 {last_trade}")
    print(f"  データ本数: {data_rows:,}")
    print(f"  初期資金: ¥{INITIAL_CAPITAL:,.0f}")
    print(f"  最終資金: ¥{final_capital:,.0f}")
    print(f"  純損益: ¥{net_profit:,.0f} ({net_profit/INITIAL_CAPITAL:.1%})")

    if stopped:
        print(f"  ⚠ システム停止: {stop_reason}")

    print(f"\n  --- 全体統計 ---")
    print(f"  総取引数:     {total_trades}")
    print(f"  勝ち:         {len(wins)}")
    print(f"  負け:         {len(losses)}")
    print(f"  引分(BE):     {len(breakevens)}")
    print(f"  勝率:         {win_rate:.1f}%")
    print(f"  PF:           {pf:.2f}")
    print(f"  最大DD:       {max_dd:.1%}")
    print(f"  平均利益:     ¥{gross_profit/len(wins):,.0f}" if wins else "  平均利益:     N/A")
    print(f"  平均損失:     ¥{gross_loss/len(losses):,.0f}" if losses else "  平均損失:     N/A")

    # 決済理由別内訳
    exit_types = defaultdict(int)
    for t in trades:
        exit_types[t['result']] += 1
    print(f"\n  --- 決済理由別 ---")
    for et in ['TP', 'SL', 'BE', 'TRAIL', 'TIME']:
        if exit_types[et] > 0:
            et_pnl = sum(t['pnl'] for t in trades if t['result'] == et)
            print(f"  {et:6s}: {exit_types[et]:4d}件  損益 ¥{et_pnl:>+12,.0f}")

    avg_bars = np.mean([t['bars_held'] for t in trades])
    print(f"  平均保有期間:  {avg_bars:.1f}本")

    # シグナル別統計
    print(f"\n  --- シグナル別 ---")
    for sig_type in ['BB_reversal', 'Fast_BB', 'Pullback']:
        sig_trades = [t for t in trades if t['signal'] == sig_type]
        if not sig_trades:
            print(f"  {sig_type:15s}: 0件")
            continue
        sig_wins = sum(1 for t in sig_trades if t['pnl'] > 0)
        sig_pnl = sum(t['pnl'] for t in sig_trades)
        print(f"  {sig_type:15s}: {len(sig_trades):4d}件  "
              f"勝率 {sig_wins/len(sig_trades)*100:5.1f}%  "
              f"損益 ¥{sig_pnl:>+12,.0f}")

    # マーチン段階別統計
    print(f"\n  --- マーチン段階別 ---")
    for stage in range(MAX_MARTIN_STAGE):
        st_trades = [t for t in trades if t['stage'] == stage + 1]
        if not st_trades:
            print(f"  Stage {stage+1} (×{MARTIN_MULTIPLIERS[stage]:.2f}): 0件")
            continue
        st_wins = sum(1 for t in st_trades if t['pnl'] > 0)
        st_pnl = sum(t['pnl'] for t in st_trades)
        print(f"  Stage {stage+1} (×{MARTIN_MULTIPLIERS[stage]:.2f}): {len(st_trades):4d}件  "
              f"勝率 {st_wins/len(st_trades)*100:5.1f}%  "
              f"損益 ¥{st_pnl:>+12,.0f}")

    # 月次損益
    if monthly_pnl:
        print(f"\n  --- 月次損益 ---")
        months_sorted = sorted(monthly_pnl.keys())
        monthly_returns = []
        running_cap = INITIAL_CAPITAL
        for m in months_sorted:
            pnl = monthly_pnl[m]
            ret = pnl / running_cap if running_cap > 0 else 0
            monthly_returns.append(ret)
            print(f"  {m}: ¥{pnl:>+12,.0f}  ({ret:>+6.1%})")
            running_cap += pnl

        if monthly_returns:
            med_ret = np.median(monthly_returns)
            avg_ret = np.mean(monthly_returns)
            positive_months = sum(1 for r in monthly_returns if r > 0)
            target_months = sum(1 for r in monthly_returns if r >= 0.08)
            print(f"\n  月利中央値:     {med_ret:+.1%}")
            print(f"  月利平均:       {avg_ret:+.1%}")
            print(f"  プラス月:       {positive_months}/{len(monthly_returns)}")
            print(f"  月利8%達成月:   {target_months}/{len(monthly_returns)}")

    # 直近トレードログ（最大20件）
    print(f"\n  --- 直近トレードログ（最大20件）---")
    recent = trades[-20:]
    for t in recent:
        print(f"  {str(t['entry_time'])[:16]} {t['direction']:4s} "
              f"{t['signal']:12s} Stage{t['stage']} "
              f"Entry:{t['entry_price']:.3f} SL:{t['sl']:.3f} TP:{t['tp']:.3f} "
              f"→ {t['result']} ¥{t['pnl']:>+10,.0f}")

    print()


# ============================================================
# メイン
# ============================================================
def main():
    print("USD/JPY BB逆張り＋マーチン バックテスト")
    print("実チャートデータ（yfinance）使用")
    print("=" * 50)

    configs = [
        ("15分足（直近60日）", "60d", "15m"),
        ("1時間足（約2.8年）", "730d", "1h"),
    ]

    for label, period, interval in configs:
        print(f"\n>>> データ取得中: {label} ...")
        try:
            df = yf.download('USDJPY=X', period=period, interval=interval, progress=False)
            if df.empty:
                print(f"  データ取得失敗: {label}")
                continue

            # マルチレベルカラムをフラット化
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            print(f"  取得: {len(df)}本 ({df.index[0]} 〜 {df.index[-1]})")

            # インジケーター計算
            df = calc_indicators(df)

            # NaN行を除外してバックテスト対象を特定
            valid = df.dropna(subset=['SMA200', 'ATR_MA100'])
            print(f"  有効データ: {len(valid)}本（SMA200算出後）")

            # バックテスト実行
            trades, final_cap, monthly_pnl, stopped, stop_reason = run_backtest(df)

            # レポート
            print_report(trades, final_cap, monthly_pnl, stopped, stop_reason, label, len(df))

        except Exception as e:
            print(f"  エラー: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
