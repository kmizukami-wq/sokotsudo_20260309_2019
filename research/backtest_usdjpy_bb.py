#!/usr/bin/env python3
"""
FX BB逆張り＋押し目 × 3段マーチン バックテスト（複数通貨ペア対応）
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
RR_RATIO = 2.0               # リスクリワード比
ATR_FILTER_MULT = 2.2        # ATRフィルター倍率

# ペア基本設定（FXTF実取扱ペアに対応）
PAIR_BASE = {
    'USDJPY=X': {'name': 'USD/JPY', 'pip': 0.01,  'quote_to_jpy': 1.0},
    'EURUSD=X': {'name': 'EUR/USD', 'pip': 0.0001, 'quote_to_jpy': 150.0},
    'EURJPY=X': {'name': 'EUR/JPY', 'pip': 0.01,  'quote_to_jpy': 1.0},
    'GBPUSD=X': {'name': 'GBP/USD', 'pip': 0.0001, 'quote_to_jpy': 150.0},
    'GBPJPY=X': {'name': 'GBP/JPY', 'pip': 0.01,  'quote_to_jpy': 1.0},
    'AUDJPY=X': {'name': 'AUD/JPY', 'pip': 0.01,  'quote_to_jpy': 1.0},
    'NZDJPY=X': {'name': 'NZD/JPY', 'pip': 0.01,  'quote_to_jpy': 1.0},
    'ZARJPY=X': {'name': 'ZAR/JPY', 'pip': 0.01,  'quote_to_jpy': 1.0},
    'CHFJPY=X': {'name': 'CHF/JPY', 'pip': 0.01,  'quote_to_jpy': 1.0},
    'USDCHF=X': {'name': 'USD/CHF', 'pip': 0.0001, 'quote_to_jpy': 170.0},
}

# FXTF スプレッド設定（pips）
BROKERS = {
    'FXTF(MT4)': {
        'desc': 'MT4対応, 低コスト(スプレッド+手数料), EA利用可',
        'api': 'MT4 EA',
        'spreads': {
            'USDJPY=X': 0.3, 'EURUSD=X': 0.3, 'GBPUSD=X': 0.7,
            'EURJPY=X': 0.5, 'GBPJPY=X': 1.0, 'AUDJPY=X': 0.5,
            'NZDJPY=X': 1.0, 'ZARJPY=X': 1.5, 'CHFJPY=X': 0.8,
            'USDCHF=X': 1.0,
        },
    },
}

# デフォルトPAIRS（後で業者別に差し替え）
PAIRS = {k: {**v, 'spread_pips': 0.3} for k, v in PAIR_BASE.items()}

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
        # 買い: トレンド上向き + 前足がBB下限タッチ + 当足がBB下限を回復 + RSI < 42
        if sma200_up and prev_close <= prev_row['BB_lower'] and close > row['BB_lower'] and rsi < 42:
            return ('BUY', 'BB_reversal')
        # 売り: トレンド下向き + 前足がBB上限タッチ + 当足がBB上限を下抜け + RSI > 58
        if not sma200_up and prev_close >= prev_row['BB_upper'] and close < row['BB_upper'] and rsi > 58:
            return ('SELL', 'BB_reversal')

    # --- シグナル2: 高速BB逆張り（サブ）---
    sma50_up = row['SMA50_up']
    # 買い: SMA200・SMA50ともに上向き + BB下限タッチ + RSI < 48
    if sma200_up and sma50_up and close <= row['FBB_lower'] and rsi < 48:
        return ('BUY', 'Fast_BB')
    # 売り: SMA200・SMA50ともに下向き + BB上限タッチ + RSI > 52
    if not sma200_up and not sma50_up and close >= row['FBB_upper'] and rsi > 52:
        return ('SELL', 'Fast_BB')

    # --- シグナル3: 押し目・戻り売り（条件厳格化）---
    sma20 = row['SMA20']
    sma50 = row['SMA50']
    atr = row['ATR']
    sma_gap = abs(sma20 - sma50)
    # 追加条件: SMA20-SMA50の間隔がATR×2以上（十分な押し目幅）
    if sma_gap >= atr * 2:
        # 買い: SMA200上向き + 価格がSMA20とSMA50の間 + RSI 30〜50
        if sma200_up:
            lower_band = min(sma20, sma50)
            upper_band = max(sma20, sma50)
            if lower_band <= close <= upper_band and 30 <= rsi <= 50:
                return ('BUY', 'Pullback')
        # 売り: SMA200下向き + 価格がSMA20とSMA50の間 + RSI 50〜70
        if not sma200_up:
            lower_band = min(sma20, sma50)
            upper_band = max(sma20, sma50)
            if lower_band <= close <= upper_band and 50 <= rsi <= 70:
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


def run_backtest(df, spread, quote_to_jpy=1.0, label=""):
    """メインバックテストループ（spread: 価格ベースのスプレッド, quote_to_jpy: 損益の円換算レート）"""
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
                    position['sl'] = entry + spread  # BEに移動（スプレッド分のみ）
                    sl = position['sl']

                # 部分利確: 1:1.5 RR到達
                if not position['partial_closed'] and unrealized_move >= sl_distance * PARTIAL_CLOSE_RR:
                    partial_pnl = (sl_distance * PARTIAL_CLOSE_RR - spread) * lots * PARTIAL_CLOSE_PCT * quote_to_jpy
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
                    pnl = (sl - entry - spread) * lots * quote_to_jpy
                    closed = True
                    if position['be_activated'] and not position['partial_closed']:
                        result = 'BE'
                    elif position['partial_closed']:
                        result = 'TRAIL'
                    else:
                        result = 'SL'
                # TP判定
                elif high >= tp:
                    pnl = (tp - entry - spread) * lots * quote_to_jpy
                    closed = True
                    result = 'TP'

            else:  # SELL
                unrealized_move = entry - low
                if not position['be_activated'] and unrealized_move >= sl_distance * BE_TRIGGER_RR:
                    position['be_activated'] = True
                    position['sl'] = entry - spread
                    sl = position['sl']

                if not position['partial_closed'] and unrealized_move >= sl_distance * PARTIAL_CLOSE_RR:
                    partial_pnl = (sl_distance * PARTIAL_CLOSE_RR - spread) * lots * PARTIAL_CLOSE_PCT * quote_to_jpy
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
                    pnl = (entry - sl - spread) * lots * quote_to_jpy
                    closed = True
                    if position['be_activated'] and not position['partial_closed']:
                        result = 'BE'
                    elif position['partial_closed']:
                        result = 'TRAIL'
                    else:
                        result = 'SL'
                elif low <= tp:
                    pnl = (entry - tp - spread) * lots * quote_to_jpy
                    closed = True
                    result = 'TP'

            # 最大保有期間チェック
            if not closed and position['bars_held'] >= MAX_HOLDING_BARS:
                if direction == 'BUY':
                    pnl = (close - entry - spread) * lots * quote_to_jpy
                else:
                    pnl = (entry - close - spread) * lots * quote_to_jpy
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
            entry_price = close + spread / 2
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            entry_price = close - spread / 2
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        risk_amount = capital * RISK_PER_TRADE
        martin_mult = MARTIN_MULTIPLIERS[martin_stage]
        risk_amount_martin = risk_amount * martin_mult

        if sl_distance <= 0:
            continue
        lots = risk_amount_martin / (sl_distance * quote_to_jpy)

        margin_required = (lots * entry_price * quote_to_jpy) / 25
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
def print_report(trades, final_capital, monthly_pnl, stopped, stop_reason, label, data_rows, pair_name="", pip_value=0.01):
    """バックテスト結果を表示"""
    price_fmt = '.3f' if pip_value >= 0.01 else '.5f'
    print(f"\n{'='*70}")
    print(f"  {pair_name} BB逆張り＋マーチン バックテスト結果 [{label}]")
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
              f"Entry:{t['entry_price']:{price_fmt}} SL:{t['sl']:{price_fmt}} TP:{t['tp']:{price_fmt}} "
              f"→ {t['result']} ¥{t['pnl']:>+10,.0f}")

    print()


# ============================================================
# ポートフォリオシミュレーション（複数ペア同時稼働）
# ============================================================
def run_portfolio_simulation(all_pair_trades, excluded_pairs=None):
    """
    全ペアのトレードログを時系列マージし、共有資金でのポートフォリオ成績を算出。
    各トレードのPnLは個別BT時の初期資金ベースなので、共有資金上での比率で再スケールする。
    """
    if excluded_pairs is None:
        excluded_pairs = set()

    # 全トレードをマージ（ペア名付き）
    merged = []
    for pair_name, trades in all_pair_trades.items():
        if pair_name in excluded_pairs:
            continue
        for t in trades:
            merged.append({**t, 'pair': pair_name})

    if not merged:
        return

    # exit_timeでソート
    merged.sort(key=lambda t: t['exit_time'])

    n_pairs = len(all_pair_trades) - len(excluded_pairs)
    capital = float(INITIAL_CAPITAL)
    peak = capital
    max_dd = 0.0
    monthly_pnl = defaultdict(float)

    for t in merged:
        # PnLを現在の資金ベースにスケール
        # 個別BTでは初期資金¥1Mの0.8%リスク。ポートフォリオでは同じ¥1Mを共有。
        # → 各トレードのPnLは初期資金に対する比率で適用
        pnl_ratio = t['pnl'] / INITIAL_CAPITAL
        scaled_pnl = pnl_ratio * capital

        capital += scaled_pnl
        month_key = f"{t['exit_time'].year}-{t['exit_time'].month:02d}"
        monthly_pnl[month_key] += scaled_pnl

        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak
        if dd > max_dd:
            max_dd = dd

    # レポート出力
    total = len(merged)
    wins = sum(1 for t in merged if t['pnl'] > 0)
    losses = sum(1 for t in merged if t['pnl'] < 0)
    net = capital - INITIAL_CAPITAL

    print(f"\n{'='*85}")
    print(f"  ポートフォリオシミュレーション（{n_pairs}ペア同時稼働・共有資金）")
    print(f"{'='*85}")
    print(f"  初期資金: ¥{INITIAL_CAPITAL:,.0f}")
    print(f"  最終資金: ¥{capital:,.0f}")
    print(f"  純損益:   ¥{net:,.0f} ({net/INITIAL_CAPITAL:.1%})")
    print(f"  総取引数: {total}  勝ち: {wins}  負け: {losses}  勝率: {wins/total*100:.1f}%")
    print(f"  最大DD:   {max_dd:.1%}")

    # 月次ブレークダウン
    months_sorted = sorted(monthly_pnl.keys())
    monthly_returns = []
    run_cap = INITIAL_CAPITAL
    print(f"\n  --- 月次損益 ---")
    for m in months_sorted:
        pnl = monthly_pnl[m]
        ret = pnl / run_cap if run_cap > 0 else 0
        monthly_returns.append(ret)
        month_trades = sum(1 for t in merged if f"{t['exit_time'].year}-{t['exit_time'].month:02d}" == m)
        print(f"  {m}: ¥{pnl:>+12,.0f}  ({ret:>+6.1%})  [{month_trades}件]")
        run_cap += pnl

    if monthly_returns:
        med = np.median(monthly_returns)
        avg = np.mean(monthly_returns)
        pos = sum(1 for r in monthly_returns if r > 0)
        target = sum(1 for r in monthly_returns if r >= 0.08)
        print(f"\n  月利中央値:     {med:+.1%}")
        print(f"  月利平均:       {avg:+.1%}")
        print(f"  プラス月:       {pos}/{len(monthly_returns)}")
        print(f"  月利8%達成月:   {target}/{len(monthly_returns)}")

        # 年利換算
        if avg > 0:
            annual_compound = (1 + avg) ** 12 - 1
            print(f"  年利換算(複利): {annual_compound:+.0%}")

    # ペア別貢献度
    print(f"\n  --- ペア別貢献度 ---")
    pair_contrib = defaultdict(lambda: {'pnl': 0, 'count': 0})
    for t in merged:
        pair_contrib[t['pair']]['pnl'] += t['pnl'] / INITIAL_CAPITAL * INITIAL_CAPITAL
        pair_contrib[t['pair']]['count'] += 1

    sorted_contrib = sorted(pair_contrib.items(), key=lambda x: x[1]['pnl'], reverse=True)
    for pair, data in sorted_contrib:
        print(f"  {pair:<10s}: {data['count']:4d}件  損益 ¥{data['pnl']:>+12,.0f}")

    print()
    return {
        'capital': capital,
        'net': net,
        'max_dd': max_dd,
        'monthly_returns': monthly_returns,
        'total_trades': total,
    }


# ============================================================
# 為替レート動的取得
# ============================================================
def get_quote_to_jpy_rates():
    """USD/JPY等のレートを動的取得し、quote_to_jpy変換レートを更新"""
    rates = {}
    try:
        usdjpy = yf.download('USDJPY=X', period='1d', interval='1d', progress=False)
        if isinstance(usdjpy.columns, pd.MultiIndex):
            usdjpy.columns = usdjpy.columns.get_level_values(0)
        usdjpy_rate = float(usdjpy['Close'].iloc[-1])
        rates['USD'] = usdjpy_rate

        chfjpy = yf.download('CHFJPY=X', period='1d', interval='1d', progress=False)
        if isinstance(chfjpy.columns, pd.MultiIndex):
            chfjpy.columns = chfjpy.columns.get_level_values(0)
        rates['CHF'] = float(chfjpy['Close'].iloc[-1])

        cadjpy = yf.download('CADJPY=X', period='1d', interval='1d', progress=False)
        if isinstance(cadjpy.columns, pd.MultiIndex):
            cadjpy.columns = cadjpy.columns.get_level_values(0)
        rates['CAD'] = float(cadjpy['Close'].iloc[-1])

        print(f"  為替レート取得: USD/JPY={rates['USD']:.1f}, CHF/JPY={rates['CHF']:.1f}, CAD/JPY={rates['CAD']:.1f}")
    except Exception:
        rates = {'USD': 150.0, 'CHF': 170.0, 'CAD': 110.0}
        print(f"  為替レート取得失敗、デフォルト値使用")

    # PairsのquoteToJPYを更新
    for ticker, cfg in PAIRS.items():
        name = cfg['name']
        if name.endswith('/JPY'):
            cfg['quote_to_jpy'] = 1.0
        elif name.endswith('/USD'):
            cfg['quote_to_jpy'] = rates.get('USD', 150.0)
        elif name.endswith('/CHF'):
            cfg['quote_to_jpy'] = rates.get('CHF', 170.0)
        elif name.endswith('/CAD'):
            cfg['quote_to_jpy'] = rates.get('CAD', 110.0)

    return rates


# ============================================================
# ペア別バックテスト＆集計
# ============================================================
def run_all_pairs(broker_name, broker_spreads, data_cache):
    """指定業者のスプレッドで全ペアをバックテスト"""
    results = []
    all_pair_trades = {}

    for ticker, base in PAIR_BASE.items():
        pair_name = base['name']
        spread_pips = broker_spreads.get(ticker)
        if spread_pips is None:
            continue
        spread = spread_pips * base['pip']
        q2j = base['quote_to_jpy']

        if ticker not in data_cache:
            continue
        df = data_cache[ticker].copy()

        df = calc_indicators(df)
        trades, final_cap, monthly_pnl, stopped, stop_reason = run_backtest(
            df, spread=spread, quote_to_jpy=q2j
        )

        if trades:
            all_pair_trades[pair_name] = trades
            total = len(trades)
            wins = sum(1 for t in trades if t['pnl'] > 0)
            gp = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            gl = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
            pf = gp / gl if gl > 0 else float('inf')

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

            net = final_cap - INITIAL_CAPITAL
            if monthly_pnl:
                run_cap = INITIAL_CAPITAL
                m_rets = []
                for m in sorted(monthly_pnl.keys()):
                    r = monthly_pnl[m] / run_cap if run_cap > 0 else 0
                    m_rets.append(r)
                    run_cap += monthly_pnl[m]
                med_monthly = np.median(m_rets)
            else:
                med_monthly = 0

            results.append({
                'pair': pair_name, 'trades': total,
                'win_rate': wins / total * 100, 'pf': pf,
                'max_dd': max_dd, 'net_pnl': net,
                'med_monthly': med_monthly, 'stopped': stopped,
                'spread': spread_pips,
            })
        else:
            results.append({
                'pair': pair_name, 'trades': 0, 'win_rate': 0, 'pf': 0,
                'max_dd': 0, 'net_pnl': 0, 'med_monthly': 0, 'stopped': False,
                'spread': spread_pips,
            })

    return results, all_pair_trades


# ============================================================
# メイン
# ============================================================
def main():
    print("FX BB逆張り＋マーチン バックテスト（国内業者別・複数通貨ペア）")
    print("実チャートデータ（yfinance）15分足使用")
    print("=" * 70)

    # 為替レート動的取得
    get_quote_to_jpy_rates()

    # データ一括取得（業者間で共有）
    print("\n>>> 全ペアのデータ一括取得中 ...")
    data_cache = {}
    for ticker, base in PAIR_BASE.items():
        try:
            df = yf.download(ticker, period='60d', interval='15m', progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data_cache[ticker] = df
            print(f"  {base['name']}: {len(df)}本")
        except Exception as e:
            print(f"  {base['name']}: エラー - {e}")
    print(f"  取得完了: {len(data_cache)}/{len(PAIR_BASE)}ペア")

    # ============================================================
    # 業者別バックテスト
    # ============================================================
    broker_results = {}
    broker_portfolios = {}

    for broker_name, broker_cfg in BROKERS.items():
        print(f"\n{'#'*85}")
        print(f"  {broker_name}")
        print(f"  {broker_cfg['desc']}")
        print(f"  API: {broker_cfg['api']}")
        print(f"{'#'*85}")

        results, all_pair_trades = run_all_pairs(
            broker_name, broker_cfg['spreads'], data_cache
        )
        broker_results[broker_name] = results

        # ペア別サマリー
        if results:
            print(f"\n  {'ペア':<10s} {'SP':>4s} {'取引':>4s} {'勝率':>6s} {'PF':>6s} {'DD':>6s} {'純損益':>12s} {'月利中央':>7s}")
            print(f"  {'-'*10} {'-'*4} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*12} {'-'*7}")
            for r in sorted(results, key=lambda x: x['pf'], reverse=True):
                print(f"  {r['pair']:<10s} {r['spread']:4.1f} {r['trades']:4d} {r['win_rate']:5.1f}% "
                      f"{r['pf']:6.2f} {r['max_dd']:5.1%} ¥{r['net_pnl']:>+11,.0f} {r['med_monthly']:>+6.1%}")

            # ポートフォリオ
            if all_pair_trades:
                port = run_portfolio_simulation(all_pair_trades)
                broker_portfolios[broker_name] = port

    # ============================================================
    # 業者間比較サマリー
    # ============================================================
    if broker_portfolios:
        print(f"\n{'='*85}")
        print(f"  国内FX業者 比較サマリー（全ペア同時稼働・ポートフォリオ）")
        print(f"{'='*85}")
        print(f"  {'業者':<16s} {'API':>12s} {'取引数':>5s} {'月利平均':>8s} {'月利中央':>8s} {'最大DD':>7s} {'純損益':>14s}")
        print(f"  {'-'*16} {'-'*12} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*14}")

        for broker_name in BROKERS:
            if broker_name not in broker_portfolios:
                continue
            port = broker_portfolios[broker_name]
            api = BROKERS[broker_name]['api']
            avg_m = np.mean(port['monthly_returns']) if port['monthly_returns'] else 0
            med_m = np.median(port['monthly_returns']) if port['monthly_returns'] else 0
            print(f"  {broker_name:<16s} {api:>12s} {port['total_trades']:5d} "
                  f"{avg_m:>+7.1%} {med_m:>+7.1%} {port['max_dd']:6.1%} "
                  f"¥{port['net']:>+13,.0f}")

        # 推奨業者
        best_broker = max(broker_portfolios.items(), key=lambda x: x[1]['net'])
        lowest_dd = min(broker_portfolios.items(), key=lambda x: x[1]['max_dd'])
        print(f"\n  推奨（利益最大）: {best_broker[0]} — 純損益 ¥{best_broker[1]['net']:+,.0f}")
        print(f"  推奨（DD最小）:   {lowest_dd[0]} — 最大DD {lowest_dd[1]['max_dd']:.1%}")

        # プラスペア数比較
        print(f"\n  --- プラスペア数 ---")
        for broker_name, results in broker_results.items():
            profitable = sum(1 for r in results if r['net_pnl'] > 0 and r['trades'] > 0)
            total_p = sum(1 for r in results if r['trades'] > 0)
            losers = [r['pair'] for r in results if r['net_pnl'] < 0 and r['trades'] > 0]
            loser_str = f"  (マイナス: {', '.join(losers)})" if losers else ""
            print(f"  {broker_name:<16s}: {profitable}/{total_p}ペア{loser_str}")

        print()


if __name__ == '__main__':
    main()
