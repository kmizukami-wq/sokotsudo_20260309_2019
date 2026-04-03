"""
Technical indicators for BTC/JPY momentum bot v2.
Pure numpy implementation — no external TA library dependency.
"""

import numpy as np
from collections import deque


def true_range(highs, lows, closes):
    """Calculate True Range array.

    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    prev_closes = np.roll(closes, 1)
    prev_closes[0] = closes[0]

    hl = highs - lows
    hc = np.abs(highs - prev_closes)
    lc = np.abs(lows - prev_closes)

    return np.maximum(hl, np.maximum(hc, lc))


def atr(highs, lows, closes, period=14):
    """Average True Range using Wilder's smoothing."""
    tr = true_range(highs, lows, closes)

    if len(tr) < period:
        return np.full_like(tr, np.nan)

    result = np.full_like(tr, np.nan, dtype=float)
    # Initial ATR = simple average of first `period` TRs
    result[period - 1] = np.mean(tr[:period])

    # Wilder's smoothing: ATR = (prev_ATR * (period-1) + TR) / period
    for i in range(period, len(tr)):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


def rsi(closes, period=14):
    """Relative Strength Index."""
    deltas = np.diff(closes)

    if len(deltas) < period:
        return np.full(len(closes), np.nan)

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    result = np.full(len(closes), np.nan)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


def adx(highs, lows, closes, period=14):
    """Average Directional Index.

    Returns (adx_array, plus_di_array, minus_di_array).
    ADX > 25 = trending, ADX < 20 = ranging.
    """
    n = len(closes)
    if n < period * 2 + 1:
        nans = np.full(n, np.nan)
        return nans, nans, nans

    # +DM / -DM
    high_diff = np.diff(highs)
    low_diff = -np.diff(lows)  # note: negative diff of lows

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    tr = true_range(highs[1:], lows[1:], closes[1:])
    # Prepend a TR for alignment (use first available)
    tr_full = true_range(highs, lows, closes)

    # Wilder's smoothing for TR, +DM, -DM
    def wilder_smooth(data, p):
        result = np.full(len(data), np.nan, dtype=float)
        result[p - 1] = np.sum(data[:p])
        for i in range(p, len(data)):
            result[i] = result[i - 1] - (result[i - 1] / p) + data[i]
        return result

    smooth_tr = wilder_smooth(tr_full, period)
    smooth_plus_dm = wilder_smooth(
        np.concatenate([[0.0], plus_dm]), period
    )
    smooth_minus_dm = wilder_smooth(
        np.concatenate([[0.0], minus_dm]), period
    )

    # +DI / -DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * smooth_plus_dm / smooth_tr
        minus_di = 100.0 * smooth_minus_dm / smooth_tr

        # DX
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = np.where(di_sum != 0, 100.0 * di_diff / di_sum, 0.0)

    # ADX = Wilder's smoothing of DX
    adx_result = np.full(n, np.nan)

    # First valid DX index
    first_valid = period - 1
    # We need `period` valid DX values to start ADX
    adx_start = first_valid + period

    if adx_start < n:
        adx_result[adx_start] = np.nanmean(dx[first_valid:adx_start + 1])
        for i in range(adx_start + 1, n):
            if not np.isnan(dx[i]) and not np.isnan(adx_result[i - 1]):
                adx_result[i] = (adx_result[i - 1] * (period - 1) + dx[i]) / period

    return adx_result, plus_di, minus_di


def momentum(closes, lookback):
    """Price momentum as percentage change over lookback periods.

    mom = (price_now - price_lookback_ago) / price_lookback_ago
    """
    result = np.full(len(closes), np.nan)
    for i in range(lookback, len(closes)):
        if closes[i - lookback] != 0:
            result[i] = (closes[i] - closes[i - lookback]) / closes[i - lookback]
    return result


def bollinger_band_width(closes, period=20, num_std=2.0):
    """Bollinger Band width as percentage of middle band.

    High width = high volatility, low width = low volatility (squeeze).
    """
    result = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        ma = np.mean(window)
        std = np.std(window, ddof=1)
        if ma != 0:
            result[i] = (2 * num_std * std) / ma
    return result


def realized_volatility(closes, period=30):
    """Annualized realized volatility from log returns.

    For hourly data: annualize by sqrt(24*365).
    For 5-min data: annualize by sqrt(288*365).
    Returns as decimal (e.g., 0.50 = 50%).
    """
    log_returns = np.diff(np.log(closes))
    result = np.full(len(closes), np.nan)

    for i in range(period, len(closes)):
        window = log_returns[i - period:i]
        result[i] = np.std(window, ddof=1)

    return result


class IncrementalATR:
    """Incremental ATR calculator for live trading.

    Maintains state across updates without recalculating from scratch.
    """

    def __init__(self, period=14):
        self.period = period
        self.prev_close = None
        self.tr_buffer = deque(maxlen=period)
        self.current_atr = None
        self.count = 0

    def update(self, high, low, close):
        """Update with new candle. Returns current ATR or None if not ready."""
        if self.prev_close is not None:
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close)
            )
        else:
            tr = high - low

        self.prev_close = close
        self.tr_buffer.append(tr)
        self.count += 1

        if self.count < self.period:
            return None

        if self.current_atr is None:
            self.current_atr = sum(self.tr_buffer) / self.period
        else:
            self.current_atr = (self.current_atr * (self.period - 1) + tr) / self.period

        return self.current_atr


class IncrementalRSI:
    """Incremental RSI calculator for live trading."""

    def __init__(self, period=14):
        self.period = period
        self.prev_close = None
        self.gains = deque(maxlen=period)
        self.losses = deque(maxlen=period)
        self.avg_gain = None
        self.avg_loss = None
        self.count = 0

    def update(self, close):
        """Update with new close price. Returns RSI or None if not ready."""
        if self.prev_close is not None:
            delta = close - self.prev_close
            gain = max(delta, 0)
            loss = max(-delta, 0)
            self.gains.append(gain)
            self.losses.append(loss)
            self.count += 1

        self.prev_close = close

        if self.count < self.period:
            return None

        if self.avg_gain is None:
            self.avg_gain = sum(self.gains) / self.period
            self.avg_loss = sum(self.losses) / self.period
        else:
            self.avg_gain = (self.avg_gain * (self.period - 1) + self.gains[-1]) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + self.losses[-1]) / self.period

        if self.avg_loss == 0:
            return 100.0

        rs = self.avg_gain / self.avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
