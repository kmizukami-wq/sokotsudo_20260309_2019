#!/usr/bin/env python3
"""
BTC/JPY CFD Momentum Bot v2 — Optimized

Improvements over v1:
  Phase 1: Risk management overhaul
    - Fractional Kelly position sizing (20% vs 90%)
    - ATR-based dynamic trailing stop (replaces fixed 2.5%)
    - Equity curve scaling (graduated reduction after drawdown)

  Phase 2: Entry filters
    - ADX trend filter (>25 = trending market)
    - RSI confirmation (no long when overbought, no short when oversold)
    - Multi-timeframe momentum confirmation (72h + 24h)

  Phase 3: Strategy enhancements
    - Volatility regime detection (LOW/NORMAL/HIGH)
    - Adaptive entry threshold (0.75% - 1.5% based on regime)
    - Partial profit taking (scale-out at +5%, +10%)
    - Funding rate awareness (log cost at settlement times)

Environment variables (via .env.mm):
  BITFLYER_API_KEY
  BITFLYER_API_SECRET
"""

import os
import sys
import time
import math
import hmac
import hashlib
import logging
import json
from datetime import datetime, timezone, timedelta
from collections import deque
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

import numpy as np


def load_env():
    """Load environment variables from .env or .env.mm file (no dotenv needed)."""
    for env_file in [".env", ".env.mm", "../.env", "../.env.mm"]:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), env_file)
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
            return path
    return None


_env_path = load_env()


from indicators import (
    atr, rsi, adx, momentum, realized_volatility,
    IncrementalATR, IncrementalRSI,
)
from risk_manager import (
    VolatilityRegime, PositionSizer, DynamicStopLoss,
    ScaleOutManager, EntryFilter, AdaptiveThreshold,
)

try:
    from ml_model import FeatureBuilder, SignalPredictor, features_to_matrix
    HAS_ML = True
except ImportError:
    HAS_ML = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRODUCT = "FX_BTC_JPY"

# Momentum parameters
LB_LONG = 864          # 72h lookback in 5-min bars (72*12)
LB_SHORT = 288         # 24h lookback in 5-min bars (24*12) — confirmation
WARMUP_DAYS = 4        # Days of history to fetch on startup

# Risk parameters
CAPITAL_RATIO = 0.35   # Half-Kelly safe: ruin-proof with 2x leverage + B2.2
MAX_LOSS_RATIO = 0.15  # Emergency stop at -15%

# XGBoost hybrid thresholds (V4+ optimized via 34k param sweep)
ML_PROB_BOOST = 0.50   # Above this → 2.2x size (ML confident)
ML_PROB_SKIP = 0.45    # Below this → skip trade (ML disagrees)
ML_BOOST_MULT = 2.2    # Size multiplier for high-confidence trades

# Timing
INTV = 300             # Check interval: 5 minutes
MIN_SIZE = 0.01        # bitFlyer minimum trade size

# Logging
LOG_FILE = os.path.expanduser("~/momentum_bot_v2.log")

# bitFlyer funding rate settlement times (JST)
FUNDING_HOURS_JST = [6, 14, 22]

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("bot_v2")

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("BITFLYER_API_KEY", "")
API_SECRET = os.environ.get("BITFLYER_API_SECRET", "")
BF_BASE = "https://api.bitflyer.com"
BB_BASE = "https://public.bitbank.cc"


def _bf_headers(method, path, body=""):
    """Generate bitFlyer authentication headers."""
    timestamp = str(int(time.time()))
    text = timestamp + method + path + body
    sign = hmac.new(
        API_SECRET.encode(), text.encode(), hashlib.sha256
    ).hexdigest()
    return {
        "ACCESS-KEY": API_KEY,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-SIGN": sign,
        "Content-Type": "application/json",
    }


def bf_get(path):
    """Authenticated GET to bitFlyer."""
    url = BF_BASE + path
    headers = _bf_headers("GET", path)
    req = Request(url, headers=headers)
    with urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def bf_post(path, body_dict):
    """Authenticated POST to bitFlyer."""
    body = json.dumps(body_dict)
    url = BF_BASE + path
    headers = _bf_headers("POST", path, body)
    req = Request(url, data=body.encode(), headers=headers, method="POST")
    with urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def public_get(url):
    """Unauthenticated GET."""
    req = Request(url)
    with urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def get_mid_price():
    """Get FX_BTC_JPY mid price (bid+ask)/2."""
    data = public_get(f"{BF_BASE}/v1/getboard?product_code={PRODUCT}")
    bid = data["bids"][0]["price"]
    ask = data["asks"][0]["price"]
    return (bid + ask) / 2.0


def get_collateral():
    """Get collateral from bitFlyer."""
    data = bf_get("/v1/me/getcollateral")
    return data["collateral"]


def get_position():
    """Get current FX position. Returns (side, size, avg_price) or (None, 0, 0)."""
    positions = bf_get(f"/v1/me/getpositions?product_code={PRODUCT}")
    if not positions:
        return None, 0.0, 0.0

    # Aggregate — bitFlyer can return multiple position entries
    total_buy = 0.0
    total_sell = 0.0
    buy_cost = 0.0
    sell_cost = 0.0

    for p in positions:
        size = p["size"]
        price = p["price"]
        if p["side"] == "BUY":
            total_buy += size
            buy_cost += size * price
        else:
            total_sell += size
            sell_cost += size * price

    net = total_buy - total_sell
    if abs(net) < 0.001:
        return None, 0.0, 0.0
    elif net > 0:
        avg = buy_cost / total_buy if total_buy > 0 else 0
        return "LONG", round(net, 8), avg
    else:
        avg = sell_cost / total_sell if total_sell > 0 else 0
        return "SHORT", round(abs(net), 8), avg


def send_order(side, size):
    """Send market order. side = 'BUY' or 'SELL'."""
    body = {
        "product_code": PRODUCT,
        "child_order_type": "MARKET",
        "side": side,
        "size": round(size, 8),
    }
    logger.info("ORDER: %s %.4f BTC", side, size)
    result = bf_post("/v1/me/sendchildorder", body)
    logger.info("ORDER result: %s", result)
    return result


def fetch_bitbank_ohlc(days=4):
    """Fetch hourly OHLC from bitbank for warmup.

    Returns list of dicts with keys: open, high, low, close, timestamp
    sorted by time ascending.
    """
    candles = []
    today = datetime.now(timezone.utc)

    for d in range(days, -1, -1):
        date = today - timedelta(days=d)
        date_str = date.strftime("%Y%m%d")
        url = f"{BB_BASE}/btc_jpy/candlestick/1hour/{date_str}"
        try:
            data = public_get(url)
            ohlc_list = data["data"]["candlestick"][0]["ohlcv"]
            for row in ohlc_list:
                candles.append({
                    "open": float(row[0]),
                    "high": float(row[1]),
                    "low": float(row[2]),
                    "close": float(row[3]),
                    "volume": float(row[4]),
                    "timestamp": int(row[5]),
                })
        except Exception as e:
            logger.warning("bitbank fetch failed for %s: %s", date_str, e)

    candles.sort(key=lambda c: c["timestamp"])
    return candles


def interpolate_to_5min(hourly_candles):
    """Linearly interpolate hourly candles to 5-minute resolution.

    Returns dict of arrays: close, high, low, volume (each 5-min bar).
    """
    closes_1h = [c["close"] for c in hourly_candles]
    highs_1h = [c["high"] for c in hourly_candles]
    lows_1h = [c["low"] for c in hourly_candles]
    volumes_1h = [c["volume"] for c in hourly_candles]

    n = len(closes_1h)
    # 12 five-min bars per hour
    total_bars = (n - 1) * 12

    closes = np.interp(
        np.arange(total_bars),
        np.arange(0, total_bars + 1, 12)[:n],
        closes_1h,
    )
    highs = np.interp(
        np.arange(total_bars),
        np.arange(0, total_bars + 1, 12)[:n],
        highs_1h,
    )
    lows = np.interp(
        np.arange(total_bars),
        np.arange(0, total_bars + 1, 12)[:n],
        lows_1h,
    )
    volumes = np.interp(
        np.arange(total_bars),
        np.arange(0, total_bars + 1, 12)[:n],
        volumes_1h,
    )

    return {
        "close": closes,
        "high": highs,
        "low": lows,
        "volume": volumes,
    }


# ---------------------------------------------------------------------------
# Main Bot
# ---------------------------------------------------------------------------

class MomentumBotV2:
    """Optimized momentum bot with multi-layer risk management."""

    def __init__(self):
        # Price queues (5-min bars)
        self.prices = deque(maxlen=LB_LONG + 100)  # Extra buffer for indicators
        self.highs = deque(maxlen=LB_LONG + 100)
        self.lows = deque(maxlen=LB_LONG + 100)
        self.volumes = deque(maxlen=LB_LONG + 100)

        # Risk management modules
        self.vol_regime = VolatilityRegime()
        self.position_sizer = PositionSizer(capital_ratio=CAPITAL_RATIO)
        self.stop_loss = DynamicStopLoss()
        self.scale_out = ScaleOutManager()
        self.entry_filter = EntryFilter()
        self.adaptive_threshold = AdaptiveThreshold()

        # Incremental indicators for live updates
        self.inc_atr = IncrementalATR(period=14)
        self.inc_rsi = IncrementalRSI(period=14)

        # XGBoost hybrid model
        self.ml_predictor = None
        self.feature_builder = None
        if HAS_ML:
            self.feature_builder = FeatureBuilder()
            self.ml_predictor = SignalPredictor()
            if self.ml_predictor.load():
                logger.info("ML model loaded from %s", self.ml_predictor.model_path)
            else:
                logger.info("No ML model found — running without XGBoost filter")

        # State
        self.position_side = None   # 'LONG', 'SHORT', or None
        self.position_size = 0.0
        self.original_size = 0.0    # For scale-out calculation
        self.entry_price = 0.0
        self.peak = 0.0             # Trailing high (for long)
        self.trough = float("inf")  # Trailing low (for short)

        # Equity tracking
        self.initial_collateral = 0.0
        self.peak_equity = 0.0

        # PnL
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

    def warmup(self):
        """Fetch historical data and initialize indicator queues."""
        logger.info("=== Momentum Bot V2 Starting ===")
        logger.info("Fetching %d days of bitbank data for warmup...", WARMUP_DAYS)

        candles = fetch_bitbank_ohlc(WARMUP_DAYS)
        if len(candles) < 24:
            logger.error("Insufficient warmup data: %d candles", len(candles))
            sys.exit(1)

        data = interpolate_to_5min(candles)
        logger.info("Interpolated to %d 5-min bars", len(data["close"]))

        for i in range(len(data["close"])):
            self.prices.append(data["close"][i])
            self.highs.append(data["high"][i])
            self.lows.append(data["low"][i])
            self.volumes.append(data["volume"][i])
            self.inc_atr.update(data["high"][i], data["low"][i], data["close"][i])
            self.inc_rsi.update(data["close"][i])

        # Initialize equity tracking
        self.initial_collateral = get_collateral()
        self.peak_equity = self.initial_collateral
        self.loss_limit = self.initial_collateral * MAX_LOSS_RATIO

        # Check existing position
        self.position_side, self.position_size, self.entry_price = get_position()
        if self.position_side:
            self.original_size = self.position_size
            current = self.prices[-1] if self.prices else 0
            self.peak = max(current, self.entry_price)
            self.trough = min(current, self.entry_price)
            logger.info(
                "Existing position detected: %s %.4f BTC @ %.0f",
                self.position_side, self.position_size, self.entry_price,
            )

        logger.info(
            "Warmup complete. Collateral=%.0f JPY, Loss limit=%.0f JPY, "
            "Bars=%d",
            self.initial_collateral, self.loss_limit, len(self.prices),
        )

    def compute_indicators(self):
        """Compute all indicators from current price queue.

        Returns dict with all indicator values.
        """
        closes = np.array(self.prices)
        highs_arr = np.array(self.highs)
        lows_arr = np.array(self.lows)
        volumes_arr = np.array(self.volumes)
        n = len(closes)

        result = {}

        # Momentum (72h and 24h)
        if n > LB_LONG:
            result["mom_long"] = (closes[-1] - closes[-LB_LONG]) / closes[-LB_LONG]
        else:
            result["mom_long"] = None

        if n > LB_SHORT:
            result["mom_short"] = (closes[-1] - closes[-LB_SHORT]) / closes[-LB_SHORT]
        else:
            result["mom_short"] = None

        # ATR (incremental — already updated)
        result["atr"] = self.inc_atr.current_atr

        # RSI (incremental)
        result["rsi"] = self.inc_rsi.avg_gain is not None and (
            100.0 - (100.0 / (1.0 + self.inc_rsi.avg_gain / self.inc_rsi.avg_loss))
            if self.inc_rsi.avg_loss > 0 else 100.0
        ) if self.inc_rsi.count >= 14 else None

        # ADX (batch on last 60 bars for efficiency)
        window = min(n, 60)
        if window >= 30:
            adx_vals, _, _ = adx(
                highs_arr[-window:], lows_arr[-window:], closes[-window:], period=14
            )
            valid_adx = adx_vals[~np.isnan(adx_vals)]
            result["adx"] = float(valid_adx[-1]) if len(valid_adx) > 0 else None
        else:
            result["adx"] = None

        # Realized volatility (30-bar hourly equivalent → use 360 5-min bars)
        vol_window = min(n, 360)
        if vol_window >= 30:
            rv = realized_volatility(closes[-vol_window:], period=30)
            valid_rv = rv[~np.isnan(rv)]
            if len(valid_rv) > 0:
                # Annualize from 5-min: multiply by sqrt(288*365)
                hourly_vol = float(valid_rv[-1]) * math.sqrt(288 * 365)
                result["realized_vol"] = hourly_vol
            else:
                result["realized_vol"] = None
        else:
            result["realized_vol"] = None

        # Volume check (current vs 20-bar average)
        if n >= 20:
            avg_vol = np.mean(volumes_arr[-20:])
            result["volume_ratio"] = float(volumes_arr[-1] / avg_vol) if avg_vol > 0 else 1.0
        else:
            result["volume_ratio"] = 1.0

        # Current price
        result["price"] = float(closes[-1])

        return result

    def check_funding_warning(self):
        """Log warning near bitFlyer funding rate settlement times."""
        jst = datetime.now(timezone(timedelta(hours=9)))
        for fh in FUNDING_HOURS_JST:
            target = jst.replace(hour=fh, minute=0, second=0, microsecond=0)
            diff = abs((jst - target).total_seconds())
            if diff < 600:  # Within 10 minutes
                logger.info(
                    "FUNDING: Near settlement time %02d:00 JST. "
                    "Position costs may apply.", fh
                )
                return True
        return False

    def try_entry(self, indicators):
        """Evaluate entry conditions and open position if signals align."""
        mom_long = indicators["mom_long"]
        mom_short = indicators["mom_short"]
        price = indicators["price"]

        if mom_long is None:
            return

        # Volatility regime
        regime = self.vol_regime.classify(indicators.get("realized_vol"))
        threshold = self.adaptive_threshold.get_threshold(regime)

        logger.info(
            "Indicators: mom72h=%.3f%%, mom24h=%s, ADX=%s, RSI=%s, "
            "regime=%s, threshold=%.2f%%",
            mom_long * 100,
            f"{mom_short * 100:.3f}%" if mom_short is not None else "N/A",
            f"{indicators['adx']:.1f}" if indicators["adx"] is not None else "N/A",
            f"{indicators['rsi']:.1f}" if indicators["rsi"] is not None else "N/A",
            regime, threshold * 100,
        )

        # === LONG signal ===
        if mom_long > threshold:
            # Multi-timeframe confirmation: 24h momentum should also be positive
            if mom_short is not None and mom_short <= 0:
                logger.info("SKIP LONG: 24h momentum negative (%.3f%%)", mom_short * 100)
                return

            # Entry filter (ADX + RSI)
            ok, reason = self.entry_filter.check_long(
                adx_value=indicators.get("adx"),
                rsi_value=indicators.get("rsi"),
            )
            if not ok:
                logger.info("SKIP LONG: %s", reason)
                return

            # ML confidence check
            ml_mult = self._ml_size_mult("LONG")

            if ml_mult <= 0:
                logger.info("SKIP LONG: ML confidence below %.2f", ML_PROB_SKIP)
                return

            # Position sizing
            collateral = get_collateral()
            self.peak_equity = max(self.peak_equity, collateral)
            size = self.position_sizer.calculate(
                collateral, price, regime,
                self.peak_equity, collateral, MIN_SIZE,
            )
            size *= ml_mult
            if size < MIN_SIZE:
                return

            send_order("BUY", size)
            self.position_side = "LONG"
            self.position_size = size
            self.original_size = size
            self.entry_price = price
            self.peak = price
            self.scale_out.reset()
            self.trade_count += 1
            logger.info(
                "OPENED LONG: %.4f BTC @ %.0f (mom=%.3f%%, th=%.2f%%, ml=%.1fx)",
                size, price, mom_long * 100, threshold * 100, ml_mult,
            )

        # === SHORT signal ===
        elif mom_long < -threshold:
            if mom_short is not None and mom_short >= 0:
                logger.info("SKIP SHORT: 24h momentum positive (%.3f%%)", mom_short * 100)
                return

            ok, reason = self.entry_filter.check_short(
                adx_value=indicators.get("adx"),
                rsi_value=indicators.get("rsi"),
            )
            if not ok:
                logger.info("SKIP SHORT: %s", reason)
                return

            # ML confidence check
            ml_mult = self._ml_size_mult("SHORT")

            if ml_mult <= 0:
                logger.info("SKIP SHORT: ML confidence below %.2f", ML_PROB_SKIP)
                return

            collateral = get_collateral()
            self.peak_equity = max(self.peak_equity, collateral)
            size = self.position_sizer.calculate(
                collateral, price, regime,
                self.peak_equity, collateral, MIN_SIZE,
            )
            size *= ml_mult
            if size < MIN_SIZE:
                return

            send_order("SELL", size)
            self.position_side = "SHORT"
            self.position_size = size
            self.original_size = size
            self.entry_price = price
            self.trough = price
            self.scale_out.reset()
            self.trade_count += 1
            logger.info(
                "OPENED SHORT: %.4f BTC @ %.0f (mom=%.3f%%, th=%.2f%%, ml=%.1fx)",
                size, price, mom_long * 100, threshold * 100, ml_mult,
            )

    def _ml_size_mult(self, direction):
        """Get ML-based size multiplier for a trade direction.

        Returns:
            1.2 if ML confident (prob > ML_PROB_BOOST)
            1.0 if ML neutral (ML_PROB_SKIP < prob <= ML_PROB_BOOST)
            0.0 if ML disagrees (prob <= ML_PROB_SKIP)
        """
        if self.ml_predictor is None or not self.ml_predictor.is_ready:
            return 1.0  # No ML model → use V3 sizing as-is

        if self.feature_builder is None or len(self.prices) < FeatureBuilder.MIN_BARS:
            return 1.0

        try:
            closes = np.array(self.prices)
            highs_arr = np.array(self.highs)
            lows_arr = np.array(self.lows)
            volumes_arr = np.array(self.volumes)

            features = self.feature_builder.build(
                closes, highs_arr, lows_arr, volumes_arr)
            X, valid = features_to_matrix(
                features, self.feature_builder.feature_names)

            if not valid[-1]:
                return 1.0

            x_i = X[-1:, :]
            prob = self.ml_predictor.predict_proba(x_i, direction=direction)
            prob_val = float(prob[0]) if hasattr(prob, '__len__') else float(prob)

            if prob_val > ML_PROB_BOOST:
                logger.info("ML: %s confidence=%.3f → BOOST (%.1fx)",
                            direction, prob_val, ML_BOOST_MULT)
                return ML_BOOST_MULT
            elif prob_val > ML_PROB_SKIP:
                logger.info("ML: %s confidence=%.3f → NORMAL (1.0x)",
                            direction, prob_val)
                return 1.0
            else:
                logger.info("ML: %s confidence=%.3f → SKIP",
                            direction, prob_val)
                return 0.0
        except Exception as e:
            logger.warning("ML prediction error: %s", e)
            return 1.0  # Fallback to V3

    def manage_position(self, indicators):
        """Manage existing position: trailing stop + scale-out."""
        price = indicators["price"]
        regime = self.vol_regime.classify(indicators.get("realized_vol"))

        # Dynamic stop distance
        stop_pct = self.stop_loss.stop_distance_pct(
            indicators.get("atr"), price, regime,
        )

        # --- Scale-out check ---
        partial_close = self.scale_out.check(
            self.entry_price, price, self.position_side, self.original_size,
        )
        if partial_close > 0 and partial_close <= self.position_size:
            if self.position_side == "LONG":
                send_order("SELL", partial_close)
            else:
                send_order("BUY", partial_close)
            self.position_size -= partial_close
            self.position_size = round(self.position_size, 8)
            logger.info(
                "SCALE-OUT: closed %.4f BTC, remaining %.4f BTC",
                partial_close, self.position_size,
            )
            if self.position_size < MIN_SIZE:
                self._close_position(price, "SCALE-OUT (fully closed)")
                return

        # --- Trailing stop ---
        if self.position_side == "LONG":
            self.peak = max(self.peak, price)
            drawdown = (price - self.peak) / self.peak
            logger.info(
                "LONG: price=%.0f, peak=%.0f, dd=%.2f%%, stop=%.2f%%",
                price, self.peak, drawdown * 100, stop_pct * 100,
            )
            if drawdown <= -stop_pct:
                self._close_position(price, "TRAILING STOP")

        elif self.position_side == "SHORT":
            self.trough = min(self.trough, price)
            runup = (price - self.trough) / self.trough
            logger.info(
                "SHORT: price=%.0f, trough=%.0f, runup=%.2f%%, stop=%.2f%%",
                price, self.trough, runup * 100, stop_pct * 100,
            )
            if runup >= stop_pct:
                self._close_position(price, "TRAILING STOP")

    def _close_position(self, price, reason):
        """Close current position and record PnL."""
        if self.position_side == "LONG":
            pnl_pct = (price - self.entry_price) / self.entry_price
            send_order("SELL", self.position_size)
        else:
            pnl_pct = (self.entry_price - price) / self.entry_price
            send_order("BUY", self.position_size)

        pnl_jpy = pnl_pct * self.entry_price * self.position_size
        self.total_pnl += pnl_jpy

        if pnl_jpy > 0:
            self.win_count += 1

        logger.info(
            "CLOSED %s: %.4f BTC @ %.0f → %.0f (%s) PnL=%.0f JPY (%.2f%%) "
            "Total PnL=%.0f JPY [%d/%d wins]",
            self.position_side, self.position_size, self.entry_price, price,
            reason, pnl_jpy, pnl_pct * 100, self.total_pnl,
            self.win_count, self.trade_count,
        )

        self.position_side = None
        self.position_size = 0.0
        self.original_size = 0.0
        self.entry_price = 0.0
        self.peak = 0.0
        self.trough = float("inf")

    def check_emergency_stop(self):
        """Check cumulative loss against limit. Returns True if must stop."""
        if self.total_pnl < -self.loss_limit:
            logger.critical(
                "EMERGENCY STOP: total_pnl=%.0f < -%.0f (limit)",
                self.total_pnl, self.loss_limit,
            )
            # Force close any open position
            if self.position_side:
                price = get_mid_price()
                self._close_position(price, "EMERGENCY STOP")
            return True
        return False

    def run(self):
        """Main loop."""
        self.warmup()

        logger.info("=== Starting main loop (interval=%ds) ===", INTV)

        while True:
            try:
                # Get current price
                price = get_mid_price()

                # Update queues (use price as proxy for H/L/C in 5-min bar)
                self.prices.append(price)
                self.highs.append(price)  # Approximate — real OHLC would be better
                self.lows.append(price)
                self.volumes.append(0)    # Volume not available from board

                # Update incremental indicators
                self.inc_atr.update(price, price, price)
                self.inc_rsi.update(price)

                # Compute all indicators
                indicators = self.compute_indicators()

                # Funding rate warning
                self.check_funding_warning()

                # Emergency stop check
                if self.check_emergency_stop():
                    logger.critical("Bot stopped due to emergency loss limit.")
                    sys.exit(0)

                # Position management or new entry
                if self.position_side:
                    self.manage_position(indicators)
                else:
                    self.try_entry(indicators)

            except (HTTPError, URLError, ConnectionError) as e:
                logger.error("Network error: %s", e)
            except Exception as e:
                logger.exception("Unexpected error: %s", e)

            time.sleep(INTV)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        logger.error("BITFLYER_API_KEY and BITFLYER_API_SECRET must be set")
        sys.exit(1)

    bot = MomentumBotV2()
    bot.run()
