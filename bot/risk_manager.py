"""
Risk management module for BTC/JPY momentum bot v2.

Handles:
- Fractional Kelly position sizing
- ATR-based dynamic stop loss
- Equity curve scaling
- Volatility regime detection
- Partial profit taking (scale-out)
"""

import math
import logging

logger = logging.getLogger("risk_manager")


# ---------------------------------------------------------------------------
# Volatility Regime
# ---------------------------------------------------------------------------

class VolatilityRegime:
    """Classifies market into LOW / NORMAL / HIGH volatility regimes.

    Uses annualized realized volatility (hourly basis).
    """
    LOW = "LOW"           # < low_threshold
    NORMAL = "NORMAL"     # low_threshold .. high_threshold
    HIGH = "HIGH"         # > high_threshold

    def __init__(self, low_threshold=0.30, high_threshold=0.50):
        """
        Args:
            low_threshold: Annualized vol below this = LOW (default 30%)
            high_threshold: Annualized vol above this = HIGH (default 50%)
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def classify(self, annualized_vol):
        """Returns regime string given annualized volatility (decimal)."""
        if annualized_vol is None or math.isnan(annualized_vol):
            return self.NORMAL  # default to normal if unknown
        if annualized_vol < self.low_threshold:
            return self.LOW
        elif annualized_vol > self.high_threshold:
            return self.HIGH
        else:
            return self.NORMAL


# ---------------------------------------------------------------------------
# Position Sizer
# ---------------------------------------------------------------------------

class PositionSizer:
    """Calculates optimal position size using Fractional Kelly + ATR + equity scaling.

    Layers:
    1. Base size = Fractional Kelly (capital_ratio param, default 20%)
    2. Volatility adjustment: scale down in HIGH vol, up in LOW vol
    3. Equity curve scaling: reduce size after drawdown from peak
    """

    def __init__(
        self,
        capital_ratio=0.20,           # Base: 20% of collateral
        vol_regime_multipliers=None,  # {regime: multiplier}
        equity_drawdown_levels=None,  # [(drawdown_pct, size_multiplier), ...]
    ):
        self.capital_ratio = capital_ratio

        self.vol_multipliers = vol_regime_multipliers or {
            VolatilityRegime.LOW: 1.2,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: 0.6,
        }

        # Ordered list: if drawdown exceeds X%, multiply size by Y
        self.equity_levels = equity_drawdown_levels or [
            (0.05, 0.80),  # -5% drawdown → 80% size
            (0.10, 0.60),  # -10% drawdown → 60% size
            (0.15, 0.00),  # -15% drawdown → stop trading
        ]
        # Sort by drawdown ascending
        self.equity_levels.sort(key=lambda x: x[0])

    def calculate(self, collateral, price, vol_regime, peak_equity, current_equity,
                  min_size=0.01):
        """Calculate position size in BTC.

        Args:
            collateral: Current collateral in JPY
            price: Current BTC/JPY price
            vol_regime: VolatilityRegime.LOW / NORMAL / HIGH
            peak_equity: Historical peak equity (for drawdown calc)
            current_equity: Current equity
            min_size: Minimum order size in BTC

        Returns:
            Position size in BTC (0.0 if below min or stopped)
        """
        if price <= 0 or collateral <= 0:
            return 0.0

        # 1. Base size
        base_jpy = collateral * self.capital_ratio

        # 2. Volatility adjustment
        vol_mult = self.vol_multipliers.get(vol_regime, 1.0)
        adjusted_jpy = base_jpy * vol_mult

        # 3. Equity curve scaling
        equity_mult = 1.0
        if peak_equity > 0:
            drawdown = (peak_equity - current_equity) / peak_equity
            for dd_level, dd_mult in self.equity_levels:
                if drawdown >= dd_level:
                    equity_mult = dd_mult
                # Keep checking — last matching level wins

        final_jpy = adjusted_jpy * equity_mult
        size_btc = final_jpy / price

        if size_btc < min_size:
            logger.info(
                "Position size %.4f BTC < min %.4f, skip",
                size_btc, min_size
            )
            return 0.0

        # Round down to 0.01 precision (bitFlyer minimum increment)
        size_btc = math.floor(size_btc * 100) / 100

        logger.info(
            "Position: base=%.0f JPY, vol_mult=%.1f(%s), eq_mult=%.2f(dd=%.1f%%), "
            "final=%.4f BTC",
            base_jpy, vol_mult, vol_regime, equity_mult,
            ((peak_equity - current_equity) / peak_equity * 100) if peak_equity > 0 else 0,
            size_btc,
        )
        return size_btc


# ---------------------------------------------------------------------------
# Dynamic Stop Loss
# ---------------------------------------------------------------------------

class DynamicStopLoss:
    """ATR-based trailing stop loss.

    stop_distance = ATR * multiplier
    Adapts automatically: tighter in low vol, wider in high vol.
    Also supports regime-specific multiplier overrides.
    """

    def __init__(
        self,
        atr_multiplier=2.0,
        min_stop_pct=0.015,   # Floor: 1.5%
        max_stop_pct=0.06,    # Cap: 6%
        regime_multipliers=None,
    ):
        self.atr_multiplier = atr_multiplier
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct
        self.regime_multipliers = regime_multipliers or {
            VolatilityRegime.LOW: 1.5,
            VolatilityRegime.NORMAL: 2.0,
            VolatilityRegime.HIGH: 2.5,
        }

    def stop_distance_pct(self, current_atr, current_price, vol_regime=None):
        """Calculate stop distance as percentage of price.

        Args:
            current_atr: Current ATR value (in JPY terms)
            current_price: Current BTC/JPY price
            vol_regime: Optional regime for multiplier override

        Returns:
            Stop distance as decimal (e.g., 0.025 = 2.5%)
        """
        if current_atr is None or current_price <= 0:
            return 0.025  # Fallback to fixed 2.5%

        if vol_regime and vol_regime in self.regime_multipliers:
            mult = self.regime_multipliers[vol_regime]
        else:
            mult = self.atr_multiplier

        stop_jpy = current_atr * mult
        stop_pct = stop_jpy / current_price

        # Clamp to min/max
        stop_pct = max(self.min_stop_pct, min(self.max_stop_pct, stop_pct))

        return stop_pct


# ---------------------------------------------------------------------------
# Scale-Out Manager
# ---------------------------------------------------------------------------

class ScaleOutManager:
    """Manages partial profit taking (3-tier scale-out).

    Levels are defined as (profit_pct, exit_fraction).
    exit_fraction is the fraction of the ORIGINAL position to close.
    """

    def __init__(self, levels=None):
        self.levels = levels or [
            (0.05, 0.25),   # +5% → close 25%
            (0.10, 0.25),   # +10% → close another 25%
            # Remaining 50% rides with trailing stop
        ]
        self.levels.sort(key=lambda x: x[0])
        self.triggered = set()  # Track which levels have been triggered

    def reset(self):
        """Reset for new trade."""
        self.triggered = set()

    def check(self, entry_price, current_price, side, original_size):
        """Check if any scale-out level is triggered.

        Args:
            entry_price: Original entry price
            current_price: Current price
            side: 'LONG' or 'SHORT'
            original_size: Original position size in BTC

        Returns:
            Size to close (BTC), or 0.0 if no level triggered.
        """
        if entry_price <= 0:
            return 0.0

        if side == "LONG":
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price

        close_size = 0.0
        for level_pct, exit_frac in self.levels:
            level_key = level_pct
            if profit_pct >= level_pct and level_key not in self.triggered:
                self.triggered.add(level_key)
                partial = original_size * exit_frac
                # Round down
                partial = math.floor(partial * 100) / 100
                close_size += partial
                logger.info(
                    "Scale-out triggered: profit=%.2f%% >= level=%.2f%%, "
                    "closing %.4f BTC",
                    profit_pct * 100, level_pct * 100, partial,
                )

        return close_size


# ---------------------------------------------------------------------------
# Entry Filter
# ---------------------------------------------------------------------------

class EntryFilter:
    """Composite entry filter combining ADX + RSI + volume checks.

    All filters must pass for entry signal to be valid.
    """

    def __init__(
        self,
        adx_threshold=25.0,     # ADX > 25 = trending
        rsi_overbought=70.0,    # Don't go long above this
        rsi_oversold=30.0,      # Don't go short below this
        enable_adx=True,
        enable_rsi=True,
    ):
        self.adx_threshold = adx_threshold
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.enable_adx = enable_adx
        self.enable_rsi = enable_rsi

    def check_long(self, adx_value=None, rsi_value=None):
        """Check if LONG entry is allowed.

        Returns (allowed: bool, reason: str).
        """
        if self.enable_adx and adx_value is not None:
            if adx_value < self.adx_threshold:
                return False, f"ADX={adx_value:.1f} < {self.adx_threshold} (ranging)"

        if self.enable_rsi and rsi_value is not None:
            if rsi_value > self.rsi_overbought:
                return False, f"RSI={rsi_value:.1f} > {self.rsi_overbought} (overbought)"

        return True, "OK"

    def check_short(self, adx_value=None, rsi_value=None):
        """Check if SHORT entry is allowed.

        Returns (allowed: bool, reason: str).
        """
        if self.enable_adx and adx_value is not None:
            if adx_value < self.adx_threshold:
                return False, f"ADX={adx_value:.1f} < {self.adx_threshold} (ranging)"

        if self.enable_rsi and rsi_value is not None:
            if rsi_value < self.rsi_oversold:
                return False, f"RSI={rsi_value:.1f} < {self.rsi_oversold} (oversold)"

        return True, "OK"


# ---------------------------------------------------------------------------
# Adaptive Entry Threshold
# ---------------------------------------------------------------------------

class AdaptiveThreshold:
    """Adjusts momentum entry threshold based on volatility regime.

    Low vol → lower threshold (more sensitive)
    High vol → higher threshold (filter noise)
    """

    def __init__(self, base_threshold=0.01, regime_thresholds=None):
        self.base_threshold = base_threshold
        self.thresholds = regime_thresholds or {
            VolatilityRegime.LOW: 0.0075,    # 0.75%
            VolatilityRegime.NORMAL: 0.01,   # 1.0%
            VolatilityRegime.HIGH: 0.015,    # 1.5%
        }

    def get_threshold(self, vol_regime):
        """Returns threshold for current regime."""
        return self.thresholds.get(vol_regime, self.base_threshold)
