"""
XGBoost-based signal confidence predictor for BTC/JPY momentum bot.

Used as a hybrid filter on top of V3 rule-based signals.
V3 generates entry signals; this model scores their confidence (0.0-1.0).
High confidence → full size, medium → half, low → skip.
"""

import os
import pickle
import math
import numpy as np

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from indicators import (
    atr, rsi, adx, momentum, bollinger_band_width,
    realized_volatility, macd, sma, bollinger_bands,
)


class FeatureBuilder:
    """Builds feature matrix from OHLCV arrays for XGBoost."""

    # Minimum bars needed for all features
    MIN_BARS = 200

    def build(self, closes, highs, lows, volumes, timestamps=None):
        """Build feature matrix from OHLCV arrays.

        Args:
            closes, highs, lows, volumes: numpy arrays of equal length
            timestamps: optional unix ms timestamps (for time features)

        Returns:
            features: dict of {name: numpy_array} (length = len(closes))
            All arrays have NaN where insufficient history.
        """
        n = len(closes)
        features = {}

        # --- Price momentum (6) ---
        features["mom_4h"] = momentum(closes, 4)
        features["mom_24h"] = momentum(closes, 24)
        features["mom_72h"] = momentum(closes, 72)
        features["mom_168h"] = momentum(closes, 168)

        sma_20 = sma(closes, 20)
        sma_50 = sma(closes, 50)
        with np.errstate(divide='ignore', invalid='ignore'):
            features["price_vs_sma20"] = closes / sma_20 - 1.0
            features["price_vs_sma50"] = closes / sma_50 - 1.0

        # --- Volatility (5) ---
        atr_14 = atr(highs, lows, closes, period=14)
        with np.errstate(divide='ignore', invalid='ignore'):
            features["atr_14_norm"] = atr_14 / closes

        features["bb_width"] = bollinger_band_width(closes, period=20)
        features["realized_vol_30"] = realized_volatility(closes, period=30)

        # ATR change (acceleration)
        atr_change = np.full(n, np.nan)
        for i in range(1, n):
            if not np.isnan(atr_14[i]) and not np.isnan(atr_14[i - 1]) and atr_14[i - 1] > 0:
                atr_change[i] = (atr_14[i] - atr_14[i - 1]) / atr_14[i - 1]
        features["atr_change"] = atr_change

        # 24h high-low range normalized
        hl_range = np.full(n, np.nan)
        for i in range(24, n):
            h = np.max(highs[i - 24:i + 1])
            l = np.min(lows[i - 24:i + 1])
            if closes[i] > 0:
                hl_range[i] = (h - l) / closes[i]
        features["high_low_range"] = hl_range

        # --- Oscillators (4) ---
        features["rsi_14"] = rsi(closes, period=14)
        features["rsi_28"] = rsi(closes, period=28)

        macd_line, signal_line, histogram = macd(closes)
        features["macd_signal_diff"] = macd_line - signal_line

        # MACD histogram slope (3-bar)
        macd_slope = np.full(n, np.nan)
        for i in range(3, n):
            if not np.isnan(histogram[i]) and not np.isnan(histogram[i - 3]):
                macd_slope[i] = histogram[i] - histogram[i - 3]
        features["macd_hist_slope"] = macd_slope

        # --- Trend strength (3) ---
        adx_arr, plus_di, minus_di = adx(highs, lows, closes, period=14)
        features["adx_14"] = adx_arr
        features["plus_di_minus_di"] = plus_di - minus_di

        # Trend consistency: fraction of last 12h that closed up
        trend_cons = np.full(n, np.nan)
        for i in range(12, n):
            ups = 0
            for j in range(i - 11, i + 1):
                if closes[j] > closes[j - 1]:
                    ups += 1
            trend_cons[i] = ups / 12.0
        features["trend_consistency"] = trend_cons

        # --- Volume (2) ---
        if volumes is not None and len(volumes) == n:
            vol_sma20 = sma(volumes, 20)
            with np.errstate(divide='ignore', invalid='ignore'):
                features["volume_ratio"] = volumes / vol_sma20

            vol_sma5 = sma(volumes, 5)
            vol_trend = np.full(n, np.nan)
            for i in range(1, n):
                if not np.isnan(vol_sma5[i]) and not np.isnan(vol_sma5[i - 1]) and vol_sma5[i - 1] > 0:
                    vol_trend[i] = (vol_sma5[i] - vol_sma5[i - 1]) / vol_sma5[i - 1]
            features["volume_trend"] = vol_trend
        else:
            features["volume_ratio"] = np.full(n, np.nan)
            features["volume_trend"] = np.full(n, np.nan)

        # --- New momentum features (2) ---
        sma_8 = sma(closes, 8)
        with np.errstate(divide='ignore', invalid='ignore'):
            features["funding_rate_proxy"] = (closes - sma_8) / sma_8

        rvol = realized_volatility(closes, period=30)
        vol_speed = np.full(n, np.nan)
        for i in range(6, n):
            if not np.isnan(rvol[i]) and not np.isnan(rvol[i - 6]) and rvol[i - 6] > 0:
                vol_speed[i] = (rvol[i] - rvol[i - 6]) / rvol[i - 6]
        features["vol_regime_speed"] = vol_speed

        # --- Time features (3) ---
        if timestamps is not None and len(timestamps) == n:
            from datetime import datetime, timezone
            hours = np.zeros(n)
            dows = np.zeros(n)
            for i in range(n):
                dt = datetime.fromtimestamp(timestamps[i] / 1000, tz=timezone.utc)
                hours[i] = dt.hour
                dows[i] = dt.weekday()
            features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
            features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
            features["day_of_week"] = dows
        else:
            features["hour_sin"] = np.full(n, 0.0)
            features["hour_cos"] = np.full(n, 0.0)
            features["day_of_week"] = np.full(n, 0.0)

        return features

    def build_labels(self, closes, horizon=5, threshold=0.01):
        """Build target labels.

        target = 1 if max rise in next `horizon` bars >= threshold
        target = -1 if max drop in next `horizon` bars >= threshold
        target = 0 otherwise

        Args:
            closes: numpy array of close prices
            horizon: look-ahead window (bars)
            threshold: minimum move for +1/-1 label

        Returns:
            labels: numpy array (length = len(closes), last `horizon` are NaN)
        """
        n = len(closes)
        labels = np.full(n, np.nan)

        for i in range(n - horizon):
            future = closes[i + 1:i + 1 + horizon]
            max_up = (np.max(future) - closes[i]) / closes[i]
            max_down = (np.min(future) - closes[i]) / closes[i]

            if max_up >= threshold:
                labels[i] = 1
            elif max_down <= -threshold:
                labels[i] = -1
            else:
                labels[i] = 0

        return labels

    @property
    def feature_names(self):
        """Ordered list of feature names."""
        return [
            "mom_4h", "mom_24h", "mom_72h", "mom_168h",
            "price_vs_sma20", "price_vs_sma50",
            "atr_14_norm", "bb_width", "realized_vol_30",
            "atr_change", "high_low_range",
            "rsi_14", "rsi_28", "macd_signal_diff", "macd_hist_slope",
            "adx_14", "plus_di_minus_di", "trend_consistency",
            "volume_ratio", "volume_trend",
            "hour_sin", "hour_cos", "day_of_week",
            "funding_rate_proxy", "vol_regime_speed",
        ]


class SignalPredictor:
    """XGBoost model wrapper for signal confidence prediction."""

    def __init__(self, model_path=None):
        self.model = None
        self.feature_builder = FeatureBuilder()
        self.model_path = model_path or os.path.expanduser("~/xgb_model.pkl")
        self.is_ready = False

    def train(self, X, y):
        """Train XGBoost classifier.

        Args:
            X: numpy array (n_samples, n_features)
            y: numpy array of labels (-1, 0, 1)
        """
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed. pip install xgboost")

        # Map labels: -1→0, 0→1, 1→2 for multi-class
        y_mapped = y + 1  # -1→0, 0→1, 1→2

        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        self.model.fit(X, y_mapped)
        self.is_ready = True

    def predict_proba(self, X, direction="LONG"):
        """Predict relative confidence for signal direction.

        Uses P(direction) / (P(direction) + P(opposite)) to normalize
        out the neutral class, which dominates in 3-class problems.

        Args:
            X: numpy array (1, n_features) or (n_samples, n_features)
            direction: 'LONG' or 'SHORT'

        Returns:
            float or array: confidence (0.0-1.0), >0.5 means model
            favors the direction over the opposite.
        """
        if not self.is_ready or self.model is None:
            return np.full(X.shape[0], 0.5) if len(X.shape) > 1 else 0.5

        # Predict probabilities: [P(down), P(neutral), P(up)]
        probs = self.model.predict_proba(X)

        if direction == "LONG":
            p_dir = probs[:, 2]   # P(up)
            p_opp = probs[:, 0]   # P(down)
        else:
            p_dir = probs[:, 0]   # P(down)
            p_opp = probs[:, 2]   # P(up)

        # Relative confidence: P(dir) / (P(dir) + P(opp))
        # This ignores neutral and asks: among directional outcomes,
        # how much does the model favor this direction?
        denom = p_dir + p_opp
        confidence = np.where(denom > 0, p_dir / denom, 0.5)
        return confidence

    def save(self, path=None):
        """Save trained model to disk."""
        path = path or self.model_path
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path=None):
        """Load model from disk."""
        path = path or self.model_path
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            self.is_ready = True
            return True
        return False

    def feature_importance(self):
        """Return dict of feature name → importance score."""
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        names = self.feature_builder.feature_names
        return dict(sorted(
            zip(names, importances),
            key=lambda x: x[1], reverse=True,
        ))


class MRFeatureBuilder:
    """Feature builder for mean reversion strategy.

    Extends base features with BB position, RSI speed, and mean distance.
    """

    MIN_BARS = 200

    def __init__(self):
        self._base = FeatureBuilder()

    def build(self, closes, highs, lows, volumes, timestamps=None):
        features = self._base.build(closes, highs, lows, volumes, timestamps)
        n = len(closes)

        # BB position: (close - BB_lower) / (BB_upper - BB_lower)
        bb_upper, bb_middle, bb_lower = bollinger_bands(closes, period=20)
        with np.errstate(divide='ignore', invalid='ignore'):
            bb_range = bb_upper - bb_lower
            features["bb_position"] = np.where(
                bb_range > 0, (closes - bb_lower) / bb_range, 0.5
            )

        # RSI speed: RSI[i] - RSI[i-3]
        rsi_arr = features["rsi_14"]
        rsi_speed = np.full(n, np.nan)
        for i in range(3, n):
            if not np.isnan(rsi_arr[i]) and not np.isnan(rsi_arr[i - 3]):
                rsi_speed[i] = rsi_arr[i] - rsi_arr[i - 3]
        features["rsi_speed"] = rsi_speed

        # Mean distance: (close - SMA_20) / SMA_20
        sma_20 = sma(closes, 20)
        with np.errstate(divide='ignore', invalid='ignore'):
            features["mean_distance"] = (closes - sma_20) / sma_20

        return features

    def build_labels(self, closes, horizon=3, threshold=0.005):
        return self._base.build_labels(closes, horizon=horizon, threshold=threshold)

    @property
    def feature_names(self):
        return self._base.feature_names + [
            "bb_position", "rsi_speed", "mean_distance",
        ]


class MRSignalPredictor:
    """XGBoost predictor for mean reversion signals."""

    def __init__(self, model_path=None):
        self.model = None
        self.feature_builder = MRFeatureBuilder()
        self.model_path = model_path or os.path.expanduser("~/xgb_model_mr.pkl")
        self.is_ready = False

    def train(self, X, y):
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed")
        y_mapped = y + 1
        self.model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, objective="multi:softprob",
            num_class=3, eval_metric="mlogloss", random_state=42,
            n_jobs=1, verbosity=0,
        )
        self.model.fit(X, y_mapped)
        self.is_ready = True

    def predict_proba(self, X, direction="LONG"):
        if not self.is_ready or self.model is None:
            return np.full(X.shape[0], 0.5) if len(X.shape) > 1 else 0.5
        probs = self.model.predict_proba(X)
        if direction == "LONG":
            p_dir, p_opp = probs[:, 2], probs[:, 0]
        else:
            p_dir, p_opp = probs[:, 0], probs[:, 2]
        denom = p_dir + p_opp
        return np.where(denom > 0, p_dir / denom, 0.5)

    def save(self, path=None):
        path = path or self.model_path
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path=None):
        path = path or self.model_path
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            self.is_ready = True
            return True
        return False


def features_to_matrix(features, feature_names):
    """Convert feature dict to numpy matrix.

    Args:
        features: dict of {name: numpy_array}
        feature_names: ordered list of feature names

    Returns:
        X: numpy array (n_samples, n_features)
        valid_mask: boolean array (True where all features are finite)
    """
    n = len(next(iter(features.values())))
    X = np.column_stack([features[name] for name in feature_names])

    # Mask rows with any NaN/inf
    valid_mask = np.all(np.isfinite(X), axis=1)

    # Replace NaN with 0 for the matrix (masked rows won't be used)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, valid_mask
