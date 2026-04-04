#!/usr/bin/env python3
"""
Precision sweep: find max annual return with MaxDD constrained to -18~20%.
Tests both CFD-off (L1.0) and CFD 2x (L2.0) with fine-grained boost_mult.
Reuses ML cache from sweep_200.py.
"""

import sys
import math
import logging
import numpy as np
from datetime import datetime

logging.getLogger("risk_manager").setLevel(logging.CRITICAL)

from backtest import fetch_hourly_data, Trade, atr, rsi, adx, realized_volatility
from ml_model import FeatureBuilder, SignalPredictor, features_to_matrix
from sweep_200 import pretrain_walk_forward, fast_backtest


def main():
    print("=" * 90)
    print("  PRECISION SWEEP: MaxDD ≤ -20% constraint, maximize annual return")
    print("=" * 90)

    data = fetch_hourly_data(days=365 * 3)
    closes = data["close"]
    highs = data["high"]
    lows = data["low"]
    volumes = data["volume"]
    timestamps = data["timestamps"]
    n = len(closes)

    if n < 1000:
        print("ERROR: Not enough data")
        sys.exit(1)

    print("\nPre-computing indicators...")
    atr_arr = atr(highs, lows, closes, period=14)
    rsi_arr = rsi(closes, period=14)
    adx_arr, _, _ = adx(highs, lows, closes, period=14)
    rvol_arr = realized_volatility(closes, period=30)
    rvol_annual = rvol_arr * math.sqrt(24 * 365)

    print("Pre-computing ML features...")
    fb = FeatureBuilder()
    features = fb.build(closes, highs, lows, volumes, timestamps)

    # Pre-train ML (horizon=5 only, best from prior sweep)
    print("\nPre-training walk-forward ML (horizon=5h)...")
    labels = fb.build_labels(closes, horizon=5, threshold=0.01)
    X, valid = features_to_matrix(features, fb.feature_names)
    lv = np.isfinite(labels)
    fm = valid & lv
    prob_long, prob_short = pretrain_walk_forward(
        X, valid, labels, fm, n,
        train_hours=24*180, embargo_hours=24*7, retrain_interval=24*30,
    )

    # ===================================================================
    # Fine-grained sweep parameters
    # ===================================================================
    boost_mults_no_cfd = [round(x * 0.1, 1) for x in range(10, 26)]   # 1.0 to 2.5
    boost_mults_cfd2x = [round(x * 0.1, 1) for x in range(10, 16)]    # 1.0 to 1.5
    prob_skips = [0.30, 0.35, 0.40, 0.45, 0.50]
    prob_boosts = [0.45, 0.50, 0.55, 0.60]
    lb_longs = [72]
    lb_shorts = [12, 24]
    adx_ths = [25.0]

    DD_LIMIT = -20.0  # MaxDD must be > this (less negative)

    all_results = []

    # --- Phase 1: CFD OFF (L1.0) ---
    print(f"\n{'='*90}")
    print(f"  PHASE 1: CFD OFF (L1.0) — boost_mult 1.0~2.5, DD > {DD_LIMIT}%")
    print(f"{'='*90}")

    count = 0
    for bm in boost_mults_no_cfd:
        for ps in prob_skips:
            for pb in prob_boosts:
                if ps >= pb:
                    continue
                for lbl in lb_longs:
                    for lbs in lb_shorts:
                        for ath in adx_ths:
                            r = fast_backtest(
                                closes, n, atr_arr, rsi_arr, adx_arr, rvol_annual,
                                prob_long, prob_short,
                                leverage=1.0, adx_th=ath, sl_mult=2.5,
                                lb_long=lbl, lb_short=lbs,
                                prob_skip=ps, prob_boost=pb, boost_mult=bm,
                            )
                            count += 1
                            tag = f"L1.0 ADX{int(ath)} LB{lbl}/{lbs} P{ps:.2f}/{pb:.2f} B{bm:.1f}"
                            all_results.append(("CFD_OFF", tag, r, bm, ps, pb, lbl, lbs))

    print(f"  Tested: {count} combinations")

    # --- Phase 2: CFD 2x (L2.0) ---
    print(f"\n{'='*90}")
    print(f"  PHASE 2: CFD 2x (L2.0) — boost_mult 1.0~1.5, DD > {DD_LIMIT}%")
    print(f"{'='*90}")

    count2 = 0
    for bm in boost_mults_cfd2x:
        for ps in prob_skips:
            for pb in prob_boosts:
                if ps >= pb:
                    continue
                for lbl in lb_longs:
                    for lbs in lb_shorts:
                        for ath in adx_ths:
                            r = fast_backtest(
                                closes, n, atr_arr, rsi_arr, adx_arr, rvol_annual,
                                prob_long, prob_short,
                                leverage=2.0, adx_th=ath, sl_mult=2.5,
                                lb_long=lbl, lb_short=lbs,
                                prob_skip=ps, prob_boost=pb, boost_mult=bm,
                            )
                            count2 += 1
                            tag = f"L2.0 ADX{int(ath)} LB{lbl}/{lbs} P{ps:.2f}/{pb:.2f} B{bm:.1f}"
                            all_results.append(("CFD_2x", tag, r, bm, ps, pb, lbl, lbs))

    print(f"  Tested: {count2} combinations")

    # ===================================================================
    # RESULTS: Filter by DD constraint
    # ===================================================================
    within_dd = [(mode, tag, r, bm, ps, pb, lbl, lbs)
                 for mode, tag, r, bm, ps, pb, lbl, lbs in all_results
                 if r["max_dd"] > DD_LIMIT and r["trades"] > 50]
    within_dd.sort(key=lambda x: x[2]["annual"], reverse=True)

    # V4 current baseline
    v4_base = fast_backtest(
        closes, n, atr_arr, rsi_arr, adx_arr, rvol_annual,
        prob_long, prob_short,
        leverage=1.0, adx_th=25.0, sl_mult=2.5,
        lb_long=72, lb_short=24,
        prob_skip=0.40, prob_boost=0.55, boost_mult=1.2,
    )

    print(f"\n{'='*120}")
    print(f"  V4 CURRENT BASELINE")
    print(f"{'='*120}")
    print(f"  Annual={v4_base['annual']:.1f}%  Return={v4_base['total_ret']:.1f}%  "
          f"Trades={v4_base['trades']}  WR={v4_base['win_rate']:.1f}%  "
          f"MaxDD={v4_base['max_dd']:.1f}%  Sharpe={v4_base['sharpe']:.2f}")

    print(f"\n{'='*120}")
    print(f"  TOP STRATEGIES: MaxDD > {DD_LIMIT}% (total: {len(within_dd)} found)")
    print(f"{'='*120}")
    hdr = (f"{'#':>3} {'Mode':<8} {'Annual%':>8} {'Return%':>9} {'Trades':>7} "
           f"{'WinR%':>7} {'AvgW%':>7} {'AvgL%':>7} {'MaxDD%':>8} {'Sharpe':>7} {'Config'}")
    print(hdr)
    print("-" * 120)

    # Show CFD OFF top 15
    cfd_off = [(m, t, r, bm, ps, pb, lbl, lbs) for m, t, r, bm, ps, pb, lbl, lbs in within_dd if m == "CFD_OFF"]
    print("\n  --- CFD OFF (L1.0) ---")
    for rank, (mode, tag, r, bm, ps, pb, lbl, lbs) in enumerate(cfd_off[:15], 1):
        print(f"{rank:>3} {mode:<8} {r['annual']:>7.1f}% {r['total_ret']:>8.1f}% {r['trades']:>7d} "
              f"{r['win_rate']:>6.1f}% {r['avg_win']:>6.2f}% {r['avg_loss']:>6.2f}% "
              f"{r['max_dd']:>7.1f}% {r['sharpe']:>7.2f}  {tag}")

    # Show CFD 2x top 15
    cfd_on = [(m, t, r, bm, ps, pb, lbl, lbs) for m, t, r, bm, ps, pb, lbl, lbs in within_dd if m == "CFD_2x"]
    print(f"\n  --- CFD 2x (L2.0) ---")
    for rank, (mode, tag, r, bm, ps, pb, lbl, lbs) in enumerate(cfd_on[:15], 1):
        print(f"{rank:>3} {mode:<8} {r['annual']:>7.1f}% {r['total_ret']:>8.1f}% {r['trades']:>7d} "
              f"{r['win_rate']:>6.1f}% {r['avg_win']:>6.2f}% {r['avg_loss']:>6.2f}% "
              f"{r['max_dd']:>7.1f}% {r['sharpe']:>7.2f}  {tag}")

    print(f"\n{'='*120}")

    # === Comparison table ===
    best_off = cfd_off[0] if cfd_off else None
    best_on = cfd_on[0] if cfd_on else None

    print(f"\n{'='*120}")
    print(f"  FINAL COMPARISON: V4 Current vs Best CFD-OFF vs Best CFD-2x  (DD > {DD_LIMIT}%)")
    print(f"{'='*120}")

    rows = [("V4 Current", v4_base, "L1.0 B1.2 P0.40/0.55")]
    if best_off:
        rows.append(("V4+ CFD OFF", best_off[2], best_off[1]))
    if best_on:
        rows.append(("V4+ CFD 2x", best_on[2], best_on[1]))

    print(f"{'Strategy':<16} {'Annual%':>8} {'3yr Ret%':>10} {'Trades':>7} "
          f"{'WinR%':>7} {'MaxDD%':>8} {'Sharpe':>7} {'Config'}")
    print("-" * 120)
    for name, r, cfg in rows:
        print(f"{name:<16} {r['annual']:>7.1f}% {r['total_ret']:>9.1f}% {r['trades']:>7d} "
              f"{r['win_rate']:>6.1f}% {r['max_dd']:>7.1f}% {r['sharpe']:>7.2f}  {cfg}")
    print(f"{'='*120}")

    # === boost_mult vs DD curve (CFD OFF) ===
    print(f"\n{'='*80}")
    print(f"  BOOST_MULT vs MaxDD CURVE (CFD OFF, best prob per boost)")
    print(f"{'='*80}")
    print(f"{'boost':>7} {'Annual%':>8} {'MaxDD%':>8} {'Sharpe':>7} {'Config'}")
    print("-" * 60)

    for bm in boost_mults_no_cfd:
        subset = [(t, r) for m, t, r, bm2, ps, pb, lbl, lbs in all_results
                  if m == "CFD_OFF" and bm2 == bm and r["trades"] > 50]
        if not subset:
            continue
        # Best annual for this boost_mult
        subset.sort(key=lambda x: x[1]["annual"], reverse=True)
        best_tag, best_r = subset[0]
        print(f"{bm:>7.1f} {best_r['annual']:>7.1f}% {best_r['max_dd']:>7.1f}% "
              f"{best_r['sharpe']:>7.2f}  {best_tag}")

    # === boost_mult vs DD curve (CFD 2x) ===
    print(f"\n{'='*80}")
    print(f"  BOOST_MULT vs MaxDD CURVE (CFD 2x, best prob per boost)")
    print(f"{'='*80}")
    print(f"{'boost':>7} {'Annual%':>8} {'MaxDD%':>8} {'Sharpe':>7} {'Config'}")
    print("-" * 60)

    for bm in boost_mults_cfd2x:
        subset = [(t, r) for m, t, r, bm2, ps, pb, lbl, lbs in all_results
                  if m == "CFD_2x" and bm2 == bm and r["trades"] > 50]
        if not subset:
            continue
        subset.sort(key=lambda x: x[1]["annual"], reverse=True)
        best_tag, best_r = subset[0]
        print(f"{bm:>7.1f} {best_r['annual']:>7.1f}% {best_r['max_dd']:>7.1f}% "
              f"{best_r['sharpe']:>7.2f}  {best_tag}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
