"""
meta_label_optimizer.py — Automated Meta-Labeling Optimization Engine.

Identifies the mathematically optimal ML confidence threshold to maximize
Profit Factor and Net PnL while maintaining statistical significance (N >= 100).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

# Silence non-critical matplotlib cache warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_report import compute_metrics, save_report, verdict
from audit_strategy import standardize_trade_columns, generate_html_tear_sheet


def run_meta_labeling_optimization(
    csv_path: str,
    output_dir: str = "reports/meta_labeled",
    min_trades_hurdle: int = 100,
) -> dict:
    raw_df = pd.read_csv(csv_path)
    trades, _ = standardize_trade_columns(raw_df)

    # Check for confidence column
    conf_col = None
    for candidate in ["confidence", "conf", "prob", "probability", "score"]:
        for c in raw_df.columns:
            if c.strip().lower() == candidate:
                conf_col = c
                break
        if conf_col:
            break

    if conf_col is None:
        raise ValueError("No confidence or probability column found in input CSV.")

    raw_df["_conf"] = pd.to_numeric(raw_df[conf_col], errors="coerce").fillna(0.5)

    # 1. Base Strategy Performance (Threshold = 0.0)
    base_metrics = compute_metrics(trades)
    base_verdict, base_reasons = verdict(base_metrics)

    # 2. Grid Search for Optimal Confidence Hurdle
    grid_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    sweep_results = []
    best_candidate = None
    highest_score = -float("inf")

    for tau in grid_thresholds:
        mask = raw_df["_conf"] >= tau
        sub_trades = trades[mask].copy().reset_index(drop=True)
        if len(sub_trades) < 20:
            continue

        m = compute_metrics(sub_trades)
        v, _ = verdict(m)

        # Composite score: Profit Factor * Log(Trades) penalized by Drawdown
        score = (m["profit_factor"] * np.log10(m["n_trades"])) / max(1.0, m["max_drawdown_pct"])

        record = {
            "threshold": tau,
            "n_trades": m["n_trades"],
            "win_rate_pct": round(m["win_rate"] * 100, 2),
            "profit_factor": round(m["profit_factor"], 2),
            "net_pnl": round(m["net_pnl"], 2),
            "expectancy": round(m["expectancy"], 2),
            "max_drawdown_pct": round(m["max_drawdown_pct"], 2),
            "verdict": v,
            "meets_min_trades": m["n_trades"] >= min_trades_hurdle,
        }
        sweep_results.append(record)

        # Optimize for best score while respecting minimum sample size
        if record["meets_min_trades"] and score > highest_score:
            highest_score = score
            best_candidate = (tau, sub_trades, m, v)

    # Fallback if no hurdle satisfies min_trades
    if best_candidate is None and sweep_results:
        best_candidate = (sweep_results[0]["threshold"], trades, base_metrics, base_verdict)

    opt_tau, opt_trades, opt_metrics, opt_verdict = best_candidate

    # 3. Save Comparative Artifacts
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_report(opt_metrics, opt_trades, out_path)

    # 4. Generate Interactive Standalone HTML Factsheet
    html_file = generate_html_tear_sheet(
        verdict=opt_verdict,
        reasons=[],
        metrics=opt_metrics,
        regime_stats=[],
        side_stats=[],
        out_dir=out_path,
        strategy_name=f"ML Meta-Labeled Strategy (Conf >= {opt_tau:.2f})",
    )

    # 5. Generate Comparative Before-and-After Equity Curve
    plt.figure(figsize=(10, 6))
    base_cum = trades["net_pnl"].cumsum().values
    opt_cum = opt_trades["net_pnl"].cumsum().values

    plt.plot(
        base_cum,
        label=f"Original Strategy (PF: {base_metrics['profit_factor']:.2f}, WR: {base_metrics['win_rate']*100:.1f}%)",
        color="#94a3b8",
        linewidth=1.8,
        linestyle="--",
    )
    plt.plot(
        opt_cum,
        label=f"ML Meta-Labeled @ Conf >= {opt_tau} (PF: {opt_metrics['profit_factor']:.2f}, WR: {opt_metrics['win_rate']*100:.1f}%)",
        color="#10b981",
        linewidth=2.5,
    )

    plt.title(
        f"Meta-Labeling Optimization: Original vs. ML-Filtered Equity Curve",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Trade Execution Count")
    plt.ylabel("Cumulative Net PnL ($)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    chart_file = out_path / "comparative_equity_curve.png"
    plt.savefig(chart_file, dpi=150)
    plt.close()

    return {
        "base_metrics": base_metrics,
        "base_verdict": base_verdict,
        "optimal_threshold": opt_tau,
        "optimal_metrics": opt_metrics,
        "optimal_verdict": opt_verdict,
        "sweep": sweep_results,
        "chart_path": str(chart_file),
        "html_file": str(html_file),
        "out_dir": str(out_path),
    }


def print_meta_labeling_card(res: dict):
    b = res["base_metrics"]
    o = res["optimal_metrics"]
    tau = res["optimal_threshold"]
    bar = "=" * 68

    print("\n" + bar)
    print("      AUTOMATED ML META-LABELING OPTIMIZATION TEAR SHEET        ")
    print(bar)
    print(f"  Optimal Confidence Hurdle Selected : [ >= {tau:.2f} ]")
    print(f"  Official Verdict After ML Filtering : [ {res['optimal_verdict']} ]")
    print(bar)
    print("  METRIC COMPARISON (BEFORE VS AFTER ML FILTERING):")
    print(f"  Metric              | Original Strategy | ML Filtered (Conf >= {tau:.2f}) | Net Change")
    print("  --------------------+-------------------+-------------------------+-----------")
    print(f"  Total Trades        | {b['n_trades']:17d} | {o['n_trades']:23d} | -{b['n_trades'] - o['n_trades']} trades")
    print(f"  Win Rate (%)        | {b['win_rate']*100:16.2f}% | {o['win_rate']*100:22.2f}% | +{(o['win_rate'] - b['win_rate'])*100:+.2f}%")
    print(f"  Profit Factor       | {b['profit_factor']:17.2f} | {o['profit_factor']:23.2f} | {o['profit_factor'] - b['profit_factor']:+.2f}")
    print(f"  Total Net PnL ($)   | ${b['net_pnl']:16,.2f} | ${o['net_pnl']:22,.2f} | +${o['net_pnl'] - b['net_pnl']:+,.2f}")
    print(f"  Expectancy / Trade  | ${b['expectancy']:16,.2f} | ${o['expectancy']:22,.2f} | +${o['expectancy'] - b['expectancy']:+,.2f}")
    print(f"  Max Drawdown (%)    | {b['max_drawdown_pct']:16.2f}% | {o['max_drawdown_pct']:22.2f}% | {o['max_drawdown_pct'] - b['max_drawdown_pct']:+.2f}%")
    print(bar)

    print("\n--- CONFIDENCE GRID SWEEP RESULTS ---")
    sweep_df = pd.DataFrame(res["sweep"])
    print(sweep_df.to_string(index=False))

    print(bar)
    print(f"  Comparative Equity Chart saved to : {res['chart_path']}")
    if res.get("html_file"):
        print(f"  Interactive HTML Factsheet        : {res['html_file']}")
    print(f"  Optimized Audit Artifacts in      : {res['out_dir']}")
    print(bar + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Meta-Labeling optimization on a strategy CSV")
    parser.add_argument("--csv", type=str, default="reports/client_strategy/client_trades.csv", help="Path to strategy CSV")
    parser.add_argument("--out", type=str, default="reports/meta_labeled", help="Output directory")
    parser.add_argument("--min-trades", type=int, default=100, help="Minimum sample size requirement")
    args = parser.parse_args()

    results = run_meta_labeling_optimization(args.csv, args.out, min_trades_hurdle=args.min_trades)
    print_meta_labeling_card(results)
