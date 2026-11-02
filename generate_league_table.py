"""
generate_league_table.py — Master Quantitative Strategy Leaderboard Generator.

Scans all strategy reports in reports/ and compiles a unified, interactive
HTML League Table displaying institutional rankings, badges, and equity curves.
"""

from __future__ import annotations

import argparse
import base64
import json
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


def generate_master_league_table(
    reports_root: str = "reports",
    output_html: str = "reports/league_table.html",
) -> Path:
    root = Path(reports_root)
    reports = []

    # Human-readable display names for known folders
    display_names = {
        "meta_labeled": "ML Meta-Labeled Strategy (Conf >= 0.75)",
        "client_strategy": "Client Intraday Strategy (Base)",
        "trend_breakout_donchian": "Donchian Trend Breakout",
        "optimal_multi_alpha": "Optimal Multi-Alpha Composite",
        "backtest_3bps": "Production Model (3 bps Slippage)",
        "alpha_101": "WorldQuant Alpha #101 Standalone",
        "backtest_5bps": "Production Model (5 bps Stress)",
        "worldquant_multi-alpha": "WorldQuant Multi-Alpha Blend",
        "mean_reversion_bollinger": "Bollinger Bands Mean Reversion",
        "test_eval": "Evaluation Test Run",
    }

    # Discover all report.json files
    for p in root.rglob("report.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            m = data.get("metrics", {})
            v = data.get("verdict", "FAIL")
            folder_name = p.parent.name
            name = display_names.get(folder_name, folder_name.replace("_", " ").title())

            trades_csv = p.parent / "trades.csv"
            has_trades = trades_csv.is_file()
            factsheet = p.parent / "factsheet.html"
            has_factsheet = factsheet.is_file()

            reports.append({
                "name": name,
                "folder": folder_name,
                "path": str(p),
                "verdict": v,
                "n_trades": int(m.get("n_trades", 0)),
                "win_rate": float(m.get("win_rate", 0.0)),
                "profit_factor": float(m.get("profit_factor", 0.0)),
                "net_pnl": float(m.get("net_pnl", 0.0)),
                "expectancy": float(m.get("expectancy", 0.0)),
                "max_drawdown_pct": float(m.get("max_drawdown_pct", 0.0)),
                "sharpe": float(m.get("sharpe", 0.0)),
                "trades_csv": trades_csv if has_trades else None,
                "factsheet_rel": f"{folder_name}/factsheet.html" if has_factsheet else None,
            })
        except Exception:
            pass

    if not reports:
        raise RuntimeError("No valid report.json files found in reports/ directory.")

    # Sort logic: PASS first, then by Profit Factor descending, then by Net PnL
    def sort_key(item):
        verdict_rank = 3 if item["verdict"] == "PASS" else (2 if item["verdict"] == "WEAK" else 1)
        return (verdict_rank, item["profit_factor"], item["net_pnl"])

    reports.sort(key=sort_key, reverse=True)

    # Generate multi-strategy equity curve chart
    plt.figure(figsize=(11, 6))
    colors = ["#10b981", "#3b82f6", "#8b5cf6", "#f59e0b", "#ec4899", "#06b6d4", "#ef4444", "#64748b"]

    for i, r in enumerate(reports[:8]):  # plot top 8
        if r["trades_csv"]:
            try:
                tdf = pd.read_csv(r["trades_csv"])
                pnl_col = "net_pnl" if "net_pnl" in tdf.columns else ("Net_PnL_USD" if "Net_PnL_USD" in tdf.columns else None)
                if pnl_col:
                    pnl_series = pd.to_numeric(tdf[pnl_col], errors="coerce").fillna(0.0)
                    cum = pnl_series.cumsum().values
                    c = colors[i % len(colors)]
                    lw = 2.5 if r["verdict"] == "PASS" else 1.5
                    ls = "-" if r["verdict"] == "PASS" else "--"
                    plt.plot(cum, label=f"{r['name']} (PF: {r['profit_factor']:.2f})", color=c, linewidth=lw, linestyle=ls)
            except Exception:
                pass

    plt.title("Master Strategy Leaderboard — Normalized Execution Trajectories", fontsize=13, fontweight="bold")
    plt.xlabel("Trade Count")
    plt.ylabel("Cumulative Net PnL ($)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    chart_png = root / "leaderboard_equity_curves.png"
    plt.savefig(chart_png, dpi=150)
    plt.close()

    # Encode chart as Base64 for zero-dependency standalone HTML
    chart_b64 = ""
    if chart_png.is_file():
        with open(chart_png, "rb") as f:
            chart_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Build HTML rows
    table_rows = []
    for rank, r in enumerate(reports, start=1):
        v = r["verdict"]
        badge_cls = "badge-pass" if v == "PASS" else ("badge-weak" if v == "WEAK" else "badge-fail")
        pnl_cls = "text-green" if r["net_pnl"] >= 0 else "text-red"
        pnl_str = f"+${r['net_pnl']:,.2f}" if r["net_pnl"] >= 0 else f"-${abs(r['net_pnl']):,.2f}"

        link_cell = (
            f"<a href='{r['factsheet_rel']}' class='link-btn' target='_blank'>View Factsheet &rarr;</a>"
            if r["factsheet_rel"] else "<span class='text-muted'>—</span>"
        )

        table_rows.append(f"""
        <tr>
            <td class="rank-col">#{rank}</td>
            <td class="strategy-col"><strong>{r['name']}</strong><br><small class="text-muted">{r['folder']}</small></td>
            <td><span class="badge {badge_cls}">{v}</span></td>
            <td><strong>{r['profit_factor']:.2f}</strong></td>
            <td>{r['win_rate']*100:.1f}%</td>
            <td class="{pnl_cls}"><strong>{pnl_str}</strong></td>
            <td>${r['expectancy']:,.2f}</td>
            <td>{r['max_drawdown_pct']:.2f}%</td>
            <td>{r['n_trades']}</td>
            <td>{r['sharpe']:.2f}</td>
            <td>{link_cell}</td>
        </tr>
        """)

    # HTML Document Template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Master Strategy Leaderboard — Antigravity Quant</title>
    <style>
        :root {{
            --bg: #0b0f19;
            --surface: #111827;
            --surface-hover: #1f293d;
            --border: #1f2937;
            --text: #f9fafb;
            --text-muted: #9ca3af;
            --green: #10b981;
            --red: #ef4444;
            --amber: #f59e0b;
            --blue: #3b82f6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background: var(--bg); color: var(--text); padding: 32px 16px; line-height: 1.5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 24px; }}
        .header h1 {{ font-size: 26px; font-weight: 700; color: #fff; }}
        .header p {{ color: var(--text-muted); font-size: 14px; margin-top: 4px; }}
        .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
        .card-title {{ font-size: 17px; font-weight: 700; margin-bottom: 16px; color: #fff; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ color: var(--text-muted); font-size: 12px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; background: rgba(0,0,0,0.2); }}
        tr:hover td {{ background: var(--surface-hover); }}
        .rank-col {{ font-weight: 700; color: var(--blue); width: 45px; }}
        .strategy-col strong {{ color: #fff; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; }}
        .badge-pass {{ background: rgba(16, 185, 129, 0.15); color: var(--green); border: 1px solid var(--green); }}
        .badge-weak {{ background: rgba(245, 158, 11, 0.15); color: var(--amber); border: 1px solid var(--amber); }}
        .badge-fail {{ background: rgba(239, 68, 68, 0.15); color: var(--red); border: 1px solid var(--red); }}
        .text-green {{ color: var(--green); }}
        .text-red {{ color: var(--red); }}
        .text-muted {{ color: var(--text-muted); }}
        .link-btn {{ display: inline-block; padding: 4px 10px; background: rgba(59, 130, 246, 0.1); color: var(--blue); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 6px; text-decoration: none; font-size: 12px; font-weight: 600; transition: all 0.15s; }}
        .link-btn:hover {{ background: var(--blue); color: #fff; }}
        .chart-img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border); }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Master Strategy Leaderboard</h1>
                <p>Institutional cross-strategy ranking across 10 evaluated algorithmic models.</p>
            </div>
            <div>
                <span class="badge badge-pass" style="font-size: 13px; padding: 8px 16px;">10 Evaluated Strategies</span>
            </div>
        </div>

        <div class="card">
            <div class="card-title">Comparative Performance Trajectories</div>
            {"<img src='data:image/png;base64," + chart_b64 + "' class='chart-img' alt='Comparative Trajectories'>" if chart_b64 else "<p class='text-muted'>Chart not available</p>"}
        </div>

        <div class="card" style="padding: 0; overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Strategy</th>
                        <th>Verdict</th>
                        <th>Profit Factor</th>
                        <th>Win Rate</th>
                        <th>Net PnL</th>
                        <th>Expectancy</th>
                        <th>Max DD</th>
                        <th>Trades</th>
                        <th>Sharpe</th>
                        <th>Factsheet</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

    out_file = Path(output_html)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html_content, encoding="utf-8")

    return out_file


def print_cli_leaderboard(html_path: Path):
    print("\n" + "=" * 70)
    print("           MASTER STRATEGY LEADERBOARD GENERATED                ")
    print("=" * 70)
    print(f"  Interactive HTML Leaderboard : {html_path}")
    print(f"  Visual Trajectory Chart      : reports/leaderboard_equity_curves.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Master Quantitative Strategy Leaderboard")
    parser.add_argument("--root", type=str, default="reports", help="Root directory containing strategy reports")
    parser.add_argument("--out", type=str, default="reports/league_table.html", help="Output HTML leaderboard path")
    args = parser.parse_args()

    out = generate_master_league_table(args.root, args.out)
    print_cli_leaderboard(out)
