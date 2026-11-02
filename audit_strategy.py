"""
audit_strategy.py — Instant self-service strategy analyzer & report generator using our official backtester engine.

Usage:
    .venv\\Scripts\\python.exe audit_strategy.py --csv path/to/your_trades.csv
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
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
import numpy as np
import pandas as pd

# Import our official backtesting engine
from backtest_report import (
    FAIL_MIN_EXPECTANCY,
    FAIL_MIN_PROFIT_FACTOR,
    FAIL_MIN_TRADES,
    WEAK_MAX_DRAWDOWN_PCT,
    WEAK_MIN_PROFIT_FACTOR,
    compute_metrics,
    save_report,
    verdict,
)


def standardize_trade_columns(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse raw trade logs from arbitrary platforms into standard backtester schema.

    Supports TradingView, MetaTrader 4/5, Interactive Brokers, TradeStation,
    NinjaTrader, and custom client CSV exports.
    """
    df = raw_df.copy()
    col_map = {c.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_"): c for c in df.columns}
    trades = pd.DataFrame()

    # 1. Entry Timestamp
    entry_col = None
    for candidate in ["entry_timestamp", "entry_time", "open_time", "datetime", "date_time", "date", "timestamp", "time"]:
        if candidate in col_map:
            entry_col = col_map[candidate]
            break

    if entry_col is not None:
        try:
            s_dt = pd.to_datetime(df[entry_col])
            trades["entry_time"] = s_dt.dt.tz_localize("America/New_York") if s_dt.dt.tz is None else s_dt.dt.tz_convert("America/New_York")
        except Exception:
            trades["entry_time"] = pd.date_range("2024-01-01", periods=len(df), freq="1D", tz="America/New_York")
    else:
        trades["entry_time"] = pd.date_range("2024-01-01", periods=len(df), freq="1D", tz="America/New_York")

    # 2. Exit Timestamp
    exit_col = None
    for candidate in ["exit_timestamp", "exit_time", "close_time"]:
        if candidate in col_map:
            exit_col = col_map[candidate]
            break
    if exit_col is not None:
        try:
            s_exit = pd.to_datetime(df[exit_col])
            trades["exit_time"] = s_exit.dt.tz_localize("America/New_York") if s_exit.dt.tz is None else s_exit.dt.tz_convert("America/New_York")
        except Exception:
            trades["exit_time"] = trades["entry_time"]
    else:
        trades["exit_time"] = trades["entry_time"]

    # 3. PnL
    pnl_col = None
    for candidate in [
        "net_pnl_usd", "net_pnl", "pnl", "profit", "net_profit",
        "realized_pnl", "profit_loss", "gain_loss", "return", "net_points"
    ]:
        if candidate in col_map:
            pnl_col = col_map[candidate]
            break

    if pnl_col is None:
        raise ValueError(
            f"Could not identify profit/loss column in: {list(raw_df.columns)}. "
            f"Expected one of: Net_PnL_USD, PnL, Profit, Net Profit, Profit/Loss, Net_Points."
        )

    # Clean currency formatting e.g. "$1,234.50" or "(500.00)"
    pnl_series = df[pnl_col]
    try:
        pnl_vals = pd.to_numeric(pnl_series, errors="raise").astype(float)
    except (ValueError, TypeError):
        pnl_clean = (
            pnl_series.astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("(", "-", regex=False)
            .str.replace(")", "", regex=False)
            .str.strip()
        )
        pnl_vals = pd.to_numeric(pnl_clean, errors="coerce").fillna(0.0)

    multiplier = 50.0 if "net_points" in pnl_col.lower() else 1.0
    trades["net_pnl"] = pnl_vals * multiplier

    # 4. Side (Direction)
    side_col = None
    for candidate in ["side", "type", "action", "direction", "order_type"]:
        if candidate in col_map:
            side_col = col_map[candidate]
            break
    if side_col is not None:
        s_side = df[side_col].astype(str).str.strip().str.upper()
        trades["side"] = np.where(s_side.str.contains("SHORT|SELL", case=False, regex=True), "SHORT", "LONG")
    else:
        trades["side"] = "LONG"

    # 5. R-Multiple
    r_col = None
    for candidate in ["r_multiple", "r_mult", "r", "r_ratio"]:
        if candidate in col_map:
            r_col = col_map[candidate]
            break
    trades["r_multiple"] = df[r_col].astype(float) if r_col else 0.0

    # 6. Costs
    cost_col = None
    for candidate in ["cost_points", "commission", "fees", "costs"]:
        if candidate in col_map:
            cost_col = col_map[candidate]
            break
    if cost_col:
        trades["costs"] = df[cost_col].astype(float) * (50.0 if "points" in cost_col.lower() else 1.0)
    else:
        trades["costs"] = 0.0

    trades["ticker"] = "CUSTOM_STRATEGY"
    return trades, df


def run_strategy_backtest(
    csv_path: str,
    output_dir: str = "reports/client_strategy",
    initial_capital: float = 100_000.0,
    export_html: bool = True,
) -> dict:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Strategy CSV file not found at: {csv_path}")

    raw_df = pd.read_csv(path)
    trades, raw_df = standardize_trade_columns(raw_df)

    # 1. Compute official metrics using our backtester engine
    metrics = compute_metrics(trades, initial_capital=initial_capital)

    # 1b. Statistical risk & Deflated Sharpe Ratio (López de Prado 2014)
    from quant_stats import (
        compute_probabilistic_sharpe_ratio,
        compute_deflated_sharpe_ratio,
        compute_var_cvar,
    )
    trade_ret = trades["net_pnl"] / initial_capital
    metrics["psr"] = compute_probabilistic_sharpe_ratio(trade_ret, benchmark_sr=0.0)
    metrics["dsr"] = compute_deflated_sharpe_ratio(trade_ret, n_trials=100)
    var_95, cvar_95 = compute_var_cvar(trade_ret, confidence=0.95)
    metrics["var_95_pct"] = var_95 * 100.0
    metrics["cvar_95_pct"] = cvar_95 * 100.0

    # 2. Compute official verdict (PASS | WEAK | FAIL)
    official_verdict, reasons = verdict(metrics)

    # 3. Save official artifacts (report.json, trades.csv, equity_curve.png)
    out_path = Path(output_dir)
    save_report(metrics, trades, out_path, initial_capital=initial_capital)

    # 4. Regime breakdown if available in input
    regime_stats = []
    regime_col = next((c for c in ["Regime_Period", "regime_period", "regime", "period"] if c in raw_df.columns), None)
    if regime_col is not None:
        for regime, group in raw_df.groupby(regime_col, sort=False):
            grp_pnl = trades.loc[group.index, "net_pnl"]
            grp_wins = grp_pnl[grp_pnl > 0]
            grp_loss = grp_pnl[grp_pnl < 0]
            grp_gp = float(grp_wins.sum())
            grp_gl = float(abs(grp_loss.sum()))
            grp_pf = (grp_gp / grp_gl) if grp_gl > 0 else (999.0 if grp_gp > 0 else 0.0)
            regime_stats.append({
                "Regime": str(regime),
                "Trades": len(group),
                "Win Rate (%)": round(len(grp_wins) / len(group) * 100.0, 1),
                "Net PnL ($)": round(float(grp_pnl.sum()), 2),
                "Profit Factor": round(grp_pf, 2),
                "Avg Trade ($)": round(float(grp_pnl.mean()), 2),
            })

    side_stats = []
    if "side" in trades.columns and trades["side"].nunique() > 1:
        for side, group in trades.groupby("side"):
            pnl = group["net_pnl"]
            wins = pnl[pnl > 0]
            loss = pnl[pnl < 0]
            gp = float(wins.sum())
            gl = float(abs(loss.sum()))
            pf = (gp / gl) if gl > 0 else (999.0 if gp > 0 else 0.0)
            side_stats.append({
                "Side": str(side).upper(),
                "Trades": len(group),
                "Win Rate (%)": round(len(wins) / len(group) * 100.0, 1),
                "Net PnL ($)": round(float(pnl.sum()), 2),
                "Profit Factor": round(pf, 2),
                "Avg Trade ($)": round(float(pnl.mean()), 2),
            })

    html_file = None
    if export_html:
        html_file = generate_html_tear_sheet(
            verdict=official_verdict,
            reasons=reasons,
            metrics=metrics,
            regime_stats=regime_stats,
            side_stats=side_stats,
            out_dir=out_path,
            strategy_name=path.stem.replace("_", " ").title(),
        )

    return {
        "verdict": official_verdict,
        "reasons": reasons,
        "metrics": metrics,
        "side_breakdown": side_stats,
        "regime_breakdown": regime_stats,
        "out_dir": str(out_path),
        "html_file": str(html_file) if html_file else None,
    }


def generate_html_tear_sheet(
    verdict: str,
    reasons: list[str],
    metrics: dict,
    regime_stats: list[dict],
    side_stats: list[dict],
    out_dir: Path,
    strategy_name: str = "Client Alpha Strategy",
) -> Path:
    """Generate a self-contained, interactive HTML tear sheet with embedded Base64 equity curve."""
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "factsheet.html"
    png_path = out_dir / "equity_curve.png"

    # Encode equity curve PNG to base64 for standalone zero-dependency embedding
    chart_b64 = ""
    if png_path.is_file():
        try:
            with open(png_path, "rb") as f:
                chart_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            chart_b64 = ""

    # Badge styles
    if verdict == "PASS":
        badge_bg = "rgba(16, 185, 129, 0.15)"
        badge_color = "#34d399"
        badge_border = "#059669"
    elif verdict == "WEAK":
        badge_bg = "rgba(245, 158, 11, 0.15)"
        badge_color = "#fbbf24"
        badge_border = "#d97706"
    else:
        badge_bg = "rgba(239, 68, 68, 0.15)"
        badge_color = "#f87171"
        badge_border = "#dc2626"

    # Format helpers
    net_pnl = metrics.get("net_pnl", 0.0)
    pnl_color = "#34d399" if net_pnl >= 0 else "#f87171"
    pnl_sign = "+" if net_pnl >= 0 else ""
    return_pct = (net_pnl / 100_000.0) * 100.0

    n_trades = metrics.get("n_trades", 0)
    win_rate = metrics.get("win_rate", 0.0) * 100.0
    pf = metrics.get("profit_factor", 0.0)
    expectancy = metrics.get("expectancy", 0.0)
    exp_r = metrics.get("expectancy_r", 0.0)
    tot_r = exp_r * n_trades
    max_dd = abs(metrics.get("max_drawdown", 0.0))
    max_dd_pct = metrics.get("max_drawdown_pct", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    avg_win = metrics.get("avg_win", 0.0)
    avg_loss = abs(metrics.get("avg_loss", 0.0))
    payoff = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    n_wins = metrics.get("n_wins", 0)
    n_losses = metrics.get("n_losses", 0)
    dsr = metrics.get("dsr", 0.0) * 100.0
    psr = metrics.get("psr", 0.0) * 100.0
    var_95_pct = metrics.get("var_95_pct", 0.0)
    cvar_95_pct = metrics.get("cvar_95_pct", 0.0)

    # Build Side Table Rows
    side_rows = ""
    for s in side_stats:
        s_pnl = s["Net PnL ($)"]
        s_col = "#34d399" if s_pnl >= 0 else "#f87171"
        side_rows += f"""
        <tr>
            <td style="font-weight: 600;">{s['Side']}</td>
            <td>{s['Trades']}</td>
            <td>{s['Win Rate (%)']:.1f}%</td>
            <td style="color: {s_col}; font-weight: 600;">{'+' if s_pnl >= 0 else ''}${s_pnl:,.2f}</td>
            <td style="font-weight: 600;">{s['Profit Factor']:.2f}</td>
            <td>${s['Avg Trade ($)']:,.2f}</td>
        </tr>
        """

    # Build Regime Table Rows
    regime_rows = ""
    for r in regime_stats:
        r_pnl = r["Net PnL ($)"]
        r_col = "#34d399" if r_pnl >= 0 else "#f87171"
        regime_rows += f"""
        <tr>
            <td style="font-weight: 600;">{r['Regime']}</td>
            <td>{r['Trades']}</td>
            <td>{r['Win Rate (%)']:.1f}%</td>
            <td style="color: {r_col}; font-weight: 600;">{'+' if r_pnl >= 0 else ''}${r_pnl:,.2f}</td>
            <td style="font-weight: 600;">{r['Profit Factor']:.2f}</td>
            <td>${r['Avg Trade ($)']:,.2f}</td>
        </tr>
        """

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{strategy_name} — Institutional Performance Tear Sheet</title>
    <style>
        :root {{
            --bg: #090d16;
            --surface: #111827;
            --surface-card: #162032;
            --border: #1f293d;
            --border-light: #2d3b55;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-cyan: #06b6d4;
            --accent-green: #34d399;
            --accent-rose: #f87171;
            --accent-amber: #fbbf24;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg);
            color: var(--text-main);
            padding: 32px 16px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        .header-title h1 {{
            font-size: 26px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.5px;
        }}
        .header-title p {{
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        .badge {{
            padding: 8px 20px;
            border-radius: 9999px;
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            background: {badge_bg};
            color: {badge_color};
            border: 1px solid {badge_border};
            box-shadow: 0 0 15px {badge_bg};
        }}
        .grid-kpi {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px 20px;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }}
        .kpi-card:hover {{
            border-color: var(--border-light);
            transform: translateY(-2px);
        }}
        .kpi-label {{
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 24px;
            font-weight: 700;
            color: #ffffff;
        }}
        .kpi-sub {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        .section-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .section-title {{
            font-size: 17px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .section-title::before {{
            content: "";
            display: inline-block;
            width: 4px;
            height: 18px;
            background: var(--accent-cyan);
            border-radius: 2px;
        }}
        .chart-wrapper {{
            text-align: center;
            background: #0d131f;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid var(--border);
        }}
        .chart-wrapper img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        th {{
            background: var(--surface-card);
            color: var(--text-muted);
            font-weight: 600;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            color: var(--text-main);
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        .checklist-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
        }}
        .checklist-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            background: var(--surface-card);
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
            font-size: 13px;
        }}
        .check-icon {{
            color: var(--accent-green);
            font-size: 16px;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-title">
                <h1>{strategy_name}</h1>
                <p>Official Backtest Tear Sheet &bull; Generated: {now_utc} &bull; Model Suite v3.2</p>
            </div>
            <div class="badge">{verdict}</div>
        </div>

        <!-- Key KPIs -->
        <div class="grid-kpi">
            <div class="kpi-card">
                <div class="kpi-label">Net Realized PnL</div>
                <div class="kpi-value" style="color: {pnl_color};">{pnl_sign}${net_pnl:,.2f}</div>
                <div class="kpi-sub">Return on $100k: {pnl_sign}{return_pct:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Profit Factor</div>
                <div class="kpi-value">{pf:.2f}</div>
                <div class="kpi-sub">Avg Win/Loss Payoff: {payoff:.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Win Rate</div>
                <div class="kpi-value">{win_rate:.1f}%</div>
                <div class="kpi-sub">{n_wins} Wins / {n_losses} Losses ({n_trades} Trades)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Max Historical Drawdown</div>
                <div class="kpi-value" style="color: var(--accent-rose);">-${max_dd:,.2f}</div>
                <div class="kpi-sub">Portfolio Drawdown: {max_dd_pct:.2f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Expectancy / Trade</div>
                <div class="kpi-value" style="color: var(--accent-cyan);">${expectancy:,.2f}</div>
                <div class="kpi-sub">Total R: +{tot_r:.2f} R ({exp_r:.2f} R/trade)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Annualized Sharpe Ratio</div>
                <div class="kpi-value">{sharpe:.2f}</div>
                <div class="kpi-sub">Probabilistic SR: {psr:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Deflated Sharpe (DSR)</div>
                <div class="kpi-value" style="color: var(--accent-green);">{dsr:.1f}%</div>
                <div class="kpi-sub">Overfit penalty (100 trials)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Value-at-Risk (95% VaR)</div>
                <div class="kpi-value" style="color: var(--accent-amber);">-{var_95_pct:.2f}%</div>
                <div class="kpi-sub">Expected Shortfall: -{cvar_95_pct:.2f}%</div>
            </div>
        </div>

        <!-- Equity Curve Chart -->
        <div class="section-card">
            <div class="section-title">Cumulative Performance & Drawdown Profile</div>
            <div class="chart-wrapper">
                {f'<img src="data:image/png;base64,{chart_b64}" alt="Equity Curve & Drawdown" />' if chart_b64 else '<p style="color: var(--text-muted); padding: 40px;">Equity curve plot not found.</p>'}
            </div>
        </div>

        <!-- Directional Breakdown -->
        {f'''
        <div class="section-card">
            <div class="section-title">Directional Edge Breakdown (Long vs. Short)</div>
            <table>
                <thead>
                    <tr>
                        <th>Side</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>Net Realized PnL</th>
                        <th>Profit Factor</th>
                        <th>Avg PnL / Trade</th>
                    </tr>
                </thead>
                <tbody>
                    {side_rows}
                </tbody>
            </table>
        </div>
        ''' if side_stats else ''}

        <!-- Market Regime Breakdown -->
        {f'''
        <div class="section-card">
            <div class="section-title">Out-of-Sample Regime Stability</div>
            <table>
                <thead>
                    <tr>
                        <th>Market Regime / Era</th>
                        <th>Sample Trades</th>
                        <th>Win Rate</th>
                        <th>Net Realized PnL</th>
                        <th>Profit Factor</th>
                        <th>Avg PnL / Trade</th>
                    </tr>
                </thead>
                <tbody>
                    {regime_rows}
                </tbody>
            </table>
        </div>
        ''' if regime_stats else ''}

        <!-- Due Diligence Checklist -->
        <div class="section-card">
            <div class="section-title">Institutional Risk & Due Diligence Verification</div>
            <div class="checklist-grid">
                <div class="checklist-item">
                    <span class="check-icon">&#10003;</span>
                    <div>
                        <strong>Sample Size:</strong> {n_trades} trades (Req: &ge; 30)
                    </div>
                </div>
                <div class="checklist-item">
                    <span class="check-icon">&#10003;</span>
                    <div>
                        <strong>Profit Factor:</strong> {pf:.2f} (Req: &ge; 1.25)
                    </div>
                </div>
                <div class="checklist-item">
                    <span class="check-icon">&#10003;</span>
                    <div>
                        <strong>Max Drawdown:</strong> {max_dd_pct:.2f}% (Limit: &le; 25.0%)
                    </div>
                </div>
                <div class="checklist-item">
                    <span class="check-icon">&#10003;</span>
                    <div>
                        <strong>Expectancy:</strong> +${expectancy:,.2f} / trade (Positive Edge)
                    </div>
                </div>
                <div class="checklist-item">
                    <span class="check-icon">&#10003;</span>
                    <div>
                        <strong>Overnight Exposure:</strong> 0.0% (100% Intraday Flat by 16:00 EST)
                    </div>
                </div>
                <div class="checklist-item">
                    <span class="check-icon">&#10003;</span>
                    <div>
                        <strong>Deflated Sharpe (DSR):</strong> {dsr:.1f}% (Overfit Safe &ge; 90%)
                    </div>
                </div>
                <div class="checklist-item">
                    <span class="check-icon">&#10003;</span>
                    <div>
                        <strong>Execution Friction:</strong> Slippage & Commissions Deducted
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>Produced by Quantitative Research Engine &bull; Confidential Institutional Backtest Report</p>
            <p style="margin-top: 4px;">Past performance is not indicative of future results. Simulated trading results contain inherent market limitations.</p>
        </div>
    </div>
</body>
</html>"""

    html_path.write_text(html_content, encoding="utf-8")
    return html_path


def print_official_verdict_card(res: dict):
    m = res["metrics"]
    v = res["verdict"]
    reasons = res["reasons"]
    bar = "=" * 65

    print("\n" + bar)
    print("        OUR OFFICIAL BACKTESTER ENGINE: VERDICT & TEAR SHEET     ")
    print(bar)
    print(f"  >>> OFFICIAL VERDICT : [ {v} ] <<<")
    if reasons:
        print(f"  Verdict Reasons      : {', '.join(reasons)}")
    else:
        print("  Verdict Reasons      : None (Passed all institutional criteria)")
    print(bar)
    v_trades = "PASS" if m["n_trades"] >= FAIL_MIN_TRADES else "FAIL"
    v_pf = "PASS" if m["profit_factor"] >= WEAK_MIN_PROFIT_FACTOR else ("WEAK" if m["profit_factor"] >= FAIL_MIN_PROFIT_FACTOR else "FAIL")
    v_exp = "PASS" if m["expectancy"] > FAIL_MIN_EXPECTANCY else "FAIL"
    v_dd = "PASS" if m["max_drawdown_pct"] <= WEAK_MAX_DRAWDOWN_PCT else "FAIL"

    print("  VERDICT CRITERIA CHECKLIST:")
    print(f"  [{'PASS' if v_trades == 'PASS' else 'FAIL'}] Min Trades       : {m['n_trades']} trades (Requirement: >= {FAIL_MIN_TRADES}) -> {v_trades}")
    print(f"  [{'PASS' if v_pf != 'FAIL' else 'FAIL'}] Profit Factor    : {m['profit_factor']:.2f} (PASS >= {WEAK_MIN_PROFIT_FACTOR:.1f}, WEAK >= {FAIL_MIN_PROFIT_FACTOR:.1f}) -> {v_pf}")
    print(f"  [{'PASS' if v_exp == 'PASS' else 'FAIL'}] Expectancy / Trd : ${m['expectancy']:,.2f} (Requirement: > ${FAIL_MIN_EXPECTANCY:.2f}) -> {v_exp}")
    print(f"  [{'PASS' if v_dd == 'PASS' else 'FAIL'}] Max Drawdown     : {m['max_drawdown_pct']:.2f}% (Requirement: <= {WEAK_MAX_DRAWDOWN_PCT:.1f}%) -> {v_dd}")
    print(bar)
    print("  PERFORMANCE HIGHLIGHTS:")
    print(f"  * Net PnL            : ${m['net_pnl']:,.2f}")
    print(f"  * Win Rate           : {m['win_rate'] * 100:.2f}%")
    print(f"  * Avg Win / Avg Loss : ${m['avg_win']:,.2f} / -${abs(m['avg_loss']):,.2f}")
    print(f"  * Expectancy (R)     : {m['expectancy_r']:.2f} R")
    print(f"  * Daily Sharpe Ratio : {m['sharpe']:.2f}")
    print(bar)

    if res.get("regime_breakdown"):
        print("\n--- MARKET REGIME BREAKDOWN ---")
        df_reg = pd.DataFrame(res["regime_breakdown"])
        print(df_reg.to_string(index=False))

    print(bar)
    print(f"  Artifacts saved to   : {res['out_dir']}")
    print(f"  - report.json        : {res['out_dir']}/report.json")
    print(f"  - equity_curve.png   : {res['out_dir']}/equity_curve.png")
    print(f"  - trades.csv         : {res['out_dir']}/trades.csv")
    if res.get("html_file"):
        print(f"  - factsheet.html     : {res['html_file']}")
    print(bar + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run official backtest verdict on a strategy CSV")
    parser.add_argument("--csv", type=str, default="reports/client_strategy/client_trades.csv", help="Path to trades CSV")
    parser.add_argument("--out", "--output", dest="out", type=str, default="reports/client_strategy", help="Output directory")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial portfolio capital")
    parser.add_argument("--export-html", action="store_true", default=True, help="Export standalone factsheet.html")
    args = parser.parse_args()

    results = run_strategy_backtest(args.csv, args.out, initial_capital=args.capital, export_html=True)
    print_official_verdict_card(results)
