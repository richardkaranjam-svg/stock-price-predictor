"""
benchmark_strategies.py — Ready-to-Present Institutional Benchmark Strategies & Comparative Auditing.

Generates and audits 3 foundational institutional trading strategies:
1. Trend Breakout (20-bar Donchian Channel Breakout with ATR stops)
2. Mean Reversion (Bollinger Band & RSI extremes with reversion targets)
3. Multi-Alpha Ensemble (WorldQuant Kakushadze multi-factor signal portfolio)

Outputs single-file HTML tear sheets in reports/benchmarks/<strategy>/factsheet.html
and prints an executive comparative scorecard.

Usage:
    .venv\\Scripts\\python.exe benchmark_strategies.py
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

from audit_strategy import run_strategy_backtest
import formulaic_alphas as fa


def generate_market_data(n_bars: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic intraday/daily bars covering multi-year market regimes."""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-03 09:30", periods=n_bars, freq="B", tz="America/New_York")

    # Drift with volatility regimes (bull, bear, chop) proportional to n_bars
    n1 = int(n_bars * 0.35)
    n2 = int(n_bars * 0.25)
    n3 = n_bars - n1 - n2
    regimes = np.concatenate([np.full(n1, 0.0008), np.full(n2, -0.0006), np.full(n3, 0.0003)])
    vols = np.concatenate([np.full(n1, 0.012), np.full(n2, 0.024), np.full(n3, 0.015)])
    returns = np.random.normal(regimes, vols)

    close = 150.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.abs(np.random.normal(0, 0.008, n_bars)))
    low = close * (1.0 - np.abs(np.random.normal(0, 0.008, n_bars)))
    open_ = low + np.random.uniform(0.2, 0.8, n_bars) * (high - low)
    volume = np.random.lognormal(mean=14, sigma=0.5, size=n_bars)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)

    # 14-period ATR
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    df["ATR"] = pd.Series(tr, index=df.index).rolling(14, min_periods=1).mean()
    return df


def simulate_trend_breakout_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy 1: 20-period Donchian Channel Breakout with 1.5x ATR trailing stop."""
    upper = df["High"].rolling(20).max().shift(1)
    lower = df["Low"].rolling(20).min().shift(1)

    trades = []
    position = None

    for i in range(25, len(df)):
        ts = df.index[i]
        c = df["Close"].iloc[i]
        h = df["High"].iloc[i]
        l = df["Low"].iloc[i]
        atr = df["ATR"].iloc[i]

        if position is not None:
            # Check stops
            if position["side"] == "LONG":
                if l <= position["sl"]:
                    exit_p = position["sl"]
                    pnl = (exit_p - position["entry"]) * position["qty"]
                    trades.append({
                        "Entry_Timestamp": position["entry_time"].strftime("%Y-%m-%d %H:%M"),
                        "Exit_Timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                        "Side": "LONG",
                        "Net_PnL_USD": pnl,
                        "R_Multiple": round(pnl / (position["init_risk"] * position["qty"]), 2),
                        "Regime_Period": position["entry_time"].strftime("%Y-H1") if position["entry_time"].month <= 6 else position["entry_time"].strftime("%Y-H2"),
                    })
                    position = None
                    continue
                else:
                    # Trail stop higher
                    position["sl"] = max(position["sl"], c - 1.5 * atr)
            elif position["side"] == "SHORT":
                if h >= position["sl"]:
                    exit_p = position["sl"]
                    pnl = (position["entry"] - exit_p) * position["qty"]
                    trades.append({
                        "Entry_Timestamp": position["entry_time"].strftime("%Y-%m-%d %H:%M"),
                        "Exit_Timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                        "Side": "SHORT",
                        "Net_PnL_USD": pnl,
                        "R_Multiple": round(pnl / (position["init_risk"] * position["qty"]), 2),
                        "Regime_Period": position["entry_time"].strftime("%Y-H1") if position["entry_time"].month <= 6 else position["entry_time"].strftime("%Y-H2"),
                    })
                    position = None
                    continue
                else:
                    # Trail stop lower
                    position["sl"] = min(position["sl"], c + 1.5 * atr)

        if position is None:
            if c > upper.iloc[i]:
                risk = 1.5 * atr
                position = {
                    "side": "LONG",
                    "entry": c,
                    "entry_time": ts,
                    "qty": 100,
                    "sl": c - risk,
                    "init_risk": risk,
                }
            elif c < lower.iloc[i]:
                risk = 1.5 * atr
                position = {
                    "side": "SHORT",
                    "entry": c,
                    "entry_time": ts,
                    "qty": 100,
                    "sl": c + risk,
                    "init_risk": risk,
                }

    return pd.DataFrame(trades)


def simulate_mean_reversion_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy 2: 20-period Bollinger Band (2.0 std) Mean Reversion with 1.0x ATR TP."""
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20

    trades = []
    position = None

    for i in range(25, len(df)):
        ts = df.index[i]
        c = df["Close"].iloc[i]
        h = df["High"].iloc[i]
        l = df["Low"].iloc[i]
        atr = df["ATR"].iloc[i]

        if position is not None:
            # Check TP / SL
            if position["side"] == "LONG":
                if h >= position["tp"]:
                    pnl = (position["tp"] - position["entry"]) * position["qty"]
                    trades.append({
                        "Entry_Timestamp": position["entry_time"].strftime("%Y-%m-%d %H:%M"),
                        "Exit_Timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                        "Side": "LONG",
                        "Net_PnL_USD": pnl,
                        "R_Multiple": 1.25,
                        "Regime_Period": position["entry_time"].strftime("%Y-H1") if position["entry_time"].month <= 6 else position["entry_time"].strftime("%Y-H2"),
                    })
                    position = None
                    continue
                elif l <= position["sl"]:
                    pnl = (position["sl"] - position["entry"]) * position["qty"]
                    trades.append({
                        "Entry_Timestamp": position["entry_time"].strftime("%Y-%m-%d %H:%M"),
                        "Exit_Timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                        "Side": "LONG",
                        "Net_PnL_USD": pnl,
                        "R_Multiple": -1.0,
                        "Regime_Period": position["entry_time"].strftime("%Y-H1") if position["entry_time"].month <= 6 else position["entry_time"].strftime("%Y-H2"),
                    })
                    position = None
                    continue
            elif position["side"] == "SHORT":
                if l <= position["tp"]:
                    pnl = (position["entry"] - position["tp"]) * position["qty"]
                    trades.append({
                        "Entry_Timestamp": position["entry_time"].strftime("%Y-%m-%d %H:%M"),
                        "Exit_Timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                        "Side": "SHORT",
                        "Net_PnL_USD": pnl,
                        "R_Multiple": 1.25,
                        "Regime_Period": position["entry_time"].strftime("%Y-H1") if position["entry_time"].month <= 6 else position["entry_time"].strftime("%Y-H2"),
                    })
                    position = None
                    continue
                elif h >= position["sl"]:
                    pnl = (position["entry"] - position["sl"]) * position["qty"]
                    trades.append({
                        "Entry_Timestamp": position["entry_time"].strftime("%Y-%m-%d %H:%M"),
                        "Exit_Timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                        "Side": "SHORT",
                        "Net_PnL_USD": pnl,
                        "R_Multiple": -1.0,
                        "Regime_Period": position["entry_time"].strftime("%Y-H1") if position["entry_time"].month <= 6 else position["entry_time"].strftime("%Y-H2"),
                    })
                    position = None
                    continue

        if position is None:
            # Overextended below lower band -> Buy reversal
            if c < lower.iloc[i]:
                position = {
                    "side": "LONG",
                    "entry": c,
                    "entry_time": ts,
                    "qty": 100,
                    "tp": c + 1.25 * atr,
                    "sl": c - 1.0 * atr,
                }
            # Overextended above upper band -> Sell reversal
            elif c > upper.iloc[i]:
                position = {
                    "side": "SHORT",
                    "entry": c,
                    "entry_time": ts,
                    "qty": 100,
                    "tp": c - 1.25 * atr,
                    "sl": c + 1.0 * atr,
                }

    return pd.DataFrame(trades)


def simulate_multi_alpha_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy 3: WorldQuant Multi-Alpha Portfolio (Ensemble of Alpha 001, 002, 101)."""
    alphas = fa.compute_all_alphas(df)
    composite = (
        0.40 * alphas["alpha_001"] +
        0.35 * -alphas["alpha_002"] +  # Alpha 2 has negative IC -> flip sign
        0.25 * alphas["alpha_101"]
    )
    return fa.simulate_alpha_trades(composite, df["Close"], horizon=5, top_quantile=0.80)


def run_all_benchmarks(out_base: str = "reports/benchmarks") -> pd.DataFrame:
    """Execute, audit, and compare all benchmark strategies."""
    os.makedirs(out_base, exist_ok=True)
    df_market = generate_market_data(n_bars=1200)

    strategies = {
        "Trend Breakout (Donchian)": simulate_trend_breakout_strategy(df_market),
        "Mean Reversion (Bollinger)": simulate_mean_reversion_strategy(df_market),
        "WorldQuant Multi-Alpha": simulate_multi_alpha_strategy(df_market),
    }

    summary_rows = []

    print("\n" + "=" * 70)
    print("      INSTITUTIONAL BENCHMARK STRATEGIES: PERFORMANCE AUDIT       ")
    print("=" * 70)

    for name, trades_df in strategies.items():
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        strat_dir = Path(out_base) / slug
        strat_dir.mkdir(parents=True, exist_ok=True)

        csv_path = strat_dir / "trades.csv"
        trades_df.to_csv(csv_path, index=False)

        audit_res = run_strategy_backtest(
            str(csv_path),
            output_dir=str(strat_dir),
            initial_capital=100_000.0,
            export_html=True,
        )

        m = audit_res["metrics"]
        v = audit_res["verdict"]

        summary_rows.append({
            "Strategy Name": name,
            "Trades": m["n_trades"],
            "Win Rate (%)": round(m["win_rate"] * 100, 1),
            "Profit Factor": round(m["profit_factor"], 2),
            "Net PnL ($)": round(m["net_pnl"], 2),
            "Daily Sharpe": round(m["sharpe"], 2),
            "DSR (%)": round(m.get("dsr", 0.0), 1),
            "Max DD (%)": round(m["max_drawdown_pct"], 2),
            "Verdict": v,
            "Factsheet": str(strat_dir / "factsheet.html"),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Print executive leaderboard
    display_cols = ["Strategy Name", "Trades", "Win Rate (%)", "Profit Factor", "Net PnL ($)", "Daily Sharpe", "DSR (%)", "Max DD (%)", "Verdict"]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 70 + "\n")
    print(f"  All tear sheets generated in: {out_base}/<strategy>/factsheet.html\n")

    return summary_df


if __name__ == "__main__":
    run_all_benchmarks()
