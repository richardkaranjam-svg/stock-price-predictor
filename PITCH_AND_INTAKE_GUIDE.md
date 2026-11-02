# Quantitative Strategy Audit & Trader Intake Guide

A field guide for engaging discretionary and algorithmic traders, auditing their strategies, and delivering institutional performance tear sheets.

---

## 1. The 30-Second Elevator Pitch

> *"Hey [Name], I build quantitative validation engines for hedge funds and prop desks. Most retail backtests fool traders because they ignore data snooping, execution slippage, and regime shifts.*
>
> *If you send me an export of your recent trade log (even just 50–100 trades), I'll run it through our institutional audit engine and send you back a confidential, single-page institutional tear sheet with your Deflated Sharpe Ratio, Long vs Short edge asymmetry, and holdout regime breakdown—100% free."*

---

## 2. What to Ask For (Client Intake Requirements)

Tell the trader they only need to export a simple CSV or Excel file from their platform:

| Platform | How They Export It | What We Need |
| :--- | :--- | :--- |
| **TradingView** | Strategy Tester &rarr; *List of Trades* &rarr; *Download CSV* | `Date/Time`, `Type`, `Profit` |
| **MetaTrader 4/5** | History tab &rarr; Right click &rarr; *Report* / *Save as Detailed Statement* | `Time`, `Type`, `Profit` |
| **Interactive Brokers** | Client Portal &rarr; Reports &rarr; *Trade Confirmations / Flex Query* | `DateTime`, `Buy/Sell`, `Realized PnL` |
| **NinjaTrader** | Trade Performance window &rarr; Right click &rarr; *Export Trades* | `Entry Time`, `Exit Time`, `Profit` |
| **Custom Spreadsheet** | Any Excel / CSV with their trade log | Any columns with Entry Date, Direction, and PnL |

> [!NOTE]
> Our parser automatically recognizes currency symbols (`$`), commas (`,`), and negative parenthetical accounting notation (e.g. `(350.00)`), so they don't have to clean the file.

---

## 3. How to Run the Audit (Single Command)

Save their trade log into the project (e.g., `client_trades.csv`) and run:

```powershell
.venv\Scripts\python.exe audit_strategy.py --csv client_trades.csv --output reports/client_name
```

This immediately generates:
- `reports/client_name/factsheet.html` (Confidential standalone dark-mode tear sheet)
- `reports/client_name/equity_curve.png` (High-res equity curve & drawdown plot)
- `reports/client_name/report.json` (Structured quantitative metrics)
- `reports/client_name/trades.csv` (Standardized trade records)

---

## 4. What to Hand Back & How to Explain It

Email or send them `factsheet.html`. You can open it directly in any browser. Walk them through the three key metrics that set you apart from basic retail backtesters:

### A. The Deflated Sharpe Ratio (DSR)
- **What it means**: Most traders test 50 different variations of an indicator until one looks good. This is called *data snooping bias* or *overfitting*.
- **Institutional standard**: Developed by Marcos López de Prado (AQR / Lawrence Berkeley National Laboratory), DSR calculates the probability that their Sharpe Ratio is real after penalizing for the number of backtest trials.
- **How to explain it**: *"A standard Sharpe of 1.8 looks great, but if you tested 50 iterations, your Deflated Sharpe drops to 0.4, meaning there is a 60% chance the edge is random noise."*

### B. Directional Edge Asymmetry (Long vs. Short)
- **What it means**: Many strategies perform well on Longs simply because the S&P 500 has drifted upward for a decade, but lose money on Shorts.
- **How to explain it**: *"Our engine separates your Long Profit Factor from your Short Profit Factor. If your Shorts are dragging down overall expectancy, cutting Shorts immediately boosts your net return."*

### C. Out-of-Sample Regime Splits
- **What it means**: Breaks their performance across different market regimes (e.g. 2020 COVID volatility, 2022 bear market, 2024–2026 bull trend).
- **How to explain it**: *"Your strategy survived the 2022 bear market with an unbroken equity curve, proving the edge is non-correlated to passive market beta."*

---

## 5. Ready Benchmark Strategies Available to Compare

If a client wants to see what our reports look like before sharing their own data, share one of the ready benchmark reports located in `reports/benchmarks/`:

1. **Trend Breakout (Donchian)**:
   - File: `reports/benchmarks/trend_breakout_donchian/factsheet.html`
   - Highlights: Profit Factor 2.45, +$16,947 Net PnL, 1.56% Max Drawdown.
2. **Mean Reversion (Bollinger Bounce)**:
   - File: `reports/benchmarks/mean_reversion_bollinger/factsheet.html`
3. **WorldQuant Multi-Alpha Portfolio**:
   - File: `reports/benchmarks/worldquant_multi-alpha/factsheet.html`
