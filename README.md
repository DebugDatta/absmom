# absmom — Indian Absolute Momentum Backtester

GitHub: https://github.com/DebugDatta/absmom

Live Demo: https://signalsync.streamlit.app/

## Overview

absmom is a Python-based backtesting engine for Gary Antonacci's Absolute Momentum strategy, applied to Indian market indices and ETFs. It tests risk-on/risk-off switching between equities and cash, generating performance charts, heatmaps, and multi-asset comparison reports.

The strategy follows Antonacci's Absolute Momentum research: when a risk-on asset shows positive momentum over a lookback period, remain invested; otherwise rotate to a risk-off asset (cash). The approach aims to reduce drawdowns while preserving upside participation.

Paper Reference:

Antonacci, G. (2014). *Absolute Momentum: A Simple Rule-Based Strategy and Universal Trend-Following Overlay.*

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2244633

## Features

* Dual Interface — CLI pipeline (`main.py`) for batch reports and an interactive Streamlit dashboard (`app.py`) for live parameter exploration.
* Multi-Asset — Backtests Nifty 50, Bank Nifty, Nifty Next 50, GoldBees, and AGG.
* Absolute Momentum — Risk-on/risk-off switching based on N-month price momentum.
* 60/40 Portfolio Variant — Tests a momentum-enhanced 60/40 equity/cash blend against the traditional static allocation.
* Comprehensive Metrics — CAGR, Sharpe Ratio, Sortino Ratio, Max Drawdown, Alpha, Beta, Win Rate.
* Charts & Visualizations — Equity curves, drawdown profiles, 36-month rolling Sharpe, monthly returns heatmap, lookback optimization, and multi-asset Sharpe comparison.
* Lookback Optimization — Compares 2, 3, 4, 6, 8, 10, 12, and 18-month formation periods.

## The Strategy: Absolute Momentum

Absolute Momentum measures an asset's own performance over a lookback window.

The core rule:

1. Compute Momentum = (Price_today / Price_{today - N months}) - 1
2. If Momentum > 0 → Risk-On: stay invested in the equity/index.
3. If Momentum ≤ 0 → Risk-Off: switch to cash (or a low-risk asset).

The signal is evaluated monthly and the position is taken with a one-month lag (shifted) to avoid look-ahead bias. Transaction costs are deducted on each switch based on portfolio turnover.

This is distinct from Relative Momentum, which ranks multiple assets against each other. Absolute Momentum answers:

"Is this asset in an uptrend based on its own historical performance?"

## Glossary

| Term              | Definition                                                                                                                              |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Absolute Momentum | A strategy that switches between risk-on and risk-off assets based on whether an asset's own return over a lookback period is positive. |
| Dual Momentum     | Gary Antonacci's framework combining Absolute Momentum with Relative Momentum. This project focuses on the absolute component.          |
| Risk-On           | The actively held investment when momentum is positive, typically an equity index such as Nifty 50.                                     |
| Risk-Off          | The safe-haven allocation held when momentum is negative, represented here by cash.                                                     |
| CAGR              | Compound Annual Growth Rate.                                                                                                            |
| Sharpe Ratio      | Risk-adjusted return metric based on excess return divided by volatility.                                                               |
| Sortino Ratio     | Similar to Sharpe Ratio but only penalizes downside volatility.                                                                         |
| Max Drawdown      | Largest peak-to-trough decline in the equity curve.                                                                                     |
| Alpha             | Excess return beyond what benchmark exposure would predict.                                                                             |
| Beta              | Sensitivity of the strategy to market movements.                                                                                        |
| Win Rate          | Percentage of months producing positive returns.                                                                                        |
| Lookback Period   | Number of months used to compute the momentum signal.                                                                                   |
| Transaction Cost  | Cost deducted whenever the portfolio switches regimes.                                                                                  |
| Turnover          | Proportion of the portfolio changed during a rebalance period.                                                                          |
| Equity Curve      | Growth of a hypothetical ₹100 investment over time.                                                                                     |
| Drawdown          | Distance below the previous equity peak.                                                                                                |
| Rolling Sharpe    | Sharpe ratio calculated over a rolling time window.                                                                                     |
| Monthly Heatmap   | Year-by-month visualization of monthly returns.                                                                                         |
| 60/40 Portfolio   | 60% equity and 40% cash allocation.                                                                                                     |
| Regime            | Current market state: Risk-On or Risk-Off.                                                                                              |
| Signal            | Binary output of the momentum rule.                                                                                                     |
| Position          | Actual holding chosen after applying signal lag.                                                                                        |
| Backtesting       | Simulating historical strategy performance using past market data.                                                                      |
| Resampling (ME)   | Converting daily prices to month-end observations using `resample('ME').last()`.                                                        |
| Streamlit         | Framework used to build the interactive dashboard.                                                                                      |
| yfinance          | Library used to download historical market data from Yahoo Finance.                                                                     |
| Kaleido           | Library used to export Plotly charts as image files.                                                                                    |

## Installation

```bash
git clone https://github.com/DebugDatta/absmom.git
cd absmom
pip install -r requirements.txt
```

## Requirements

* Python 3.14+
* yfinance
* pandas
* numpy
* plotly
* kaleido
* streamlit

## How to Run Locally

### CLI Pipeline (Batch Report Generation)

```bash
python main.py
```

This runs the full backtest pipeline end-to-end with default parameters:

* Start date: 2008-01-01
* Lookback: 12 months
* Transaction cost: 0.10%
* Risk-free rate: 6%

All charts and CSV reports are saved to the `report_artifacts/` directory.

### Interactive Dashboard

```bash
streamlit run app.py
```

Opens a browser-based GUI at `http://localhost:8501` where you can:

* Select any asset pair (risk-on and risk-off)
* Adjust lookback period (1–24 months)
* Modify transaction cost and risk-free rate
* Set custom start dates
* Navigate live charts and metrics

## Output Location

All CLI-generated files are saved to:

```text
report_artifacts/
├── equity_curve.png
├── drawdown_chart.png
├── rolling_sharpe.png
├── monthly_heatmap.png
├── lookback_optimization.csv
├── lookback_comparison.png
├── multi_asset_results.csv
├── sharpe_comparison.png
├── 6040_portfolio.csv
├── 6040_equity_curve.png
├── performance_summary.csv
├── corr_matrix_no_mom.csv
├── corr_matrix_with_mom.csv
└── trade_log_nifty.csv
```

## Parameters

| Parameter       | Default          | Description                                                   |
| --------------- | ---------------- | ------------------------------------------------------------- |
| start_date      | 2008-01-01       | Beginning of the backtest period                              |
| lookback_months | 12               | Number of months for momentum calculation                     |
| tx_cost         | 0.10%            | Transaction cost deducted per trade                           |
| risk_free_rate  | 6.0%             | Annual risk-free rate used in Sharpe and Sortino calculations |
| risk_on_asset   | Nifty 50 (^NSEI) | Primary equity index/ETF                                      |
| risk_off_asset  | Cash (0% return) | Safe-haven asset when momentum is negative                    |

## Paper Reference

Antonacci, G. (2014). *Absolute Momentum: A Simple Rule-Based Strategy and Universal Trend-Following Overlay.*

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2244633

The local copy of the paper is included as `ssrn-2244633.pdf` in this repository.

## Repository

https://github.com/DebugDatta/absmom

## Live App

https://absmom.streamlit.app/
