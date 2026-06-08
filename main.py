import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

output_dir = "report_artifacts"
os.makedirs(output_dir, exist_ok=True)
print(f"Directory '{output_dir}' ready. Starting data pipeline...")

start_date = "2008-01-01"
tx_cost = 0.0010
risk_free_rate = 0.06

assets = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Nifty Next 50": "^NSMIDCP",
    "GoldBees": "GOLDBEES.NS",
    "GiltBees": "AGG"
}

def fetch_data(ticker, start):
    df = yf.download(ticker, start=start, progress=False)
    if df.empty: return pd.Series(dtype='float64')
    close = df['Close'][ticker] if isinstance(df.columns, pd.MultiIndex) else df['Close']
    return close.resample('M').last()

def run_backtest(lookback, risk_on, risk_off, cost):
    df = pd.DataFrame({'RiskOn': risk_on, 'RiskOff': risk_off}).dropna()
    if df.empty: return df
    df['Ret_ON'] = df['RiskOn'].pct_change()
    df['Ret_OFF'] = df['RiskOff'].pct_change()
    df['Momentum'] = df['RiskOn'] / df['RiskOn'].shift(lookback) - 1
    df['Signal'] = np.where(df['Momentum'] > 0, 1, 0)
    df['Position'] = df['Signal'].shift(1)
    df['Turnover'] = df['Position'].diff().abs().fillna(0)
    df['Strat_Ret'] = np.where(df['Position'] == 1, df['Ret_ON'], df['Ret_OFF'])
    df['Strat_Ret'] = df['Strat_Ret'] - (df['Turnover'] * cost)
    df = df.dropna()
    df['BnH_Eq'] = (1 + df['Ret_ON']).cumprod() * 100
    df['Strat_Eq'] = (1 + df['Strat_Ret']).cumprod() * 100
    df['BnH_Peak'] = df['BnH_Eq'].cummax()
    df['BnH_DD'] = (df['BnH_Eq'] - df['BnH_Peak']) / df['BnH_Peak']
    df['Strat_Peak'] = df['Strat_Eq'].cummax()
    df['Strat_DD'] = (df['Strat_Eq'] - df['Strat_Peak']) / df['Strat_Peak']
    return df

def calc_metrics(returns, benchmark_returns=None):
    if len(returns) == 0: return 0, 0, 0, 0, 0, 0, 0
    cagr = ((1 + returns).prod() ** (12 / len(returns))) - 1
    vol = returns.std() * np.sqrt(12)
    sharpe = (cagr - risk_free_rate) / vol if vol > 0 else 0
    downside = returns[returns < 0]
    sortino = (cagr - risk_free_rate) / (downside.std() * np.sqrt(12)) if len(downside) > 0 else 0
    win_rate = len(returns[returns > 0]) / len(returns)
    alpha, beta = 0, 1
    if benchmark_returns is not None:
        cov = np.cov(returns, benchmark_returns)[0][1]
        var = np.var(benchmark_returns)
        beta = cov / var if var > 0 else 1
        b_cagr = ((1 + benchmark_returns).prod() ** (12 / len(benchmark_returns))) - 1
        alpha = cagr - (risk_free_rate + beta * (b_cagr - risk_free_rate))
    return cagr, vol, sharpe, sortino, alpha, beta, win_rate

print("Downloading market data...")
market_data = {name: fetch_data(ticker, start_date) for name, ticker in assets.items()}
cash_proxy = pd.Series(1.0, index=market_data["Nifty 50"].index)

print("Processing Section 7: Nifty 50 Baseline...")
df_nifty = run_backtest(12, market_data["Nifty 50"], cash_proxy, tx_cost)

df_nifty[['RiskOn', 'Momentum', 'Signal', 'Position', 'Strat_Ret', 'Strat_Eq']].to_csv(f"{output_dir}/trade_log_nifty.csv")

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=df_nifty.index, y=df_nifty['Strat_Eq'], name='Absolute Momentum', line=dict(color='blue')))
fig_eq.add_trace(go.Scatter(x=df_nifty.index, y=df_nifty['BnH_Eq'], name='Buy& Hold Nifty', line=dict(color='gray', dash='dash')))
fig_eq.update_layout(title="Equity Curve (Log Scale)", yaxis_type="log", yaxis_title="Growth of ₹100", template="plotly_white")
fig_eq.write_image(f"{output_dir}/equity_curve.png", width=1000, height=500)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=df_nifty.index, y=df_nifty['Strat_DD'], fill='tozeroy', name='Strategy DD', line=dict(color='red')))
fig_dd.add_trace(go.Scatter(x=df_nifty.index, y=df_nifty['BnH_DD'], mode='lines', name='BnH DD', line=dict(color='gray')))
fig_dd.update_layout(title="Drawdown Profile", yaxis_tickformat='.1%', template="plotly_white")
fig_dd.write_image(f"{output_dir}/drawdown_chart.png", width=1000, height=400)

rolling_strat = df_nifty['Strat_Ret'].rolling(36)
rolling_sharpe = (rolling_strat.mean() * 12 - risk_free_rate) / (rolling_strat.std() * np.sqrt(12))
rolling_bnh = df_nifty['Ret_ON'].rolling(36)
rolling_bnh_sharpe = (rolling_bnh.mean() * 12 - risk_free_rate) / (rolling_bnh.std() * np.sqrt(12))

fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(x=df_nifty.index, y=rolling_sharpe, name='Strategy Sharpe', line=dict(color='green')))
fig_roll.add_trace(go.Scatter(x=df_nifty.index, y=rolling_bnh_sharpe, name='Benchmark Sharpe', line=dict(color='orange')))
fig_roll.update_layout(title="36-Month Rolling Sharpe Ratio", template="plotly_white")
fig_roll.write_image(f"{output_dir}/rolling_sharpe.png", width=1000, height=400)

df_heat = df_nifty.copy()
df_heat['Year'] = df_heat.index.year
df_heat['Month'] = df_heat.index.strftime('%b')
heatmap_data = df_heat.pivot_table(index='Year', columns='Month', values='Strat_Ret', aggfunc='sum')
months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
heatmap_data = heatmap_data.reindex(columns=[m for m in months_order if m in heatmap_data.columns])

fig_heat = px.imshow(heatmap_data * 100, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
fig_heat.update_layout(title="Monthly Returns Heatmap (%)")
fig_heat.write_image(f"{output_dir}/monthly_heatmap.png", width=1000, height=600)

print("Processing Section 8: Lookback Optimization...")
lookbacks = [2, 3, 4, 6, 8, 10, 12, 18]
opt_results = []

for lb in lookbacks:
    test_df = run_backtest(lb, market_data["Nifty 50"], cash_proxy, tx_cost)
    c, v, s, sort, alpha, beta, wr = calc_metrics(test_df['Strat_Ret'], test_df['Ret_ON'])
    mdd = test_df['Strat_DD'].min()
    opt_results.append({
        "Lookback (Months)": lb,
        "CAGR": round(c, 4),
        "Sharpe": round(s, 2),
        "Sortino": round(sort, 2),
        "Max DD": round(mdd, 4),
        "Win Rate": round(wr, 2),
        "Alpha": round(alpha, 4),
        "Beta": round(beta, 2)
    })
pd.DataFrame(opt_results).to_csv(f"{output_dir}/lookback_optimization.csv", index=False)

fig_lb = go.Figure()
fig_lb.add_trace(go.Bar(x=[r["Lookback (Months)"] for r in opt_results], y=[r["Sharpe"] for r in opt_results],
 name='Sharpe Ratio', marker_color='steelblue'))
fig_lb.update_layout(title="Lookback Optimization: Sharpe Ratio by Formation Period",
                     xaxis_title="Lookback (Months)", yaxis_title="Sharpe Ratio",
                     template="plotly_white")
fig_lb.write_image(f"{output_dir}/lookback_comparison.png", width=900, height=450)

print("Processing Section 9: Multi-Asset Testing...")
ma_results = []
test_assets = ["Nifty 50", "Bank Nifty", "GoldBees", "GiltBees"]

for asset in test_assets:
    df_asset = run_backtest(12, market_data[asset], cash_proxy, tx_cost)
    if df_asset.empty:
        print(f"  Skipping {asset} — no data")
        continue
    strat_cagr, strat_vol, strat_sharpe, strat_sortino, strat_alpha, strat_beta, strat_wr = calc_metrics(df_asset['Strat_Ret'])
    bnh_cagr, bnh_vol, bnh_sharpe, bnh_sortino, bnh_alpha, bnh_beta, bnh_wr = calc_metrics(df_asset['Ret_ON'])
    ma_results.append({
        "Asset": asset,
        "BnH CAGR": round(bnh_cagr, 4),
        "Strat CAGR": round(strat_cagr, 4),
        "BnH Volatility": round(bnh_vol, 4),
        "Strat Volatility": round(strat_vol, 4),
        "BnH Sharpe": round(bnh_sharpe, 2),
        "Strat Sharpe": round(strat_sharpe, 2),
        "BnH Sortino": round(bnh_sortino, 2),
        "Strat Sortino": round(strat_sortino, 2),
        "BnH Max DD": round(df_asset['BnH_DD'].min(), 4),
        "Strat Max DD": round(df_asset['Strat_DD'].min(), 4),
        "BnH Win Rate": round(bnh_wr, 2),
        "Strat Win Rate": round(strat_wr, 2),
        "BnH Alpha": round(bnh_alpha, 4),
        "Strat Alpha": round(strat_alpha, 4),
        "BnH Beta": round(bnh_beta, 2),
        "Strat Beta": round(strat_beta, 2)
    })
pd.DataFrame(ma_results).to_csv(f"{output_dir}/multi_asset_results.csv", index=False)

fig_sharpe = go.Figure()
x = [r["Asset"] for r in ma_results]
fig_sharpe.add_trace(go.Bar(x=x, y=[r["Strat Sharpe"] for r in ma_results], name='Strategy Sharpe', marker_color='steelblue'))
fig_sharpe.add_trace(go.Bar(x=x, y=[r["BnH Sharpe"] for r in ma_results], name='Buy & Hold Sharpe', marker_color='gray'))
fig_sharpe.update_layout(title="Sharpe Ratio: Strategy vs Buy & Hold", yaxis_title="Sharpe Ratio", template="plotly_white")
fig_sharpe.write_image(f"{output_dir}/sharpe_comparison.png", width=900, height=450)

print("Processing Section 10: 60/40 Portfolio...")
df_6040 = pd.DataFrame({
    'Nifty': market_data["Nifty 50"],
    'Cash': cash_proxy
}).dropna()

df_6040['Ret_Nifty'] = df_6040['Nifty'].pct_change()
df_6040['Ret_Cash'] = 0.0
df_6040['Trad_6040_Ret'] = (0.6 * df_6040['Ret_Nifty']) + (0.4 * df_6040['Ret_Cash'])

df_6040['Nifty_Mom'] = df_6040['Nifty'] / df_6040['Nifty'].shift(12) - 1
df_6040['Signal'] = np.where(df_6040['Nifty_Mom'] > 0, 1, 0)
df_6040['Position'] = df_6040['Signal'].shift(1)
df_6040['Turnover'] = df_6040['Position'].diff().abs().fillna(0)

df_6040['Mom_6040_Ret'] = np.where(
    df_6040['Position'] == 1,
    (0.6 * df_6040['Ret_Nifty']) + (0.4 * df_6040['Ret_Cash']),
    df_6040['Ret_Cash']
)
df_6040['Mom_6040_Ret'] = df_6040['Mom_6040_Ret'] - (df_6040['Turnover'] * tx_cost)
df_6040 = df_6040.dropna()

trad_cagr, trad_vol, trad_sharpe, trad_sortino, _, _, trad_wr = calc_metrics(df_6040['Trad_6040_Ret'])
mom_cagr, mom_vol, mom_sharpe, mom_sortino, _, _, mom_wr = calc_metrics(df_6040['Mom_6040_Ret'])

df_6040['Trad_Eq'] = (1 + df_6040['Trad_6040_Ret']).cumprod() * 100
df_6040['Mom_Eq'] = (1 + df_6040['Mom_6040_Ret']).cumprod() * 100

trad_mdd = ((df_6040['Trad_Eq'] - df_6040['Trad_Eq'].cummax()) / df_6040['Trad_Eq'].cummax()).min()
mom_mdd = ((df_6040['Mom_Eq'] - df_6040['Mom_Eq'].cummax()) / df_6040['Mom_Eq'].cummax()).min()

port_results = pd.DataFrame([{
    "Portfolio": "Traditional 60/40",
    "CAGR": round(trad_cagr, 4),
    "Sharpe": round(trad_sharpe, 2),
    "Sortino": round(trad_sortino, 2),
    "Volatility": round(trad_vol, 4),
    "Max DD": round(trad_mdd, 4),
    "Win Rate": round(trad_wr, 2)
}, {
    "Portfolio": "Momentum 60/40",
    "CAGR": round(mom_cagr, 4),
    "Sharpe": round(mom_sharpe, 2),
    "Sortino": round(mom_sortino, 2),
    "Volatility": round(mom_vol, 4),
    "Max DD": round(mom_mdd, 4),
    "Win Rate": round(mom_wr, 2)
}])
port_results.to_csv(f"{output_dir}/6040_portfolio.csv", index=False)

fig_6040 = go.Figure()
fig_6040.add_trace(go.Scatter(x=df_6040.index, y=df_6040['Mom_Eq'], name='Momentum 60/40', line=dict(color='blue')))
fig_6040.add_trace(go.Scatter(x=df_6040.index, y=df_6040['Trad_Eq'], name='Traditional 60/40', line=dict(color='gray', dash='dash')))
fig_6040.update_layout(title="60/40 Portfolio: Traditional vs Momentum Enhanced", yaxis_title="Growth of ₹100", template="plotly_white")
fig_6040.write_image(f"{output_dir}/6040_equity_curve.png", width=1000, height=500)

print("Processing Section 11: Performance Summary...")
perf_rows = []
for name, ticker in assets.items():
    series = market_data[name]
    if series.empty: continue
    df_b = run_backtest(12, series, cash_proxy, tx_cost)
    if df_b.empty: continue
    strat_ret = df_b['Strat_Ret']
    bnh_ret = df_b['Ret_ON']
    sc, sv, ss, sso, sa, sb, sw = calc_metrics(strat_ret, bnh_ret)
    bc, bv, bs, bso, ba, bb, bw = calc_metrics(bnh_ret)
    for label, c, v, s, so, a, b, w in [
        ("Strategy", sc, sv, ss, sso, sa, sb, sw),
        ("Buy & Hold", bc, bv, bs, bso, ba, bb, bw)
    ]:
        perf_rows.append({
            "Asset": name,
            "Method": label,
            "CAGR": round(c, 4),
            "Volatility": round(v, 4),
            "Sharpe": round(s, 2),
            "Sortino": round(so, 2),
            "Max DD": round(df_b['Strat_DD'].min() if label == "Strategy" else df_b['BnH_DD'].min(), 4),
            "Win Rate": round(w, 2),
            "Alpha": round(a, 4),
            "Beta": round(b, 2)
        })
pd.DataFrame(perf_rows).to_csv(f"{output_dir}/performance_summary.csv", index=False)

print("Processing Section 12: Correlation Matrices...")
all_returns = pd.DataFrame()
all_strat_returns = pd.DataFrame()

for name, ticker in assets.items():
    series = market_data[name]
    if series.empty: continue
    df_b = run_backtest(12, series, cash_proxy, tx_cost)
    if df_b.empty: continue
    all_returns[name] = df_b['Ret_ON']
    all_strat_returns[name] = df_b['Strat_Ret']

all_returns = all_returns.dropna()
all_strat_returns = all_strat_returns.dropna()

all_returns.to_csv(f"{output_dir}/corr_matrix_no_mom.csv")
all_strat_returns.to_csv(f"{output_dir}/corr_matrix_with_mom.csv")

print(f"Pipeline complete! All files saved to {os.path.abspath(output_dir)}")
