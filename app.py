import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Indian Absolute Momentum", layout="wide")
st.title("📈 Advanced Indian Absolute Momentum")

st.sidebar.header("Strategy Parameters")

assets = {
    "NIFTY 50": "^NSEI",
    "NIFTY Next 50": "^NSMIDCP",
    "Bank Nifty": "^NSEBANK",
    "Gold ETF (GOLDBEES)": "GOLDBEES.NS",
    "Long Bond ETF (AGG)": "AGG"
}

risk_on_name = st.sidebar.selectbox("Risk-On Asset", list(assets.keys()), index=0)
risk_off_name = st.sidebar.selectbox("Risk-Off Asset", ["Cash (0% Return)"] + list(assets.keys()), index=0)

risk_on_ticker = assets[risk_on_name]
risk_off_ticker = assets.get(risk_off_name, None)

lookback_months = st.sidebar.slider("Primary Lookback (Months)", 1, 24, 12)
tx_cost = st.sidebar.number_input("Transaction Cost per Trade (%)", value=0.10, step=0.05) / 100.0
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (Annual %)", value=6.0, step=0.5) / 100.0
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2008-01-01"))

@st.cache_data
def fetch_data(ticker, start):
    df = yf.download(ticker, start=start, progress=False)
    if df.empty: return pd.Series()
    close = df['Close'][ticker] if isinstance(df.columns, pd.MultiIndex) and ticker in df['Close'].columns else df['Close']
    return close.resample('M').last()

with st.spinner("Fetching market data..."):
    risk_on_prices = fetch_data(risk_on_ticker, start_date)
    risk_off_prices = fetch_data(risk_off_ticker, start_date) if risk_off_ticker else pd.Series(1.0, index=risk_on_prices.index)

def run_backtest(lookback, risk_on, risk_off, cost):
    df = pd.DataFrame({'RiskOn': risk_on, 'RiskOff': risk_off}).dropna()
    if df.empty: return pd.DataFrame()
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
    if len(returns) == 0: return 0, 0, 0, 0, 0, 0
    cagr = ((1 + returns).prod() ** (12 / len(returns))) - 1
    vol = returns.std() * np.sqrt(12)
    sharpe = (cagr - risk_free_rate) / vol if vol > 0 else 0
    downside = returns[returns < 0]
    sortino = (cagr - risk_free_rate) / (downside.std() * np.sqrt(12)) if len(downside) > 0 else 0
    alpha, beta = 0, 1
    if benchmark_returns is not None:
        cov = np.cov(returns, benchmark_returns)[0][1]
        var = np.var(benchmark_returns)
        beta = cov / var if var > 0 else 1
        alpha = cagr - (risk_free_rate + beta * (((1 + benchmark_returns).prod() ** (12 / len(benchmark_returns))) - 1 - risk_free_rate))
    return cagr, vol, sharpe, sortino, alpha, beta

df = run_backtest(lookback_months, risk_on_prices, risk_off_prices, tx_cost)
if df.empty:
    st.error("No data available for backtesting. Check ticker symbols and date range.")
    st.stop()

strat_cagr, strat_vol, strat_sharpe, strat_sortino, alpha, beta = calc_metrics(df['Strat_Ret'], df['Ret_ON'])
bnh_cagr, bnh_vol, bnh_sharpe, bnh_sortino, _, _ = calc_metrics(df['Ret_ON'])
strat_mdd = df['Strat_DD'].min()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Regime", "🟢 ON" if df['Signal'].iloc[-1] == 1 else "🔴 OFF")
col2.metric("Strategy CAGR", f"{strat_cagr*100:.2f}%")
col3.metric("Sharpe Ratio", f"{strat_sharpe:.2f}")
col4.metric("Alpha (Annual)", f"{alpha*100:.2f}%")
col5.metric("Beta", f"{beta:.2f}")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Equity Curve", "Drawdown", "Rolling Risk", "Monthly Heatmap",
    "Parameter Optimization", "Performance Summary", "Multi-Asset", "60/40 Portfolio"
])

with tab1:
    st.subheader("Equity Curve (Net of Costs)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Strat_Eq'], name='Strategy', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BnH_Eq'], name='Benchmark', line=dict(color='gray', dash='dash')))
    fig.update_layout(yaxis_type="log", yaxis_title="Growth of ₹100", template="plotly_white")
    st.plotly_chart(fig, width='stretch')

with tab2:
    st.subheader("Drawdown Profile")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=df.index, y=df['Strat_DD'], fill='tozeroy', name='Strategy DD', line=dict(color='red')))
    fig_dd.add_trace(go.Scatter(x=df.index, y=df['BnH_DD'], mode='lines', name='BnH DD', line=dict(color='gray')))
    fig_dd.update_layout(title="Drawdown Profile", yaxis_tickformat='.1%', template="plotly_white")
    st.plotly_chart(fig_dd, width='stretch')

with tab3:
    st.subheader("36-Month Rolling Sharpe Ratio")
    rolling_strat = df['Strat_Ret'].rolling(36)
    rolling_sharpe = (rolling_strat.mean() * 12 - risk_free_rate) / (rolling_strat.std() * np.sqrt(12))
    rolling_bnh = df['Ret_ON'].rolling(36)
    rolling_bnh_sharpe = (rolling_bnh.mean() * 12 - risk_free_rate) / (rolling_bnh.std() * np.sqrt(12))
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=df.index, y=rolling_sharpe, name='Strategy Sharpe', line=dict(color='green')))
    fig_roll.add_trace(go.Scatter(x=df.index, y=rolling_bnh_sharpe, name='Benchmark Sharpe', line=dict(color='orange')))
    fig_roll.update_layout(template="plotly_white", yaxis_title="Sharpe Ratio")
    st.plotly_chart(fig_roll, width='stretch')

with tab4:
    st.subheader("Monthly Returns Heatmap")
    df_heat = df.copy()
    df_heat['Year'] = df_heat.index.year
    df_heat['Month'] = df_heat.index.strftime('%b')
    heatmap_data = df_heat.pivot_table(index='Year', columns='Month', values='Strat_Ret', aggfunc='sum')
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    heatmap_data = heatmap_data.reindex(columns=[m for m in months_order if m in heatmap_data.columns])
    fig_heat = px.imshow(heatmap_data * 100, text_auto=".1f", aspect="auto",
                         color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                         labels=dict(color="Return %"))
    st.plotly_chart(fig_heat, width='stretch')

with tab5:
    st.subheader("Lookback Parameter Optimization")
    lookbacks = [2, 3, 4, 6, 8, 10, 12, 18]
    opt_results = []
    with st.spinner("Running optimization..."):
        for lb in lookbacks:
            test_df = run_backtest(lb, risk_on_prices, risk_off_prices, tx_cost)
            c, v, s, sort, a, b = calc_metrics(test_df['Strat_Ret'], test_df['Ret_ON'])
            mdd = test_df['Strat_DD'].min()
            opt_results.append({
                "Lookback (M)": lb,
                "CAGR (%)": round(c*100, 2),
                "Sharpe": round(s, 2),
                "Sortino": round(sort, 2),
                "Max DD (%)": round(mdd*100, 2),
                "Alpha (%)": round(a*100, 2)
            })
    opt_df = pd.DataFrame(opt_results).set_index("Lookback (M)").round(2)
    st.dataframe(opt_df.style.highlight_max(subset=['CAGR (%)', 'Sharpe', 'Alpha (%)'], color='lightgreen')
                             .highlight_max(subset=['Max DD (%)'], color='lightcoral'), width='stretch')
    fig_lb = go.Figure()
    fig_lb.add_trace(go.Bar(x=[r["Lookback (M)"] for r in opt_results], y=[r["Sharpe"] for r in opt_results],
                          marker_color='steelblue'))
    fig_lb.update_layout(title="Sharpe Ratio by Lookback Period",
 xaxis_title="Lookback (Months)", yaxis_title="Sharpe Ratio",
                         template="plotly_white")
    st.plotly_chart(fig_lb, width='stretch')

with tab6:
    st.subheader("Performance Summary")
    strat_cagr, strat_vol, strat_sharpe, strat_sortino, alpha, beta = calc_metrics(df['Strat_Ret'], df['Ret_ON'])
    bnh_cagr, bnh_vol, bnh_sharpe, bnh_sortino, _, _ = calc_metrics(df['Ret_ON'])
    strat_mdd = df['Strat_DD'].min()
    bnh_mdd = df['BnH_DD'].min()
    strat_wr = len(df['Strat_Ret'][df['Strat_Ret'] > 0]) / len(df['Strat_Ret'])
    bnh_wr = len(df['Ret_ON'][df['Ret_ON'] > 0]) / len(df['Ret_ON'])
    summary_data = {
        "Metric": ["CAGR", "Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Win Rate", "Alpha (Annual)", "Beta"],
        "Strategy": [
            f"{strat_cagr*100:.2f}%",
            f"{strat_vol*100:.2f}%",
            f"{strat_sharpe:.2f}",
            f"{strat_sortino:.2f}",
            f"{strat_mdd*100:.2f}%",
            f"{strat_wr*100:.1f}%",
            f"{alpha*100:.2f}%",
            f"{beta:.2f}"
        ],
        "Buy& Hold": [
            f"{bnh_cagr*100:.2f}%",
            f"{bnh_vol*100:.2f}%",
            f"{bnh_sharpe:.2f}",
            f"{bnh_sortino:.2f}",
            f"{bnh_mdd*100:.2f}%",
            f"{bnh_wr*100:.1f}%",
            "—",
            "1.00"
        ]
    }
    st.dataframe(pd.DataFrame(summary_data).set_index("Metric"), width='stretch')

with tab7:
    st.subheader("Multi-Asset Comparison")
    ma_results = []
    app_assets = {
        "Nifty 50": "^NSEI",
        "Bank Nifty": "^NSEBANK",
        "GoldBees": "GOLDBEES.NS",
        "AGG (Bond)": "AGG"
    }
    with st.spinner("Loading multi-asset data..."):
        for name, ticker in app_assets.items():
            series = fetch_data(ticker, start_date)
            if series.empty: continue
            df_a = run_backtest(12, series, risk_off_prices, tx_cost)
            if df_a.empty: continue
            sc, sv, ss, sso, sa, sb = calc_metrics(df_a['Strat_Ret'], df_a['Ret_ON'])
            bc, bv, bs, bso, ba, bb = calc_metrics(df_a['Ret_ON'])
            sw = len(df_a['Strat_Ret'][df_a['Strat_Ret'] > 0]) / len(df_a['Strat_Ret'])
            bw = len(df_a['Ret_ON'][df_a['Ret_ON'] > 0]) / len(df_a['Ret_ON'])
            ma_results.append({
                "Asset": name,
                "Strat CAGR": f"{sc*100:.2f}%",
                "BnH CAGR": f"{bc*100:.2f}%",
                "Strat Sharpe": f"{ss:.2f}",
                "BnH Sharpe": f"{bs:.2f}",
                "Strat Max DD": f"{df_a['Strat_DD'].min()*100:.2f}%",
                "BnH Max DD": f"{df_a['BnH_DD'].min()*100:.2f}%",
                "Strat Win Rate": f"{sw*100:.1f}%",
                "BnH Win Rate": f"{bw*100:.1f}%"
            })
    if ma_results:
        st.dataframe(pd.DataFrame(ma_results).set_index("Asset"), width='stretch')
        if len(ma_results) > 1:
            fig_ma = go.Figure()
            assets = [r["Asset"] for r in ma_results]
            fig_ma.add_trace(go.Bar(x=assets, y=[float(r["Strat Sharpe"]) for r in ma_results],
                                   name='Strategy', marker_color='steelblue'))
            fig_ma.add_trace(go.Bar(x=assets, y=[float(r["BnH Sharpe"]) for r in ma_results],
                                   name='Buy & Hold', marker_color='gray'))
            fig_ma.update_layout(title="Sharpe Ratio: Strategy vs Buy & Hold",
                                 yaxis_title="Sharpe Ratio", template="plotly_white")
            st.plotly_chart(fig_ma, width='stretch')

with tab8:
    st.subheader("60/40 Portfolio Analysis")
    df_6040 = pd.DataFrame({
        'Equity': risk_on_prices,
        'Cash': risk_off_prices
    }).dropna()
    if not df_6040.empty:
        df_6040['Ret_Equity'] = df_6040['Equity'].pct_change()
        df_6040['Ret_Cash'] = 0.0
        df_6040['Trad_Ret'] = (0.6 * df_6040['Ret_Equity']) + (0.4 * df_6040['Ret_Cash'])
        df_6040['Mom'] = df_6040['Equity'] / df_6040['Equity'].shift(12) - 1
        df_6040['Signal'] = np.where(df_6040['Mom'] > 0, 1, 0)
        df_6040['Position'] = df_6040['Signal'].shift(1)
        df_6040['Turnover'] = df_6040['Position'].diff().abs().fillna(0)
        df_6040['Mom_Ret'] = np.where(
            df_6040['Position'] == 1,
            (0.6 * df_6040['Ret_Equity']) + (0.4 * df_6040['Ret_Cash']),
            df_6040['Ret_Cash']
        )
        df_6040['Mom_Ret'] = df_6040['Mom_Ret'] - (df_6040['Turnover'] * tx_cost)
        df_6040 = df_6040.dropna()
        if not df_6040.empty:
            tc, tv, ts, tso, _, _ = calc_metrics(df_6040['Trad_Ret'])
            mc, mv, ms, mso, _, _ = calc_metrics(df_6040['Mom_Ret'])
            df_6040['Trad_Eq'] = (1 + df_6040['Trad_Ret']).cumprod() * 100
            df_6040['Mom_Eq'] = (1 + df_6040['Mom_Ret']).cumprod() * 100
            trad_mdd = ((df_6040['Trad_Eq'] - df_6040['Trad_Eq'].cummax()) / df_6040['Trad_Eq'].cummax()).min()
            mom_mdd = ((df_6040['Mom_Eq'] - df_6040['Mom_Eq'].cummax()) / df_6040['Mom_Eq'].cummax()).min()
            tw = len(df_6040['Trad_Ret'][df_6040['Trad_Ret'] > 0]) / len(df_6040['Trad_Ret'])
            mw = len(df_6040['Mom_Ret'][df_6040['Mom_Ret'] > 0]) / len(df_6040['Mom_Ret'])
            port_data = {
                "Portfolio": ["Traditional 60/40", "Momentum 60/40"],
                "CAGR": [f"{tc*100:.2f}%", f"{mc*100:.2f}%"],
                "Sharpe": [f"{ts:.2f}", f"{ms:.2f}"],
                "Sortino": [f"{tso:.2f}", f"{mso:.2f}"],
                "Max DD": [f"{trad_mdd*100:.2f}%", f"{mom_mdd*100:.2f}%"],
                "Win Rate": [f"{tw*100:.1f}%", f"{mw*100:.1f}%"]
            }
            st.dataframe(pd.DataFrame(port_data).set_index("Portfolio"), width='stretch')
            fig_6040 = go.Figure()
            fig_6040.add_trace(go.Scatter(x=df_6040.index, y=df_6040['Mom_Eq'],
                                          name='Momentum 60/40', line=dict(color='blue')))
            fig_6040.add_trace(go.Scatter(x=df_6040.index, y=df_6040['Trad_Eq'],
                                          name='Traditional 60/40', line=dict(color='gray', dash='dash')))
            fig_6040.update_layout(title="60/40 Portfolio: Traditional vs Momentum Enhanced",
                                   yaxis_title="Growth of ₹100", template="plotly_white")
            st.plotly_chart(fig_6040, width='stretch')
