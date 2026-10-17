import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import black_scholes as bs_model
import volatility as vol_model
import the_greeks as gr_model


@st.cache_data(ttl=3600)
def treasury_yield_curve():
    curve = []
    for symbol, maturity_years in [("^IRX", 0.25), ("^FVX", 5.0), ("^TNX", 10.0)]:
        history = yf.Ticker(symbol).history(period="5d")
        closes = history["Close"].dropna() if not history.empty else pd.Series(dtype=float)
        if not closes.empty:
            curve.append((maturity_years, float(closes.iloc[-1]) / 100))
    if not curve:
        raise ValueError("Treasury yield data is unavailable")
    return curve


@st.cache_data(ttl=30)
def latest_underlying_quote(symbol):
    ticker = yf.Ticker(symbol)
    try:
        intraday = ticker.history(period="1d", interval="1m")
        closes = intraday["Close"].dropna() if not intraday.empty else pd.Series(dtype=float)
        if not closes.empty and np.isfinite(float(closes.iloc[-1])) and float(closes.iloc[-1]) > 0:
            quote_time = pd.to_datetime(closes.index[-1], utc=True).strftime("%Y-%m-%d %H:%M UTC")
            return float(closes.iloc[-1]), "latest 1-minute bar", quote_time
    except Exception:
        pass

    try:
        fast_price = float(ticker.fast_info["last_price"])
        if np.isfinite(fast_price) and fast_price > 0:
            return fast_price, "Yahoo Finance fast quote", "timestamp unavailable"
    except Exception:
        pass

    daily = ticker.history(period="5d")
    closes = daily["Close"].dropna() if not daily.empty else pd.Series(dtype=float)
    if not closes.empty and np.isfinite(float(closes.iloc[-1])) and float(closes.iloc[-1]) > 0:
        quote_time = pd.to_datetime(closes.index[-1], utc=True).strftime("%Y-%m-%d %H:%M UTC")
        return float(closes.iloc[-1]), "daily close fallback", quote_time
    raise ValueError(f"No usable price quote found for {symbol}")


st.set_page_config(page_title="Black-Scholes Calculator", layout="wide")
st.title("The Black-Scholes Calculator")
st.write("Pick an option from the market and let us calculate its value and its Greeks")
st.info("Research-only estimate. Dividend dates and amounts are user supplied; execution prices, costs, taxes, and market risk may differ in practice.")
#sidebar
st.sidebar.header("1. Choose action")

ticker_symbol = st.sidebar.text_input("Please enter your action's ticker( AAPL, TSLA etc.)", value = "AAPL")

try:
    stock = yf.Ticker(ticker_symbol)
    u, underlying_quote_source, underlying_quote_time = latest_underlying_quote(ticker_symbol)
    exp_dates = stock.options
    if not exp_dates:
        st.error("Ticker doesn't have any valid options")
        st.stop()
        
    selected_expiry = st.sidebar.selectbox("Date of options expiry", exp_dates)
    exp_dt = datetime.strptime(selected_expiry, "%Y-%m-%d")
    days_to_exp = (exp_dt - datetime.now()).days
    t = max(days_to_exp / 365.0, 0.001)
    
    option_type = st.sidebar.selectbox("Type of option", ["call", "put"])
    chain = stock.option_chain(selected_expiry)
    active_chain = chain.calls if option_type == "call" else chain.puts
    strikes_list = active_chain['strike'].tolist()
    default_idx = int(np.argmin(np.abs(np.array(strikes_list) - u)))
    s = st.sidebar.selectbox("Strike price (s)", strikes_list, index=default_idx)
    market_row = active_chain[active_chain['strike'] == s].iloc[0]
    market_price = pd.to_numeric(market_row.get('lastPrice', np.nan), errors="coerce")
    bid_price = pd.to_numeric(market_row.get('bid', np.nan), errors="coerce")
    ask_price = pd.to_numeric(market_row.get('ask', np.nan), errors="coerce")
    last_trade_timestamp = pd.to_datetime(market_row.get('lastTradeDate', pd.NaT), utc=True, errors="coerce")

    implied_vol = float(market_row['impliedVolatility'])
    if np.isfinite(implied_vol) and implied_vol > 0:
        v = implied_vol
        volatility_source = "implied"
    else:
        v = vol_model.volatility(ticker_symbol, period="1y")
        volatility_source = "historical"

except Exception as e:
    st.sidebar.error(f"Greška: {e}")
    st.stop()

try:
    treasury_curve = treasury_yield_curve()
    r = float(np.interp(t, [point[0] for point in treasury_curve], [point[1] for point in treasury_curve]))
    rate_source = "Treasury yield curve estimate"
except Exception:
    r = st.sidebar.number_input("Fallback risk-free rate (%)", min_value=-5.0, max_value=25.0, value=5.0, step=0.1) / 100
    rate_source = "manual fallback"

st.sidebar.caption(f"Risk-free rate: {r:.2%} ({rate_source})")
st.sidebar.subheader("Expected cash dividends")
dividend_schedule = st.sidebar.data_editor(
    pd.DataFrame({
        "ex_date": pd.Series(dtype="datetime64[ns]"),
        "cash_per_share": pd.Series(dtype="float64"),
    }),
    num_rows="dynamic",
    column_config={
        "ex_date": st.column_config.DateColumn("Ex-dividend date"),
        "cash_per_share": st.column_config.NumberColumn("Cash per share", min_value=0.0, step=0.01, format="$%.2f"),
    },
    hide_index=True,
    key="expected_dividend_schedule",
)
dividend_events = []
for _, row in dividend_schedule.iterrows():
    ex_date = pd.to_datetime(row["ex_date"], errors="coerce")
    cash_per_share = pd.to_numeric(row["cash_per_share"], errors="coerce")
    if pd.notna(ex_date) and pd.notna(cash_per_share) and cash_per_share > 0:
        if datetime.now().date() <= ex_date.date() <= exp_dt.date():
            time_to_dividend = max((ex_date.to_pydatetime() - datetime.now()).total_seconds() / (365.0 * 24 * 60 * 60), 0)
            dividend_events.append((ex_date.date().isoformat(), float(cash_per_share), time_to_dividend))

dividend_pv = sum(amount * np.exp(-r * time_to_dividend) for _, amount, time_to_dividend in dividend_events)
adjusted_spot = u - dividend_pv
if adjusted_spot <= 0:
    st.sidebar.error("The present value of entered dividends must be less than the current share price.")
    st.stop()
st.sidebar.caption(f"Included dividends: {len(dividend_events)}; present value ${dividend_pv:.2f} per share.")

trade_side = st.sidebar.selectbox("Trade side", ["Buy (long)", "Sell (short)"])
contracts = st.sidebar.number_input("Number of contracts", min_value=1, step=1, value=1)
contract_multiplier = st.sidebar.number_input("Shares per contract", min_value=1, step=1, value=100)
fee_per_contract = st.sidebar.number_input("Fee per contract per side ($)", min_value=0.0, step=0.05, value=0.65)
tax_rate = st.sidebar.number_input("Tax rate (%)", min_value=0.0, max_value=100.0, step=0.1, value=15.0) / 100
has_valid_quote = (
    pd.notna(bid_price) and pd.notna(ask_price)
    and np.isfinite(bid_price) and np.isfinite(ask_price)
    and 0 < bid_price <= ask_price
)
market_mid = (bid_price + ask_price) / 2 if has_valid_quote else market_price
quote_spread = ask_price - bid_price if has_valid_quote else np.nan
entry_price = (ask_price if trade_side == "Buy (long)" else bid_price) if has_valid_quote else market_price
if not np.isfinite(market_mid) or market_mid <= 0:
    st.error("No usable option bid/ask or last-trade price is available for this contract.")
    st.stop()
bs_value = bs_model.black_scholes(adjusted_spot, s, t, r, v, option_type)
mc_value, mc_standard_error = bs_model.monte_carlo_european_option(
    adjusted_spot, s, t, r, v, option_type, n_paths=200000, random_seed=42,
    return_standard_error=True,
)
position_bs_value = bs_value * contracts * contract_multiplier
position_mc_value = mc_value * contracts * contract_multiplier
position_market_value = market_mid * contracts * contract_multiplier
trade_direction = 1 if trade_side == "Buy (long)" else -1
gross_model_pnl = trade_direction * (bs_value - market_mid) * contracts * contract_multiplier
execution_costs = quote_spread * contracts * contract_multiplier + 2 * fee_per_contract * contracts if has_valid_quote else np.nan
profit_without_tax = gross_model_pnl - execution_costs if has_valid_quote else np.nan
profit_with_tax = profit_without_tax - max(profit_without_tax, 0) * tax_rate if has_valid_quote else np.nan
col1, col2, col3, col4 = st.columns(4)
col1.metric("Black-Scholes Price", f"${bs_value:.2f}")
col2.metric("Monte Carlo Comparison", f"${mc_value:.2f}", f"95% error +/- ${1.96 * mc_standard_error:.3f}")
col3.metric("Option Quote Mid" if has_valid_quote else "Last Trade (fallback)", f"${market_mid:.2f}")
col4.metric("BS - MC", f"${bs_value - mc_value:.2f}")

st.caption(f"Underlying quote: {underlying_quote_source}, {underlying_quote_time}.")
if has_valid_quote:
    if pd.notna(last_trade_timestamp):
        last_trade_age = max((pd.Timestamp.now(tz="UTC") - last_trade_timestamp).total_seconds() / 60, 0)
        st.caption(f"Option bid/ask: ${bid_price:.2f} / ${ask_price:.2f}; last trade was {last_trade_age:.0f} minutes ago.")
    else:
        st.caption(f"Option bid/ask: ${bid_price:.2f} / ${ask_price:.2f}; last-trade time unavailable.")
else:
    st.warning("No valid two-sided bid/ask quote. The last trade is shown for reference; executable P/L and spread costs are unavailable.")

st.subheader("Position Value")
pos_col1, pos_col2, pos_col3 = st.columns(3)
pos_col1.metric("Contracts", f"{int(contracts)}")
pos_col2.metric("BS Position Value", f"${position_bs_value:.2f}")
pos_col3.metric("Market Mid Position Value", f"${position_market_value:.2f}")

if volatility_source == "implied":
    st.caption(f"Volatility source: implied volatility from the selected option chain ({v:.4f}).")
else:
    st.caption(f"Volatility source: fallback historical volatility ({v:.4f}) because implied volatility was unavailable or invalid.")
st.caption(f"Risk-free rate source: {rate_source}; maturity-interpolated Treasury yields are an approximation.")
st.caption("Discrete dividends use the present-value spot adjustment; enter expected ex-dividend dates and cash amounts before expiry.")

st.subheader("Profit")
profit_col1, profit_col2, profit_col3 = st.columns(3)
profit_col1.metric("Estimated P/L after costs, pre-tax", f"${profit_without_tax:.2f}" if np.isfinite(profit_without_tax) else "N/A")
profit_col2.metric("Estimated P/L after tax", f"${profit_with_tax:.2f}" if np.isfinite(profit_with_tax) else "N/A")
profit_col3.metric("Estimated round-trip costs", f"${execution_costs:.2f}" if np.isfinite(execution_costs) else "N/A")
if has_valid_quote:
    st.caption(f"Estimated entry premium: ${entry_price:.2f}/share; costs include the current full bid/ask spread and two fee sides.")

st.divider()

#here we show the greeks
st.subheader("The Greeks for chosen option")
g_col1, g_col2, g_col3, g_col4, g_col5 = st.columns(5)
delta_val = gr_model.delta(adjusted_spot, s, t, r, v, option_type)
gamma_val = gr_model.gamma(adjusted_spot, s, t, r, v)
theta_val = gr_model.theta(adjusted_spot, s, t, r, v, option_type)
vega_val = gr_model.vega(adjusted_spot, s, t, r, v)
rho_val = gr_model.rho(adjusted_spot, s, t, r, v, option_type)

g_col1.metric("Delta ", f"{delta_val:.4f}")
g_col2.metric("Gamma ", f"{gamma_val:.4f}")
g_col3.metric("Theta ", f"{theta_val:.4f}")
g_col4.metric("Vega ", f"{vega_val:.4f}")
g_col5.metric("Rho ", f"{rho_val:.4f}")

call_chain = chain.calls[chain.calls['strike'] == s]
put_chain = chain.puts[chain.puts['strike'] == s]
call_market = call_chain['lastPrice'].iloc[0] if not call_chain.empty else np.nan
put_market = put_chain['lastPrice'].iloc[0] if not put_chain.empty else np.nan
parity_rhs = adjusted_spot - s * np.exp(-r * t)
parity_gap = (call_market - put_market) - parity_rhs
st.subheader("Call-Put Parity Check")
parity_col1, parity_col2, parity_col3 = st.columns(3)
parity_col1.metric("Parity RHS", f"${parity_rhs:.2f}")
parity_col2.metric("Market C - P", f"${call_market - put_market:.2f}")
parity_col3.metric("Parity Gap", f"${parity_gap:.2f}")

if np.isfinite(parity_gap) and abs(parity_gap) < 0.05:
    st.success("Call-put parity is consistent within a small tolerance.")
elif np.isfinite(parity_gap):
    st.warning("Call-put parity deviates from the no-arbitrage relation.")
else:
    st.info("Call-put parity check is unavailable for this strike because the counter-side option is missing.")

export_df = pd.DataFrame([
    {
        "ticker": ticker_symbol,
        "option_type": option_type,
        "contracts": int(contracts),
        "spot_price": u,
        "underlying_quote_source": underlying_quote_source,
        "underlying_quote_time": underlying_quote_time,
        "strike": s,
        "time_to_expiry_years": t,
        "risk_free_rate": r,
        "volatility": v,
        "dividend_events": json.dumps([{"ex_date": date, "cash_per_share": amount} for date, amount, _ in dividend_events]),
        "dividend_present_value_per_share": dividend_pv,
        "black_scholes_price_per_contract": bs_value,
        "monte_carlo_comparison_price_per_contract": mc_value,
        "monte_carlo_standard_error": mc_standard_error,
        "market_last_price_per_share": market_price,
        "market_mid_per_share": market_mid,
        "option_bid_per_share": bid_price,
        "option_ask_per_share": ask_price,
        "last_trade_timestamp": str(last_trade_timestamp) if pd.notna(last_trade_timestamp) else "",
        "estimated_entry_price_per_share": entry_price,
        "trade_side": trade_side,
        "contract_multiplier": int(contract_multiplier),
        "estimated_execution_costs": execution_costs,
        "estimated_pnl_before_tax": profit_without_tax,
        "estimated_pnl_after_tax": profit_with_tax,
        "bs_position_value": position_bs_value,
        "market_position_value": position_market_value,
        "price_difference": bs_value - market_mid,
        "delta": delta_val,
        "gamma": gamma_val,
        "theta": theta_val,
        "vega": vega_val,
        "rho": rho_val,
        "parity_rhs": parity_rhs,
        "call_market": call_market,
        "put_market": put_market,
        "parity_gap": parity_gap,
    }
])
st.download_button(
    label="Download CSV",
    data=export_df.to_csv(index=False),
    file_name=f"{ticker_symbol}_{option_type}_option_metrics.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Sensitivity of Greeks")
spot_range = np.linspace(u * 0.5, u * 1.5, 100)
greeks_df = pd.DataFrame({"Spot Price": spot_range})
greeks_df["Delta"] = [gr_model.delta(max(price - dividend_pv, 0.01), s, t, r, v, option_type) for price in spot_range]
greeks_df["Gamma"] = [gr_model.gamma(max(price - dividend_pv, 0.01), s, t, r, v) for price in spot_range]
greeks_df["Theta"] = [gr_model.theta(max(price - dividend_pv, 0.01), s, t, r, v, option_type) for price in spot_range]
greeks_df["Vega"] = [gr_model.vega(max(price - dividend_pv, 0.01), s, t, r, v) for price in spot_range]
greeks_df["Rho"] = [gr_model.rho(max(price - dividend_pv, 0.01), s, t, r, v, option_type) for price in spot_range]
st.line_chart(greeks_df.set_index("Spot Price"), use_container_width=True, height=400)

st.divider()
st.subheader("Black-Scholes Heatmap")
spot_grid = np.linspace(max(u * 0.7, 0.01), u * 1.3, 40)
vol_grid = np.linspace(max(v * 0.5, 0.05), max(v * 1.5, 0.5), 40)
heatmap = np.empty((len(vol_grid), len(spot_grid)))
for i, vol in enumerate(vol_grid):
    for j, spot in enumerate(spot_grid):
        heatmap[i, j] = bs_model.black_scholes(max(spot - dividend_pv, 0.01), s, t, r, vol, option_type)

fig, ax = plt.subplots(figsize=(9, 5))
img = ax.imshow(heatmap, aspect='auto', cmap='viridis', origin='lower')
ax.set_xticks(np.linspace(0, len(spot_grid) - 1, 5))
ax.set_xticklabels([f"{x:.2f}" for x in np.linspace(spot_grid.min(), spot_grid.max(), 5)])
ax.set_yticks(np.linspace(0, len(vol_grid) - 1, 5))
ax.set_yticklabels([f"{y:.2f}" for y in np.linspace(vol_grid.min(), vol_grid.max(), 5)])
ax.set_xlabel("Spot Price")
ax.set_ylabel("Volatility")
ax.set_title(f"{option_type.capitalize()} Option Value")
fig.colorbar(img, ax=ax, label="Option Price")
plt.tight_layout()
st.pyplot(fig)

st.divider()
st.subheader("Volatility Smile / Skew")
chain_df = stock.option_chain(selected_expiry)
if option_type == "call":
    chain_df = chain_df.calls[["strike", "lastPrice", "impliedVolatility"]].dropna()
else:
    chain_df = chain_df.puts[["strike", "lastPrice", "impliedVolatility"]].dropna()

if not chain_df.empty:
    chain_df = chain_df.sort_values("strike")
    st.line_chart(chain_df.set_index("strike")["impliedVolatility"], use_container_width=True, height=400)
    st.caption("This plot shows the implied volatility across strikes for the chosen expiry, revealing smile/skew effects.")
else:
    st.info("No option chain data available for implied volatility smile/skew.")

st.divider()
#payoff diagram
st.subheader("Payoff Diagram")
strike_range = np.linspace(s * 0.5, s * 1.5, 100)
payoff = np.maximum(0, strike_range - s) if option_type == "call" else np.maximum(0, s - strike_range)
payoff_df = pd.DataFrame({"Strike": strike_range, "Payoff": payoff})
st.line_chart(payoff_df.set_index("Strike"), use_container_width=True, height=400, width=800)
