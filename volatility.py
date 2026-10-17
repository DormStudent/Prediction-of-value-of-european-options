import numpy as np
import yfinance as yf


def volatility(symbol, period="1y"):
    """Calculate annualized historical volatility for a stock ticker."""
    stock = yf.Ticker(symbol)
    history = stock.history(period=period)

    if history.empty or "Close" not in history.columns:
        raise ValueError(f"No price history available for symbol '{symbol}'.")

    history = history[["Close"]].dropna()
    if len(history) < 2:
        raise ValueError(f"Not enough daily price data for '{symbol}' to calculate volatility.")

    history["Log_Returns"] = np.log(history["Close"] / history["Close"].shift(1)).dropna()
    log_returns = history["Log_Returns"].dropna()
    if log_returns.empty:
        raise ValueError(f"No valid daily log returns for '{symbol}'.")

    daily_volatility = log_returns.std(ddof=1)
    annual_volatility = float(daily_volatility * np.sqrt(252))

    if not np.isfinite(annual_volatility):
        raise ValueError(f"Calculated volatility is not finite for '{symbol}'.")

    return annual_volatility
