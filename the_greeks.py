import numpy as np
from scipy.stats import norm


def _validate_inputs(u, s, t, r, v, option_type=None, dividend_yield=0.0):
    if u <= 0:
        raise ValueError("Spot price must be positive.")
    if s <= 0:
        raise ValueError("Strike price must be positive.")
    if t <= 0:
        raise ValueError("Time to expiry must be positive.")
    if v <= 0:
        raise ValueError("Volatility must be positive.")
    if dividend_yield < 0:
        raise ValueError("Dividend yield cannot be negative.")
    if option_type is not None:
        option_type = str(option_type).strip().lower()
        if option_type not in {"call", "put"}:
            raise ValueError("Parameter must be 'put' or 'call'.")
        return option_type
    return None


def delta(u, s, t, r, v, option_type, dividend_yield=0.0):  # sensitivity that depends on the change of the stock value
    option_type = _validate_inputs(u, s, t, r, v, option_type, dividend_yield)
    d1 = (np.log(u / s) + (r - dividend_yield + v ** 2 / 2) * t) / (v * np.sqrt(t))
    if option_type == "call":
        return np.exp(-dividend_yield * t) * norm.cdf(d1)
    return -np.exp(-dividend_yield * t) * norm.cdf(-d1)


def gamma(u, s, t, r, v, dividend_yield=0.0):  # the speed of change of delta
    _validate_inputs(u, s, t, r, v, dividend_yield=dividend_yield)
    d1 = (np.log(u / s) + (r - dividend_yield + v ** 2 / 2) * t) / (v * np.sqrt(t))
    return np.exp(-dividend_yield * t) * norm.pdf(d1) / (u * v * np.sqrt(t))


def theta(u, s, t, r, v, option_type, dividend_yield=0.0):  # time decay of the value of an option
    option_type = _validate_inputs(u, s, t, r, v, option_type, dividend_yield)
    d1 = (np.log(u / s) + (r - dividend_yield + v ** 2 / 2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    if option_type == "call":
        return -(u * np.exp(-dividend_yield * t) * norm.pdf(d1) * v) / (2 * np.sqrt(t)) - r * s * np.exp(-r * t) * norm.cdf(d2) + dividend_yield * u * np.exp(-dividend_yield * t) * norm.cdf(d1)
    return -(u * np.exp(-dividend_yield * t) * norm.pdf(d1) * v) / (2 * np.sqrt(t)) + r * s * np.exp(-r * t) * norm.cdf(-d2) - dividend_yield * u * np.exp(-dividend_yield * t) * norm.cdf(-d1)


def vega(u, s, t, r, v, dividend_yield=0.0):  # describes sensitivity to volatility
    _validate_inputs(u, s, t, r, v, dividend_yield=dividend_yield)
    d1 = (np.log(u / s) + (r - dividend_yield + v ** 2 / 2) * t) / (v * np.sqrt(t))
    return u * np.exp(-dividend_yield * t) * np.sqrt(t) * norm.pdf(d1)


def rho(u, s, t, r, v, option_type, dividend_yield=0.0):
    option_type = _validate_inputs(u, s, t, r, v, option_type, dividend_yield)
    d1 = (np.log(u / s) + (r - dividend_yield + v ** 2 / 2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    if option_type == "call":
        return s * t * np.exp(-r * t) * norm.cdf(d2)
    return -s * t * np.exp(-r * t) * norm.cdf(-d2)
