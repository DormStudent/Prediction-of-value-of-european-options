import numpy as np
from scipy.stats import norm


def _validate_inputs(u, s, t, r, v, option_type, dividend_yield=0.0):
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
    if option_type is None:
        raise ValueError("Option type cannot be None.")
    option_type = str(option_type).strip().lower()
    if option_type not in {"call", "put"}:
        raise ValueError("Parameter must be 'put' or 'call'.")
    return option_type


def black_scholes(u, s, t, r, v, option_type, dividend_yield=0.0):
    option_type = _validate_inputs(u, s, t, r, v, option_type, dividend_yield)

    d1 = (np.log(u / s) + (r - dividend_yield + v ** 2 / 2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)

    if option_type == "call":
        return u * np.exp(-dividend_yield * t) * norm.cdf(d1) - s * np.exp(-r * t) * norm.cdf(d2)

    return s * np.exp(-r * t) * norm.cdf(-d2) - u * np.exp(-dividend_yield * t) * norm.cdf(-d1)


def monte_carlo_european_option(
    u,
    s,
    t,
    r,
    v,
    option_type,
    n_paths=100000,
    random_seed=None,
    dividend_yield=0.0,
    return_standard_error=False,
):
    option_type = _validate_inputs(u, s, t, r, v, option_type, dividend_yield)
    if n_paths <= 0:
        raise ValueError("Number of Monte Carlo paths must be positive.")

    rng = np.random.default_rng(random_seed)
    n_pairs = (n_paths + 1) // 2
    z = rng.standard_normal(n_pairs)
    drift = (r - dividend_yield - 0.5 * v ** 2) * t
    diffusion = v * np.sqrt(t) * z
    terminal_up = u * np.exp(drift + diffusion)
    terminal_down = u * np.exp(drift - diffusion)

    if option_type == "call":
        payoffs = (np.maximum(terminal_up - s, 0.0) + np.maximum(terminal_down - s, 0.0)) / 2
    else:
        payoffs = (np.maximum(s - terminal_up, 0.0) + np.maximum(s - terminal_down, 0.0)) / 2

    discount = np.exp(-r * t)
    discounted_payoffs = discount * payoffs
    discounted_stock = discount * (terminal_up + terminal_down) / 2
    expected_discounted_stock = u * np.exp(-dividend_yield * t)

    if n_pairs > 1:
        control_beta = np.cov(discounted_payoffs, discounted_stock, ddof=1)[0, 1] / np.var(discounted_stock, ddof=1)
        discounted_payoffs = discounted_payoffs - control_beta * (discounted_stock - expected_discounted_stock)

    price = float(discounted_payoffs.mean())
    if return_standard_error:
        standard_error = float(discounted_payoffs.std(ddof=1) / np.sqrt(n_pairs)) if n_pairs > 1 else 0.0
        return price, standard_error
    return price