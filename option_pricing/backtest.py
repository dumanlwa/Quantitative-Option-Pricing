"""Small, explicit delta-hedged option backtesting primitive."""

import math

import numpy as np
import pandas as pd

from .analytical import BlackScholesPricer
from .models import Option


def delta_hedged_backtest(
    option: Option,
    spot_path: np.ndarray,
    volatility: float,
    position: int = 1,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Backtest a continuously re-marked, daily delta-hedged option position.

    ``position=1`` buys the option and ``position=-1`` sells it. The option is
    entered at the model value using ``volatility`` and settled at the final
    path point. This is a research primitive, not an execution simulator.
    """
    if option.exercise_style != "european":
        raise ValueError("the delta-hedged backtest supports European options only")
    if len(spot_path) < 2 or np.any(np.asarray(spot_path) <= 0):
        raise ValueError("spot_path must contain at least two positive prices")
    if volatility <= 0 or position not in (-1, 1) or transaction_cost_bps < 0:
        raise ValueError("invalid volatility, position, or transaction cost")

    spots = np.asarray(spot_path, dtype=float)
    pricer = BlackScholesPricer()
    dt = option.maturity / (len(spots) - 1)
    cash = -position * pricer.price(option.with_market_data(spot=spots[0], volatility=volatility))
    previous_delta = 0.0
    rows: list[dict[str, float]] = []

    for index, spot in enumerate(spots):
        remaining = max(option.maturity - index * dt, 0.0)
        marked = option.with_market_data(spot=float(spot), maturity=remaining, volatility=volatility)
        delta = 0.0 if remaining == 0 else pricer.greeks(marked)["delta"]
        target_shares = position * delta
        share_trade = target_shares - previous_delta
        cost = abs(share_trade) * spot * transaction_cost_bps / 10_000
        cash -= share_trade * spot + cost
        previous_delta = target_shares
        option_value = 0.0 if remaining == 0 else pricer.price(marked)
        portfolio_value = cash + target_shares * spot + position * option_value
        rows.append({"step": index, "spot": spot, "option_value": option_value, "delta": delta, "shares": target_shares, "cash": cash, "transaction_cost": cost, "portfolio_value": portfolio_value})

    result = pd.DataFrame(rows)
    result["pnl"] = result["portfolio_value"] - result["portfolio_value"].iloc[0]
    return result
