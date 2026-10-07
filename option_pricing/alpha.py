"""Implied-volatility and volatility relative-value signals."""

from dataclasses import dataclass
from typing import Literal

from scipy.optimize import brentq

from .analytical import BlackScholesPricer
from .models import Option


@dataclass(frozen=True)
class VolatilitySignal:
    """A transparent option-market mispricing signal."""

    market_implied_volatility: float
    forecast_volatility: float
    volatility_spread: float
    direction: Literal["long_option", "short_option", "flat"]


def implied_volatility(
    option: Option,
    market_price: float,
    lower_bound: float = 1e-8,
    upper_bound: float = 5.0,
) -> float:
    """Invert Black-Scholes to obtain volatility from a European market price."""
    if option.exercise_style != "european":
        raise ValueError("implied volatility requires a European option")
    if market_price < 0:
        raise ValueError("market_price cannot be negative")
    pricer = BlackScholesPricer()
    intrinsic = pricer._intrinsic(option)
    if market_price < intrinsic:
        raise ValueError("market_price is below intrinsic value")

    def objective(volatility: float) -> float:
        return pricer.price(option.with_market_data(volatility=volatility)) - market_price

    low_value = objective(lower_bound)
    high_value = objective(upper_bound)
    if low_value * high_value > 0:
        raise ValueError("market_price is outside the supported Black-Scholes volatility range")
    return float(brentq(objective, lower_bound, upper_bound, xtol=1e-10))


def volatility_signal(
    option: Option,
    market_price: float,
    forecast_volatility: float,
    threshold: float = 0.01,
) -> VolatilitySignal:
    """Compare market implied volatility with a forecast and classify direction."""
    if forecast_volatility <= 0 or threshold < 0:
        raise ValueError("forecast_volatility must be positive and threshold non-negative")
    market_iv = implied_volatility(option, market_price)
    spread = market_iv - forecast_volatility
    if spread > threshold:
        direction: Literal["long_option", "short_option", "flat"] = "short_option"
    elif spread < -threshold:
        direction = "long_option"
    else:
        direction = "flat"
    return VolatilitySignal(market_iv, forecast_volatility, spread, direction)
