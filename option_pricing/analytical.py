"""Black-Scholes analytical pricing and Greeks."""

import math
from typing import TypedDict

from scipy.stats import norm

from .models import Option


class Greeks(TypedDict):
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class BlackScholesPricer:
    """Black-Scholes-Merton pricer for European vanilla options."""

    def price(self, option: Option) -> float:
        """Return the analytical European option value."""
        self._validate(option)
        if option.maturity == 0:
            return self._intrinsic(option)
        d1, d2 = self._d1_d2(option)
        discount = math.exp(-option.rate * option.maturity)
        carry_discount = math.exp(-option.dividend_yield * option.maturity)
        if option.option_type == "call":
            return option.spot * carry_discount * norm.cdf(d1) - option.strike * discount * norm.cdf(d2)
        return option.strike * discount * norm.cdf(-d2) - option.spot * carry_discount * norm.cdf(-d1)

    def greeks(self, option: Option) -> Greeks:
        """Return analytical Delta, Gamma, Vega, Theta, and Rho."""
        self._validate(option)
        if option.maturity == 0 or option.volatility == 0:
            raise ValueError("Greeks require positive maturity and volatility")
        d1, d2 = self._d1_d2(option)
        s, t, sigma = option.spot, option.maturity, option.volatility
        discount = math.exp(-option.rate * t)
        carry_discount = math.exp(-option.dividend_yield * t)
        pdf = norm.pdf(d1)
        gamma = carry_discount * pdf / (s * sigma * math.sqrt(t))
        vega = s * carry_discount * pdf * math.sqrt(t)
        if option.option_type == "call":
            delta = carry_discount * norm.cdf(d1)
            theta = (
                -s * carry_discount * pdf * sigma / (2 * math.sqrt(t))
                - option.rate * option.strike * discount * norm.cdf(d2)
                + option.dividend_yield * s * carry_discount * norm.cdf(d1)
            )
            rho = option.strike * t * discount * norm.cdf(d2)
        else:
            delta = carry_discount * (norm.cdf(d1) - 1)
            theta = (
                -s * carry_discount * pdf * sigma / (2 * math.sqrt(t))
                + option.rate * option.strike * discount * norm.cdf(-d2)
                - option.dividend_yield * s * carry_discount * norm.cdf(-d1)
            )
            rho = -option.strike * t * discount * norm.cdf(-d2)
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

    def _d1_d2(self, option: Option) -> tuple[float, float]:
        t_sqrt = math.sqrt(option.maturity)
        d1 = (
            math.log(option.spot / option.strike)
            + (option.rate - option.dividend_yield + 0.5 * option.volatility**2) * option.maturity
        ) / (option.volatility * t_sqrt)
        return d1, d1 - option.volatility * t_sqrt

    @staticmethod
    def _intrinsic(option: Option) -> float:
        return max(option.spot - option.strike, 0.0) if option.option_type == "call" else max(option.strike - option.spot, 0.0)

    @staticmethod
    def _validate(option: Option) -> None:
        if option.exercise_style != "european":
            raise ValueError("Black-Scholes is only available for European options")
