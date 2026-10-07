"""Vectorized Monte Carlo pricing with variance reduction."""

from dataclasses import dataclass
import math

import numpy as np

from .models import Option


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    simulations: int

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """Approximate two-sided 95% normal confidence interval."""
        margin = 1.96 * self.standard_error
        return self.price - margin, self.price + margin


class MonteCarloPricer:
    """Risk-neutral terminal-price Monte Carlo pricer.

    Antithetic pairs are always used. Control variates use the discounted
    terminal underlying, whose risk-neutral expectation is the dividend-adjusted
    spot price.
    """

    def price(
        self,
        option: Option,
        simulations: int = 100_000,
        seed: int | None = 42,
        antithetic: bool = True,
        control_variate: bool = True,
        return_result: bool = False,
    ) -> float | MonteCarloResult:
        """Estimate a European option value using vectorized NumPy operations."""
        if option.exercise_style != "european":
            raise ValueError("Monte Carlo pricer supports European options only")
        if simulations < 2:
            raise ValueError("simulations must be at least 2")
        rng = np.random.default_rng(seed)
        pair_count = (simulations + 1) // 2 if antithetic else simulations
        z = rng.standard_normal(pair_count)
        shocks = np.concatenate((z, -z))[:simulations] if antithetic else z
        terminal = option.spot * np.exp(
            (option.rate - option.dividend_yield - 0.5 * option.volatility**2) * option.maturity
            + option.volatility * math.sqrt(option.maturity) * shocks
        )
        if antithetic:
            payoff_count = min(pair_count, simulations // 2)
            terminal_pairs = terminal[:payoff_count * 2].reshape(payoff_count, 2)
            payoffs = self._payoff(option, terminal_pairs).mean(axis=1)
            controls = terminal_pairs.mean(axis=1) * math.exp(-option.rate * option.maturity)
        else:
            payoffs = self._payoff(option, terminal)
            controls = terminal * math.exp(-option.rate * option.maturity)
        expected_control = option.spot * math.exp(-option.dividend_yield * option.maturity)
        discounted_payoffs = payoffs * math.exp(-option.rate * option.maturity)
        if control_variate and len(discounted_payoffs) > 1:
            covariance = np.cov(discounted_payoffs, controls, ddof=1)[0, 1]
            variance = np.var(controls, ddof=1)
            beta = covariance / variance if variance > 0 else 0.0
            adjusted = discounted_payoffs - beta * (controls - expected_control)
        else:
            adjusted = discounted_payoffs
        result = MonteCarloResult(float(np.mean(adjusted)), float(np.std(adjusted, ddof=1) / math.sqrt(len(adjusted))), len(adjusted))
        return result if return_result else result.price

    @staticmethod
    def _payoff(option: Option, spot: np.ndarray) -> np.ndarray:
        if option.option_type == "call":
            return np.maximum(spot - option.strike, 0.0)
        return np.maximum(option.strike - spot, 0.0)

    def finite_difference_greeks(self, option: Option, simulations: int = 100_000, seed: int | None = 42, spot_bump: float = 0.01, volatility_bump: float = 0.0001) -> dict[str, float]:
        """Estimate Delta, Gamma, and Vega using common random numbers."""
        up = option.with_market_data(spot=option.spot + spot_bump)
        down = option.with_market_data(spot=option.spot - spot_bump)
        p_up = self.price(up, simulations, seed)
        p = self.price(option, simulations, seed)
        p_down = self.price(down, simulations, seed)
        v_up = self.price(option.with_market_data(volatility=option.volatility + volatility_bump), simulations, seed)
        v_down = self.price(option.with_market_data(volatility=max(0.0, option.volatility - volatility_bump)), simulations, seed)
        return {"delta": (p_up - p_down) / (2 * spot_bump), "gamma": (p_up - 2 * p + p_down) / spot_bump**2, "vega": (v_up - v_down) / (2 * volatility_bump)}
