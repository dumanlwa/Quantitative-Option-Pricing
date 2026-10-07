"""Cox-Ross-Rubinstein binomial tree pricing."""

import math

from .models import Option


class BinomialPricer:
    """CRR tree pricer for European and American vanilla options."""

    def price(self, option: Option, steps: int = 200) -> float:
        """Return the discounted CRR tree value, with early exercise if needed."""
        if steps < 1:
            raise ValueError("steps must be positive")
        if option.maturity == 0:
            return self._intrinsic(option)
        dt = option.maturity / steps
        up = math.exp(option.volatility * math.sqrt(dt))
        down = 1.0 / up
        growth = math.exp((option.rate - option.dividend_yield) * dt)
        probability = (growth - down) / (up - down)
        if not 0 <= probability <= 1:
            raise ValueError("invalid CRR probability; increase steps or check inputs")
        discount = math.exp(-option.rate * dt)
        values = [self._payoff(option, option.spot * up**j * down ** (steps - j)) for j in range(steps + 1)]
        for level in range(steps - 1, -1, -1):
            values = [
                discount * (probability * values[j + 1] + (1 - probability) * values[j])
                for j in range(level + 1)
            ]
            if option.exercise_style == "american":
                values = [
                    max(value, self._payoff(option, option.spot * up**j * down ** (level - j)))
                    for j, value in enumerate(values)
                ]
        return values[0]

    def finite_difference_greeks(self, option: Option, steps: int = 200, spot_bump: float = 0.01, volatility_bump: float = 0.0001) -> dict[str, float]:
        """Estimate Delta, Gamma, and Vega with central finite differences."""
        p_up = self.price(option.with_market_data(spot=option.spot + spot_bump), steps)
        p = self.price(option, steps)
        p_down = self.price(option.with_market_data(spot=option.spot - spot_bump), steps)
        v_up = self.price(option.with_market_data(volatility=option.volatility + volatility_bump), steps)
        v_down = self.price(option.with_market_data(volatility=max(0.0, option.volatility - volatility_bump)), steps)
        return {"delta": (p_up - p_down) / (2 * spot_bump), "gamma": (p_up - 2 * p + p_down) / spot_bump**2, "vega": (v_up - v_down) / (2 * volatility_bump)}

    @staticmethod
    def _payoff(option: Option, spot: float) -> float:
        return max(spot - option.strike, 0.0) if option.option_type == "call" else max(option.strike - spot, 0.0)

    @staticmethod
    def _intrinsic(option: Option) -> float:
        return BinomialPricer._payoff(option, option.spot)
