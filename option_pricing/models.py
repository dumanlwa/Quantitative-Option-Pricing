"""Core market and option data models."""

from dataclasses import dataclass
from typing import Literal

OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


@dataclass(frozen=True)
class Option:
    """Plain-vanilla option contract and market inputs.

    Args:
        spot: Current underlying price.
        strike: Strike price.
        maturity: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualized volatility.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuously compounded dividend yield.
        exercise_style: ``"european"`` or ``"american"``.
    """

    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    option_type: OptionType = "call"
    dividend_yield: float = 0.0
    exercise_style: ExerciseStyle = "european"

    def __post_init__(self) -> None:
        if self.spot <= 0 or self.strike <= 0:
            raise ValueError("spot and strike must be positive")
        if self.maturity < 0:
            raise ValueError("maturity must be non-negative")
        if self.volatility < 0:
            raise ValueError("volatility cannot be negative")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.exercise_style not in ("european", "american"):
            raise ValueError("exercise_style must be 'european' or 'american'")

    def with_market_data(self, **changes: float | str) -> "Option":
        """Return a copy with selected market or contract fields changed."""
        values = {
            "spot": self.spot,
            "strike": self.strike,
            "maturity": self.maturity,
            "rate": self.rate,
            "volatility": self.volatility,
            "option_type": self.option_type,
            "dividend_yield": self.dividend_yield,
            "exercise_style": self.exercise_style,
        }
        values.update(changes)
        return Option(**values)
