import math

import pytest

import numpy as np

from option_pricing import (
    BinomialPricer,
    BlackScholesPricer,
    MonteCarloPricer,
    Option,
    delta_hedged_backtest,
    implied_volatility,
    volatility_signal,
)


@pytest.fixture
def european_call() -> Option:
    return Option(spot=100, strike=100, maturity=1, rate=0.05, volatility=0.2)


def test_black_scholes_known_call(european_call: Option) -> None:
    assert BlackScholesPricer().price(european_call) == pytest.approx(10.4506, abs=1e-3)


def test_black_scholes_put_call_parity(european_call: Option) -> None:
    put = european_call.with_market_data(option_type="put")
    call_price = BlackScholesPricer().price(european_call)
    put_price = BlackScholesPricer().price(put)
    assert call_price - put_price == pytest.approx(100 - 100 * math.exp(-0.05), abs=1e-10)


def test_binomial_converges_to_black_scholes(european_call: Option) -> None:
    bs = BlackScholesPricer().price(european_call)
    tree = BinomialPricer().price(european_call, steps=500)
    assert tree == pytest.approx(bs, abs=0.03)


def test_american_put_is_at_least_european(european_call: Option) -> None:
    put_eu = european_call.with_market_data(option_type="put")
    put_am = put_eu.with_market_data(exercise_style="american")
    assert BinomialPricer().price(put_am, steps=200) >= BinomialPricer().price(put_eu, steps=200)


def test_monte_carlo_is_close(european_call: Option) -> None:
    result = MonteCarloPricer().price(european_call, simulations=200_000, seed=7, return_result=True)
    benchmark = BlackScholesPricer().price(european_call)
    assert result.price == pytest.approx(benchmark, abs=0.08)
    assert abs(result.price - benchmark) <= 1.96 * result.standard_error


def test_monte_carlo_control_variate_handles_dividends() -> None:
    option = Option(100, 100, 1, 0.05, 0.2, dividend_yield=0.03)
    result = MonteCarloPricer().price(option, simulations=200_000, seed=7, return_result=True)
    assert result.price == pytest.approx(BlackScholesPricer().price(option), abs=0.08)


def test_implied_volatility_recovers_input(european_call: Option) -> None:
    price = BlackScholesPricer().price(european_call)
    assert implied_volatility(european_call, price) == pytest.approx(0.2, abs=1e-7)


def test_volatility_signal_classifies_expensive_option(european_call: Option) -> None:
    market_price = BlackScholesPricer().price(european_call.with_market_data(volatility=0.3))
    signal = volatility_signal(european_call, market_price, forecast_volatility=0.2)
    assert signal.direction == "short_option"
    assert signal.volatility_spread == pytest.approx(0.1, abs=1e-6)


def test_delta_hedged_backtest_returns_path(european_call: Option) -> None:
    path = np.linspace(100, 105, 11)
    result = delta_hedged_backtest(european_call, path, volatility=0.2)
    assert len(result) == len(path)
    assert result["pnl"].iloc[0] == pytest.approx(0.0)
