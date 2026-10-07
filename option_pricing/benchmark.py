"""Benchmark, convergence, and sensitivity helpers."""

import time
from collections.abc import Iterable

import pandas as pd

from .analytical import BlackScholesPricer
from .binomial import BinomialPricer
from .models import Option
from .monte_carlo import MonteCarloPricer


def benchmark_models(
    option: Option,
    simulations: Iterable[int] = (5_000, 20_000, 100_000),
    steps: Iterable[int] = (25, 50, 100, 200),
    seed: int | None = 42,
) -> pd.DataFrame:
    """Compare model values, runtime in milliseconds, and BS absolute error."""
    bs = BlackScholesPricer().price(option)
    rows: list[dict[str, float | int | str]] = [{"model": "Black-Scholes", "parameter": 0, "price": bs, "execution_time_ms": 0.0, "error": 0.0}]
    mc = MonteCarloPricer()
    for count in simulations:
        start = time.perf_counter()
        price = mc.price(option, simulations=count, seed=seed)
        rows.append({"model": "Monte Carlo", "parameter": count, "price": price, "execution_time_ms": (time.perf_counter() - start) * 1000, "error": abs(price - bs)})
    tree = BinomialPricer()
    for count in steps:
        start = time.perf_counter()
        price = tree.price(option, steps=count)
        rows.append({"model": "Binomial CRR", "parameter": count, "price": price, "execution_time_ms": (time.perf_counter() - start) * 1000, "error": abs(price - bs)})
    return pd.DataFrame(rows, columns=["model", "parameter", "price", "execution_time_ms", "error"])


def sensitivity_data(option: Option, spots: Iterable[float], volatilities: Iterable[float], steps: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return price and analytical/numerical Greek curves over spot and volatility."""
    bs = BlackScholesPricer()
    tree = BinomialPricer()
    rows: list[dict[str, float]] = []
    for spot in spots:
        current = option.with_market_data(spot=float(spot))
        row = {"spot": float(spot), "price": bs.price(current), **bs.greeks(current)}
        row.update({f"tree_{key}": value for key, value in tree.finite_difference_greeks(current, steps).items()})
        rows.append(row)
    vol_rows = [{"volatility": float(vol), "price": bs.price(option.with_market_data(volatility=float(vol)))} for vol in volatilities]
    return pd.DataFrame(rows), pd.DataFrame(vol_rows)
