# Option Pricing Engine

This project provides a typed, class-based implementation of:

- Black-Scholes-Merton European pricing and Delta, Gamma, Vega, Theta, and Rho.
- Vectorized NumPy Monte Carlo pricing with antithetic variates and a discounted-terminal-spot control variate.
- Cox-Ross-Rubinstein binomial pricing for European and American options, including early exercise.
- Finite-difference Delta, Gamma, and Vega for the numerical models.
- Pandas benchmarking/convergence tables and Matplotlib/Seaborn plots.
- Implied-volatility inversion, volatility mispricing signals, and a transparent
  delta-hedged backtesting primitive.

## Install and test

```bash
python -m pip install -e ".[test]"
pytest
```

## Example

```python
from option_pricing import Option, BlackScholesPricer, MonteCarloPricer, BinomialPricer
from option_pricing.benchmark import benchmark_models

option = Option(100, 100, 1, 0.05, 0.20, option_type="call")
print(BlackScholesPricer().price(option))
print(MonteCarloPricer().price(option, simulations=100_000))
print(BinomialPricer().price(option, steps=200))
print(benchmark_models(option))
```

## Alpha research extension

The pricing engine can be used as the valuation layer for volatility
relative-value research:

```python
import numpy as np
from option_pricing import (
    BlackScholesPricer,
    Option,
    delta_hedged_backtest,
    volatility_signal,
)

option = Option(100, 100, 1, 0.05, 0.20)
market_price = BlackScholesPricer().price(option.with_market_data(volatility=0.25))
signal = volatility_signal(option, market_price, forecast_volatility=0.20)
print(signal)  # market IV is above forecast: short-option signal

path = np.array([100, 101, 99, 102, 103], dtype=float)
history = delta_hedged_backtest(option, path, volatility=0.20, position=-1)
print(history[["step", "spot", "pnl"]])
```

This is a research building block, not proof of an investable strategy. A
production backtest should add historical option quotes, bid-ask spreads,
slippage, liquidity filters, contract rolls, borrow costs, and out-of-sample
evaluation while avoiding look-ahead bias.

## How to assess accuracy

- Use the Black-Scholes value as the benchmark for European options.
- For Monte Carlo, request `return_result=True` and inspect both
  `standard_error` and `confidence_interval_95`. Increase simulations until the
  confidence interval is narrower than your required tolerance.
- Use the same random seed when comparing successive Monte Carlo runs, and
  report the seed for reproducibility.
- For CRR, increase `steps` and check that the price stabilizes. Compare the
  converged European tree price with Black-Scholes; American options do not
  have a general Black-Scholes benchmark.
- Validate put-call parity and the American-option lower bound
  (American value must be at least the corresponding European value).

For example:

```python
result = MonteCarloPricer().price(option, simulations=200_000, return_result=True)
print(result.price, result.standard_error, result.confidence_interval_95)
```
