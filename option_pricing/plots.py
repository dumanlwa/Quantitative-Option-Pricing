"""Matplotlib/Seaborn visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd


def plot_convergence(benchmark: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """Plot model prices against simulation/tree resolution."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, group in benchmark[benchmark["model"] != "Black-Scholes"].groupby("model"):
        sns.lineplot(data=group, x="parameter", y="price", marker="o", label=model, ax=ax)
    bs = benchmark.loc[benchmark["model"] == "Black-Scholes", "price"].iloc[0]
    ax.axhline(bs, color="black", linestyle="--", label="Black-Scholes")
    ax.set(xlabel="Simulations / tree steps", ylabel="Option price", title="Pricing convergence")
    return fig, ax


def plot_sensitivities(price_greeks: pd.DataFrame, volatility_data: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
    """Plot option price and Greeks versus spot, plus price versus volatility."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    axes = axes.flatten()
    axes[0].plot(price_greeks["spot"], price_greeks["price"])
    axes[0].set(title="Price vs spot", xlabel="Spot", ylabel="Price")
    axes[1].plot(volatility_data["volatility"], volatility_data["price"])
    axes[1].set(title="Price vs volatility", xlabel="Volatility", ylabel="Price")
    for axis, key in zip(axes[2:], ("delta", "gamma", "vega", "theta", "rho")):
        axis.plot(price_greeks["spot"], price_greeks[key])
        axis.set(title=key.title(), xlabel="Spot", ylabel=key.title())
    for axis in axes[7:]:
        axis.set_visible(False)
    fig.tight_layout()
    return fig, axes
