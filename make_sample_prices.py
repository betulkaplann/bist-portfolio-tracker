"""
make_sample_prices.py - create SIMULATED data for offline testing.

These numbers are random (a simple one-factor model), NOT real market data.
They only let the code run when the data websites cannot be reached.
It creates:
    data/sample_prices.csv  - daily prices for every stock in holdings_sample.csv,
                              the BIST 100 and USD/TRY
    data/sample_macro.csv   - monthly CPI inflation and the policy rate
Run:  python make_sample_prices.py
"""

import numpy as np
import pandas as pd

from tracker import BENCHMARK, DATA_DIR, FX_TICKER, load_holdings

rng = np.random.default_rng(seed=42)           # fixed seed = same result every run
dates = pd.bdate_range("2024-01-02", "2026-10-16", name="Date")
n = len(dates)

# Market factor: the common movement of the whole market each day
market = rng.normal(0.0012, 0.015, n)

# Each stock = beta x market + its own random noise, with a random beta
prices = {}
for ticker in load_holdings()["yahoo"]:
    beta = rng.uniform(0.6, 1.4)
    daily = beta * market + rng.normal(0, 0.012, n)
    prices[ticker] = rng.uniform(20, 400) * np.cumprod(1 + daily)

prices[BENCHMARK] = 7500 * np.cumprod(1 + market)
# USD/TRY drifts upwards (lira depreciation), with small noise
prices[FX_TICKER] = 29.5 * np.cumprod(1 + rng.normal(0.0006, 0.002, n))
pd.DataFrame(prices, index=dates).round(4).to_csv(DATA_DIR / "sample_prices.csv")

# Monthly macro data: inflation slowly falling, policy rate high then cut
months = pd.date_range("2024-01-01", "2026-10-01", freq="MS", name="Date")
t = np.arange(len(months))
macro = pd.DataFrame({
    "monthly_inflation_pct": np.clip(4.0 - 0.08 * t + rng.normal(0, 0.4, len(t)), 0.8, None),
    "policy_rate_pct": np.clip(45 - 0.4 * np.maximum(t - 10, 0), 30, None),
}, index=months).round(2)
macro.to_csv(DATA_DIR / "sample_macro.csv")
print(f"Saved simulated prices ({n} days, {len(prices)} series) and macro data")
