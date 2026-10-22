"""Tests for the BIST tracker and the economics functions. Run with:  pytest -q
All tests run offline on small hand-made data or the simulated sample files."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import econ                # noqa: E402
import tracker as tr       # noqa: E402

DAYS = pd.bdate_range("2025-01-01", periods=6)


def test_time_weighted_return_ignores_new_money():
    """Prices never move, so the return must be zero even though a second
    stock is bought on day 3 (the portfolio VALUE jumps, the INDEX must not)."""
    prices = pd.DataFrame({"AAA.IS": 10.0, "BBB.IS": 20.0}, index=DAYS)
    holdings = pd.DataFrame({"ticker": ["AAA", "BBB"], "yahoo": ["AAA.IS", "BBB.IS"],
                             "shares": [10, 5], "buy_date": [DAYS[0], DAYS[2]]})
    index = tr.portfolio_index(holdings, prices)
    assert np.allclose(index, 100)
    assert tr.portfolio_value(holdings, prices).iloc[-1] == 200      # 10x10 + 5x20


def test_time_weighted_return_matches_price_growth():
    prices = pd.DataFrame({"AAA.IS": [10, 11, 12, 12, 13, 15.0]}, index=DAYS)
    holdings = pd.DataFrame({"ticker": ["AAA"], "yahoo": ["AAA.IS"], "shares": [7],
                             "buy_date": [DAYS[0]]})
    index = tr.portfolio_index(holdings, prices)
    assert np.isclose(index.iloc[-1], 150)


def test_max_drawdown():
    assert np.isclose(tr.max_drawdown(pd.Series([100, 120, 90, 130, 117])), -0.25)


def test_real_index_is_flat_when_prices_grow_with_inflation():
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    macro = econ._add_cpi_index(pd.DataFrame({"monthly_inflation_pct": 2.0,
                                              "policy_rate_pct": 40.0}, index=months))
    days = pd.bdate_range("2024-02-01", "2024-11-29")
    nominal = econ.daily_cpi(macro, days).rename("nominal")      # grows exactly like CPI
    real = econ.real_index(nominal, macro)
    assert np.allclose(real, 100, atol=1e-6)


def test_event_study_has_zero_abnormal_return_without_news():
    days = pd.bdate_range("2024-01-01", periods=300)
    r = pd.Series(0.001, index=days)                              # constant return
    ar = econ.event_returns(r, None, [days[200]], model="constant")
    assert np.allclose(ar.values, 0)


def test_full_report_runs_offline():
    holdings = tr.load_holdings()
    tickers = list(holdings["yahoo"]) + [tr.BENCHMARK, tr.FX_TICKER]
    prices = tr.get_prices(tickers, start=holdings["buy_date"].min(), offline=True)
    holdings = tr.fill_buy_prices(holdings, prices)
    value = tr.portfolio_index(holdings, prices)
    metrics = tr.performance_metrics(value, prices[tr.BENCHMARK])
    assert -1 < metrics["Total return"] < 10
    table = tr.holdings_table(holdings, prices)
    assert np.isclose(table["weight"].sum(), 1)
