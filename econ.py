"""
econ.py - economic and financial-econometric analysis of the portfolio.

Used by bist_economics.ipynb. Four building blocks:
    1. Macro data   : Turkish CPI inflation and the TCMB policy rate
    2. Real returns : Fisher relation - deflate nominal TRY returns by CPI
    3. CAPM         : OLS regressions with Newey-West standard errors
    4. Markowitz    : mean-variance efficient frontier (long-only)
"""

import numpy as np
import pandas as pd

from tracker import DATA_DIR

WEEKS_PER_YEAR = 52


# ---------------------------------------------------------------------------
# 1. Macro data
# ---------------------------------------------------------------------------
def get_macro(start, offline=False):
    """Monthly CPI index (start = 100) and the policy rate (% per year).

    Live data comes from TCMB via the borsapy package:
      * Inflation().tufe()  -> monthly CPI (TUFE) inflation in %
      * TCMB().history()    -> one-week repo (policy) rate decisions
    If that fails, it uses SIMULATED data from data/sample_macro.csv.
    Returns a DataFrame with monthly rows: monthly_inflation_pct,
    policy_rate_pct, cpi_index, and attrs["source"].
    """
    start = pd.Timestamp(start).to_period("M").to_timestamp()
    if not offline:
        try:
            import borsapy as bp

            tufe = bp.Inflation().tufe(start=start.strftime("%Y-%m-%d"))
            infl = pd.Series(tufe["MonthlyInflation"].astype(float).values,
                             index=pd.to_datetime(tufe.index).to_period("M").to_timestamp())
            rate = bp.TCMB().history("policy", period="max")["lending"].astype(float)
            rate.index = pd.to_datetime(rate.index)
            # policy rate on the first day of each month = last decision before it
            months = infl.sort_index().index
            rate_m = rate.sort_index().reindex(rate.index.union(months)).ffill().loc[months]
            macro = pd.DataFrame({"monthly_inflation_pct": infl.sort_index(),
                                  "policy_rate_pct": rate_m})
            macro.attrs["source"] = "TCMB via borsapy"
            return _add_cpi_index(macro.dropna())
        except Exception as error:
            print(f"Macro download failed ({type(error).__name__}: {error}) - using simulated data.")

    macro = pd.read_csv(DATA_DIR / "sample_macro.csv", index_col="Date", parse_dates=True)
    macro = macro.loc[macro.index >= start]
    macro.attrs["source"] = "SIMULATED sample data (not real TCMB data)"
    return _add_cpi_index(macro)


def _add_cpi_index(macro):
    """CPI level index from monthly inflation: chain the monthly changes."""
    source = macro.attrs.get("source", "")
    macro = macro.copy()
    macro["cpi_index"] = 100 * (1 + macro["monthly_inflation_pct"] / 100).cumprod()
    macro.attrs["source"] = source
    return macro


# ---------------------------------------------------------------------------
# 2. Real returns (Fisher relation)
# ---------------------------------------------------------------------------
def daily_cpi(macro, dates):
    """Spread the monthly CPI index over trading days (log-linear interpolation).

    Monthly inflation for month m is assigned to the END of month m, so the
    index for date t uses only price changes up to that point.
    """
    cpi = macro["cpi_index"].copy()
    cpi.index = cpi.index + pd.offsets.MonthEnd(0)              # end of month
    first = cpi.index[0] - pd.offsets.MonthBegin(1)             # start = 100
    cpi = pd.concat([pd.Series([100.0], index=[first]), cpi])
    all_days = cpi.index.union(dates)
    log_cpi = np.log(cpi).reindex(all_days).interpolate(method="time").ffill()
    return np.exp(log_cpi).reindex(dates)


def real_index(nominal, macro):
    """Real (inflation-adjusted) value index: nominal / CPI, rebased to 100.

    Fisher relation: (1 + real return) = (1 + nominal return) / (1 + inflation)
    """
    cpi = daily_cpi(macro, nominal.index)
    real = nominal / cpi
    return (real / real.dropna().iloc[0] * 100).rename(nominal.name)


# ---------------------------------------------------------------------------
# 3. CAPM regressions
# ---------------------------------------------------------------------------
def weekly_returns(prices):
    """Weekly (Friday-to-Friday) simple returns. Weekly data reduces noise from
    thin trading and non-synchronous prices compared with daily data."""
    weekly = prices.ffill().resample("W-FRI").last()
    if weekly.index[-1] > prices.index[-1]:      # drop the unfinished current week
        weekly = weekly.iloc[:-1]
    return weekly.pct_change().dropna(how="all")


def weekly_risk_free(macro, weeks):
    """Weekly risk-free rate from the annual policy rate: (1 + r)^(1/52) - 1."""
    annual = macro["policy_rate_pct"] / 100
    annual = annual.reindex(annual.index.union(weeks)).ffill().reindex(weeks)
    return (1 + annual) ** (1 / WEEKS_PER_YEAR) - 1


def capm_table(stock_returns, market_returns, rf, sectors=None, hac_lags=4):
    """Estimate the CAPM for each stock by OLS:

        r_i - r_f = alpha_i + beta_i (r_m - r_f) + e_i

    Standard errors are Newey-West (HAC) to allow for heteroskedasticity
    and autocorrelation in weekly returns.
    """
    import statsmodels.api as sm

    rows = []
    mkt_excess = (market_returns - rf).rename("market_excess")
    for ticker in stock_returns.columns:
        y = (stock_returns[ticker] - rf).rename("excess")
        data = pd.concat([y, mkt_excess], axis=1).dropna()
        if len(data) < 30:                      # too few observations
            continue
        X = sm.add_constant(data["market_excess"])
        fit = sm.OLS(data["excess"], X).fit(cov_type="HAC",
                                            cov_kwds={"maxlags": hac_lags})
        rows.append({
            "ticker": ticker.replace(".IS", ""),
            "sector": (sectors or {}).get(ticker, ""),
            "alpha (annual)": fit.params["const"] * WEEKS_PER_YEAR,
            "alpha t-stat": fit.tvalues["const"],
            "beta": fit.params["market_excess"],
            "beta t-stat": fit.tvalues["market_excess"],
            "R2": fit.rsquared,
            "weeks": int(fit.nobs),
            "avg excess return (annual)": data["excess"].mean() * WEEKS_PER_YEAR,
        })
    return pd.DataFrame(rows).set_index("ticker").sort_values("beta", ascending=False)


# ---------------------------------------------------------------------------
# 4. Markowitz mean-variance optimisation
# ---------------------------------------------------------------------------
def annual_moments(weekly):
    """Annualised mean vector and covariance matrix from weekly returns."""
    weekly = weekly.dropna()
    return weekly.mean() * WEEKS_PER_YEAR, weekly.cov() * WEEKS_PER_YEAR


def portfolio_stats(w, mu, cov):
    """Expected return and volatility of portfolio weights w."""
    return float(w @ mu), float(np.sqrt(w @ cov @ w))


def efficient_frontier(mu, cov, n_points=40, max_weight=1.0):
    """Long-only efficient frontier: for each target return, minimise variance.

        min_w  w' S w   s.t.  sum(w) = 1,  w' mu = target,  0 <= w <= max_weight
    """
    from scipy.optimize import minimize

    n = len(mu)
    bounds = [(0, max_weight)] * n
    w0 = np.repeat(1 / n, n)
    budget = {"type": "eq", "fun": lambda w: w.sum() - 1}

    def min_var(extra=None):
        cons = [budget] + ([extra] if extra else [])
        res = minimize(lambda w: w @ cov.values @ w, w0, bounds=bounds,
                       constraints=cons, method="SLSQP")
        return res.x if res.success else None

    w_minvar = min_var()
    lo = float(w_minvar @ mu)
    hi = float(mu.max()) if max_weight >= 1 else float(np.sort(mu.values)[::-1][
        :int(np.ceil(1 / max_weight))].mean())
    frontier = []
    for target in np.linspace(lo, hi, n_points):
        w = min_var({"type": "eq", "fun": lambda w, t=target: w @ mu.values - t})
        if w is not None:
            r, s = portfolio_stats(w, mu.values, cov.values)
            frontier.append({"return": r, "volatility": s, "weights": w})
    return pd.DataFrame(frontier), w_minvar


def max_sharpe(mu, cov, rf_annual, max_weight=1.0):
    """Tangency portfolio: maximise (w'mu - rf) / sqrt(w' S w), long-only."""
    from scipy.optimize import minimize

    n = len(mu)

    def neg_sharpe(w):
        r, s = portfolio_stats(w, mu.values, cov.values)
        return -(r - rf_annual) / s

    res = minimize(neg_sharpe, np.repeat(1 / n, n), bounds=[(0, max_weight)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP")
    return res.x


# ---------------------------------------------------------------------------
# 5. Dividends: price return vs total return
# ---------------------------------------------------------------------------
def price_and_total_return(tickers, start, offline=False):
    """Two price tables from Yahoo Finance:
      * price : the quoted closing price (price return only)
      * total : the 'adjusted close', which reinvests dividends (total return)
    The difference between the two is the return from dividends.
    Offline: simulated prices with an assumed 3% dividend yield.
    """
    if not offline:
        try:
            import yfinance as yf

            raw = yf.download(tickers, start=start, auto_adjust=False, progress=False)
            price, total = raw["Close"].dropna(how="all"), raw["Adj Close"].dropna(how="all")
            if not price.empty:
                price.attrs["source"] = total.attrs["source"] = "Yahoo Finance"
                return price, total
        except Exception as error:
            print(f"Download failed ({error}) - using simulated data.")
    price = pd.read_csv(DATA_DIR / "sample_prices.csv", index_col="Date", parse_dates=True)
    price = price.loc[price.index >= pd.Timestamp(start), tickers]
    growth = (1 + 0.03) ** (np.arange(len(price)) / 252)
    total = price.copy()
    stocks = [t for t in tickers if t.endswith(".IS") and t != "XU100.IS"]   # the index has no dividends
    total[stocks] = price[stocks].mul(growth, axis=0)
    price.attrs["source"] = total.attrs["source"] = "SIMULATED (assumed 3% dividend yield)"
    return price, total


def annualised(prices):
    """Average yearly growth between the first and last available price."""
    first = prices.apply(lambda s: s.dropna().iloc[0])
    last = prices.apply(lambda s: s.dropna().iloc[-1])
    years = prices.apply(lambda s: (s.dropna().index[-1] - s.dropna().index[0]).days / 365.25)
    return (last / first) ** (1 / years) - 1


# ---------------------------------------------------------------------------
# 6. Event study: TCMB policy-rate decisions
# ---------------------------------------------------------------------------
def policy_rate_changes(start, offline=False):
    """Dates on which the TCMB policy rate changed, with the change in % points.

    Live: every decision in TCMB().history("policy"); we keep the dates where
    the rate differs from the previous decision (hikes and cuts).
    """
    if not offline:
        try:
            import borsapy as bp

            rate = bp.TCMB().history("policy", period="max")["lending"].astype(float)
            rate.index = pd.to_datetime(rate.index)
            rate = rate.sort_index()
            source = "TCMB via borsapy"
        except Exception as error:
            print(f"Policy-rate download failed ({error}) - using simulated data.")
            offline = True
    if offline:
        rate = pd.read_csv(DATA_DIR / "sample_macro.csv", index_col="Date",
                           parse_dates=True)["policy_rate_pct"]
        source = "SIMULATED"
    change = rate.diff()
    events = pd.DataFrame({"rate_after": rate, "change_pp": change})
    events = events[(events["change_pp"].abs() > 1e-9) & (events.index >= pd.Timestamp(start))]
    events["direction"] = np.where(events["change_pp"] > 0, "hike", "cut")
    events.attrs["source"] = source
    return events


def event_returns(returns, market, event_dates, est_window=(-150, -11), event_window=(-5, 10),
                  model="market"):
    """Abnormal returns around events, in trading days relative to day 0.

    model="market":   AR = r - (a + b * r_market), a and b estimated by OLS in
                      the estimation window (MacKinlay 1997)
    model="constant": AR = r - mean(r) in the estimation window
                      (used for the index itself, which IS the market)
    Returns a DataFrame: rows = event date, columns = relative day.
    """
    days = returns.index
    out = {}
    for date in event_dates:
        pos = days.searchsorted(pd.Timestamp(date))      # day 0 = first trading day >= event
        if pos + est_window[0] < 0 or pos + event_window[1] >= len(days):
            continue                                    # not enough data around this event
        est = slice(pos + est_window[0], pos + est_window[1] + 1)
        win = slice(pos + event_window[0], pos + event_window[1] + 1)
        r_est, r_win = returns.iloc[est], returns.iloc[win]
        if model == "market":
            m_est = market.iloc[est]
            ok = r_est.notna() & m_est.notna()
            if ok.sum() < 60:
                continue
            b, a = np.polyfit(m_est[ok], r_est[ok], 1)
            ar = r_win - (a + b * market.iloc[win])
        else:
            ar = r_win - r_est.mean()
        out[pd.Timestamp(date)] = pd.Series(ar.values,
                                            index=range(event_window[0], event_window[1] + 1))
    return pd.DataFrame(out).T


def car_test(ar, window):
    """Cumulative abnormal return over a window (inclusive) for each event,
    plus the average and a simple cross-sectional t-statistic."""
    car = ar.loc[:, window[0]:window[1]].sum(axis=1)
    t = car.mean() / (car.std(ddof=1) / np.sqrt(len(car))) if len(car) > 1 else np.nan
    return car, car.mean(), t


# ---------------------------------------------------------------------------
# Event dates on the Borsa Istanbul calendar, and a placebo test
# ---------------------------------------------------------------------------
def align_to_sessions(event_dates, announce_time="14:00"):
    """Map each TCMB decision to the first Borsa Istanbul session that could react.

    The Monetary Policy Committee announces at 14:00 Istanbul time.
      * Normal trading day: the market is open at 14:00, so day 0 = that day.
      * Half day (e.g. the eve of a holiday, closing 12:30): the market is
        already closed, so day 0 = the next session.
      * Holiday or weekend: day 0 = the next session.
    Uses the official exchange calendar "XIST" from the exchange_calendars
    package. Without it, falls back to plain weekdays (holidays ignored).
    Returns a DataFrame: announcement date, day-0 session, and whether it moved.
    """
    rows = []
    try:
        import exchange_calendars as xc
        cal = xc.get_calendar("XIST")
        for date in pd.to_datetime(event_dates):
            announce = pd.Timestamp(f"{date.date()} {announce_time}", tz="Europe/Istanbul")
            if cal.is_session(date) and announce < cal.session_close(date):
                session = date                                    # market still open
            else:
                session = cal.date_to_session(date, direction="next")
                if session == date:                               # half day, already closed
                    session = cal.next_session(date)
            rows.append((date, session.tz_localize(None)))
        source = "XIST exchange calendar"
    except ImportError:
        for date in pd.to_datetime(event_dates):
            rows.append((date, date + pd.offsets.BDay(0)))        # weekend -> Monday
        source = "weekdays only (install exchange_calendars for holidays)"
    out = pd.DataFrame(rows, columns=["announced", "day0_session"])
    out["moved"] = out["announced"] != out["day0_session"]
    out.attrs["source"] = source
    return out


def placebo_car(returns, market, event_dates, window=(0, 1), n_draws=1000, exclude_days=10,
                model="market", seed=1, **kwargs):
    """Is the average CAR around real events unusual compared with random days?

    1. Draw as many random trading days as there are real events, avoiding
       +/- `exclude_days` around any real event.
    2. Compute the average CAR on those fake dates, exactly as for the real ones.
    3. Repeat n_draws times. The empirical p-value is the share of draws whose
       average CAR is at least as large (in absolute value) as the real one.
    This needs no normality assumption, unlike the t-test in car_test.
    """
    rng = np.random.default_rng(seed)
    days = returns.dropna().index
    real_ar = event_returns(returns, market, event_dates, model=model, **kwargs)
    _, real_mean, _ = car_test(real_ar, window)

    # Candidate placebo days: not close to any real event, with enough history
    pos_real = days.searchsorted(pd.to_datetime(event_dates))
    blocked = np.zeros(len(days), dtype=bool)
    for p in pos_real:
        blocked[max(p - exclude_days, 0): p + exclude_days + 1] = True
    candidates = days[~blocked][160:-15]               # room for the estimation window

    k = len(real_ar)
    draws = []
    for _ in range(n_draws):
        fake = rng.choice(candidates, size=k, replace=False)
        ar = event_returns(returns, market, fake, model=model, **kwargs)
        draws.append(car_test(ar, window)[1])
    draws = pd.Series(draws)
    p_value = (draws.abs() >= abs(real_mean)).mean()
    return {"real mean CAR": real_mean, "placebo draws": draws,
            "empirical p-value": p_value, "events": k}
