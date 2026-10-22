"""
run_report.py - build the full report with one command.

Examples (run in the Terminal, inside this folder):
    python run_report.py                          # sample portfolio, live prices
    python run_report.py --holdings data/my_holdings.csv --rf 0.35
    python run_report.py --offline                # simulated prices, no internet

Outputs go to the output/ folder: an Excel report and four PNG charts.
"""

import argparse

import pandas as pd

import tracker as tr

# ---- 1. Read the options typed after "python run_report.py" ---------------
parser = argparse.ArgumentParser(description="BIST portfolio report")
parser.add_argument("--holdings", default=str(tr.DATA_DIR / "holdings_sample.csv"),
                    help="CSV with columns ticker, shares, buy_price, buy_date")
parser.add_argument("--rf", type=float, default=0.0,
                    help="annual risk-free rate in TRY as a decimal, e.g. 0.35")
parser.add_argument("--sources", default="yahoo,borsapy,isyatirim",
                    help="data sources in order, comma-separated")
parser.add_argument("--offline", action="store_true",
                    help="use simulated sample prices instead of downloading")
args = parser.parse_args()

# ---- 2. Holdings and prices -----------------------------------------------
holdings = tr.load_holdings(args.holdings)
start = holdings["buy_date"].min()
tickers = list(holdings["yahoo"]) + [tr.BENCHMARK, tr.FX_TICKER]
prices = tr.get_prices(tickers, start=start, offline=args.offline,
                       sources=args.sources.split(","))
source = prices.attrs.get("source", "")
holdings = tr.fill_buy_prices(holdings, prices)

# ---- 3. Calculations --------------------------------------------------------
# Time-weighted index (removes the effect of buying new stocks over time)
value_try = tr.portfolio_index(holdings, prices)
bench_try = prices[tr.BENCHMARK].ffill().rename("BIST 100")
value_usd = tr.to_usd(value_try, prices[tr.FX_TICKER])
bench_usd = tr.to_usd(bench_try, prices[tr.FX_TICKER])

table = tr.holdings_table(holdings, prices)
metrics_try = tr.performance_metrics(value_try, bench_try, args.rf)
metrics_usd = tr.performance_metrics(value_usd, bench_usd, 0.0)

# ---- 4. Outputs -------------------------------------------------------------
out_dir = tr.PROJECT_DIR / "output"
out_dir.mkdir(exist_ok=True)
tr.make_charts(value_try, bench_try, value_usd, bench_usd, table, out_dir, source)
tr.export_excel(table, metrics_try, metrics_usd, value_try, bench_try,
                out_dir / "portfolio_report.xlsx", source)

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
print(f"\nPrice source: {source}\n")
print(table.to_string(index=False))
print("\n", pd.DataFrame({"TRY": metrics_try, "USD": metrics_usd}).to_string())
print(f"\nSaved Excel report and charts to {out_dir}")
