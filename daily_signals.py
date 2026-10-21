"""Compare signal frequencies: monthly vs daily vs daily-check/trade-on-change.

Same momentum rule, same 5 bps cost model, same no-lookahead lag.
Writes daily_results.json and prints a comparison table.
"""
import json

import pandas as pd

from strategy import backtest, backtest_daily, summarize

if __name__ == "__main__":
    prices = pd.read_parquet("prices.parquet")
    runs = {
        "monthly (reference)": backtest(prices, use_regime_filter=False),
        "daily rebalance": backtest_daily(prices, on_change_only=False),
        "daily check, trade on change": backtest_daily(prices, on_change_only=True),
    }
    out = {}
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    for name, res in runs.items():
        s = summarize(res, name)
        s.pop("avg_monthly_turnover", None)  # meaningless for daily runs
        s["turnover_per_year"] = round(float(res["turnover"].sum() / years), 2)
        s["cost_drag_total_pct"] = round(float(
            (res["turnover"] * 5 / 1e4).sum() * 100), 1)
        out[name] = s
        print(f"{name:32s} CAGR {s['CAGR']:.2%}  avg/mo {s['avg_monthly']:.2%}  "
              f"Sharpe {s['sharpe_annual']:.2f}  maxDD {s['max_drawdown']:.1%}  "
              f"turnover/yr {s['turnover_per_year']:.1f}x")
    with open("daily_results.json", "w") as f:
        json.dump(out, f, indent=2)
