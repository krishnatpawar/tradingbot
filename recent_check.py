"""Recent-window check: run the strategy over the last N years only.

Usage: python recent_check.py [years]   (default 2)

Signals are computed on full history (momentum needs its 126-day lookback),
then performance is measured from (latest close − N years) onward — the
way you'd have experienced it entering with the then-current signal.
Costs included, same engine as the full backtest.
"""
import json
import sys

import numpy as np
import pandas as pd

from strategy import backtest, backtest_daily

STARTING = 100_000.0


def stats(daily: pd.Series, label: str) -> dict:
    daily = daily.dropna()
    eq = (1 + daily).cumprod()
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    monthly = daily.resample("ME").apply(lambda x: (1 + x).prod() - 1).iloc[1:]  # full months only
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    mid = daily.index[0] + (daily.index[-1] - daily.index[0]) / 2
    h1 = (1 + daily[daily.index < mid]).prod() - 1
    h2 = (1 + daily[daily.index >= mid]).prod() - 1
    return {
        "label": label,
        "total_return": round(float(eq.iloc[-1] - 1), 4),
        "CAGR": round(float(eq.iloc[-1] ** (1 / years) - 1), 4),
        "ending_value_on_100k": round(STARTING * float(eq.iloc[-1]), 2),
        "sharpe_annual": round(float(monthly.mean() / monthly.std() * np.sqrt(12)), 2),
        "max_drawdown": round(float(dd), 4),
        "pct_months_positive": round(float((monthly > 0).mean()), 3),
        "pct_months_ge_5pct": round(float((monthly >= 0.05).mean()), 3),
        "best_month": round(float(monthly.max()), 4),
        "worst_month": round(float(monthly.min()), 4),
        "first_half_return": round(float(h1), 4),
        "second_half_return": round(float(h2), 4),
    }


if __name__ == "__main__":
    years = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    prices = pd.read_parquet("prices.parquet")
    end = prices.index[-1]
    start = end - pd.DateOffset(years=years)
    print(f"Window: {start.date()} -> {end.date()}")

    monthly = backtest(prices, use_regime_filter=False)["daily_returns"]
    monthly_r = backtest(prices, use_regime_filter=True)["daily_returns"]
    daily = backtest_daily(prices, on_change_only=False)["daily_returns"]
    spy = prices["SPY"].pct_change()

    out = {}
    for label, series in [("monthly momentum-only", monthly),
                          ("monthly + regime filter", monthly_r),
                          ("daily signals", daily),
                          ("SPY buy-and-hold", spy)]:
        s = stats(series[series.index >= start], label)
        out[label] = s
        print(f"{label:26s} total {s['total_return']:+.1%}  CAGR {s['CAGR']:.1%}  "
              f"$100k -> ${s['ending_value_on_100k']:,.0f}  Sharpe {s['sharpe_annual']:.2f}  "
              f"maxDD {s['max_drawdown']:.1%}  H1 {s['first_half_return']:+.1%}  H2 {s['second_half_return']:+.1%}")
    with open(f"recent_{years}y.json", "w") as f:
        json.dump(out, f, indent=2)
