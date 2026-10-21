"""
Sector Momentum Rotation strategy ("sectrot").

Rules (fixed a priori from momentum literature — no parameter fitting):
  - Universe: 11 SPDR sector ETFs (XLF, XLK, XLE, XLV, XLI, XLP, XLU, XLY, XLB, XLRE, XLC)
  - Signal: 126-trading-day (≈6-month) total return, measured at month-end.
  - Position: equal-weight top-3 sectors by momentum, rebalanced monthly.
  - Regime filter: if SPY closes below its 200-day MA at month-end, hold cash (T-bill proxy = 0%).
  - Execution: trades at month-end close; signal uses data through that close
    (implementable in practice with market-on-close orders).

Costs: 5 bps per one-way trade on notional traded (turnover-based).

Benchmark: buy-and-hold SPY with dividends (adjusted closes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = ["XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB", "XLRE", "XLC"]
MOM_LOOKBACK = 126      # trading days ≈ 6 months
TOP_N = 3
TREND_LOOKBACK = 200    # SPY 200-day MA regime filter
COST_BPS = 5            # one-way transaction cost in basis points


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month present in the index."""
    s = pd.Series(1, index=index)
    return s.resample("ME").last().index.intersection(index)


def build_weights(prices: pd.DataFrame, use_regime_filter: bool = True) -> pd.DataFrame:
    """Target weights at each month-end. Columns = tickers + 'CASH'."""
    cols = SECTORS + ["CASH"]
    weights = pd.DataFrame(0.0, index=month_ends(prices.index), columns=cols)

    mom = prices[SECTORS].pct_change(MOM_LOOKBACK)
    spy_trend = prices["SPY"] > prices["SPY"].rolling(TREND_LOOKBACK).mean()

    for dt in weights.index:
        if use_regime_filter and not bool(spy_trend.loc[dt]):
            weights.loc[dt, "CASH"] = 1.0
            continue
        ranked = mom.loc[dt].dropna().sort_values(ascending=False)
        picks = ranked.head(TOP_N).index.tolist()
        if not picks:  # not enough history yet
            weights.loc[dt, "CASH"] = 1.0
            continue
        weights.loc[dt, picks] = 1.0 / len(picks)
    return weights


def backtest(prices: pd.DataFrame, use_regime_filter: bool = True) -> dict:
    """Monthly-rebalanced backtest. Returns equity curves and stats."""
    rets = prices.pct_change()
    w = build_weights(prices, use_regime_filter)

    # Holdings apply from the rebalance date's close to next rebalance.
    holdings = w.reindex(prices.index, method="ffill").fillna(0.0)
    # No-lookahead: weights decided at close t first earn the return of t+1.
    # (Without this shift, the new portfolio would earn the very day whose
    # close produced the signal — a 1-day leak that flatters results.)
    holdings = holdings.shift(1).fillna(0.0)
    # Turnover at each rebalance date (sum of |weight changes| / 2 = one-way turnover)
    trade_dates = w.index
    prev = w.shift(1).fillna(0.0)
    one_way_turnover = (w - prev).abs().sum(axis=1) / 2.0

    asset_rets = holdings[SECTORS].mul(rets[SECTORS].fillna(0.0)).sum(axis=1)
    cost = pd.Series(0.0, index=prices.index)
    cost.loc[trade_dates] = one_way_turnover * (COST_BPS / 1e4)
    strat_rets = asset_rets - cost

    # Benchmark: SPY buy & hold from first valid date
    bench_rets = rets["SPY"].copy()

    equity = (1 + strat_rets).cumprod()
    bench_equity = (1 + bench_rets).cumprod()

    monthly = strat_rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_b = bench_rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    return {
        "weights": w,
        "daily_returns": strat_rets,
        "equity": equity,
        "bench_equity": bench_equity,
        "monthly": monthly,
        "monthly_bench": monthly_b,
        "turnover": one_way_turnover,
    }


def daily_targets(prices: pd.DataFrame, on_change_only: bool = False) -> pd.DataFrame:
    """Top-N momentum target weights computed at EVERY close.

    on_change_only=False: target is re-struck daily (rebalance daily).
    on_change_only=True: target only moves when the top-N membership changes.
    """
    mom = prices[SECTORS].pct_change(MOM_LOOKBACK)
    mask = mom.rank(axis=1, ascending=False) <= TOP_N
    counts = mask.sum(axis=1)
    tw = mask.div(counts.where(counts > 0), axis=0).fillna(0.0)
    tw["CASH"] = (counts == 0).astype(float)
    if on_change_only:
        sets = mask.apply(lambda r: tuple(sorted(r.index[r])), axis=1)
        changed = sets != sets.shift(1)
        changed.iloc[0] = True
        tw = tw.where(changed).ffill().fillna(0.0)
    return tw


def backtest_daily(prices: pd.DataFrame, on_change_only: bool = False) -> dict:
    """Daily-signal backtest, same cost model and no-lookahead lag as monthly."""
    rets = prices.pct_change()
    tw = daily_targets(prices, on_change_only)
    holdings = tw.shift(1).fillna(0.0)  # signal at close t earns from t+1
    one_way = (tw - tw.shift(1).fillna(0.0)).abs().sum(axis=1) / 2.0
    cost = one_way * (COST_BPS / 1e4)
    strat_rets = holdings[SECTORS].mul(rets[SECTORS].fillna(0.0)).sum(axis=1) - cost
    bench_rets = rets["SPY"].copy()
    return {
        "weights": tw,
        "daily_returns": strat_rets,
        "equity": (1 + strat_rets).cumprod(),
        "bench_equity": (1 + bench_rets).cumprod(),
        "monthly": strat_rets.resample("ME").apply(lambda x: (1 + x).prod() - 1),
        "monthly_bench": bench_rets.resample("ME").apply(lambda x: (1 + x).prod() - 1),
        "turnover": one_way,
    }


def summarize(res: dict, name: str) -> dict:
    m = res["monthly"].dropna()
    n_years = len(m) / 12
    cagr = (res["equity"].iloc[-1]) ** (1 / n_years) - 1
    vol_m = m.std()
    sharpe_m = (m.mean() / vol_m * np.sqrt(12)) if vol_m > 0 else 0.0
    downside = m[m < 0].std()
    sortino_m = (m.mean() / downside * np.sqrt(12)) if downside > 0 else 0.0
    roll_max = res["equity"].cummax()
    max_dd = ((res["equity"] - roll_max) / roll_max).min()
    return {
        "strategy": name,
        "period": f"{m.index[0].date()} → {m.index[-1].date()}",
        "months": len(m),
        "CAGR": round(float(cagr), 4),
        "avg_monthly": round(float(m.mean()), 4),
        "median_monthly": round(float(m.median()), 4),
        "pct_months_ge_5pct": round(float((m >= 0.05).mean()), 3),
        "pct_months_positive": round(float((m > 0).mean()), 3),
        "best_month": round(float(m.max()), 4),
        "worst_month": round(float(m.min()), 4),
        "sharpe_annual": round(float(sharpe_m), 2),
        "sortino_annual": round(float(sortino_m), 2),
        "max_drawdown": round(float(max_dd), 4),
        "avg_monthly_turnover": round(float(res["turnover"].mean()), 3),
    }


if __name__ == "__main__":
    import json

    prices = pd.read_parquet("prices.parquet")
    results = {}
    for variant, filt in [("momentum+regime", True), ("momentum-only", False)]:
        res = backtest(prices, use_regime_filter=filt)
        results[variant] = summarize(res, variant)
        # out-of-sample split: params fixed a priori, but report 2020+ separately
        oos = {k: v for k, v in res.items()}
        mask = res["monthly"].index >= "2020-01-01"
        oos_monthly = res["monthly"][mask]
        tmp = dict(res)
        tmp["monthly"] = oos_monthly
        tmp["equity"] = (1 + res["daily_returns"][res["daily_returns"].index >= "2020-01-01"]).cumprod()
        results[variant + "_2020plus"] = summarize(tmp, variant + " (2020+)")

    # SPY benchmark summary
    bench_monthly = res["monthly_bench"].dropna()
    n_years = len(bench_monthly) / 12
    results["benchmark_spy"] = {
        "strategy": "SPY buy-and-hold",
        "CAGR": round(float(res["bench_equity"].iloc[-1] ** (1 / n_years) - 1), 4),
        "avg_monthly": round(float(bench_monthly.mean()), 4),
        "pct_months_ge_5pct": round(float((bench_monthly >= 0.05).mean()), 3),
        "sharpe_annual": round(float(bench_monthly.mean() / bench_monthly.std() * np.sqrt(12)), 2),
        "max_drawdown": round(
            float(((res["bench_equity"] - res["bench_equity"].cummax()) / res["bench_equity"].cummax()).min()), 4),
    }

    print(json.dumps(results, indent=2))
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
