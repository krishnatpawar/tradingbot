"""Correctness tests for the strategy engine and paper trader.

Run: venv/bin/python -m pytest test_strategy.py -q
Uses the real downloaded panel (prices.parquet) plus small synthetic
frames for the leak/lag checks. No network access required.
"""
import numpy as np
import pandas as pd
import pytest

import strategy
from strategy import SECTORS, backtest, build_weights, daily_targets


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return pd.read_parquet("prices.parquet")


# ---------- weights ----------

def test_monthly_weights_sum_to_one(prices):
    w = build_weights(prices, use_regime_filter=False)
    assert len(w) > 100
    assert np.allclose(w.sum(axis=1), 1.0)


def test_monthly_weights_only_hold_top_n(prices):
    w = build_weights(prices, use_regime_filter=False)
    held = (w[SECTORS] > 0).sum(axis=1)
    assert (held <= strategy.TOP_N).all()


def test_regime_filter_goes_cash_in_crash(prices):
    # Mar 2020: SPY was below its 200DMA at month-end -> regime variant in cash
    w = build_weights(prices, use_regime_filter=True)
    mar2020 = w.loc["2020-03-31"]
    assert mar2020["CASH"] == 1.0


# ---------- no lookahead ----------

def test_weights_unchanged_by_future_data(prices):
    """Truncating the future must not change any past weight — the core
    anti-lookahead property, checked for both monthly and daily targets."""
    cut = prices.index[2000]
    w_full = build_weights(prices, use_regime_filter=False)
    w_trunc = build_weights(prices.loc[:cut], use_regime_filter=False)
    pd.testing.assert_frame_equal(w_full.loc[w_trunc.index], w_trunc)

    d_full = daily_targets(prices)
    d_trunc = daily_targets(prices.loc[:cut])
    pd.testing.assert_frame_equal(d_full.loc[d_trunc.index], d_trunc)


def test_signal_day_return_belongs_to_old_portfolio(monkeypatch):
    """Weights decided at close t must NOT earn day t's return."""
    idx = pd.bdate_range("2024-01-01", periods=10)
    px = pd.DataFrame({s: 100.0 for s in SECTORS + ["SPY"]}, index=idx)
    px["XLF"] = 100.0
    px["XLK"] = [100, 100, 100, 100, 200, 200, 200, 200, 200, 200]  # +100% on day 4
    flip = idx[4]
    w = pd.DataFrame(0.0, index=[idx[0], flip], columns=SECTORS + ["CASH"])
    w.loc[idx[0], "XLF"] = 1.0
    w.loc[flip, "XLK"] = 1.0
    monkeypatch.setattr(strategy, "build_weights",
                        lambda p, use_regime_filter=True: w)
    # month_ends() must return our two dates
    monkeypatch.setattr(strategy, "month_ends", lambda i: w.index)
    res = backtest(px, use_regime_filter=False)
    # Day of the flip: still holding XLF (flat) -> return is ONLY the switch
    # cost (-5 bps), crucially NOT XLK's +100% jump that produced the signal.
    assert res["daily_returns"].loc[flip] == pytest.approx(
        -strategy.COST_BPS / 1e4, abs=1e-9)
    # Day after: now holding XLK, which is flat -> still 0 (no phantom gains)
    assert res["daily_returns"].loc[idx[5]] == pytest.approx(0.0, abs=1e-9)


# ---------- costs ----------

def test_costs_reduce_equity(prices, monkeypatch):
    monkeypatch.setattr(strategy, "COST_BPS", 0)
    free = backtest(prices, use_regime_filter=False)
    monkeypatch.setattr(strategy, "COST_BPS", 5)
    paid = backtest(prices, use_regime_filter=False)
    assert paid["turnover"].sum() > 0
    assert paid["equity"].iloc[-1] < free["equity"].iloc[-1]


# ---------- daily targets ----------

def test_daily_targets_sum_to_one(prices):
    tw = daily_targets(prices)
    assert np.allclose(tw.sum(axis=1), 1.0)


def test_on_change_targets_move_less(prices):
    d = daily_targets(prices, on_change_only=False)
    oc = daily_targets(prices, on_change_only=True)
    moves_d = (d.diff().abs().sum(axis=1) > 0).sum()
    moves_oc = (oc.diff().abs().sum(axis=1) > 0).sum()
    assert moves_oc <= moves_d


# ---------- paper trader validation ----------

def test_validate_panel_rejects_stale():
    import paper_trader
    old = pd.DataFrame({"SPY": [100.0]}, index=pd.to_datetime(["2020-01-02"]))
    with pytest.raises(RuntimeError, match="stale"):
        paper_trader.validate_panel(old)


def test_validate_panel_rejects_corrupt_spy(prices):
    import paper_trader
    bad = prices.copy()
    bad.iloc[-1, bad.columns.get_loc("SPY")] = np.nan
    with pytest.raises(RuntimeError, match="corrupt"):
        paper_trader.validate_panel(bad)


def test_validate_panel_accepts_real_data(prices):
    import paper_trader
    paper_trader.validate_panel(prices)  # must not raise
