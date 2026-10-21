"""Paper-trading runner for the sector momentum rotation strategy.

A forward test with real market prices and a virtual account — no broker,
no real money. Each run:
  1. Fetches fresh daily data and computes the strategy's latest
     month-end target weights (same rules as the backtest).
  2. If the target changed since the last run (or the account is new),
     rebalances the virtual portfolio at the latest close, charging
     5 bps one-way on traded notional — matching backtest assumptions.
  3. Marks the portfolio to market and appends to the trade/equity log.

State lives in paper_state.json; history in paper_log.csv. Both are
committed to git so the paper test has an auditable trail.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone

import pandas as pd

import download_yahoo
from strategy import COST_BPS, SECTORS, build_weights

STARTING_CASH = 100_000.0
STATE_FILE = "paper_state.json"
LOG_FILE = "paper_log.csv"
UA = download_yahoo.UA


def fetch_with_retry(ticker: str, tries: int = 3) -> pd.Series:
    """download_yahoo.fetch with retries — a flaky network must never
    silently produce a half-empty panel."""
    last_exc: Exception | None = None
    for i in range(tries):
        try:
            return download_yahoo.fetch(ticker)
        except Exception as exc:  # noqa: BLE001 — retried, then re-raised
            last_exc = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"data fetch failed for {ticker} after {tries} tries") from last_exc


def validate_panel(prices: pd.DataFrame) -> None:
    """Fail loudly on stale or corrupt data instead of trading on it."""
    if prices.empty:
        raise RuntimeError("price panel is empty")
    last_date = prices.index[-1].date()
    age = (datetime.now(timezone.utc).date() - last_date).days
    if age > 7:
        raise RuntimeError(f"stale data: last bar {last_date} is {age} days old")
    tail = prices.iloc[-5:]
    if tail["SPY"].isna().any() or (tail["SPY"] <= 0).any():
        raise RuntimeError("corrupt SPY data in latest bars")
    held_cols = [c for c in prices.columns if tail[c].notna().sum() == 0]
    if held_cols:
        raise RuntimeError(f"no recent data for: {held_cols}")


def latest_closes(tickers: list[str]) -> dict[str, float]:
    """Most recent raw (unadjusted) close per ticker — execution prices."""
    out: dict[str, float] = {}
    for t in tickers:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
               "?interval=1d&range=5d")
        last_exc: Exception | None = None
        for i in range(3):
            try:
                raw = subprocess.run(["curl", "-sS", "--max-time", "30", "-A", UA, url],
                                     capture_output=True, text=True, check=True).stdout
                closes = json.loads(raw)["chart"]["result"][0]["indicators"]["quote"][0]["close"]
                px = float([c for c in closes if c is not None][-1])
                if px <= 0:
                    raise ValueError(f"non-positive price for {t}: {px}")
                out[t] = px
                break
            except Exception as exc:  # noqa: BLE001 — retried, then re-raised
                last_exc = exc
                time.sleep(2 * (i + 1))
        else:
            raise RuntimeError(f"latest close fetch failed for {t}") from last_exc
    return out


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"cash": STARTING_CASH, "positions": {}, "target": {},
                "inception": None, "last_rebalance": None}


def save_state(state: dict) -> None:
    """Atomic write: a crash mid-save must not corrupt the account state."""
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def log(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    try:
        old = pd.read_csv(LOG_FILE)
        df = pd.concat([old, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv(LOG_FILE, index=False)


def main() -> None:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    state = load_state()
    fresh = state["inception"] is None

    # Fresh data -> latest month-end target weights from the strategy itself
    frames = {t: fetch_with_retry(t) for t in download_yahoo.TICKERS}
    prices = pd.DataFrame(frames)
    prices.index = prices.index.normalize()
    validate_panel(prices)
    w = build_weights(prices, use_regime_filter=False)
    target = {t: float(v) for t, v in w.iloc[-1].items() if v > 0}
    as_of = str(w.index[-1].date())
    if set(target) == {"CASH"}:
        # With the regime filter off, an all-cash target means broken data,
        # not a signal. Refuse to liquidate the paper account on it.
        raise RuntimeError("all-cash target with regime filter off — data problem, aborting")

    px = latest_closes(SECTORS + ["SPY"])
    equity = state["cash"] + sum(
        state["positions"].get(t, 0.0) * px[t] for t in state["positions"])

    rows: list[dict] = []
    changed = target != state.get("target", {})
    if fresh or changed:
        # Rebalance: sell everything not in target, then buy to target weights.
        new_positions: dict[str, float] = {}
        turnover = 0.0
        for t in set(state["positions"]) | set(target):
            cur_val = state["positions"].get(t, 0.0) * px[t]
            tgt_val = equity * target.get(t, 0.0)
            trade_val = tgt_val - cur_val
            turnover += abs(trade_val)
            if abs(trade_val) > 1.0:
                rows.append({"ts": now, "action": "BUY" if trade_val > 0 else "SELL",
                             "ticker": t, "value": round(trade_val, 2),
                             "price": px[t], "signal_date": as_of})
            if tgt_val > 0:
                new_positions[t] = tgt_val / px[t]
        # Match backtest convention: cost on one-way turnover (buys+sells)/2
        cost = (turnover / 2) * (COST_BPS / 1e4)
        state["positions"] = new_positions
        state["cash"] = equity - sum(v * px[t] for t, v in new_positions.items()) - cost
        state["target"] = target
        state["last_rebalance"] = now
        if fresh:
            state["inception"] = now
        rows.append({"ts": now, "action": "COST", "ticker": "", "value": round(-cost, 2),
                     "price": "", "signal_date": as_of})

    equity = state["cash"] + sum(
        state["positions"].get(t, 0.0) * px[t] for t in state["positions"])
    rows.append({"ts": now, "action": "MARK", "ticker": "", "value": round(equity, 2),
                 "price": "", "signal_date": as_of})
    save_state(state)
    log(rows)

    print(f"Paper account — {now}")
    print(f"Signal date: {as_of} | Target: "
          + ", ".join(f"{t} {v:.0%}" for t, v in sorted(target.items())))
    print(f"Cash: ${state['cash']:,.2f}")
    for t, sh in sorted(state["positions"].items()):
        print(f"  {t}: {sh:.4f} sh @ ${px[t]:.2f} = ${sh * px[t]:,.2f}")
    print(f"Equity: ${equity:,.2f} "
          f"({(equity / STARTING_CASH - 1) * 100:+.2f}% since inception)")
    print(f"SPY latest close: ${px['SPY']:.2f}")
    for r in rows:
        if r["action"] in ("BUY", "SELL"):
            print(f"  {r['action']} {r['ticker']} ${abs(r['value']):,.2f} @ ${r['price']:.2f}")


if __name__ == "__main__":
    main()
