"""Download daily adjusted closes from Yahoo's v8 chart API via curl.

yfinance is unusable in this sandbox (cookie/crumb + TLS-impersonation
timeouts), but the chart endpoint answers plain curl with a browser UA.
We request events=div,split and use the API's adjclose series, which is
split- AND dividend-adjusted (total-return prices).
"""
import datetime as dt
import json
import subprocess
import sys

import pandas as pd

TICKERS = ["SPY", "XLF", "XLK", "XLE", "XLV", "XLI",
           "XLP", "XLU", "XLY", "XLB", "XLRE", "XLC"]

START = dt.datetime(2010, 1, 1, tzinfo=dt.timezone.utc)
END = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)  # through latest close
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def fetch(ticker: str) -> pd.Series:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/{t}"
        "?interval=1d&period1={p1}&period2={p2}&events=div%2Csplit"
    ).format(t=ticker, p1=int(START.timestamp()), p2=int(END.timestamp()))
    out = subprocess.run(
        ["curl", "-sS", "--max-time", "60", "-A", UA, url],
        capture_output=True, text=True, check=True,
    ).stdout
    payload = json.loads(out)
    result = payload["chart"]["result"][0]
    idx = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None)
    adj = result["indicators"]["adjclose"][0]["adjclose"]
    s = pd.Series(adj, index=idx, name=ticker).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def main() -> None:
    frames = {}
    for t in TICKERS:
        s = fetch(t)
        frames[t] = s
        print(f"{t}: {len(s)} bars, {s.index[0].date()} -> {s.index[-1].date()}", flush=True)
    prices = pd.DataFrame(frames)  # keep full history; XLRE/XLC list later
    prices.index = prices.index.normalize()  # strip intraday time component
    print(f"panel: {prices.shape}, {prices.index[0].date()} -> {prices.index[-1].date()}")
    # sanity: SPY adjusted close on 2024-12-31 was ≈ 586.36 (Yahoo)
    print("SPY 2024-12-31 adj:", round(float(prices["SPY"].loc["2024-12-31"]), 2))
    prices.to_parquet("prices.parquet")


if __name__ == "__main__":
    sys.exit(main())
