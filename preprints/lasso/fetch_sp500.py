# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lxml==6.1.3",
#     "pandas==3.0.3",
#     "pyarrow==24.0.0",
#     "requests==2.34.2",
#     "yfinance==1.7.0",
# ]
# ///
"""Fetch the S&P 500 daily returns that the note's real-data example uses.

Takes the current S&P 500 ticker list from Wikipedia, downloads adjusted closing prices
from Yahoo Finance through yfinance, drops names missing more than 5% of trading days,
and writes the daily simple returns to ``data/sp500_pct_returns.parquet``.

The note was computed from a snapshot fetched this way: N = 494 stocks over T = 1213
trading days, 2021-07-30 to 2026-05-29, SHA-256
b5faa5222555f28d77bad5404565297bb25bbe92b0b813f416cd2bebac79937e. The data themselves are
not redistributed. A fresh fetch gives a slightly different set, because the index
constituents change and Yahoo revises adjusted prices, so the real-data numbers will
differ a little from the note's. The script prints the SHA-256 of what it wrote.

    uv run fetch_sp500.py
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

START = "2021-06-01"
END = "2026-06-01"
MISSING_THRESHOLD = 0.05  # drop tickers missing more than 5% of trading days
OUT = Path(__file__).parent / "data" / "sp500_pct_returns.parquet"
SNAPSHOT_SHA256 = "b5faa5222555f28d77bad5404565297bb25bbe92b0b813f416cd2bebac79937e"


def main() -> None:
    """Download, clean and save the S&P 500 daily-return matrix."""
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 (research script)"},
        timeout=30,
    )
    resp.raise_for_status()
    tickers = pd.read_html(io.StringIO(resp.text))[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    print(f"{len(tickers)} tickers in the S&P 500")

    prices = yf.download(
        tickers, auto_adjust=True, progress=True, threads=True, start=pd.Timestamp(START), end=pd.Timestamp(END)
    )["Close"]
    print(f"downloaded {prices.shape[0]} trading days x {prices.shape[1]} tickers")

    missing = prices.isna().mean()
    prices = prices[missing[missing <= MISSING_THRESHOLD].index].ffill().dropna()
    returns = prices.pct_change().dropna()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(OUT)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    same = "the note's snapshot" if digest == SNAPSHOT_SHA256 else "not the note's snapshot"
    print(
        f"wrote {OUT}: T = {returns.shape[0]} days x N = {returns.shape[1]} stocks, "
        f"{returns.index[0].date()} to {returns.index[-1].date()}"
    )
    print(f"SHA-256 {digest} ({same})")


if __name__ == "__main__":
    main()
