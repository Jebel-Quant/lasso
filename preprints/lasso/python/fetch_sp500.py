# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "pandas==3.0.6",
#     "pyarrow==26.0.0",
#     "yfinance==1.7.0",
# ]
# ///
"""Fetch the S&P 500 daily returns that the note's real-data example uses.

Reads the S&P 500 constituents from ``data/sp500_constituents.csv``, downloads adjusted
closing prices from Yahoo Finance through yfinance, drops names missing more than 5% of
trading days, and writes the daily simple returns to ``data/sp500_pct_returns.parquet``.

The constituents file is committed and fixed: 503 tickers, in Yahoo's spelling, with
company names, as Wikipedia's "List of S&P 500 companies" listed them in revision
1359528143 of 15 June 2026. Every revision from 4 to 17 June 2026 lists the same tickers,
and the note's snapshot was drawn from that list: its 494 stocks are the 503 less nine
listed too recently to pass the 5% filter (CEG, FDXF, GEHC, GEV, KVUE, Q, SNDK, SOLV,
VLTO).

The note was computed from a snapshot fetched this way: N = 494 stocks over T = 1213
trading days, 2021-07-30 to 2026-05-29, SHA-256
b5faa5222555f28d77bad5404565297bb25bbe92b0b813f416cd2bebac79937e. The data themselves are
not redistributed. A fresh fetch starts from the same tickers but can still differ
slightly, because Yahoo revises adjusted prices, so the real-data numbers may differ a
little from the note's. The script prints the SHA-256 of what it wrote.

    uv run python/fetch_sp500.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

START = "2021-06-01"
END = "2026-06-01"
MISSING_THRESHOLD = 0.05  # drop tickers missing more than 5% of trading days
DATA = Path(__file__).resolve().parent.parent / "data"
CONSTITUENTS = DATA / "sp500_constituents.csv"
OUT = DATA / "sp500_pct_returns.parquet"
SNAPSHOT_SHA256 = "b5faa5222555f28d77bad5404565297bb25bbe92b0b813f416cd2bebac79937e"


def main() -> None:
    """Download, clean and save the S&P 500 daily-return matrix."""
    tickers = pd.read_csv(CONSTITUENTS)["ticker"].tolist()
    print(f"{len(tickers)} tickers in {CONSTITUENTS.name}")

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
