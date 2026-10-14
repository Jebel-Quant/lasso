# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "pandas==3.0.6",
#     "requests==2.34.2",
# ]
# ///
"""Rebuild the Fama--French portfolio returns that the note's reproducible real-data check uses.

Downloads the daily returns of the 100 portfolios formed on size and book-to-market
(10 x 10, value-weighted) from Kenneth French's Data Library, keeps 2016-01-01 to
2025-12-31, drops any portfolio with a missing day in that window, and writes the returns,
in percent as the library publishes them, to ``data/french100_daily.csv``.

The note uses the snapshot committed next to this script, with the SHA-256 below. The
library rebuilds its history every month from CRSP, so a fresh download can differ in
the last digit where CRSP has revised a return; the script prints whether what it wrote
is the note's snapshot.

    uv run fetch_french.py

Source: Kenneth R. French, Data Library,
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/100_Portfolios_10x10_Daily_CSV.zip"
SECTION = "Average Value Weighted Returns -- Daily"
START, END = "2016-01-01", "2025-12-31"
MISSING = -99.99  # the library's code for a missing return (it also uses -999)
OUT = Path(__file__).parent / "data" / "french100_daily.csv"
SNAPSHOT_SHA256 = "98298ef6a28ead0ebd0b518bd8a309aa6ec4f10b23e7d1e5933802a8cbc03c43"


def value_weighted(text: str) -> pd.DataFrame:
    """The value-weighted daily returns block of the library's CSV, indexed by date."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if SECTION in line) + 1
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].strip())
    frame = pd.read_csv(io.StringIO("\n".join(lines[start:end])), index_col=0)
    frame.index = pd.to_datetime(frame.index.astype(str).str.strip(), format="%Y%m%d")
    frame.index.name = "date"
    frame.columns = frame.columns.str.strip()
    return frame


def main() -> None:
    """Download, cut and save the 100-portfolio daily returns."""
    resp = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (research script)"}, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        (name,) = archive.namelist()
        text = archive.read(name).decode("latin-1")
    print(text.splitlines()[0])  # names the CRSP vintage

    returns = value_weighted(text).loc[START:END]
    missing = (returns <= MISSING).any()
    for col in missing[missing].index:
        print(f"dropping {col}: {(returns[col] <= MISSING).sum()} missing days")
    returns = returns.loc[:, ~missing]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(OUT, float_format="%.2f", date_format="%Y-%m-%d", lineterminator="\n")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    same = "the note's snapshot" if digest == SNAPSHOT_SHA256 else "not the note's snapshot"
    print(
        f"wrote {OUT}: T = {returns.shape[0]} days x N = {returns.shape[1]} portfolios, "
        f"{returns.index[0].date()} to {returns.index[-1].date()}"
    )
    print(f"SHA-256 {digest} ({same})")


if __name__ == "__main__":
    main()
