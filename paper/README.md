# The efficient frontier from a LASSO solver: code and data

Everything needed to reproduce the figures and numbers of the note *The Efficient
Frontier from a LASSO Solver* (Schmelzer, 2026). The note computes Markowitz efficient
frontiers with a stock LASSO solver, `lars_path` from scikit-learn, using the identity
between the Critical Line Algorithm and the LASSO path proved by Schmelzer and Hastie
([arXiv:2609.25704](https://arxiv.org/abs/2609.25704)).

## Contents

```
make_figures.py     every figure and number in the note, and the checks as pytest tests
fetch_sp500.py      downloads the S&P 500 returns of the real-data example
data/test_problem/  the 20-asset test problem and its long-short path, as CSV
R/long_short.R      the R snippet of Section 5, checked against lars_path
```

## Running it

`make_figures.py` is a self-contained [PEP 723](https://peps.python.org/pep-0723/)
script with pinned dependencies, so [uv](https://docs.astral.sh/uv/) needs nothing else.
Run it from this directory:

```sh
uv run fetch_sp500.py           # download the stock data (not redistributed; see below)
uv run make_figures.py          # write figures/*.pdf and print every number the note quotes
uv run make_figures.py test     # the same numbers as pytest checks, with tolerances
uv run make_figures.py export   # rewrite data/test_problem/ for the R check
Rscript R/long_short.R          # needs the lars package: install.packages("lars")
```

Without the stock data, `make_figures.py` skips the real-data example and its test and
runs everything else. The full run takes about three minutes, almost all of it the
independent QP baselines. The tests take about twenty seconds and exit nonzero if any
number misses its tolerance. Everything except the download is seeded and runs offline.

## Data

The Yahoo Finance prices behind the real-data example are not redistributed here.
`fetch_sp500.py` downloads them: it takes the current S&P 500 ticker list from
Wikipedia, fetches adjusted closing prices with
[`yfinance`](https://github.com/ranaroussi/yfinance) from 2021-06-01 to 2026-06-01,
drops names missing more than 5% of trading days, and writes the daily simple returns to
`data/sp500_pct_returns.parquet`.

The note used a snapshot fetched this way: 494 stocks over 1213 trading days,
2021-07-30 to 2026-05-29, with SHA-256

```
b5faa5222555f28d77bad5404565297bb25bbe92b0b813f416cd2bebac79937e
```

A fresh fetch differs a little, because the index constituents change and Yahoo revises
adjusted prices. One on 2026-09-28 gave 492 stocks over the same 1213 days, 95 corners
instead of 94, and a last corner at nu = -0.88 instead of -0.18. All the checks still
pass on it; the script prints whether what it wrote is the note's snapshot.

The 20-asset test problem is simulated from a seeded five-factor model inside
`make_figures.py`; `data/test_problem/` is only a CSV copy of it for the R script.
