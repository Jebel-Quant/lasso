# The efficient frontier from a LASSO solver: source, code and data

The LaTeX source of the note *The Efficient Frontier from a LASSO Solver* (Schmelzer,
2026; [arXiv:2609.37108](https://arxiv.org/abs/2609.37108)), and everything needed to
reproduce its figures and numbers. The note computes Markowitz efficient
frontiers with a stock LASSO solver, `lars_path` from scikit-learn, using the identity
between the Critical Line Algorithm and the LASSO path proved by Schmelzer and Hastie
([arXiv:2609.25704](https://arxiv.org/abs/2609.25704)).

## Contents

```
lasso.tex           the note
Makefile            builds lasso.pdf from lasso.tex (make figures first)
python/
  make_figures.py   every figure and number in the note, and the checks as pytest tests
  fetch_french.py   downloads the Fama-French portfolio returns of Section 8 (not redistributed)
  fetch_sp500.py    downloads the S&P 500 returns of the real-data example at scale
  *.py.lock         the locked environment of each script: Python 3.12 and every package
data/test_problem/  the 20-asset test problem and its long-short path, as CSV
R/long_short.R      the R snippet of Section 5, checked against lars_path
```

## Running it

`python/make_figures.py` is a self-contained [PEP 723](https://peps.python.org/pep-0723/)
script with pinned dependencies, so [uv](https://docs.astral.sh/uv/) needs nothing else.
Each script in `python/` has a lockfile next to it (`make_figures.py.lock`,
`fetch_french.py.lock`, `fetch_sp500.py.lock`)
that fixes Python 3.12 and every transitive dependency to an exact version and hash;
`uv run` uses it automatically, and `uv run --locked` fails if it is out of date.
After changing a pin, refresh it with `uv lock --script python/make_figures.py`.
Run it from this directory:

```sh
uv run python/fetch_french.py          # download the Fama-French data (not redistributed; see below)
uv run python/fetch_sp500.py           # download the stock data (not redistributed; see below)
uv run python/make_figures.py          # write figures/*.pdf and print every number the note quotes
uv run python/make_figures.py test     # the same numbers as pytest checks, with tolerances
uv run python/make_figures.py export   # rewrite data/test_problem/ for the R check
uv run python/make_figures.py timings  # the cost ratios the note quotes, and the machine
Rscript R/long_short.R          # needs the lars package: install.packages("lars")
make pdf                        # build lasso.pdf from lasso.tex and figures/ (pdflatex)
```

Without the downloaded data, `python/make_figures.py` skips the Fama-French and S&P 500
examples and their tests and runs everything else. The full run takes about three minutes, almost all of it the
independent QP baselines; `timings` skips those and measures only the costs the note
quotes. The note gives them as ratios (one path against one QP, a grid of QPs against
one path), because the seconds depend on the machine and the ratios do not. The tests take about twenty seconds and exit nonzero if any
number misses its tolerance. Everything except the downloads is seeded and runs offline.

## Data

Section 8 checks the method on two sets of real returns.

**Fama-French portfolios (downloaded).** The returns are copyrighted by Eugene F. Fama
and Kenneth R. French and are **not** redistributed here, at their request.
`uv run python/fetch_french.py` downloads them from
[Kenneth French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
and writes `data/french100_daily.csv`, which git ignores: the daily value-weighted
returns, in percent, of the 100 portfolios formed on size and book-to-market, from
2016-01-04 to 2025-12-31. The note used the file built from the CRSP database of
August 2026. `ME9 BM10` misses 250 days in the
window and is dropped, which leaves 99 portfolios over 2514 days. SHA-256:

```
98298ef6a28ead0ebd0b518bd8a309aa6ec4f10b23e7d1e5933802a8cbc03c43
```

The script says whether what it wrote matches. The library rebuilds its history each
month from CRSP, so a later download can differ where CRSP has revised a return. Keep
the file for your own use, do not commit or share it, and cite the Data Library if you
use the data.

**S&P 500 stocks (downloaded).** The Yahoo Finance prices behind the example at scale
are not redistributed here.
`python/fetch_sp500.py` downloads them: it takes the current S&P 500 ticker list from
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
`python/make_figures.py`; `data/test_problem/` is only a CSV copy of it for the R script.
