# Replication materials for "cvxcla: The Critical Line Algorithm with an Operator Interface in Python"

This script reproduces every figure, table and number in the paper.

```
make_figures.py                 every figure and every numerical check
make_figures.py.lock            the script's locked environment
data/sp500_pct_returns.parquet  the frozen S&P 500 daily returns used by the data-driven steps
```

The package source, `cvxcla-2.7.1.tar.gz`, sits one level up. It is the release on PyPI.

## Running

Install [uv](https://docs.astral.sh/uv/). The script declares its exact dependencies
(PEP 723 inline metadata, pinned, including `cvxcla==2.7.1`), so `uv run` installs them
into an isolated environment:

```sh
uv run make_figures.py all            # every figure and every check, writes figures/
uv run make_figures.py all --quick    # the same, without the two slow scaling sweeps
uv run make_figures.py frontier       # one figure; run with no argument for the list
```

Inputs are drawn from fixed seeds and the returns are committed, so after the packages
are installed nothing is downloaded.

## Expected runtime and output

The paper's results were computed on an Apple M4 Pro (14 cores, 48 GB) under macOS 15.6.1
with Python 3.14.6. On that machine `all` takes about 40 minutes, nearly all of it in the
two scaling sweeps (Sections 7.2 and 7.4); `all --quick` takes a few minutes.

Turning-point counts, accuracy figures and KKT residuals should agree with the paper up
to round-off. Wall-clock times and memory in the scaling figures and tables depend on
the machine and the BLAS. Elsewhere the absolute numbers will differ, but the ratios
between backends should be similar.
