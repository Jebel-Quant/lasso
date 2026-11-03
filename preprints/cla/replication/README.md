# Replication materials for "cvxcla: The Critical Line Algorithm with an Operator Interface in Python"

This script reproduces every figure, table and number in the paper. It is the experiment
script of cvxcla 2.7.1 (`experiments/make_figures.py` at tag `v2.7.1`), less one
external timing baseline and two steps (covariance estimators, resampled frontiers) whose
results the paper does not use, and extended with the steps and checks listed under
Computational details in the paper (among them `worked-example`, `threads` and the KKT
certificate of the largest traces). The steps the paper links to the release, such as
`make_figures.py scaling`, compute the same results in both; `real-frontier` no longer
draws a figure, since the paper uses only its numbers.

```
make_figures.py                 every figure and every numerical check
make_figures.py.lock            the script's locked environment
custom_backend.py               the complete block-diagonal backend of Section 5.2
data/sp500_pct_returns.parquet  the frozen S&P 500 daily returns used by the data-driven steps
```

The package source, `cvxcla-2.7.1.tar.gz`, sits one level up. It is the release on PyPI.

## Running

Install [uv](https://docs.astral.sh/uv/). The script declares its exact dependencies
(PEP 723 inline metadata, pinned, including `cvxcla==2.7.1`), so `uv run` installs them
into an isolated environment:

```sh
uv run make_figures.py all            # every figure and every check, writes figures/
uv run make_figures.py all --quick    # the same, without the three slow steps
uv run make_figures.py frontier       # one figure; run with no argument for the list
uv run custom_backend.py              # the backend of Section 5.2 against the dense matrix
```

Inputs are drawn from fixed seeds and the returns are committed, so after the packages
are installed nothing is downloaded.

One step, `r-cla`, compares the turning points with the R package `CLA`. It needs R
(4.6.1 was used) and that package from CRAN, and is skipped without them:

```sh
Rscript -e 'install.packages("CLA", repos = "https://cloud.r-project.org")'
uv run make_figures.py r-cla
```

## Expected runtime and output

The paper's results were computed on an Apple M4 Pro (14 cores, 48 GB) under macOS 15.6.1
with Python 3.14.6. On that machine `all` takes about 50 minutes, nearly all of it in the
two scaling sweeps (Sections 7.2 and 7.4) and the single-thread timings (`threads`);
`all --quick` skips those three and takes a few minutes.

Turning-point counts, accuracy figures and KKT residuals should agree with the paper up
to round-off. A run of the six correctness checks (`validate-exact`, `validate-constraints`,
`validate-kkt`, `validate-scaling`, `validate-factor`, `validate-projection`) on Linux
(Debian, Python 3.12, OpenBLAS, in Docker) found the same turning points as macOS;
only round-off-level counts, such as zero-length segments and projections, moved. Wall-clock times and memory in the scaling figures and tables depend on
the machine and the BLAS. Elsewhere the absolute numbers will differ, but the ratios
between backends should be similar.
