# The constrained LASSO is a portfolio: source, code and data

The LaTeX source of the note *The Constrained LASSO is a Portfolio* (Schmelzer, 2026),
and everything needed to reproduce its figures and numbers. The note computes
constrained-LASSO paths with a stock Critical Line Algorithm, `cvxcla.Lasso` from
[`cvxcla`](https://github.com/cvxgrp/cvxcla) 2.7.0, using the identity between the CLA
and the LASSO path proved by Schmelzer and Hastie
([arXiv:2609.25704](https://arxiv.org/abs/2609.25704)). It runs that identity the other
way from the LASSO-solver note in [`../lasso/`](../lasso): there a LASSO solver computes
efficient frontiers, here a portfolio solver computes LASSO paths, under caps, signs,
equality and inequality rows and zero sums, also with more predictors than observations.
This folder holds no CLA of its own: it states the problems, calls `cvxcla`, and checks
every path without it.

## Contents

```
constrained_lasso.tex   the note
Makefile                builds constrained_lasso.pdf from constrained_lasso.tex (make figures first)
python/
  make_figures.py       every figure, table and number in the note, and the checks as pytest tests
  fetch_scd14.py        downloads the sCD14 data of Section 5 from CRAN (not redistributed)
  classo_check.py       runs c-lasso 1.0.11 on the exact zero-sum paths (Section 5.6)
  *.py.lock             the locked environment of each script: Python 3.12 and every package
julia/
  classopath.jl         the path algorithm of Gaines et al., ported from ConstrainedLasso.jl to Julia 1
  gaines_check.jl       runs it on every problem of Table 2 against the exact paths (Section 5.6)
  Project.toml          its environment, pinned with Manifest.toml (Julia 1.12)
  LICENSE.classopath    ConstrainedLasso.jl's MIT licence
R/comlasso_check.R      runs comlasso 0.2 on the exact zero-sum paths (Section 5.6, in CI too)
```

`python/make_figures.py` writes the figures to `figures/` (`example.pdf`, `path.pdf`,
`portfolio.pdf`, `scd14.pdf`), the rows of Tables 2 to 5 (`evidence.tex`, `scale.tex`,
`stress.tex`, `wide.tex`), and every number the text quotes, as `figures/*.tex` files
that the note inputs. The three checks against other path algorithms write theirs to
`figures/classo.tex`, `figures/gaines.tex` and `figures/comlasso.tex`. The text and the
tables therefore always show what the code computed.

## Running it

`python/make_figures.py` is a self-contained [PEP 723](https://peps.python.org/pep-0723/)
script with pinned dependencies, so [uv](https://docs.astral.sh/uv/) needs nothing else.
Each script in `python/` has a lockfile next to it that fixes Python 3.12 and every
transitive dependency to an exact version and hash; `uv run` uses it automatically, and
`uv run --locked` fails if it is out of date. After changing a pin, refresh it with
`uv lock --script python/make_figures.py`. Run it from this directory:

```sh
make figures                            # fetch the data, then make_figures.py and the three checks
make test                               # the numbers as pytest checks, with tolerances
make pdf                                # build constrained_lasso.pdf from the .tex and figures/
uv run python/make_figures.py           # the figures and numbers alone, without the three checks
uv run python/make_figures.py timings   # the cost ratios the note quotes, and the machine
uv run python/make_figures.py export    # write the exact zero-sum paths for the checks
```

`make figures` runs, in order, `python/fetch_scd14.py`, `python/make_figures.py`, then
the three scripts marked `# run: after figures`: `python/classo_check.py` in its own
locked environment (c-lasso needs NumPy 1.26), `julia/gaines_check.jl` in the Julia
environment that `julia/Manifest.toml` pins, and `R/comlasso_check.R` with Rscript (it
needs the CRAN package Rglpk). Each reads the paths that `make_figures.py` saved. The
tests exit nonzero if any number misses its tolerance, and everything except the
download is seeded and runs offline.

The note gives costs as ratios (a grid of interior-point solves against one path),
because the seconds depend on the machine and the ratios do not; `timings` measures
them and is the only part whose results the script does not write into the text.
Residuals near rounding differ in their last digits on another CPU or BLAS build, so the
tests check them against tolerances between 1e-12 and 1e-10.

The workflow [`comlasso-check.yml`](../../.github/workflows/comlasso-check.yml) runs the
comlasso check on every pull request that touches this preprint, with R 4.6.1 and Rglpk
from a dated CRAN snapshot, on zero-sum paths freshly exported by `make_figures.py
export`. comlasso itself is fetched from its authors' repository at a pinned commit and
checked by SHA-256.

## Data

Section 5 runs on the diabetes data, simulated problems and one real data set.

**Diabetes.** The data LARS was introduced with ship with scikit-learn, so they need no
download.

**sCD14 (downloaded).** Counts of 60 bacterial genera in 151 stool samples from a study
of the gut microbiome in HIV infection (Noguera-Julian et al., 2016), with the
inflammation marker sCD14 as the response, as distributed with the R package
[coda4microbiome](https://cran.r-project.org/package=coda4microbiome) 0.2.4 (MIT
licence). They are not ours and are not committed. `uv run python/fetch_scd14.py`
downloads the package's source tarball from CRAN, checks its SHA-256, and writes
`data/scd14.csv`, which git ignores. The script refuses a tarball with another digest
and does nothing when the file is already there. It carries the line
`# fetch: on build`, so `make figures` runs it first, here and in CI. Without the file,
`make_figures.py` skips the real data and its test, but the note needs them. SHA-256 of
`data/scd14.csv`:

```
3c6dcd6c18e10ea6f09bdd34329df3a8ce0a11dccd8026bf6c9a71f3c086d0da
```

**Simulated problems.** Every other problem of Tables 2 to 5 is simulated inside
`python/make_figures.py` from fixed seeds, so it needs no data.
