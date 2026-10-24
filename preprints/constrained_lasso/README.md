# The constrained LASSO is a portfolio: source and code

The LaTeX source of the note *The Constrained LASSO is a Portfolio*
(Schmelzer, 2026), and the code behind its figures and numbers. It runs the CLA–LASSO
identity of Schmelzer and Hastie ([arXiv:2609.25704](https://arxiv.org/abs/2609.25704))
the other way from the LASSO-solver note in [`../lasso/`](../lasso). That note computes
efficient frontiers with a LASSO solver. This one computes constrained-LASSO paths with a
Critical Line Algorithm that knows nothing about regression.

The note opens with the diabetes LASSO, capped: every coefficient must satisfy
`|β_j| ≤ u`. Without the cap, least squares plays the cholesterol measures off against
each other and gives HDL a positive coefficient. scikit-learn's `lars_path` takes no
constraint but `positive`, so it cannot cap. The CLA computes the exact path under
caps, signs, equality constraints, among them the zero sum of compositional data, and
inequality rows `Cβ ≤ h`, including rows that exclude β = 0 (`h_r < 0`). That includes the cases where an encoding starts
on a degenerate vertex (a monotone fit, caps or rows under a zero sum, weights that sum
to one, a zero sum with p > n), which cvxcla relaxes, extrapolates and certifies.

Every path comes from one call, `cvxcla.Lasso` ([`cvxcla`](https://github.com/cvxgrp/cvxcla)
2.7.0, pinned). This folder holds no CLA and no encoding of its own; it states the
problems, calls `cvxcla`, and checks the paths without it. `cvxcla.Lasso` runs the CLA in
Markowitz's general form, `min ½ x'Σx − (q + γμ)'x` subject to `Ax = b`, `Gx ≤ h`,
`l ≤ x ≤ u`, traced from `γ = ∞` down to 0, through one of the two encodings the note
describes:

- **Penalty form.** Split β into positive and negative parts, `Σ = S'GS`, `μ = −1`,
  `q = S'X'y`. The parameter `γ` is λ itself. Coefficient bounds become position limits,
  equality rows `Aβ = b` become budget rows (nondegenerate start needed, so `b ≠ 0`), and
  inequality rows `Cβ ≤ h` become rows `CSz ≤ h`. With `b ≠ 0` or some `h_r < 0` the path
  starts at the feasible point of smallest ℓ1 norm.
- **Budget form.** Rescale the ℓ1 ball to the simplex, `β = t·Sw`, and add a riskless cash
  asset for the unused budget. This is Markowitz's long-only, fully invested problem with
  `γ = 1/t`. Homogeneous constraints (`Aβ = 0`, sign constraints) carry over, and the path
  ends where cash enters.

On the diabetes data, a p > n design and compositional data with `Σβ = 0`, every LARS knot
comes back to 1e-13. The constrained cases match an interior-point solver to its own
accuracy, and every knot, and every segment from end to end, satisfies the LASSO's KKT conditions to 1e-15.

The note's own CLA was removed in favour of `cvxcla`'s. That switch exposed a bug in it:
under tight caps it returned paths that broke the optimality conditions without saying
so. The checks above are what caught it, and they are what the paths rest on now.

The same comparison runs on real data: counts of 60 genera in 151 stool samples from an
HIV study, with the inflammation marker sCD14 as the response (Noguera-Julian et al.,
2016, as shipped with the R package coda4microbiome). On each of the first ten segments
of the zero-sum path, at most one of the 60 possible reference genera gives the zero-sum
LASSO's selection.

## Contents

```
constrained_lasso.tex   the note
Makefile                builds constrained_lasso.pdf (make figures first)
python/
  make_figures.py       both forms (on cvxcla's CLA), three figures, Tables 2 and 3, both examples, and the checks as pytest tests
  make_figures.py.lock  the locked environment: Python 3.12 and every package, by version and hash
  fetch_scd14.py        downloads the sCD14 data from CRAN into data/scd14.csv (git ignores data/)
  classo_check.py       runs c-lasso 1.0.11 (NumPy 1.26, its own lock) on the saved zero-sum paths
  classo_check.py.lock  its locked environment
  fetch_scd14.py.lock   its locked environment
R/
  comlasso_check.R      runs comlasso 0.2 (Jeon et al., fetched at a pinned commit, SHA-256 checked) on the saved zero-sum paths
julia/
  classopath.jl         the path algorithm of Gaines et al., ported from ConstrainedLasso.jl to Julia 1
  gaines_check.jl       runs it on every problem of Table 2 against the exact paths
  Project.toml          its environment, pinned with Manifest.toml (Julia 1.12)
  LICENSE.classopath    ConstrainedLasso.jl's MIT licence
```

`python/make_figures.py` writes `figures/example.pdf` (the diabetes path without and with the cap, Fig. 1), `figures/path.pdf`
(the diabetes path against the LARS knots, Fig. 2), and from the real data
`figures/scd14.pdf` (Fig. 3). It also writes the rows of
Table 2 as `figures/evidence.tex`, the zero-sum problems at scale (up to 500 parts) as
Table 3 in `figures/scale.tex`, and the numbers the text quotes as `figures/example.tex`
and `figures/scd14.tex`, which the note inputs, so the text and the table always show
what the code computed.

## Running it

`python/make_figures.py` is a self-contained [PEP 723](https://peps.python.org/pep-0723/) script
with pinned dependencies, seeded and offline once the data are there (the diabetes data
ship with scikit-learn). The sCD14 data are not ours and are not committed:
`python/fetch_scd14.py` downloads coda4microbiome 0.2.4 from CRAN, checks its SHA-256 and
writes `data/scd14.csv`. It carries the line `# fetch: on build`, so `make figures` runs
it first, here and in CI; it does nothing when the file is already there. Without the
file, `make_figures.py` skips the real data and its test, but the note needs them.
`python/make_figures.py.lock` fixes Python 3.12 and every transitive dependency to an exact
version and hash; `uv run` uses it automatically. After changing a pin, refresh it with
`uv lock --script python/make_figures.py`.
Run it from this directory:

```sh
make figures    # fetch the data if needed, run make_figures.py, classo_check.py, gaines_check.jl
make test       # the same numbers as pytest checks (uv run python/make_figures.py test)
uv run python/make_figures.py timings   # the cost ratios of the note, and the machine
make pdf        # build constrained_lasso.pdf from constrained_lasso.tex and figures/
```
