# The constrained LASSO by the Critical Line Algorithm: source and code

The LaTeX source of the note *The Constrained LASSO by the Critical Line Algorithm*
(Schmelzer, 2026), and the code behind its figures and numbers. It runs the CLA–LASSO
identity of Schmelzer and Hastie ([arXiv:2609.25704](https://arxiv.org/abs/2609.25704))
the other way from the LASSO-solver note in [`../lasso/`](../lasso). That note computes
efficient frontiers with a LASSO solver. This one computes constrained-LASSO paths with a
Critical Line Algorithm that knows nothing about regression.

The CLA (`cla` in `make_figures.py`, about a hundred lines of NumPy) is Markowitz's in its
general form: `min ½ x'Σx − (q + γμ)'x` subject to `Ax = b`, `l ≤ x ≤ u`, traced from
`γ = ∞` down to 0. Two encodings feed it a LASSO:

- **Penalty form.** Split β into positive and negative parts, `Σ = S'GS`, `μ = −1`,
  `q = S'X'y`. The parameter `γ` is λ itself. Coefficient bounds become position limits,
  and equality rows `Aβ = b` become budget rows (nondegenerate start needed, so `b ≠ 0`).
- **Budget form.** Rescale the ℓ1 ball to the simplex, `β = t·Sw`, and add a riskless cash
  asset for the unused budget. This is Markowitz's long-only, fully invested problem with
  `γ = 1/t`. Homogeneous constraints (`Aβ = 0`, sign constraints) carry over, and the path
  ends where cash enters.

On the diabetes data, a p > n design and compositional data with `Σβ = 0`, every LARS knot
comes back to 1e-13. The constrained cases match an interior-point solver to its own
accuracy, and every knot and segment midpoint satisfies the LASSO's KKT conditions to 1e-15.

## Contents

```
constrained_lasso.tex   the note
Makefile                builds constrained_lasso.pdf (make figures first)
make_figures.py         the CLA, both forms, three figures, the table, and the checks as pytest tests
make_figures.py.lock    the locked environment: Python 3.12 and every package, by version and hash
```

`make_figures.py` writes `figures/path.pdf` (the diabetes path, CLA against LARS knots,
Fig. 1), `figures/portfolio.pdf` (the same path as a long/short/cash portfolio, Fig. 2) and
`figures/constrained.pdf` (box-constrained and zero-sum paths, Fig. 3).

## Running it

`make_figures.py` is a self-contained [PEP 723](https://peps.python.org/pep-0723/) script
with pinned dependencies, seeded and offline (the diabetes data ship with scikit-learn).
`make_figures.py.lock` fixes Python 3.12 and every transitive dependency to an exact
version and hash; `uv run` uses it automatically. After changing a pin, refresh it with
`uv lock --script make_figures.py`.
Run it from this directory:

```sh
make figures    # write figures/*.pdf and print Table 1 (uv run make_figures.py)
make test       # the same numbers as pytest checks (uv run make_figures.py test)
make pdf        # build constrained_lasso.pdf from constrained_lasso.tex and figures/
```
