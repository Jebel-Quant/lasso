# lasso

Preprints on computing efficient frontiers with a LASSO solver, each with its LaTeX
source and the code and data behind it.

## Preprints

- [`preprints/lasso/`](preprints/lasso) — *The Efficient Frontier from a LASSO Solver*
  ([arXiv:2609.37108](https://arxiv.org/abs/2609.37108)): one script that reproduces
  every figure and number, a script that fetches the S&P 500 returns it uses, and an R
  check. See [`preprints/lasso/README.md`](preprints/lasso/README.md).

## Layout

Each folder in `preprints/` holds one preprint: a single `.tex` file with a
`\documentclass`, a `Makefile` with `pdf`, `figures`, `test` and `clean` targets, and the
Python behind its experiments. A `make_figures.py`, if present, writes the figures to
`figures/`.

```sh
make -C preprints pdf          # build every preprint
make -C preprints/lasso pdf    # build one
```

Every pull request into `main` builds all the preprints. When it is merged, the PDFs are
published as the only files on the [`paper`](../../tree/paper) branch.
