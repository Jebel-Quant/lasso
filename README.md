# lasso

Preprints on the identity between Markowitz's Critical Line Algorithm and the LASSO
path, each with its LaTeX source and the code and data behind it.

## Preprints

- [`preprints/lasso/`](preprints/lasso) — *The Efficient Frontier from a LASSO Solver*
  ([arXiv:2609.37108](https://arxiv.org/abs/2609.37108)): one script that reproduces
  every figure and number, the Fama-French portfolio returns it checks against, a script
  that fetches the S&P 500 returns it also uses, and an R
  check. See [`preprints/lasso/README.md`](preprints/lasso/README.md).
- [`preprints/constrained_lasso/`](preprints/constrained_lasso) — *The Constrained LASSO
  by the Critical Line Algorithm*: the identity run the other way, computing
  constrained-LASSO paths with an unmodified CLA, and one script that reproduces every
  figure and number. See
  [`preprints/constrained_lasso/README.md`](preprints/constrained_lasso/README.md).

## Reproducing the results

Each preprint's numbers and figures come from one Python script, `python/make_figures.py`, run
with [uv](https://docs.astral.sh/uv/). Its dependencies are pinned in the script itself
([PEP 723](https://peps.python.org/pep-0723/)), and `python/make_figures.py.lock` next to it
fixes Python 3.12 and every package to an exact version and hash, so a rerun uses the
environment that produced the PDF. Everything is seeded and runs offline, except the
download of the S&P 500 returns in `preprints/lasso/`, which are not redistributed; its
real-data check on the Fama-French portfolios uses data committed to the repository.

```sh
cd preprints/lasso
uv run python/make_figures.py          # draw the figures and print every number in the note
uv run python/make_figures.py test     # check those numbers against tolerances
```

[`introduction.md`](introduction.md) explains the setup in full and how to add a
preprint.

## Layout

Each folder in `preprints/` holds one preprint: a single `.tex` file with a
`\documentclass`, the Python behind its experiments, and a `Makefile` that is just
`include ../Makefile`. The rules (`pdf`, `figures`, `test`, `clean`) live in
[`preprints/Makefile`](preprints/Makefile), with `arxiv` and `arxiv-check` to package the
arXiv source. The Python code sits in `python/`, each script
next to its lockfile; `python/make_figures.py`, if present, writes the
figures to `figures/`.

```sh
make -C preprints pdf          # build every preprint
make -C preprints/lasso pdf    # build one
```

Every pull request into `main` builds all the preprints. When it is merged, the PDFs are
published on the [`paper`](../../tree/paper) branch, with an `index.html` that links each
one by its title. GitHub Pages serves that branch at
<https://jebel-quant.github.io/lasso/>. Each preprint's arXiv source, its `.tex` and the
figures it includes, is published on the [`arxiv`](../../tree/arxiv) branch as
`arxiv/<preprint>.tar.gz`, ready to upload; CI first compiles each tarball on its own, as
arXiv will.
