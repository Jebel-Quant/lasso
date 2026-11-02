# cvxcla: the Critical Line Algorithm with an operator interface in Python

The LaTeX source of the software paper for [`cvxcla`](https://github.com/cvxgrp/cvxcla),
written for the *Journal of Statistical Software*, and the scripts behind its figures and
checks. The preprint is on SSRN,
[doi:10.2139/ssrn.7583179](https://doi.org/10.2139/ssrn.7583179). It moved here from
[`Jebel-Quant/homotopy`](https://github.com/Jebel-Quant/homotopy), where its history
before the move remains.

## Contents

```
cla.tex                  main file: preamble and an \input of each section
sections/                one .tex per section
refs/references.bib      the BibTeX database
cls/jss/                 the JSS class, bibliography style and logo, bundled
figures/                 the figure PDFs the paper includes, committed
figures/data/            the frozen S&P 500 returns the data-driven figures use
make_figures.py          redraws the figures and runs the numerical checks
make_figures.py.lock     the script's locked environment
replication/README.md    the README of the JSS replication materials
```

## Building

```sh
make reproduce      # every figure and check, then the PDF and the arXiv source
make pdf            # cla.pdf only: pdflatex, BibTeX, pdflatex, pdflatex
make arxiv-check    # package the arXiv source and compile it on its own
make jss            # the JSS submission: PDF, source, package sdist, replication materials
```

`make figures` runs `make_figures.py all`; `make reproduce` runs it, then builds the PDF
and checks the arXiv source.

`make jss` builds `jss-submission/` and `cla-jss-submission.zip`. They hold `cla.pdf`, the flat
LaTeX source, the `cvxcla` 2.7.1 sdist from PyPI (checked against its SHA-256), and
`replication/`: the script, its lock file and the returns snapshot, laid out as the
script expects to find them. Code is set in the plain `CodeChunk`/`CodeInput`/`CodeOutput`
environments of `jss.cls`, and the class runs without `nojss`.

The shared rules in [`../Makefile`](../Makefile) put `cls/jss/` on TeX's and BibTeX's
search paths, so no system install of the JSS files is needed, and the arXiv source
carries them at its root.

## Figures and checks

Unlike the other preprints here, this one's figures are committed, and CI does not
redraw them: the scaling figures and tables report wall-clock times, which change from
machine to machine. Redraw them by hand:

```sh
uv run make_figures.py all            # every figure and every check
uv run make_figures.py all --quick    # without the two slow scaling sweeps
uv run make_figures.py frontier       # one figure
```
