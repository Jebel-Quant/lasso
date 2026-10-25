# cvxcla: the Critical Line Algorithm with an operator interface in Python

The LaTeX source of the software paper for [`cvxcla`](https://github.com/cvxgrp/cvxcla),
written for the *Journal of Statistical Software*, and the scripts behind its figures and
checks. It moved here from
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
check_listing.py         runs the Appendix C listing as printed, against cvxcla
*.py.lock                each script's locked environment
```

## Building

```sh
make reproduce      # every figure and check, then the PDF and the arXiv source
make pdf            # cla.pdf only: pdflatex, BibTeX, pdflatex, pdflatex
make arxiv-check    # package the arXiv source and compile it on its own
```

`make figures` runs `make_figures.py all` and `make test` runs `check_listing.py`;
`make reproduce` runs both, then builds the PDF and checks the arXiv source.

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

`uv run check_listing.py` cuts the CLA listing of Appendix C (Figure 5) out of
`sections/appendix-bailey.tex`, runs it, and checks that it reproduces cvxcla's turning
points and refuses the inputs it does not cover. CI runs it on every pull request that
touches the appendix or the script (`.github/workflows/listing.yml`).
