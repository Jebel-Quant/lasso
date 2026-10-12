# Build the note "The Efficient Frontier from a LASSO Solver" from lasso.tex.
# Plain article class and an inline bibliography, so no BibTeX pass and no
# vendored class files. The figures come from make_figures.py, which writes
# them to figures/; run `make figures` first.

LATEX     := pdflatex
LATEXOPTS := -interaction=nonstopmode -halt-on-error

.PHONY: pdf figures test clean

pdf: lasso.pdf

lasso.pdf: lasso.tex $(wildcard figures/*.pdf)
	$(LATEX) $(LATEXOPTS) lasso.tex
	$(LATEX) $(LATEXOPTS) lasso.tex
	$(LATEX) $(LATEXOPTS) lasso.tex

figures:
	uv run make_figures.py

test:
	uv run make_figures.py test

clean:
	rm -f lasso.aux lasso.log lasso.out lasso.toc
