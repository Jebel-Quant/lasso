# Build the note "The Efficient Frontier from a LASSO Solver". The main source is
# whichever .tex file here has a \documentclass; the PDF takes its name.
# Plain article class and an inline bibliography, so no BibTeX pass and no
# vendored class files. The figures come from make_figures.py, which writes
# them to figures/; run `make figures` first.

LATEX     := pdflatex
LATEXOPTS := -interaction=nonstopmode -halt-on-error

MAIN := $(basename $(shell grep -l '^[[:space:]]*\\documentclass' *.tex))
ifneq ($(words $(MAIN)),1)
$(error expected one .tex file with a \documentclass, found: $(or $(MAIN),none))
endif

.PHONY: pdf figures test clean

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(wildcard figures/*.pdf)
	$(LATEX) $(LATEXOPTS) $<
	$(LATEX) $(LATEXOPTS) $<
	$(LATEX) $(LATEXOPTS) $<

figures:
	uv run make_figures.py

test:
	uv run make_figures.py test

clean:
	rm -f $(addprefix $(MAIN).,aux log out toc)
