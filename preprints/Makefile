# The rules for every preprint. Each folder here holds one preprint, and its Makefile
# includes this one. Run here, a target runs in every preprint: `make pdf` builds them
# all, `make -C lasso pdf` just one.
#
# A preprint's main source is the one .tex file in its folder with a \documentclass; the
# PDF takes its name. A make_figures.py, if present, draws the figures into figures/;
# run `make figures` before `make pdf`.

LATEX     := pdflatex
LATEXOPTS := -interaction=nonstopmode -halt-on-error
TARGETS   := pdf figures test clean

.PHONY: $(TARGETS)

MAIN := $(basename $(shell grep -l '^[[:space:]]*\\documentclass' *.tex 2>/dev/null))

ifeq ($(MAIN),)

# No main source here, so this is preprints/ itself.
PREPRINTS := $(patsubst %/Makefile,%,$(wildcard */Makefile))

$(TARGETS):
	@for dir in $(PREPRINTS); do $(MAKE) -C $$dir $@ || exit 1; done

else

ifneq ($(words $(MAIN)),1)
$(error expected one .tex file with a \documentclass, found: $(MAIN))
endif

FIGURES := $(wildcard make_figures.py)

pdf: $(MAIN).pdf

# Plain article class and an inline bibliography, so no BibTeX pass.
$(MAIN).pdf: $(MAIN).tex $(wildcard figures/*.pdf)
	$(LATEX) $(LATEXOPTS) $<
	$(LATEX) $(LATEXOPTS) $<
	$(LATEX) $(LATEXOPTS) $<

figures:
	$(if $(FIGURES),uv run $(FIGURES))

test:
	$(if $(FIGURES),uv run $(FIGURES) test)

clean:
	rm -f $(addprefix $(MAIN).,aux log out toc)

endif
