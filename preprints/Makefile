# The rules for every preprint. Each folder here holds one preprint, and its Makefile
# includes this one. Run here, a target runs in every preprint in parallel: `make pdf`
# builds them all, `make -C lasso pdf` just one.
#
# A preprint's main source is the one .tex file in its folder with a \documentclass; the
# PDF takes its name. A python/make_figures.py, if present, draws the figures into figures/;
# run `make figures` before `make pdf` or `make arxiv`. It may also write a table into
# figures/ as a .tex file for the note to \input.
#
# A python/fetch_*.py script downloads data into data/. One that carries the line
# `# fetch: on build` runs on every `make data`, and so before every `make figures`, here
# and in CI: use it only for data that may be downloaded freely and do not change. Data
# whose owner forbids redistribution, such as the Fama-French returns, are fetched by
# hand, and the scripts that use them skip what needs them.
#
# A python/*.py script that carries the line `# run: after figures` runs after
# make_figures.py in every `make figures`, in its own locked environment: for a check
# whose dependencies cannot share make_figures.py's, reading what it wrote to figures/.
# A julia/*.jl script with the same line runs after those, in the environment that
# julia/Project.toml and julia/Manifest.toml pin, for a check written in Julia. An R/*.R
# script with the same line runs last, with Rscript, for a check written in R; the R
# packages it needs are installed beforehand (CI installs them from a CRAN snapshot).
#
# A preprint may also be written for a journal, as preprints/cla/ is for the Journal of
# Statistical Software: its body in sections/*.tex, its bibliography in refs/*.bib, which
# BibTeX turns into a .bbl, and the journal's class and BibTeX style bundled with it, in
# any folder of the preprint (cla/cls/jss/). TeX and BibTeX search every folder that holds
# a .cls or .bst, and the arXiv source carries those files at its root, where arXiv looks.

LATEX     := pdflatex
LATEXOPTS := -interaction=nonstopmode -halt-on-error
TARGETS   := reproduce pdf data figures test arxiv arxiv-check clean

.PHONY: $(TARGETS)

MAIN := $(basename $(shell grep -l '^[[:space:]]*\\documentclass' *.tex 2>/dev/null))

ifeq ($(MAIN),)

# No main source here, so this is preprints/ itself. A target runs in every preprint at
# once, one job each; `lasso/pdf` and the like run it in just one.
PREPRINTS := $(patsubst %/Makefile,%,$(wildcard */Makefile))
JOBS      := $(foreach target,$(TARGETS),$(addsuffix /$(target),$(PREPRINTS)))

.PHONY: $(JOBS)

# `make reproduce` is the exception and takes one preprint at a time: cla/'s scaling
# figures time cvxcla, and would otherwise measure the other preprints' load.
$(TARGETS):
	@$(MAKE) -j$(if $(filter reproduce,$@),1,$(words $(PREPRINTS))) $(addsuffix /$@,$(PREPRINTS))

$(JOBS):
	@$(MAKE) -C $(@D) $(@F)

else

ifneq ($(words $(MAIN)),1)
$(error expected one .tex file with a \documentclass, found: $(MAIN))
endif

# A preprint's Makefile may name its figure script, the arguments it takes and its test
# command before including this one, as cla/ does; these are the defaults.
FIGURES      ?= $(wildcard python/make_figures.py)
FIGURES_ARGS ?=
TEST         ?= $(if $(FIGURES),uv run $(FIGURES) test)
FETCH   := $(shell grep -l '^\# fetch: on build$$' python/fetch_*.py 2>/dev/null)
AFTER   := $(shell grep -l '^\# run: after figures$$' python/*.py 2>/dev/null)
JULIA   := $(shell grep -l '^\# run: after figures$$' julia/*.jl 2>/dev/null)
RCHECKS := $(shell grep -l '^\# run: after figures$$' R/*.R 2>/dev/null)

SECTIONS  := $(wildcard sections/*.tex)
BIB       := $(wildcard refs/*.bib)
# With an inline bibliography there is no BibTeX pass; `true' stands in for it.
BIBTEX    := $(if $(BIB),bibtex,true)
# A bundled class or style: every file in a folder that holds a .cls or .bst. The
# trailing colon keeps the system search path.
VENDORDIR := $(sort $(dir $(shell find . \( -path ./arxiv-src -o -path ./arxiv-build \) -prune \
               -o \( -name '*.cls' -o -name '*.bst' \) -print 2>/dev/null)))
VENDOR    := $(foreach d,$(VENDORDIR),$(filter-out %/,$(wildcard $(d)*)))
ifneq ($(VENDORDIR),)
empty     :=
space     := $(empty) $(empty)
export TEXINPUTS := .:$(subst $(space),:,$(VENDORDIR)):
export BSTINPUTS := .:$(subst $(space),:,$(VENDORDIR)):
endif

# The one command that regenerates every result: the data, the figures and every number,
# the checks against them, the PDF, and the arXiv source compiled on its own.
reproduce:
	$(MAKE) figures
	$(MAKE) test
	$(MAKE) pdf
	$(MAKE) arxiv-check

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(SECTIONS) $(BIB) $(wildcard figures/*.pdf figures/*.tex)
	$(LATEX) $(LATEXOPTS) $<
	$(BIBTEX) $(MAIN)
	$(LATEX) $(LATEXOPTS) $<
	$(LATEX) $(LATEXOPTS) $<

data:
	$(foreach script,$(FETCH),uv run $(script) &&) true

figures: data
	$(if $(FIGURES),uv run $(FIGURES) $(FIGURES_ARGS))
	$(foreach script,$(AFTER),uv run $(script) &&) true
	$(if $(JULIA),julia --project=julia -e 'using Pkg; Pkg.instantiate()')
	$(foreach script,$(JULIA),julia --project=julia $(script) &&) true
	$(foreach script,$(RCHECKS),Rscript $(script) &&) true

test:
	$(TEST)

# arXiv compiles the source itself, so a submission is the .tex and the figures it
# includes, not the PDF. Only the figures the note includes go in: figures/ can hold
# drawings the note no longer uses. A table the script writes into figures/ and the note
# \inputs goes in too. The figures sit flat at the root of the submission, beside the
# .tex, so the packaged copy of the .tex (and of its sections) looks for them there: its
# \graphicspath{{figures/}} becomes {{./}} and \input{figures/...} loses the prefix. The
# sources in the repository are left as they are. The same tree is packed twice, as a
# tarball and as a zip, for whichever the upload form takes.
ARXIV_SRC     := arxiv-src
ARXIV_CHECK   := arxiv-build
ARXIV_TARBALL := $(MAIN).tar.gz
ARXIV_ZIP     := $(MAIN).zip
INCLUDED      := $(addprefix figures/,$(shell grep -hoE '\\includegraphics(\[[^]]*\])?\{[^}]*\}' \
                   $(MAIN).tex $(SECTIONS) | sed -E 's/.*\{//; s/\}//' | sort -u)) \
                 $(shell grep -hoE '\\input\{figures/[^}]*\}' $(MAIN).tex $(SECTIONS) | sed -E 's/.*\{//; s/\}//' | sort -u)

arxiv: $(ARXIV_TARBALL) $(ARXIV_ZIP)

$(ARXIV_TARBALL): $(MAIN).tex $(SECTIONS) $(BIB) $(VENDOR) $(INCLUDED)
	rm -rf $(ARXIV_SRC)
	mkdir -p $(ARXIV_SRC)
	cp $(MAIN).tex $(ARXIV_SRC)/
	$(if $(INCLUDED),cp $(INCLUDED) $(ARXIV_SRC)/)
	$(if $(SECTIONS),mkdir -p $(ARXIV_SRC)/sections && cp $(SECTIONS) $(ARXIV_SRC)/sections/)
	$(if $(BIB),mkdir -p $(ARXIV_SRC)/refs && cp $(BIB) $(ARXIV_SRC)/refs/)
	$(if $(VENDOR),cp $(VENDOR) $(ARXIV_SRC)/)
	sed -i.orig -e 's#\\graphicspath{{figures/}}#\\graphicspath{{./}}#' -e 's#\\input{figures/#\\input{#g' \
	  $(ARXIV_SRC)/$(MAIN).tex $(addprefix $(ARXIV_SRC)/,$(SECTIONS))
	find $(ARXIV_SRC) -name '*.orig' -delete
	@! grep -rnE '\\(input|includegraphics|graphicspath)[^}]*figures/' $(ARXIV_SRC) --include='*.tex' || \
	  { echo "the packaged .tex still points into figures/"; exit 1; }
	tar czf $@ -C $(ARXIV_SRC) .
	tar tzf $@

$(ARXIV_ZIP): $(ARXIV_TARBALL)
	rm -f $@
	cd $(ARXIV_SRC) && zip -qr ../$@ .
	unzip -l $@

# Compile the tarball the way arXiv will: unpacked into an empty directory and built
# there, so a file the packaging forgot fails here rather than after posting. The search
# paths are reset to the unpacked tree alone, as on arXiv, so a bundled class the tarball
# forgot is not found here either. The tail of the log is printed if a run fails, since
# the error is among its last lines.
arxiv-check: $(ARXIV_TARBALL) $(ARXIV_ZIP)
	rm -rf $(ARXIV_CHECK)
	mkdir -p $(ARXIV_CHECK)
	tar xzf $< -C $(ARXIV_CHECK)
	cd $(ARXIV_CHECK) && export TEXINPUTS=.: BSTINPUTS=.: && \
	  for step in latex bibtex latex latex; do \
	    if [ $$step = bibtex ]; then $(BIBTEX) $(MAIN) >/dev/null || { cat $(MAIN).blg; exit 1; }; \
	    else $(LATEX) $(LATEXOPTS) $(MAIN).tex >/dev/null || { tail -n 40 $(MAIN).log; exit 1; }; fi; \
	  done
	@! grep -E 'Citation .* undefined|Reference .* undefined|There were undefined' \
	  $(ARXIV_CHECK)/$(MAIN).log || { echo "undefined references in the arXiv source"; exit 1; }
	@echo "arXiv source of $(MAIN) compiles on its own: $$(grep -o 'Output written on.*' $(ARXIV_CHECK)/$(MAIN).log)"

clean:
	rm -f $(addprefix $(MAIN).,aux log out toc bbl blg)
	rm -rf $(ARXIV_SRC) $(ARXIV_CHECK) $(ARXIV_TARBALL) $(ARXIV_ZIP)

endif
