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

LATEX     := pdflatex
LATEXOPTS := -interaction=nonstopmode -halt-on-error
TARGETS   := pdf data figures test arxiv arxiv-check clean

.PHONY: $(TARGETS)

MAIN := $(basename $(shell grep -l '^[[:space:]]*\\documentclass' *.tex 2>/dev/null))

ifeq ($(MAIN),)

# No main source here, so this is preprints/ itself. A target runs in every preprint at
# once, one job each; `lasso/pdf` and the like run it in just one.
PREPRINTS := $(patsubst %/Makefile,%,$(wildcard */Makefile))
JOBS      := $(foreach target,$(TARGETS),$(addsuffix /$(target),$(PREPRINTS)))

.PHONY: $(JOBS)

$(TARGETS):
	@$(MAKE) -j$(words $(PREPRINTS)) $(addsuffix /$@,$(PREPRINTS))

$(JOBS):
	@$(MAKE) -C $(@D) $(@F)

else

ifneq ($(words $(MAIN)),1)
$(error expected one .tex file with a \documentclass, found: $(MAIN))
endif

FIGURES := $(wildcard python/make_figures.py)
FETCH   := $(shell grep -l '^\# fetch: on build$$' python/fetch_*.py 2>/dev/null)
AFTER   := $(shell grep -l '^\# run: after figures$$' python/*.py 2>/dev/null)
JULIA   := $(shell grep -l '^\# run: after figures$$' julia/*.jl 2>/dev/null)
RCHECKS := $(shell grep -l '^\# run: after figures$$' R/*.R 2>/dev/null)

pdf: $(MAIN).pdf

# Plain article class and an inline bibliography, so no BibTeX pass.
$(MAIN).pdf: $(MAIN).tex $(wildcard figures/*.pdf figures/*.tex)
	$(LATEX) $(LATEXOPTS) $<
	$(LATEX) $(LATEXOPTS) $<
	$(LATEX) $(LATEXOPTS) $<

data:
	$(foreach script,$(FETCH),uv run $(script) &&) true

figures: data
	$(if $(FIGURES),uv run $(FIGURES))
	$(foreach script,$(AFTER),uv run $(script) &&) true
	$(if $(JULIA),julia --project=julia -e 'using Pkg; Pkg.instantiate()')
	$(foreach script,$(JULIA),julia --project=julia $(script) &&) true
	$(foreach script,$(RCHECKS),Rscript $(script) &&) true

test:
	$(if $(FIGURES),uv run $(FIGURES) test)

# arXiv compiles the source itself, so a submission is the .tex and the figures it
# includes, laid out as here, not the PDF. Only the figures the note includes go in:
# figures/ can hold drawings the note no longer uses. A table the script writes into
# figures/ and the note \inputs goes in too. The same tree is packed twice, as a
# tarball and as a zip, for whichever the upload form takes.
ARXIV_SRC     := arxiv-src
ARXIV_CHECK   := arxiv-build
ARXIV_TARBALL := $(MAIN).tar.gz
ARXIV_ZIP     := $(MAIN).zip
INCLUDED      := $(addprefix figures/,$(shell grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]*\}' \
                   $(MAIN).tex | sed -E 's/.*\{//; s/\}//' | sort -u)) \
                 $(shell grep -oE '\\input\{figures/[^}]*\}' $(MAIN).tex | sed -E 's/.*\{//; s/\}//' | sort -u)

arxiv: $(ARXIV_TARBALL) $(ARXIV_ZIP)

$(ARXIV_TARBALL): $(MAIN).tex $(INCLUDED)
	rm -rf $(ARXIV_SRC)
	mkdir -p $(ARXIV_SRC)/figures
	cp $(MAIN).tex $(ARXIV_SRC)/
	$(if $(INCLUDED),cp $(INCLUDED) $(ARXIV_SRC)/figures/)
	tar czf $@ -C $(ARXIV_SRC) .
	tar tzf $@

$(ARXIV_ZIP): $(ARXIV_TARBALL)
	rm -f $@
	cd $(ARXIV_SRC) && zip -qr ../$@ .
	unzip -l $@

# Compile the tarball the way arXiv will: unpacked into an empty directory and built
# there, so a file the packaging forgot fails here rather than after posting. The tail
# of the log is printed if a run fails, since the error is among its last lines.
arxiv-check: $(ARXIV_TARBALL) $(ARXIV_ZIP)
	rm -rf $(ARXIV_CHECK)
	mkdir -p $(ARXIV_CHECK)
	tar xzf $< -C $(ARXIV_CHECK)
	cd $(ARXIV_CHECK) && for _ in 1 2 3; do \
	  $(LATEX) $(LATEXOPTS) $(MAIN).tex >/dev/null || { tail -n 40 $(MAIN).log; exit 1; }; \
	done
	@! grep -E 'Citation .* undefined|Reference .* undefined|There were undefined' \
	  $(ARXIV_CHECK)/$(MAIN).log || { echo "undefined references in the arXiv source"; exit 1; }
	@echo "arXiv source of $(MAIN) compiles on its own: $$(grep -o 'Output written on.*' $(ARXIV_CHECK)/$(MAIN).log)"

clean:
	rm -f $(addprefix $(MAIN).,aux log out toc)
	rm -rf $(ARXIV_SRC) $(ARXIV_CHECK) $(ARXIV_TARBALL) $(ARXIV_ZIP)

endif
