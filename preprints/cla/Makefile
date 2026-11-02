# Build the cvxcla software paper for the Journal of Statistical Software. The rules are
# shared by every preprint and live in ../Makefile. The JSS class, BibTeX style and logo
# are bundled under cls/jss/, the body is in sections/, and the bibliography is BibTeX's,
# from refs/references.bib.
#
# Unlike the other preprints, this one's figures are committed: its scaling figures report
# wall-clock times, so CI does not redraw them. `make figures` redraws them all and runs
# every numerical check (make_figures.py all), and `make test` runs the Appendix C listing
# (check_listing.py), which CI also runs (.github/workflows/listing.yml). `make reproduce`
# does both and builds the PDF.
FIGURES      := make_figures.py
FIGURES_ARGS := all
TEST         := uv run check_listing.py
include ../Makefile

# The JSS submission: the PDF, the flat LaTeX source (the arXiv tree), the package source
# as released on PyPI, and the replication materials, laid out as the scripts expect them
# (data/ beside make_figures.py, sections/appendix-listing.tex beside check_listing.py).
# The package sdist is checked against PyPI's SHA-256 digest.
JSS_DIR   := jss-submission
JSS_ZIP   := cla-jss-submission.zip
SDIST     := cvxcla-2.7.1.tar.gz
SDIST_URL := https://files.pythonhosted.org/packages/11/f9/2d61715327fb4cbc76269790bb8a1b05ea2afc6c0b54058154299c662c56/$(SDIST)
SDIST_SHA := b6b4d34fc3840e96a00dc4b14ed69c273d01e6c54cb7e5fbb057b37ee3a17abb

.PHONY: jss
jss: $(MAIN).pdf $(ARXIV_TARBALL)
	rm -rf $(JSS_DIR) $(JSS_ZIP)
	mkdir -p $(JSS_DIR)/replication/data $(JSS_DIR)/replication/sections
	cp $(MAIN).pdf $(JSS_DIR)/
	cp -R $(ARXIV_SRC) $(JSS_DIR)/source
	curl -fsSL -o $(JSS_DIR)/$(SDIST) $(SDIST_URL)
	echo "$(SDIST_SHA)  $(JSS_DIR)/$(SDIST)" | shasum -a 256 -c -
	cp make_figures.py make_figures.py.lock check_listing.py check_listing.py.lock replication/README.md $(JSS_DIR)/replication/
	cp figures/data/sp500_pct_returns.parquet $(JSS_DIR)/replication/data/
	cp sections/appendix-listing.tex $(JSS_DIR)/replication/sections/
	cd $(JSS_DIR) && zip -qr ../$(JSS_ZIP) .
	unzip -l $(JSS_ZIP)
