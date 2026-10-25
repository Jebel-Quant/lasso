# Build the cvxcla software paper for the Journal of Statistical Software. The rules are
# shared by every preprint and live in ../Makefile. The JSS class, BibTeX style and logo
# are bundled under cls/jss/, the body is in sections/, and the bibliography is BibTeX's,
# from refs/references.bib.
#
# Unlike the other preprints, this one's figures are committed: its scaling figures report
# wall-clock times, so CI does not redraw them. Redraw them by hand with
# `uv run make_figures.py all`, and run the Appendix C listing with
# `uv run check_listing.py`, which CI also runs (.github/workflows/listing.yml).
include ../Makefile
