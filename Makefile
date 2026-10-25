# The one command that regenerates every result in every preprint:
#
#   make reproduce
#
# runs, in each folder of preprints/, `make figures` (the data, the figures and every
# number the text quotes), `make test` (the checks against them), `make pdf` and
# `make arxiv-check`. `make -C preprints/<name> reproduce` does the same for one preprint.
.PHONY: reproduce
reproduce:
	$(MAKE) -C preprints reproduce
