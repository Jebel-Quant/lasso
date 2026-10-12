# Run a target in every preprint. Each folder here holds one preprint with its own
# Makefile; `make pdf` builds them all, `make -C lasso pdf` just one.

PREPRINTS := $(patsubst %/Makefile,%,$(wildcard */Makefile))
TARGETS   := pdf figures test clean

.PHONY: $(TARGETS)

$(TARGETS):
	@for dir in $(PREPRINTS); do $(MAKE) -C $$dir $@ || exit 1; done
