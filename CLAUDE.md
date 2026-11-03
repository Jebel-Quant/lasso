# CLAUDE.md

## Frozen preprints

`preprints/lasso/` and `preprints/constrained_lasso/` are published in the *Journal of
Reproducible Statistics* (volume 1, articles 7 and 10) and are frozen. Do not change any
file in either folder, by any tool: not with Edit or Write, and not from the shell (no
`sed -i`, redirects, `mv`, `rm`, `git mv`, formatters or regenerated figures). Reading
and running them, e.g. `make -C preprints/lasso reproduce`, is fine. If a change there
seems necessary, stop and ask.
