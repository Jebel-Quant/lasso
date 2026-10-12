# lasso

## The efficient frontier from a LASSO solver

[`paper/`](paper) holds the note *The Efficient Frontier from a LASSO Solver*
([arXiv:2609.37108](https://arxiv.org/abs/2609.37108)), its LaTeX source and the code and
data behind it: one script that reproduces every figure and number, a script that fetches
the S&P 500 returns it uses, and an R check. See [`paper/README.md`](paper/README.md).

[`docs/notebooks/bayesian_lasso.py`](docs/notebooks/bayesian_lasso.py) is a marimo
notebook that sits beside the note rather than in it: a Park–Casella Gibbs sampler on the
note's test problem, set against the exact path from `lars_path`. Open it with
`uvx marimo edit --sandbox docs/notebooks/bayesian_lasso.py`.
