# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.25.0",
#     "matplotlib==3.11.0",
#     "numpy==2.4.6",
#     "scikit-learn==1.9.0",
# ]
# ///
r"""The Bayesian LASSO on the portfolio problem of ``paper/lasso.tex``.

    uvx marimo edit --sandbox docs/notebooks/bayesian_lasso.py

A Park--Casella Gibbs sampler on the note's test problem, set against the exact path
from ``lars_path``. Not part of the note; an answer to a question about it.
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import lars_path

    return lars_path, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Bayesian LASSO as a portfolio

    The note turns a mean–variance program into a least-squares one: with
    $\Sigma = LL^\top$, set $X = L^\top$ and $y = L^{-1}\mu$, so that

    $$\tfrac12\|y - X\beta\|^2 = \tfrac12\beta^\top\Sigma\beta - \mu^\top\beta + \text{const}.$$

    The Bayesian LASSO of Park and Casella (2008) puts a Laplace prior on each weight and a
    Gaussian likelihood on $y$. Here $y$ is not data, so the "likelihood" is the Markowitz
    objective in disguise, and the posterior is a Boltzmann distribution over portfolios,

    $$p(\beta) \propto \exp\!\Big(-\tfrac{1}{T}\big[\tfrac12\beta^\top\Sigma\beta - \mu^\top\beta
    + \kappa\|\beta\|_1\big]\Big),$$

    with the noise variance $\sigma^2 = T$ acting as a temperature. Three things follow
    before any sampling:

    - **The mode is the LASSO.** At every $\kappa$ the posterior mode is the corner-portfolio
      path of the note's long–short frontier; the CLA is the zero-temperature limit.
    - **$T$ cannot be learned.** There are as many pseudo-observations as assets, so the
      unpenalised fit is exact and the residual is zero; a prior on $\sigma^2$ would be fitted
      to nothing. It is fixed here, and a slider sets it.
    - **The randomness is in the weights, for fixed $\mu$ and $\Sigma$.** Estimation error in
      $\mu$ and $\Sigma$, the uncertainty that matters for a portfolio, is not in the model.

    What sampling adds is the posterior *mean*, and how it differs from the mode.
    """)
    return


@app.cell
def _(np):
    def factor_problem(seed=42, n_assets=20, n_days=50, n_factors=5):
        """(mu, Sigma) of the note's test problem, as built in make_figures.py."""
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((n_assets, n_factors)) / np.sqrt(n_assets)
        delta = rng.uniform(0.5, 2.0, n_factors) * n_assets
        d = rng.uniform(0.5, 2.0, n_assets)
        expected = rng.uniform(0.0, 1.0, n_assets)
        factor_returns = rng.standard_normal((n_days, n_factors)) * np.sqrt(delta)
        idiosyncratic = rng.standard_normal((n_days, n_assets)) * np.sqrt(d)
        returns = expected + factor_returns @ u.T + idiosyncratic
        return returns.mean(axis=0), np.diag(d) + (u * delta) @ u.T

    mu, sigma = factor_problem()
    n = mu.size
    return mu, n, sigma


@app.cell
def _(lars_path, mu, n, np, sigma):
    # The exact path. sklearn scales the squared error by 1/n, so its alpha is kappa / n.
    chol = np.linalg.cholesky(sigma)
    x, y = chol.T, np.linalg.solve(chol, mu)
    alphas, _, betas = lars_path(x, y, method="lasso")
    corner_kappas = n * alphas

    def lasso_at(kappas):
        """The LASSO solution at each kappa: the path is piecewise linear between corners."""
        return np.array([[np.interp(k, corner_kappas[::-1], b[::-1]) for b in betas] for k in kappas])

    return corner_kappas, lasso_at


@app.cell
def _(mu, n, np, sigma):
    def gibbs(kappas, temperature, sweeps=20000, burn=2000, seed=0):
        """Park--Casella Gibbs sampler with sigma^2 fixed at the temperature.

        Run for all kappas at once. Given the latent scales, the weights are Gaussian with
        precision (Sigma + D^-1) / T; given the weights, each inverse scale is inverse
        Gaussian with mean kappa / |beta_j| and shape kappa^2 / T. Returns
        draws x kappas x assets.
        """
        rng = np.random.default_rng(seed)
        kap = np.asarray(kappas, dtype=float)[:, None]
        inv_scale = np.ones((kap.size, n))
        draws = np.empty((sweeps - burn, kap.size, n))
        for sweep in range(sweeps):
            a = sigma + inv_scale[:, :, None] * np.eye(n)
            c = np.linalg.cholesky(a)
            mean = np.linalg.solve(a, np.broadcast_to(mu, inv_scale.shape)[..., None])[..., 0]
            z = rng.standard_normal(inv_scale.shape)[..., None]
            beta = mean + np.sqrt(temperature) * np.linalg.solve(np.swapaxes(c, 1, 2), z)[..., 0]
            inv_scale = rng.wald(kap / np.abs(beta), np.broadcast_to(kap**2 / temperature, beta.shape))
            if sweep >= burn:
                draws[sweep - burn] = beta
        return draws

    return (gibbs,)


@app.cell(hide_code=True)
def _(mo):
    temperature = mo.ui.slider(
        steps=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
        value=1e-2,
        label="temperature $T = \\sigma^2$",
        show_value=True,
    )
    temperature
    return (temperature,)


@app.cell
def _(corner_kappas, gibbs, lasso_at, np, temperature):
    # From just below the first corner, where one asset is in, to a little past the last.
    kappas = np.geomspace(0.98 * corner_kappas[0], 0.3 * corner_kappas[-2], 40)
    draws = gibbs(kappas, temperature.value)
    post_mean = draws.mean(axis=0)
    post_mode = lasso_at(kappas)
    return draws, kappas, post_mean, post_mode


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mode and mean along the path

    Left, the exact path: piecewise linear in $\kappa$, with a kink wherever an asset enters,
    and exact zeros until it does. Right, the posterior mean at the chosen temperature: smooth,
    no kinks, and no weight ever exactly zero. Lower the temperature and it collapses onto
    the left panel.
    """)
    return


@app.cell
def _(corner_kappas, kappas, plt, post_mean, post_mode):
    blue, red, amber, grey = "#1f4e79", "#c00000", "#b07800", "#9aa7b4"

    def style(ax, xlabel, ylabel):
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    _left.plot(kappas, post_mode, color=blue, lw=0.9)
    for _k in corner_kappas[1:-1]:
        _left.axvline(_k, color=grey, lw=0.4, alpha=0.6)
    _left.set_title("(a) posterior mode = LASSO path", fontsize=8.5)
    _right.plot(kappas, post_mean, color=red, lw=0.9)
    _right.set_title("(b) posterior mean", fontsize=8.5)
    for _ax in (_left, _right):
        _ax.set_xscale("log")
        _ax.invert_xaxis()
        style(_ax, r"penalty $\kappa$ (decreasing)", "weight")
    _fig.tight_layout()
    _fig
    return amber, blue, grey, red, style


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One asset where the LASSO says zero

    The Laplace prior has no atom at zero, so neither does the posterior. At a penalty where
    the LASSO leaves an asset out, the posterior density peaks at zero, which is why the
    mode sits there, but its mass spreads to one side and the mean follows it.
    """)
    return


@app.cell
def _(amber, blue, red, draws, kappas, np, plt, post_mean, post_mode, style):
    # The middle of the path, and the zero weight whose posterior mean strays furthest.
    _i = kappas.size // 2
    _zeros = np.flatnonzero(post_mode[_i] == 0)
    _j = _zeros[np.argmax(np.abs(post_mean[_i, _zeros]))]
    _fig, _ax = plt.subplots(figsize=(4.6, 3.0))
    _ax.hist(draws[:, _i, _j], bins=80, density=True, color=blue, alpha=0.6)
    _ax.axvline(0.0, color=amber, lw=1.5, label="mode (LASSO): 0")
    _ax.axvline(post_mean[_i, _j], color=red, lw=1.5, label=f"mean: {post_mean[_i, _j]:+.3f}")
    _ax.set_title(f"asset {_j} at $\\kappa$ = {kappas[_i]:.3f}", fontsize=8.5)
    _ax.legend(fontsize=7, frameon=False)
    style(_ax, "weight", "posterior density")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Is the mean a good portfolio?

    Left, sparsity: the mean holds nearly every asset from the start, where the mode holds
    one or two. Right, efficiency. At a given $\kappa$ the mode minimises the penalised
    objective, so the mean can only do worse on it. The fairer test drops the penalty:
    take the mean's gross exposure $c = \|\bar\beta\|_1$, and compare it with the best
    portfolio at that cap, which is the LASSO path at the point where $\|\beta\|_1 = c$.
    The gap is the Markowitz objective $\tfrac12 w^\top\Sigma w - \mu^\top w$ given up for
    the same leverage.
    """)
    return


@app.cell
def _(blue, grey, red, corner_kappas, kappas, lasso_at, mo, mu, np, plt, post_mean, post_mode, sigma, style):
    def markowitz(w):
        return 0.5 * np.einsum("...i,ij,...j->...", w, sigma, w) - w @ mu

    # The exact path, densely, as a function of its own gross exposure.
    _dense = lasso_at(np.linspace(corner_kappas[0], 0.0, 4000))
    _caps = np.abs(_dense).sum(axis=1)
    _mean_caps = np.abs(post_mean).sum(axis=1)
    _best = markowitz(_dense[np.searchsorted(_caps, _mean_caps).clip(max=_caps.size - 1)])
    _gap = markowitz(post_mean) - _best

    def _holdings(w):
        """Positions above 1% of the portfolio's gross exposure."""
        return (np.abs(w) > 0.01 * np.abs(w).sum(axis=1, keepdims=True)).sum(axis=1)

    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    _left.step(kappas, _holdings(post_mode), where="mid", color=grey, lw=1.2, label="mode (LASSO)")
    _left.plot(kappas, _holdings(post_mean), ".", ms=4, color=red, label="posterior mean")
    _left.set_xscale("log")
    _left.invert_xaxis()
    _left.legend(fontsize=7, frameon=False)
    _left.set_title("(a) holdings above 1% of gross exposure", fontsize=8.5)
    style(_left, r"penalty $\kappa$ (decreasing)", "assets held")
    _right.plot(_mean_caps, _gap, "o-", ms=2.5, lw=0.8, color=blue)
    _right.set_title("(b) objective given up at the same gross exposure", fontsize=8.5)
    style(_right, r"gross exposure $\|\bar\beta\|_1$", "gap")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Largest gap: **{_gap.max():.2e}**, smallest: **{_gap.min():.2e}**.")])
    return (markowitz,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checking the sampler

    An independent route to the same posterior mean: draw from the Gaussian part alone,
    $\mathcal N(\Sigma^{-1}\mu,\,T\Sigma^{-1})$, and reweight each draw by the prior,
    $\exp(-\kappa\|\beta\|_1/T)$. Self-normalised importance sampling is only practical
    where the prior is weak against the likelihood, so the check uses the smallest penalty
    on the grid.
    """)
    return


@app.cell
def _(draws, kappas, mo, mu, n, np, post_mean, sigma, temperature):
    _rng = np.random.default_rng(1)
    _k = kappas[-1]
    _t = temperature.value
    _prop = _rng.multivariate_normal(np.linalg.solve(sigma, mu), _t * np.linalg.inv(sigma), size=400_000)
    _logw = -_k * np.abs(_prop).sum(axis=1) / _t
    _w = np.exp(_logw - _logw.max())
    _is_mean = (_w[:, None] * _prop).sum(axis=0) / _w.sum()
    # The chain is autocorrelated, so its Monte Carlo error comes from batch means.
    _batches = draws[:, -1].reshape(50, -1, n).mean(axis=1)
    _se = _batches.std(axis=0, ddof=1) / np.sqrt(_batches.shape[0])
    _z = np.abs(_is_mean - post_mean[-1]) / _se
    mo.md(
        f"At $\\kappa$ = {_k:.4f}, the Gibbs and importance-sampling means of the {n} weights "
        f"differ by at most **{np.abs(_is_mean - post_mean[-1]).max():.1e}**, or "
        f"**{_z.max():.1f}** Monte Carlo standard errors of the chain (median {np.median(_z):.1f})."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What it gives you

    A single dense portfolio at each $(\kappa, T)$: a smoothed average of the corner portfolios
    near the mode, sitting off the LASSO path by an amount set by the temperature. With no
    breakpoints there is nothing to say about a path; the path belongs to the mode, and the
    mode is the LASSO. Put a hyperprior on $\kappa$, as Park and Casella do, and you get one
    portfolio averaged over penalties instead of a path at all.

    The Bayesian question worth asking of a portfolio is a different one: uncertainty in
    $\mu$ and $\Sigma$, where Bayes–Stein shrinkage and Black–Litterman live.
    """)
    return


if __name__ == "__main__":
    app.run()
