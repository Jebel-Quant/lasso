# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cvxcla==2.0.1",
#     "cvxpy==1.9.2",
#     "matplotlib==3.11.0",
#     "numpy==2.4.6",
#     "osqp==1.1.3",
#     "pandas==3.0.3",
#     "pyarrow==24.0.0",
#     "pytest==8.4.2",
#     "qpsolvers==4.12.0",
#     "quadprog==0.1.13",
#     "scikit-learn==1.9.0",
#     "scipy==1.17.1",
# ]
# ///
# The pytest tests live in this script, so assert is the test idiom here too.
# ruff: noqa: S101
r"""The efficient frontier, computed by a stock LASSO solver.

Companion code for ``lasso.tex``. Everything here rests on the two
substitutions of the perspective paper: with :math:`\Sigma = X^\top X` and
:math:`\mu = X^\top y`, a mean--variance program *is* a least-squares program, so
scikit-learn's ``lars_path`` -- which knows nothing of portfolios -- traces an
efficient frontier.

    uv run make_figures.py        # the five figures, and the numbers the note quotes
    uv run make_figures.py test   # the same numbers as pytest checks, with tolerances
    uv run make_figures.py export # the test problem as CSV, for R/long_short.R

Five figures (``figures/*.pdf``) and the numerical checks behind the note, all
seeded and offline. The test problem is the one behind Figure 1 of the
perspective paper: the seed-42, 20-asset, 5-factor model of ``statsci/make_figures.py``,
reproduced here so this directory stands alone.
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import osqp
import pytest
import scipy.sparse as sparse
from cvxcla import CLA
from qpsolvers import solve_qp
from scipy.optimize import nnls
from sklearn.linear_model import lars_path, lars_path_gram

SEED = 42
OUT = Path(__file__).parent / "figures"

# A support is read off a nonnegative weight vector; anything below this is a zero that
# the solver happened to write as 1e-17.
SUPPORT_TOL = 1e-9

# The long--short baseline needs the l1 ball, which is conic rather than quadratic, so it
# goes to an interior-point solver asked for everything it has. That accuracy -- around
# 1e-7 at a corner, where the active set is on the point of changing and the central path
# is worst conditioned -- is the baseline's floor, not the homotopy's; the mid-segment
# check is the sharp one. The long-only baseline has only linear constraints and goes to
# a dual active-set QP instead, which is exact to rounding.
CONIC = {"solver": cp.CLARABEL, "tol_gap_abs": 1e-13, "tol_gap_rel": 1e-13, "tol_feas": 1e-13}


@dataclass
class Check:
    """One reported agreement number: what was compared, and how far apart."""

    name: str
    measured: float
    tol: float
    # The individual discrepancies behind ``measured``, one per point compared, kept for
    # the agreement figure; ``measured`` is their maximum.
    per_point: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    def __str__(self) -> str:
        """One line: pass or fail, the measured number, and its tolerance."""
        flag = "ok " if self.measured <= self.tol else "FAIL"
        return f"  [{flag}] {self.name}: {self.measured:.2e}  (tol {self.tol:.0e})"


# --------------------------------------------------------------------------------------
# The problem, and the substitution that turns it into a regression
# --------------------------------------------------------------------------------------
@dataclass
class FactorData:
    """The simulated problem in full: the model's parts and the returns drawn from it."""

    d: np.ndarray  # idiosyncratic variances
    u: np.ndarray  # loadings, n_assets x n_factors
    delta: np.ndarray  # factor variances
    returns: np.ndarray  # n_days x n_assets


def factor_data(seed: int = SEED, n_assets: int = 20, n_days: int = 50, n_factors: int = 5) -> FactorData:
    """Simulate the K-factor problem of the perspective paper."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_assets, n_factors)) / np.sqrt(n_assets)
    delta = rng.uniform(0.5, 2.0, n_factors) * n_assets
    d = rng.uniform(0.5, 2.0, n_assets)
    expected = rng.uniform(0.0, 1.0, n_assets)

    factor_returns = rng.standard_normal((n_days, n_factors)) * np.sqrt(delta)
    idiosyncratic = rng.standard_normal((n_days, n_assets)) * np.sqrt(d)
    returns = expected + factor_returns @ u.T + idiosyncratic
    return FactorData(d=d, u=u, delta=delta, returns=returns)


def factor_problem(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """(mu, Sigma): the sample mean of the returns, and the model covariance."""
    f = factor_data(seed)
    return f.returns.mean(axis=0), np.diag(f.d) + (f.u * f.delta) @ f.u.T


def as_regression(mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Return (X, y) with :math:`X^\top X = \Sigma` and :math:`X^\top y = \mu`.

    Any positive definite covariance factors as :math:`\Sigma = LL^\top`, so
    :math:`X = L^\top` is a design with the right Gram matrix, and the response solving
    :math:`X^\top y = Ly = \mu` comes from one triangular solve. The regression this
    builds has as many observations as assets and no noise interpretation whatever; it
    is a change of variables, not a statistical model.
    """
    chol = np.linalg.cholesky(sigma)
    return chol.T, np.linalg.solve(chol, mu)


def _corners(x: np.ndarray, y: np.ndarray, positive: bool) -> np.ndarray:
    return lars_path(x, y, method="lasso", positive=positive)[2]


def check_cholesky_free(mu: np.ndarray, sigma: np.ndarray, f: FactorData) -> list[Check]:
    r"""The routes of the note that avoid the Cholesky factor give the same paths.

    Only :math:`X^\top X` and :math:`X^\top y` enter the LASSO, so any square root of
    :math:`\Sigma` will do. Each route is compared with the Cholesky route on the same
    covariance, plain path and nonnegative path alike, corner by corner.
    """
    x, y = as_regression(mu, sigma)
    # The factor model's own square root, and an explicit response, with no factorisation.
    xf = np.vstack([np.diag(np.sqrt(f.d)), np.sqrt(f.delta)[:, None] * f.u.T])
    yf = np.concatenate([mu / np.sqrt(f.d), np.zeros(f.delta.size)])
    factor_gap = max(float(np.abs(_corners(xf, yf, pos) - _corners(x, y, pos)).max()) for pos in (False, True))
    gram_gap = max(
        float(
            np.abs(
                lars_path_gram(Xy=mu, Gram=sigma, n_samples=mu.size, method="lasso", positive=pos)[2]
                - _corners(x, y, pos)
            ).max()
        )
        for pos in (False, True)
    )
    # Centred returns against the Cholesky factor of the sample covariance they define.
    t = f.returns.shape[0]
    xs = (f.returns - f.returns.mean(axis=0)) / np.sqrt(t - 1)
    ys = np.linalg.lstsq(xs.T, mu)[0]
    xc, yc = as_regression(mu, xs.T @ xs)
    centred_gap = max(float(np.abs(_corners(xs, ys, pos) - _corners(xc, yc, pos)).max()) for pos in (False, True))
    # The shifted long-only path needs only X^T d = 1, so it works with a tall X too.
    shifted_gap = max(
        float(np.abs(long_only_path(xf, yf) - long_only_path(x, y)).max()),
        float(np.abs(long_only_path(xs, ys) - long_only_path(xc, yc)).max()),
    )
    return [
        Check("factor-model square root vs Cholesky, both paths", factor_gap, 1e-12),
        Check("lars_path_gram vs Cholesky, both paths", gram_gap, 1e-12),
        Check("centred returns vs Cholesky of sample covariance", centred_gap, 1e-12),
        Check("shifted long-only path, tall X vs Cholesky", shifted_gap, 1e-12),
    ]


# --------------------------------------------------------------------------------------
# Route A -- the gross-exposure-constrained (long-short) frontier, Theorem 1
# --------------------------------------------------------------------------------------
def long_short_frontier(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corner portfolios of the gross-exposure-constrained frontier, and their caps.

    The plain LASSO path. Theorem 1 says each of its vertices solves the
    mean-variance program capped at its own gross exposure, so the columns returned
    *are* the corner portfolios, in order of decreasing tilt.
    """
    _, _, betas = lars_path(x, y, method="lasso")
    return betas, np.abs(betas).sum(axis=0)


def check_long_short(mu: np.ndarray, sigma: np.ndarray, betas: np.ndarray, caps: np.ndarray) -> tuple[Check, Check]:
    """Re-solve the capped QP independently, at the corners and between them."""
    n = mu.shape[0]

    def capped(c: float) -> np.ndarray:
        w = cp.Variable(n)
        cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(sigma)) - mu @ w),
            [cp.norm1(w) <= c],
        ).solve(**CONIC)
        return np.asarray(w.value)

    at_corner = np.array([float(np.abs(capped(caps[k]) - betas[:, k]).max()) for k in range(1, betas.shape[1])])
    # Midpoints of each segment. The homotopy claims the whole segment and not just its
    # ends, and a cap strictly inside one is where the baseline is best conditioned.
    mid = np.array(
        [
            float(np.abs(capped(0.5 * (caps[k - 1] + caps[k])) - 0.5 * (betas[:, k - 1] + betas[:, k])).max())
            for k in range(1, betas.shape[1])
        ]
    )
    return (
        Check("long-short corners vs capped conic QP", at_corner.max(), 1e-6, at_corner),
        Check("long-short mid-segment vs capped conic QP", mid.max(), 1e-9, mid),
    )


# --------------------------------------------------------------------------------------
# Route A' -- the same path read at fixed leverage, with the tilt swept
# --------------------------------------------------------------------------------------
LEVERAGE = 2.0  # the illustrative gross-exposure budget: 150/50, give or take


def fixed_leverage_frontier(betas: np.ndarray, caps: np.ndarray, c: float = LEVERAGE) -> tuple[np.ndarray, np.ndarray]:
    r"""Corner portfolios at *fixed* gross exposure ``c``, and the tilts they solve at.

    Nobody sweeps a leverage budget; the budget is fixed and the tilt on :math:`\mu` is
    swept. That is the same curve read the other way along. Substituting
    :math:`w = \lambda v` in the tilted program and dividing the objective by
    :math:`\lambda^2` leaves the untilted one capped at :math:`c/\lambda`, so a vertex
    :math:`\beta` of the LASSO path with :math:`t = \|\beta\|_1` rescales to
    :math:`w = (c/t)\beta`, optimal at tilt :math:`\lambda = c/t`. Same vertices, each
    blown up to gross exposure ``c`` -- the long-short counterpart of the rescaling
    Proposition 3 performs long only, where the budget row makes the sum the norm.
    """
    keep = caps > SUPPORT_TOL  # the origin carries no direction to rescale
    return betas[:, keep] * (c / caps[keep]), c / caps[keep]


def tilted_capped(mu: np.ndarray, sigma: np.ndarray, lam: float, c: float) -> np.ndarray:
    """The gross-exposure-capped portfolio at tilt ``lam``, from a conic program."""
    n = mu.shape[0]
    w = cp.Variable(n)
    cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(sigma)) - lam * (mu @ w)),
        [cp.norm1(w) <= c],
    ).solve(**CONIC)
    return np.asarray(w.value)


def check_fixed_leverage(
    mu: np.ndarray, sigma: np.ndarray, betas: np.ndarray, caps: np.ndarray, c: float = LEVERAGE
) -> tuple[Check, Check]:
    """Re-solve the tilted, capped program independently, at the corners and between them.

    The mid-segment point is taken on the path, in ``beta``, and only then rescaled: it
    is ``beta`` that is affine in the cap, not the rescaled portfolio in the tilt.
    """
    weights, tilts = fixed_leverage_frontier(betas, caps, c)
    at_corner = np.array(
        [float(np.abs(tilted_capped(mu, sigma, lam, c) - weights[:, k]).max()) for k, lam in enumerate(tilts)]
    )
    mid = []
    for k in range(1, betas.shape[1]):
        half = 0.5 * (betas[:, k - 1] + betas[:, k])
        t = np.abs(half).sum()
        if t <= SUPPORT_TOL:
            continue
        mid.append(float(np.abs(tilted_capped(mu, sigma, c / t, c) - half * (c / t)).max()))
    # The corner number inherits the conic baseline's floor, amplified by the c/t
    # rescaling, which is largest exactly where the baseline is weakest. Mid-segment is
    # the sharp comparison, as it is for the uncapped sweep above.
    return (
        Check("fixed-leverage corners vs tilted conic QP", at_corner.max(), 1e-4, at_corner),
        Check("fixed-leverage mid-segment vs tilted conic QP", max(mid), 1e-9, np.array(mid)),
    )


# --------------------------------------------------------------------------------------
# Route B -- the long-only, fully invested frontier, Proposition 3
# --------------------------------------------------------------------------------------
def shift_direction(x: np.ndarray) -> np.ndarray:
    r"""A solution :math:`d` of :math:`X^\top d = \mathbf{1}`.

    Moving ``y`` by :math:`-\nu d` adds :math:`\nu\mathbf{1}^\top v` to the least-squares
    objective, up to a constant.

    Only :math:`X^\top d = \mathbf{1}` matters, so any solution will do. For square
    nonsingular :math:`X` it is :math:`(X^\top)^{-1}\mathbf{1}`; for a tall :math:`X` of full
    column rank (a factor square root, centred returns) least squares returns the
    minimum-norm one, :math:`(X^\top)^{+}\mathbf{1}`.
    """
    return np.linalg.lstsq(x.T, np.ones(x.shape[1]))[0]


def last_corner(x: np.ndarray, y: np.ndarray) -> float:
    r"""The budget multiplier :math:`\nu` of the long-only path's last corner, in closed form.

    Past the last corner the support is that of the minimum-variance portfolio, i.e. of
    the :math:`\nu\to-\infty` limit :math:`\arg\min_{u\ge0}\|Xu-d\|`. On that final
    segment :math:`v_S = a - \nu b` with :math:`a = \Sigma_{SS}^{-1}\mu_S` and
    :math:`b = \Sigma_{SS}^{-1}\mathbf{1}`. The segment stays optimal while every held weight
    is positive, :math:`\nu < a_i/b_i`, and no other asset wants in,
    :math:`(\mu - \Sigma v)_j \le \nu`, i.e. :math:`\nu \le (\mu_j - \Sigma_{jS}a)/h_j` with
    :math:`h_j = 1 - \Sigma_{jS}b < 0`. The smallest of these bounds is where the segment,
    and hence the last corner, begins.
    """
    sigma, mu = x.T @ x, x.T @ y
    support = nnls(x, shift_direction(x))[0] > SUPPORT_TOL
    s, out = np.flatnonzero(support), np.flatnonzero(~support)
    a = np.linalg.solve(sigma[np.ix_(s, s)], mu[s])
    b = np.linalg.solve(sigma[np.ix_(s, s)], np.ones(s.size))
    h = 1.0 - sigma[np.ix_(out, s)] @ b
    bounds = [a[b > 0] / b[b > 0]]
    if out.size:
        num = mu[out] - sigma[np.ix_(out, s)] @ a
        bounds.append(num[h < 0] / h[h < 0])
    return float(np.concatenate(bounds).min())


def shift_for(x: np.ndarray, y: np.ndarray) -> float:
    """A shift nu0 safely below the last corner.

    As far below it again as it lies from zero, and at least max |mu_j| below, so the
    rule scales with mu.
    """
    nu_last = last_corner(x, y)
    return nu_last - max(abs(nu_last), float(np.abs(x.T @ y).max()))


def long_only_path(x: np.ndarray, y: np.ndarray, nu0: float | None = None) -> np.ndarray:
    r"""The nonnegative LASSO path of the *shifted* response; budgets ``v`` in columns.

    Proposition 3 substitutes :math:`w = \lambda v`, which turns the fully invested
    frontier at tilt :math:`\lambda` into nonnegative least squares budgeted at
    :math:`t = 1/\lambda`; since :math:`v \ge 0` makes the budget the :math:`\ell_1`
    norm, the nonnegative LASSO traces it, with the budget multiplier :math:`\nu` as its
    penalty. A LASSO solver stops at :math:`\nu = 0`, short of the minimum-variance end,
    because it takes no negative penalty. Shifting the response first removes the stop:

    .. math:: \tfrac12\|Xv - (y - \nu_0 d)\|^2 + \nu'\mathbf{1}^\top v
              = \tfrac12\|Xv - y\|^2 + (\nu_0 + \nu')\mathbf{1}^\top v + \text{const},

    so the path in :math:`\nu' \ge 0` covers every multiplier :math:`\nu \ge \nu_0`.
    Column 0 is the origin; the last column is where the path ends, at
    :math:`\nu = \nu_0`, a point of the frontier's final segment but not a corner.
    """
    if nu0 is None:
        nu0 = shift_for(x, y)
    # lars_path stops after max_iter = 500 steps by default. The long-only path takes about
    # one step per asset, so a large problem needs a higher cap, or it ends early.
    _, _, v = lars_path(x, y - nu0 * shift_direction(x), method="lasso", positive=True, max_iter=10 * x.shape[1])
    return v


def min_variance_end(sigma: np.ndarray, v_end: np.ndarray) -> np.ndarray:
    r"""The minimum-variance portfolio, in closed form from the final segment's support.

    Past the last corner the support :math:`S` no longer changes and
    :math:`v_S = \Sigma_{SS}^{-1}(\mu_S - \nu\mathbf{1})`, so as :math:`\nu\to-\infty`
    the portfolio tends to :math:`\Sigma_{SS}^{-1}\mathbf{1}`, normalised.
    """
    s = v_end > SUPPORT_TOL
    b = np.linalg.solve(sigma[np.ix_(s, s)], np.ones(int(s.sum())))
    w = np.zeros(sigma.shape[0])
    w[s] = b / b.sum()
    return w


def long_only_frontier(
    x: np.ndarray, y: np.ndarray, sigma: np.ndarray, nu0: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corner portfolios of the long-only frontier, their tilts, and its minimum-variance end.

    Each vertex of the shifted path divided by its own sum is a corner portfolio, at
    tilt 1 / sum. ``nu0`` has to lie below the last corner, i.e. the path has to end on
    the final segment. By default it is set from the closed form of ``last_corner``; the
    support check below stays as a guard.
    """
    v = long_only_path(x, y, nu0)
    limit, _ = nnls(x, shift_direction(x))
    if set(np.flatnonzero(v[:, -1] > SUPPORT_TOL)) != set(np.flatnonzero(limit > SUPPORT_TOL)):
        msg = "the shifted path does not end on the final segment"
        raise ValueError(msg)
    corners = v[:, 1:-1]  # drop the origin and the end at nu0
    t = corners.sum(axis=0)
    return corners / t, 1.0 / t, min_variance_end(sigma, v[:, -1])


def tilted(mu: np.ndarray, sigma: np.ndarray, lam: float) -> np.ndarray:
    """The fully invested, long-only frontier portfolio at tilt ``lam``, from a QP.

    Only linear constraints here, so the baseline is a dual active-set QP
    (Goldfarb--Idnani, via ``quadprog``) rather than an interior-point method: it is
    exact to rounding at a corner, which is where the comparison has to be made.
    """
    n = mu.shape[0]
    return np.asarray(
        solve_qp(
            P=sigma,
            q=-lam * mu,
            G=-np.eye(n),
            h=np.zeros(n),
            A=np.ones((1, n)),
            b=np.ones(1),
            solver="quadprog",
        )
    )


def check_long_only(name: str, mu: np.ndarray, sigma: np.ndarray, weights: np.ndarray, tilts: np.ndarray) -> Check:
    """Re-solve the tilted, fully invested, long-only QP independently at each corner."""
    gaps = np.array([float(np.abs(tilted(mu, sigma, lam) - weights[:, k]).max()) for k, lam in enumerate(tilts)])
    return Check(name, gaps.max(), 1e-9, gaps)


def check_long_only_segments(x: np.ndarray, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> Check:
    """The same, at points strictly inside each segment of the shifted path.

    The path is affine in the budget ``t``, not in the tilt, so a midpoint is taken
    between consecutive vertices of ``v`` and only then rescaled. The segment out of the
    origin is skipped: every point on it is the same single-asset portfolio.
    """
    v = long_only_path(x, y)
    gaps = []
    for k in range(2, v.shape[1]):
        mid = 0.5 * (v[:, k - 1] + v[:, k])
        t = mid.sum()
        gaps.append(float(np.abs(tilted(mu, sigma, 1.0 / t) - mid / t).max()))
    return Check("long-only mid-segment vs tilted QP", max(gaps), 1e-9, np.array(gaps))


# --------------------------------------------------------------------------------------
# KKT certificates -- optimality checked on the path's own output, no second solver
# --------------------------------------------------------------------------------------
def kkt_capped(mu: np.ndarray, sigma: np.ndarray, w: np.ndarray, kappa: float) -> float:
    r"""Relative KKT residual of ``w`` for the penalised form of the gross-exposure cap.

    The problem is :math:`\min \tfrac12 w^\top\Sigma w - \mu^\top w + \kappa\|w\|_1`.

    With :math:`g = \mu - \Sigma w`, optimality is :math:`g_j = \kappa\,\mathrm{sign}(w_j)`
    on the support and :math:`|g_j| \le \kappa` off it. The worst violation is divided by
    :math:`\|\mu\|_\infty`, so the number is free of the units of the returns.
    """
    g = mu - sigma @ w
    held = np.abs(w) > SUPPORT_TOL
    on = np.abs(g[held] - kappa * np.sign(w[held])).max(initial=0.0)
    off = np.maximum(np.abs(g[~held]) - kappa, 0.0).max(initial=0.0)
    return float(max(on, off) / np.abs(mu).max())


def kkt_long_only(mu: np.ndarray, sigma: np.ndarray, w: np.ndarray, lam: float) -> float:
    r"""Relative KKT residual of ``w`` for the fully invested long-only program at tilt ``lam``.

    With :math:`g = \Sigma w - \lambda\mu`, optimality is :math:`g_j = \gamma` on the
    support and :math:`g_j \ge \gamma` off it, for one budget multiplier :math:`\gamma`,
    together with :math:`\mathbf 1^\top w = 1` and :math:`w \ge 0`. :math:`\gamma` is read
    off the support; the worst violation is divided by the size of the gradient terms.
    """
    g = sigma @ w - lam * mu
    held = w > SUPPORT_TOL
    gamma = g[held].mean()
    scale = max(float(np.abs(sigma @ w).max()), lam * float(np.abs(mu).max()))
    residual = max(
        np.abs(g[held] - gamma).max(),
        np.maximum(gamma - g[~held], 0.0).max(initial=0.0),
        abs(w.sum() - 1.0) * scale,
        max(0.0, -w.min()) * scale,
    )
    return float(residual / scale)


def long_short_multipliers(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The penalty nu at each column of the long-short path (lars_path reports nu / rows)."""
    alphas, _, _ = lars_path(x, y, method="lasso")
    return x.shape[0] * alphas


def check_kkt(
    mu: np.ndarray,
    sigma: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    betas: np.ndarray,
    caps: np.ndarray,
    w_long: np.ndarray,
    tilts: np.ndarray,
    w_minvar: np.ndarray,
) -> tuple[Check, Check, Check]:
    """KKT certificates for the three settings, on the portfolios the note reports.

    Long-short: each column with its own penalty. Fixed leverage: the same column
    rescaled to w = (c/t) beta at tilt lam = c/t, whose cap multiplier is lam * nu.
    Long only: every corner at its tilt, and the minimum-variance end at tilt 0.
    """
    nus = long_short_multipliers(x, y)
    ls = np.array([kkt_capped(mu, sigma, betas[:, k], nus[k]) for k in range(betas.shape[1])])
    keep = caps > SUPPORT_TOL
    weights, lams = fixed_leverage_frontier(betas, caps)
    fl = np.array(
        [
            kkt_capped(lam * mu, sigma, weights[:, k], lam * nu)
            for k, (lam, nu) in enumerate(zip(lams, nus[keep], strict=True))
        ]
    )
    lo = np.array(
        [kkt_long_only(mu, sigma, w_long[:, k], lam) for k, lam in enumerate(tilts)]
        + [kkt_long_only(mu, sigma, w_minvar, 0.0)]
    )
    return (
        Check("long-short KKT residual", ls.max(), 1e-12, ls),
        Check("fixed-leverage KKT residual", fl.max(), 1e-12, fl),
        Check("long-only KKT residual (corners and minimum-variance end)", lo.max(), 1e-12, lo),
    )


def long_only_kkt(mu: np.ndarray, sigma: np.ndarray, w: np.ndarray, tilts: np.ndarray, w_minvar: np.ndarray) -> float:
    """The worst long-only KKT residual over the corners and the minimum-variance end."""
    return max(
        max(kkt_long_only(mu, sigma, w[:, k], lam) for k, lam in enumerate(tilts)),
        kkt_long_only(mu, sigma, w_minvar, 0.0),
    )


# The margin rule of shift_for, nu0 = nu_last - delta * max(|nu_last|, max|mu_j|), with
# delta = 1. Too small a delta puts the end of the path on top of the last corner, and
# the corner is lost; too large a delta costs accuracy in proportion to |nu0|.
MARGINS = (1e-4, 1e-2, 1.0, 10.0, 100.0)


def margin_sweep(
    x: np.ndarray, y: np.ndarray, sigma: np.ndarray, margins: tuple[float, ...] = MARGINS
) -> list[tuple[float, int, float]]:
    """Algorithm 1 at several margins delta: (delta, corners found, worst KKT residual)."""
    mu = x.T @ y
    nu_last = last_corner(x, y)
    scale = max(abs(nu_last), float(np.abs(mu).max()))
    rows = []
    for delta in margins:
        try:
            w, tilts, w_minvar = long_only_frontier(x, y, sigma, nu0=nu_last - delta * scale)
        except ValueError:  # the path stopped short of the final segment
            rows.append((delta, 0, float("inf")))
            continue
        rows.append((delta, w.shape[1], long_only_kkt(mu, sigma, w, tilts, w_minvar)))
    return rows


def check_against_cla(mu: np.ndarray, sigma: np.ndarray, found: np.ndarray) -> tuple[Check, str]:
    """Match the recovered corner portfolios to the turning points the CLA returns.

    The Critical Line Algorithm is the other side of the identity, and ``cvxcla`` runs
    it on the same (mu, Sigma) without ever forming X or y. Agreement here is the whole
    claim of the note in one number.
    """
    n = mu.shape[0]
    cla = CLA(
        covariance=sigma,
        mean=mu,
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
        a=np.ones((1, n)),
        b=np.ones(1),
    )
    turning = cla.frontier.weights
    # For each turning point, the closest of the portfolios recovered here. Taken this
    # way round the number answers the question asked -- is every corner of the frontier
    # recovered. (cvxcla lists the first turning point twice; both copies match it.)
    gaps = np.abs(turning[:, None, :] - found.T[None, :, :]).max(axis=2).min(axis=1)
    summary = f"{turning.shape[0]} CLA turning points, all recovered to {gaps.max():.1e}"
    return Check("cvxcla turning points vs recovered corners", float(gaps.max()), 1e-12), summary


def budget_multiplier(mu: np.ndarray, sigma: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""The multiplier of the budget row at each column of ``v``, read off stationarity.

    On the support of a solution of
    :math:`\min \tfrac12\|Xv-y\|^2 + \nu\mathbf{1}^\top v,\ v \ge 0`, the gradient
    vanishes, so :math:`X^\top(y - Xv) = \mu - \Sigma v` equals :math:`\nu` in every
    coordinate. Where :math:`\nu \ge 0` a plain LASSO solver reaches the point; where it
    is negative only the shifted path does. Averaged over the support to wash out
    rounding.
    """
    active = v > SUPPORT_TOL
    grad = mu[:, None] - sigma @ v
    return (grad * active).sum(axis=0) / active.sum(axis=0)


def long_only_curve(x: np.ndarray, y: np.ndarray, per_segment: int = 40) -> np.ndarray:
    """The whole long-only branch, densely, as budgets ``v`` in columns.

    Between vertices the path is affine in the budget, so it is filled in exactly by
    interpolation. Past its end at ``nu0`` it stays on its final segment, so it is
    continued along the same line. Returned in order of decreasing tilt.
    """
    v = long_only_path(x, y)
    s = np.linspace(0.0, 1.0, per_segment + 1)[1:]
    inner = np.column_stack([v[:, [k - 1]] * (1 - s) + v[:, [k]] * s for k in range(1, v.shape[1])])
    tail = v[:, [-1]] + (v[:, [-1]] - v[:, [-2]]) * np.geomspace(1e-3, 1e4, 200)
    return np.column_stack([inner, tail])


def tangency(mu: np.ndarray, sigma: np.ndarray, v: np.ndarray, rate: float = 0.0) -> np.ndarray:
    r"""The maximum-Sharpe portfolio for risk-free ``rate``, read off the path.

    :math:`\tfrac12\|Xv-y\|^2 + \nu\mathbf{1}^\top v = \tfrac12 v^\top\Sigma v
    - (\mu - \nu\mathbf{1})^\top v + \text{const}`, so the point of the path where the
    budget multiplier equals :math:`r` minimises variance against *excess* return at rate
    :math:`r`, and normalised it maximises the Sharpe ratio. On a segment ``v`` is affine
    in the multiplier, so the point is found by linear interpolation between the two
    vertices that bracket it.
    """
    nu = budget_multiplier(mu, sigma, v[:, 1:])
    k = int(np.argmax(nu < rate))  # first vertex past the rate; nu decreases along the path
    s = (nu[k - 1] - rate) / (nu[k - 1] - nu[k])
    point = v[:, k] + s * (v[:, k + 1] - v[:, k])
    return point / point.sum()


def max_sharpe_qp(mu: np.ndarray, sigma: np.ndarray, rate: float = 0.0) -> np.ndarray:
    """The long-only maximum-Sharpe portfolio from the textbook QP, for comparison.

    Minimise u' Sigma u subject to (mu - rate)' u = 1, u >= 0, then normalise.
    """
    n = mu.shape[0]
    u = np.asarray(
        solve_qp(
            P=sigma, q=np.zeros(n), G=-np.eye(n), h=np.zeros(n), A=(mu - rate)[None, :], b=np.ones(1), solver="quadprog"
        )
    )
    return u / u.sum()


# --------------------------------------------------------------------------------------
# Robustness -- the long-only recipe across sizes and seeds
# --------------------------------------------------------------------------------------
ROBUST_SIZES = (20, 50, 100, 500)
ROBUST_SEEDS = (1, 2, 3)


@dataclass
class RobustRow:
    """One robustness case: problem size, seed, and how the shifted path fared."""

    n: int
    seed: int
    corners: int
    nu_last: float  # the last corner's multiplier, from the closed form
    corner_gap: float  # worst corner against the independent QP at the same tilt
    minvar_gap: float  # closed-form minimum-variance end against the minimum-variance QP
    kkt: float  # worst KKT residual over the corners and the minimum-variance end


def robustness(sizes: tuple[int, ...] = ROBUST_SIZES, seeds: tuple[int, ...] = ROBUST_SEEDS) -> list[RobustRow]:
    """Run the long-only recipe on factor models of several sizes and seeds.

    Each problem is the test problem's construction at another size and seed: five
    factors, the model covariance, and mu the sample mean of 50 simulated days. The shift
    comes from the closed-form last corner. Every corner is then compared with an
    independent active-set QP at the same tilt.
    """
    rows = []
    for n in sizes:
        for seed in seeds:
            f = factor_data(seed=seed, n_assets=n)
            mu, sigma = f.returns.mean(axis=0), np.diag(f.d) + (f.u * f.delta) @ f.u.T
            x, y = as_regression(mu, sigma)
            w, tilts, w_minvar = long_only_frontier(x, y, sigma)
            corner_gap = max(float(np.abs(tilted(mu, sigma, lam) - w[:, k]).max()) for k, lam in enumerate(tilts))
            minvar_gap = float(np.abs(tilted(mu, sigma, 0.0) - w_minvar).max())
            rows.append(
                RobustRow(
                    n,
                    seed,
                    w.shape[1],
                    last_corner(x, y),
                    corner_gap,
                    minvar_gap,
                    long_only_kkt(mu, sigma, w, tilts, w_minvar),
                )
            )
    return rows


# --------------------------------------------------------------------------------------
# Real data -- the S&P 500 sample of ../cla, through the centred-returns route
# --------------------------------------------------------------------------------------
# The frozen S&P 500 snapshot: next to this script in the public repository
# (github.com/Jebel-Quant/lasso, under paper/), or in the CLA paper's directory of the working repository.
_HERE = Path(__file__).parent
SNAPSHOT_SHA256 = "b5faa5222555f28d77bad5404565297bb25bbe92b0b813f416cd2bebac79937e"
SP500 = next(
    (
        p
        for p in (
            _HERE / "data" / "sp500_pct_returns.parquet",
            _HERE.parent / "cla" / "figures" / "data" / "sp500_pct_returns.parquet",
        )
        if p.exists()
    ),
    _HERE / "data" / "sp500_pct_returns.parquet",
)
GRID = 100  # tilts in the grid-of-QPs baseline the timing compares against


def warm_grid(mu: np.ndarray, sigma: np.ndarray, tilts: np.ndarray) -> tuple[np.ndarray, float]:
    """The long-only portfolios at ``tilts`` from one warm-started QP, and the seconds taken.

    The strongest grid baseline short of a path algorithm: OSQP is set up once, so its
    factorisation is reused, and each tilt changes only the linear term, with the solve
    starting from the previous solution. Polishing recovers the active-set solution. At
    tolerance 1e-5 it misses on some grid points of the S&P data by 1e-3; 1e-6 is the
    loosest setting that comes back to rounding at every point of the note's snapshot.
    """
    n = mu.shape[0]
    start = time.perf_counter()
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.csc_matrix(np.triu(sigma)),
        q=-tilts[0] * mu,
        A=sparse.vstack([sparse.csc_matrix(np.ones((1, n))), sparse.eye(n)], format="csc"),
        l=np.r_[1.0, np.zeros(n)],
        u=np.r_[1.0, np.full(n, np.inf)],
        eps_abs=1e-6,
        eps_rel=1e-6,
        polishing=True,
        warm_starting=True,
        verbose=False,
    )
    weights = []
    for lam in tilts:
        solver.update(q=-lam * mu)
        weights.append(solver.solve(raise_error=True).x)
    return np.array(weights).T, time.perf_counter() - start


@dataclass
class RealData:
    """The long-only recipe on real returns, and what it costs next to a grid of QPs."""

    days: int
    assets: int
    corners: int
    nu_last: float  # the last corner's multiplier, from the closed form
    mu_max: float  # max |mu_j|, the scale of mu
    held_minvar: int
    held_sharpe: int
    path_seconds: float  # one shifted lars_path call, plus the nnls check
    qp_seconds: float  # one independent QP, median over the corners
    warm_seconds: float  # a GRID-point grid from one warm-started QP
    warm_gap: float  # that grid against the cold QP, worst over every grid point
    snapshot: bool  # the data are the note's snapshot, so its numbers apply
    corner_gap: float
    minvar_gap: float
    kkt: float  # worst KKT residual over the corners and the minimum-variance end


def sp500_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, y, Sigma) for the S&P 500 sample: centred returns as X, y from least squares."""
    import pandas as pd

    r = pd.read_parquet(SP500).to_numpy()
    mu = r.mean(axis=0)
    x = (r - mu) / np.sqrt(r.shape[0] - 1)
    return x, np.linalg.lstsq(x.T, mu)[0], np.cov(r, rowvar=False)


def real_data(every: int = 1) -> RealData:
    """Run Algorithm 1 on daily S&P 500 returns, X taken as the centred returns.

    mu is the sample mean and Sigma the sample covariance, so X^T X = Sigma exactly with
    no factorisation. The shift comes from the closed-form last corner. ``every``
    compares only every so many corners with the QP (the tests use a subsample).
    """
    import pandas as pd

    r = pd.read_parquet(SP500).to_numpy()
    days, n = r.shape
    mu, sigma = r.mean(axis=0), np.cov(r, rowvar=False)
    x = (r - mu) / np.sqrt(days - 1)
    y = np.linalg.lstsq(x.T, mu)[0]
    start = time.perf_counter()
    w, tilts, w_minvar = long_only_frontier(x, y, sigma)
    path_seconds = time.perf_counter() - start
    w_sharpe = tangency(mu, sigma, long_only_path(x, y))
    gaps, seconds = [], []
    for k in range(0, w.shape[1], every):
        start = time.perf_counter()
        qp = tilted(mu, sigma, tilts[k])
        seconds.append(time.perf_counter() - start)
        gaps.append(float(np.abs(qp - w[:, k]).max()))
    grid = np.geomspace(tilts.max(), tilts.min(), GRID)
    w_grid, warm_seconds = warm_grid(mu, sigma, grid)
    warm_gap = max(float(np.abs(w_grid[:, k] - tilted(mu, sigma, grid[k])).max()) for k in range(GRID))
    return RealData(
        days=days,
        assets=n,
        corners=w.shape[1],
        nu_last=last_corner(x, y),
        mu_max=float(np.abs(mu).max()),
        held_minvar=int((w_minvar > SUPPORT_TOL).sum()),
        held_sharpe=int((w_sharpe > SUPPORT_TOL).sum()),
        path_seconds=path_seconds,
        qp_seconds=float(np.median(seconds)),
        warm_seconds=warm_seconds,
        warm_gap=warm_gap,
        snapshot=hashlib.sha256(SP500.read_bytes()).hexdigest() == SNAPSHOT_SHA256,
        corner_gap=max(gaps),
        minvar_gap=float(np.abs(tilted(mu, sigma, 0.0) - w_minvar).max()),
        kkt=long_only_kkt(mu, sigma, w, tilts, w_minvar),
    )


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
BLUE, RED, AMBER, GREY = "#1f4e79", "#c00000", "#b07800", "#9aa7b4"

# The tilt window the long-only figures show: past the most tilted corner (372.8) on the
# right, past the last corner (0.0128) on the left. Outside it the weights
# are flat and the multiplier only grows.
TILT_VIEW = (3e-3, 2e3)


def _style(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


def _save(fig: plt.Figure, name: str) -> Path:
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    out = OUT / name
    fig.savefig(out)
    plt.close(fig)
    return out


def figure(
    mu: np.ndarray,
    sigma: np.ndarray,
    betas: np.ndarray,
    w_lasso: np.ndarray,
    w_tail: np.ndarray,
    w_minvar: np.ndarray,
    w_sharpe: np.ndarray,
    budgets: np.ndarray,
) -> Path:
    """Both frontiers, each drawn from the output of a solver that knows no finance.

    The curves are exact, not corners joined by chords: between corners the weights are
    affine in the path parameter, so the variance is quadratic in it and each piece of
    the frontier is an arc.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.5))

    def var(w: np.ndarray) -> np.ndarray:
        return np.einsum("in,ij,jn->n", w, sigma, w)

    s = np.linspace(0.0, 1.0, 41)[1:]
    dense = np.column_stack([betas[:, [k - 1]] * (1 - s) + betas[:, [k]] * s for k in range(1, betas.shape[1])])
    left.plot(var(dense), mu @ dense, "-", lw=1.0, color=GREY, zorder=1)
    left.plot(var(betas), mu @ betas, "o", ms=3.0, color=BLUE, zorder=3)
    left.set_title(
        f"(a) long-short, gross exposure capped\n(plain LASSO path, {betas.shape[1] - 2} corners)",
        fontsize=8.5,
    )

    curve = budgets / budgets.sum(axis=0)
    # The same series go on the panel and into its inset, the inset being the only place
    # the last few corners are far enough apart to be seen at all.
    series = [
        (w_lasso, "o", 3.5, BLUE, "full", "corner, plain call"),
        (w_tail, "s", 4.5, RED, "none", "corner, past where it stops"),
        (w_lasso[:, :1], "^", 7.0, AMBER, "full", "highest return"),
        (w_sharpe[:, None], "D", 5.5, AMBER, "full", "max Sharpe"),
        (w_minvar[:, None], "*", 9.0, RED, "full", "min variance"),
    ]
    inset = right.inset_axes((0.5, 0.12, 0.47, 0.42))
    for ax in (right, inset):
        ax.plot(var(curve), mu @ curve, "-", lw=1.0, color=GREY, zorder=1)
        for w, marker, size, color, fill, label in series:
            ax.plot(
                var(w),
                mu @ w,
                marker,
                ms=size,
                color=color,
                fillstyle=fill,
                zorder=3,
                label=label if ax is right else None,
            )
    right.set_title(
        "(b) long only, fully invested\n(nonnegative LASSO path)",
        fontsize=8.5,
    )
    right.legend(fontsize=6, loc="upper left", frameon=False)

    # Zoom on the low-risk end: the last few corners with nu >= 0, the maximum-Sharpe
    # point where a plain path stops, the two corners with nu < 0, and the min-variance end.
    zoom = np.column_stack([w_lasso[:, -3:], w_tail, w_sharpe[:, None], w_minvar[:, None]])
    zoom_var, zoom_ret = var(zoom), mu @ zoom
    pad_x, pad_y = 0.12 * np.ptp(zoom_var) + 1e-3, 0.12 * np.ptp(zoom_ret) + 1e-3
    inset.set_xlim(zoom_var.min() - pad_x, zoom_var.max() + pad_x)
    inset.set_ylim(zoom_ret.min() - pad_y, zoom_ret.max() + pad_y)
    inset.tick_params(labelsize=5.5)
    right.indicate_inset_zoom(inset, edgecolor="#555555", lw=0.6, alpha=0.8)

    for ax in (left, right):
        _style(ax, r"Variance $w^\top\Sigma w$ (model units)", "Expected return (model units)")
    return _save(fig, "frontier.pdf")


def figure_paths(
    betas: np.ndarray,
    caps: np.ndarray,
    budgets: np.ndarray,
    is_lasso: np.ndarray,
    handover: float,
) -> list[Path]:
    """The weights themselves: the statistician's picture of both paths, one file each.

    The long-short one is the coefficient plot every LASSO user knows, drawn over gross
    exposure; the long-only one reads the path as portfolios, over the tilt, with the
    stretch a plain LASSO solver cannot reach (nu < 0) in red. Two files, so that each
    can sit beside the section it illustrates.
    """
    fig, left = plt.subplots(figsize=(4.6, 3.1))

    # One asset enters at each corner, and on this problem none ever leaves; the only
    # event worth a colour is the first short. The first and last columns of the path are
    # its ends, not corners: the empty portfolio at c = 0, and at the right, where the cap
    # goes slack, the unconstrained optimum Sigma^{-1} mu. An asset enters at a corner
    # with weight zero, so the first short is at the column *before* its weight shows.
    short = (betas < -SUPPORT_TOL).any(axis=1)
    first_short = caps[int(np.argmax((betas < -SUPPORT_TOL).any(axis=0))) - 1]
    for c in caps[1:-1]:
        left.axvline(c, color=GREY, lw=0.4, alpha=0.6, zorder=0)
    left.axhline(0.0, color="#555555", lw=0.6)
    left.plot(caps, betas[~short].T, "-", lw=0.9, color=BLUE, alpha=0.85)
    for j in np.flatnonzero(short):
        # From the corner before it enters: drawn along zero it would pass for an axis.
        enter = max(int(np.argmax(np.abs(betas[j]) > SUPPORT_TOL)) - 1, 0)
        left.plot(caps[enter:], betas[j, enter:], "-", lw=1.3, color=AMBER)
    left.text(
        caps[-1] - 0.1, betas[short, -1].min(), "goes short", color="#333333", fontsize=6.5, ha="right", va="center"
    )

    top = left.get_xaxis_transform()
    for c, style, label in [
        (LEVERAGE, "--", f"fixed budget\n$c={LEVERAGE:g}$"),
        (first_short, "-", f"first short\n$c={first_short:.2f}$"),
    ]:
        left.axvline(c, color=AMBER if style == "-" else "#555555", lw=0.8, ls=style)
        left.text(c - 0.1, 0.98, label, transform=top, ha="right", va="top", fontsize=6.5)
    left.axvline(caps[-1], color="#555555", lw=0.8, ls=":")
    left.text(
        caps[-1] + 0.15,
        0.5,
        r"cap slack: $w=\Sigma^{-1}\mu$",
        transform=top,
        rotation=90,
        ha="left",
        va="center",
        fontsize=6.5,
    )
    left.set_xlim(-0.2, caps[-1] + 0.7)
    left.set_title("long\u2013short: plain LASSO path (one asset enters at each grey line)", fontsize=8.5)
    _style(left, r"Gross exposure cap $c=\|w\|_1$", r"Weight $w_j$")
    first = _save(fig, "weights_longshort.pdf")

    fig, right = plt.subplots(figsize=(3.5, 2.7))

    tilts = 1.0 / budgets.sum(axis=0)
    view = (tilts >= TILT_VIEW[0]) & (tilts <= TILT_VIEW[1])
    budgets, is_lasso, tilts = budgets[:, view], is_lasso[view], tilts[view]
    weights = budgets / budgets.sum(axis=0)
    # Draw the nu < 0 stretch one sample into the nu >= 0 side so the two pieces meet.
    first_tail = int(np.argmin(is_lasso))
    right.plot(tilts[is_lasso], weights[:, is_lasso].T, "-", lw=0.9, color=BLUE, alpha=0.85)
    right.plot(tilts[first_tail - 1 :], weights[:, first_tail - 1 :].T, "-", lw=0.9, color=RED, alpha=0.85)
    right.axvspan(TILT_VIEW[0], handover, color=RED, alpha=0.06, lw=0)
    right.axvline(handover, color="#555555", lw=0.6, ls="--")
    right.text(
        handover * 0.8,
        0.97,
        "past the end of\nthe plain path",
        ha="right",
        va="top",
        fontsize=6.5,
        transform=right.get_xaxis_transform(),
    )
    right.set_xscale("log")
    right.set_xlim(*TILT_VIEW)
    right.set_ylim(bottom=0.0)
    right.set_title("long-only weights (most tilted on the right)", fontsize=8)
    _style(right, r"Tilt $\lambda$ (log scale)", r"Weight $w_j$")
    return [first, _save(fig, "weights_longonly.pdf")]


def figure_multiplier(
    mu: np.ndarray,
    sigma: np.ndarray,
    budgets: np.ndarray,
    is_lasso: np.ndarray,
    v_corners: np.ndarray,
    v_tail: np.ndarray,
    handover: float,
) -> Path:
    """Why a plain LASSO path stops: the budget multiplier runs through zero."""
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    tilts = 1.0 / budgets.sum(axis=0)
    view = (tilts >= TILT_VIEW[0]) & (tilts <= TILT_VIEW[1])
    budgets, is_lasso, tilts = budgets[:, view], is_lasso[view], tilts[view]
    nu = budget_multiplier(mu, sigma, budgets)
    first_tail = int(np.argmin(is_lasso))

    ax.axhline(0.0, color="#555555", lw=0.6)
    ax.axvline(handover, color="#555555", lw=0.6, ls="--")
    ax.plot(tilts[is_lasso], nu[is_lasso], "-", lw=1.2, color=BLUE, label=r"$\nu\geq0$: plain LASSO path")
    ax.plot(tilts[first_tail - 1 :], nu[first_tail - 1 :], "-", lw=1.2, color=RED, label=r"$\nu<0$: shifted path only")
    ax.plot(1.0 / v_corners.sum(axis=0), budget_multiplier(mu, sigma, v_corners), "o", ms=3.5, color=BLUE, zorder=3)
    ax.plot(
        1.0 / v_tail.sum(axis=0),
        budget_multiplier(mu, sigma, v_tail),
        "s",
        ms=4.5,
        color=RED,
        fillstyle="none",
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xlim(*TILT_VIEW)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    _style(ax, r"Tilt $\lambda$ (log scale)", r"Budget multiplier $\nu$")
    return _save(fig, "multiplier.pdf")


def figure_agreement(checks: dict[str, Check], plain: np.ndarray, min_var_gap: float) -> Path:
    """Every discrepancy in Table 1, point by point, split by the baseline it is against.

    x is the position along the path: vertex k at k, the midpoint of the segment ending
    there at k - 1/2. Zeros are drawn at the floor of the axis.
    """
    floor = 1e-17
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)

    def plot(ax: plt.Axes, gaps: np.ndarray, x: np.ndarray, color: str, filled: bool, label: str) -> None:
        ax.plot(
            x,
            np.maximum(gaps, floor),
            "o" if filled else "D",
            ms=3.8 if filled else 3.2,
            color=color,
            fillstyle="full" if filled else "none",
            lw=0,
            label=label,
        )

    ls_c, ls_m = checks["ls_corner"].per_point, checks["ls_mid"].per_point
    fl_c, fl_m = checks["fl_corner"].per_point, checks["fl_mid"].per_point
    plot(left, ls_c, np.arange(1, ls_c.size + 1), BLUE, True, "long\u2013short, corner")
    plot(left, ls_m, np.arange(1, ls_m.size + 1) - 0.5, BLUE, False, "long\u2013short, mid-segment")
    plot(left, fl_c, np.arange(1, fl_c.size + 1), AMBER, True, "fixed leverage, corner")
    plot(left, fl_m, np.arange(1, fl_m.size + 1) - 0.5, AMBER, False, "fixed leverage, mid-segment")
    left.set_title("(a) against an interior-point conic solver (Clarabel)", fontsize=8.5)

    lo_c, lo_m = checks["lo_corner"].per_point, checks["lo_mid"].per_point
    k = np.arange(1, lo_c.size + 1)
    plot(right, lo_c[plain], k[plain], BLUE, True, r"corner, $\nu\geq0$")
    plot(right, lo_c[~plain], k[~plain], RED, True, r"corner, $\nu<0$")
    plot(right, lo_m, np.arange(2, lo_m.size + 2) - 0.5, BLUE, False, "mid-segment")
    right.plot(lo_c.size + 1, max(min_var_gap, floor), "*", ms=8, color=RED, label="min-variance end (closed form)")
    right.set_title("(b) against a dual active-set QP (Goldfarb\u2013Idnani)", fontsize=8.5)

    for ax in (left, right):
        ax.set_yscale("log")
        ax.set_ylim(floor / 3, 1e-4)
        ax.axhline(np.finfo(float).eps, color=GREY, lw=0.6, ls=":")
        ax.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
        _style(ax, "Position along the path (column index)", "")
    left.set_ylabel(r"Max weight discrepancy $\|\cdot\|_\infty$", fontsize=8)
    return _save(fig, "agreement.pdf")


# --------------------------------------------------------------------------------------
def analysis(verbose: bool = True) -> tuple[dict[str, Check], Callable[[], list[Path]]]:
    """Run every computation of the note once.

    Returns the checks, keyed for the tests, and a function that draws the figures from
    the same results. With ``verbose`` the headline numbers the note quotes are printed.
    """
    say = print if verbose else (lambda *_: None)
    mu, sigma = factor_problem()
    x, y = as_regression(mu, sigma)
    factor_sqrt, gram, centred, shifted_tall = check_cholesky_free(mu, sigma, factor_data())
    checks = {
        "xtx": Check("factorisation X^T X - Sigma", float(np.abs(x.T @ x - sigma).max()), 1e-12),
        "xty": Check("factorisation X^T y - mu", float(np.abs(x.T @ y - mu).max()), 1e-12),
        "factor_sqrt": factor_sqrt,
        "gram": gram,
        "centred": centred,
        "shifted_tall": shifted_tall,
    }

    betas, caps = long_short_frontier(x, y)
    checks["ls_corner"], checks["ls_mid"] = check_long_short(mu, sigma, betas, caps)
    say(
        f"long--short: {betas.shape[1]} vertices (the empty start, {betas.shape[1] - 2} corners, "
        f"the end), gross exposure 0 -> {caps[-1]:.3f}"
    )
    # The end of the path, at nu = 0, is Sigma^{-1} mu: the unconstrained optimum, and the
    # direction of the maximum-Sharpe portfolio. The Sharpe ratio climbs towards it.
    w_star = np.linalg.solve(sigma, mu)
    corners = betas[:, 1:]
    sharpe = (mu @ corners) / np.sqrt(np.einsum("in,ij,jn->n", corners, sigma, corners))
    checks["ls_end"] = Check("long-short end vs Sigma^-1 mu", float(np.abs(betas[:, -1] - w_star).max()), 1e-12)
    checks["ls_sharpe"] = Check(
        "long-short Sharpe ratio: largest drop between corners", float(max(0.0, -np.diff(sharpe).min())), 1e-12
    )
    say(
        f"             Sharpe ratio {sharpe[0]:.3f} -> {sharpe[-1]:.3f} "
        f"(sqrt(mu' Sigma^-1 mu) = {np.sqrt(mu @ w_star):.3f}), "
        f"net exposure at the end {w_star.sum():.3f}"
    )

    _, lev_tilts = fixed_leverage_frontier(betas, caps)
    checks["fl_corner"], checks["fl_mid"] = check_fixed_leverage(mu, sigma, betas, caps)
    say(
        f"             the same {lev_tilts.size - 1} corners and the end, at fixed gross exposure "
        f"{LEVERAGE:g}, tilt {lev_tilts.min():.4f} -> {lev_tilts.max():.1f}"
    )

    w_long, tilts, w_minvar = long_only_frontier(x, y, sigma)
    plain = budget_multiplier(mu, sigma, w_long / tilts) >= 0  # reached without the shift
    checks["lo_corner"] = check_long_only("long-only corners vs tilted QP", mu, sigma, w_long, tilts)
    checks["lo_mid"] = check_long_only_segments(x, y, mu, sigma)
    w_sharpe = tangency(mu, sigma, long_only_path(x, y))  # nu = 0: where a plain path ends
    handover = float(1.0 / nnls(x, y)[0].sum())  # its tilt, from the plain NNLS fit
    say(
        f"long only:   {w_long.shape[1]} corners from one shifted path (last corner at "
        f"nu = {last_corner(x, y):.4f}, nu0 = {shift_for(x, y):.4f}), "
        f"tilt {tilts.min():.4f} -> {tilts.max():.1f}"
    )
    say(
        f"             {int(plain.sum())} with nu >= 0, down to tilt {tilts[plain].min():.4f}; "
        f"a plain path stops at {handover:.4f}; {int((~plain).sum())} with nu < 0, tilt "
        f"{tilts[~plain].min():.4f} -> {tilts[~plain].max():.4f}"
    )

    min_var_gap = float(np.abs(tilted(mu, sigma, 0.0) - w_minvar).max())
    checks["min_var"] = Check("min-variance end vs min-variance QP", min_var_gap, 1e-12)
    sharpe_gap = float(np.abs(max_sharpe_qp(mu, sigma) - w_sharpe).max())
    checks["ls_kkt"], checks["fl_kkt"], checks["lo_kkt"] = check_kkt(
        mu, sigma, x, y, betas, caps, w_long, tilts, w_minvar
    )
    checks["max_sharpe"] = Check("max-Sharpe point (nu = 0) vs max-Sharpe QP", sharpe_gap, 1e-12)

    checks["cla"], summary = check_against_cla(mu, sigma, np.column_stack([w_long, w_minvar[:, None]]))
    say(f"             {summary}")

    def draw() -> list[Path]:
        budgets = long_only_curve(x, y)
        is_lasso = budget_multiplier(mu, sigma, budgets) >= 0
        v_long = w_long / tilts
        return [
            figure(mu, sigma, betas, w_long[:, plain], w_long[:, ~plain], w_minvar, w_sharpe, budgets),
            *figure_paths(betas, caps, budgets, is_lasso, handover),
            figure_multiplier(mu, sigma, budgets, is_lasso, v_long[:, plain], v_long[:, ~plain], handover),
            figure_agreement(checks, plain, min_var_gap),
        ]

    return checks, draw


def main() -> None:
    """Draw the figures and print every number the note quotes."""
    checks, draw = analysis()
    for out in draw():
        print(f"wrote {out}")
    print("\nnumbers quoted in the note (run with `test` to check them against tolerances):")
    for c in checks.values():
        print(f"  {c.name}: {c.measured:.2e}")
    if SP500.exists():
        print_real_data(real_data())
    else:
        print(f"\nreal data skipped: {SP500} not found (fetch it with `uv run fetch_sp500.py`)")
    print("\nthe margin delta in nu0 = nu_last - delta * max(|nu_last|, max|mu_j|):")
    mu, sigma = factor_problem()
    problems = {"test problem": (*as_regression(mu, sigma), sigma)}
    if SP500.exists():
        problems["S&P 500"] = sp500_problem()
    for name, (xp, yp, sp) in problems.items():
        cells = "  ".join(
            f"{d:g}: {k} corners, KKT {e:.1e}" if k else f"{d:g}: stops short" for d, k, e in margin_sweep(xp, yp, sp)
        )
        print(f"  {name}: {cells}")
    print_robustness()


def print_real_data(rd: RealData) -> None:
    """Print the numbers of the real-returns paragraph of Section 8."""
    print(f"\nreal data: S&P 500, {rd.assets} stocks x {rd.days} days, centred returns as X")
    print(
        f"  {rd.corners} corners; last corner at nu = {rd.nu_last:.4f} "
        f"({rd.nu_last / rd.mu_max:.0f} max|mu_j|); minimum variance holds "
        f"{rd.held_minvar} names, maximum Sharpe {rd.held_sharpe}"
    )
    print(
        f"  one path: {rd.path_seconds:.2f} s; one QP: {rd.qp_seconds * 1e3:.0f} ms, so a "
        f"{GRID}-point grid of QPs: {GRID * rd.qp_seconds:.1f} s; warm-started: "
        f"{rd.warm_seconds:.1f} s, off the cold QP by {rd.warm_gap:.1e}"
    )
    print(
        f"  worst corner vs QP {rd.corner_gap:.1e}; minimum-variance end {rd.minvar_gap:.1e}; "
        f"worst KKT residual {rd.kkt:.1e}"
    )


def print_robustness() -> None:
    """Print Table 1 of the note."""
    print("\nrobustness of the long-only recipe (Table 1 of the note):")
    print("     n  seeds  corners   last corner nu     worst corner   min-variance end   KKT")
    for n in ROBUST_SIZES:
        rows = robustness(sizes=(n,))
        corners = sorted({r.corners for r in rows})
        span = f"{corners[0]}" if len(corners) == 1 else f"{corners[0]}-{corners[-1]}"
        lo, hi = min(r.nu_last for r in rows), max(r.nu_last for r in rows)
        print(
            f"  {n:4d}  {len(rows):5d}  {span:>7}  {lo:7.2f} to {hi:6.2f}"
            f"    {max(r.corner_gap for r in rows):.1e}        {max(r.minvar_gap for r in rows):.1e}"
            f"      {max(r.kkt for r in rows):.1e}"
        )


# --------------------------------------------------------------------------------------
# Tests -- uv run make_figures.py test
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def checks() -> dict[str, Check]:
    """Every check of the note, computed once per test module."""
    return analysis(verbose=False)[0]


def _within_tolerance(checks: dict[str, Check], *keys: str) -> None:
    failed = [str(checks[k]) for k in keys if checks[k].measured > checks[k].tol]
    assert not failed, "\n".join(failed)


def test_substitution(checks: dict[str, Check]) -> None:
    """X^T X = Sigma and X^T y = mu, to rounding."""
    _within_tolerance(checks, "xtx", "xty")


def test_cholesky_free_routes(checks: dict[str, Check]) -> None:
    """Factor square root, lars_path_gram and centred returns give the Cholesky path.

    The shifted long-only path works with a tall X too.
    """
    _within_tolerance(checks, "factor_sqrt", "gram", "centred", "shifted_tall")


def test_long_short_against_conic_solver(checks: dict[str, Check]) -> None:
    """The LASSO path solves the capped program, at corners and inside segments."""
    _within_tolerance(checks, "ls_corner", "ls_mid")


def test_long_short_end_is_max_sharpe(checks: dict[str, Check]) -> None:
    """The path ends at Sigma^{-1} mu, and the Sharpe ratio never falls on the way."""
    _within_tolerance(checks, "ls_end", "ls_sharpe")


def test_fixed_leverage_against_conic_solver(checks: dict[str, Check]) -> None:
    """The rescaled path solves the tilted program at fixed leverage."""
    _within_tolerance(checks, "fl_corner", "fl_mid")


def test_long_only_against_active_set_qp(checks: dict[str, Check]) -> None:
    """The shifted nonnegative path solves the long-only program, corners and segments."""
    _within_tolerance(checks, "lo_corner", "lo_mid")


def test_landmarks_in_closed_form(checks: dict[str, Check]) -> None:
    """Minimum-variance end and maximum-Sharpe point match their textbook QPs."""
    _within_tolerance(checks, "min_var", "max_sharpe")


def test_last_corner_closed_form() -> None:
    """The closed-form last corner equals the one a deeply shifted path reports.

    On the test problem, and on the case where nu0 = -10 used to be too high.
    """
    f = factor_data(seed=2, n_assets=50)
    problems = [factor_problem(), (f.returns.mean(axis=0), np.diag(f.d) + (f.u * f.delta) @ f.u.T)]
    for mu, sigma in problems:
        x, y = as_regression(mu, sigma)
        nu0 = 100.0 * last_corner(x, y)
        alphas, _, _ = lars_path(
            x, y - nu0 * shift_direction(x), method="lasso", positive=True, max_iter=10 * x.shape[1]
        )
        reported = nu0 + x.shape[0] * alphas[-2]
        assert abs(last_corner(x, y) - reported) <= 1e-9 * max(1.0, abs(reported))


def test_robustness_across_sizes_and_seeds() -> None:
    """The long-only recipe holds on other sizes and seeds.

    n <= 100 here, for speed; `uv run make_figures.py` prints the full table.
    """
    rows = robustness(sizes=(20, 50, 100))
    worst = max(rows, key=lambda r: r.corner_gap)
    assert worst.corner_gap <= 1e-10, worst
    assert max(r.minvar_gap for r in rows) <= 1e-12
    assert max(r.kkt for r in rows) <= 1e-12


@pytest.mark.skipif(not SP500.exists(), reason="no S&P 500 data; fetch it with `uv run fetch_sp500.py`")
def test_real_data() -> None:
    """Algorithm 1 on S&P 500 returns, through centred returns (every 10th corner)."""
    rd = real_data(every=10)
    assert rd.corner_gap <= 1e-10, rd
    assert rd.minvar_gap <= 1e-12, rd
    assert rd.kkt <= 1e-11, rd
    # The warm-started baseline is exact to rounding on the note's snapshot. On a fresh
    # fetch OSQP's polish can miss a point by 1e-6, which is the baseline's floor, not
    # the path's.
    if rd.snapshot:
        assert rd.warm_gap <= 1e-9, rd


def test_kkt_certificates(checks: dict[str, Check]) -> None:
    """Every reported portfolio satisfies its own optimality conditions, no solver needed."""
    _within_tolerance(checks, "ls_kkt", "fl_kkt", "lo_kkt")


def test_margin_of_the_shift() -> None:
    """Delta = 1 sits inside the range that finds every corner at full accuracy."""
    mu, sigma = factor_problem()
    x, y = as_regression(mu, sigma)
    rows = {d: (k, e) for d, k, e in margin_sweep(x, y, sigma, margins=(1e-2, 1.0, 10.0))}
    assert len({k for k, _ in rows.values()}) == 1, rows
    assert max(e for _, e in rows.values()) <= 1e-12, rows


def test_every_cla_turning_point_recovered(checks: dict[str, Check]) -> None:
    """Every turning point cvxcla reports is among the portfolios recovered here."""
    _within_tolerance(checks, "cla")


def export(out: Path = _HERE / "data" / "test_problem") -> None:
    """Write the test problem and its long-short path as CSV, for the R check."""
    mu, sigma = factor_problem()
    x, y = as_regression(mu, sigma)
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "Sigma.csv", sigma, delimiter=",", fmt="%.17g")
    np.savetxt(out / "mu.csv", mu, delimiter=",", fmt="%.17g")
    np.savetxt(out / "B_lars_path.csv", long_short_frontier(x, y)[0], delimiter=",", fmt="%.17g")
    print(f"wrote {out}/Sigma.csv, mu.csv, B_lars_path.csv")


if __name__ == "__main__":
    if sys.argv[1:2] == ["export"]:
        raise SystemExit(export())
    if sys.argv[1:2] == ["test"]:
        sys.dont_write_bytecode = True
        raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", *sys.argv[2:]]))
    main()
