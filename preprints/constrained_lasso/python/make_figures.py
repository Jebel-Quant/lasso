# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "cvxcla==2.4.0",
#     "cvxpy==1.9.3",
#     "matplotlib==3.11.2",
#     "numpy==2.5.4",
#     "pytest==9.1.1",
#     "scikit-learn==1.9.1",
#     "scipy==1.18.1",
# ]
# ///
# The pytest tests live in this script, so assert is the test idiom here too.
# ruff: noqa: S101
r"""The constrained LASSO, traced by the Critical Line Algorithm.

Companion code for ``constrained_lasso.tex``. The note runs the other way from
``../lasso``: there a LASSO solver computes efficient frontiers, here a Critical Line
Algorithm computes LASSO paths. The CLA below is written out in full (``cla``), in
Markowitz's general form, and knows nothing about regression. Two maps feed it:

* ``penalty_form``: the penalty is a cost of one unit of return per unit of gross
  position, so the CLA's risk-tolerance parameter *is* the LASSO's lambda. Bounds on the
  coefficients are bounds on the positions, and linear equality constraints are rows of
  the CLA's constraint matrix.
* ``budget_form``: the l1 ball, rescaled to the unit simplex, is the fully invested
  long-only budget, with one riskless "cash" asset as its slack. The parameter is the
  reciprocal of the l1 bound, and homogeneous constraints carry over unchanged.

    uv run python/make_figures.py        # the four figures, Table 1 as figures/evidence.tex,
                                         # the example's numbers as figures/example.tex
    uv run python/make_figures.py test   # the same numbers as pytest checks, with tolerances
    uv run python/make_figures.py timings  # the cost ratios the note quotes, and the machine

Seeded and offline: the diabetes data ship with scikit-learn.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cvxcla import CLA as CvxclaCLA
from cvxcla import Lasso as CvxclaLasso
from scipy.optimize import linprog, nnls
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lars_path

SEED = 7
OUT = Path(__file__).resolve().parent.parent / "figures"  # the preprint folder; this script lives in python/

# A position strictly inside its bounds by less than this is read as at the bound when the
# starting vertex is taken off the LP solution.
VERTEX_TOL = 1e-9

# Two events closer than this, relative to the current parameter, are one event: the
# variable that has just changed state is not allowed to fire again at the same point.
EVENT_TOL = 1e-10

# The baseline is an interior-point solve asked for everything it has; around 1e-8 relative
# is its floor, not the path's.
CONIC = {"solver": cp.CLARABEL, "tol_gap_abs": 1e-12, "tol_gap_rel": 1e-12, "tol_feas": 1e-12}


# --------------------------------------------------------------------------------------
# The Critical Line Algorithm
# --------------------------------------------------------------------------------------
@dataclass
class Turns:
    """The turning points of a CLA run, from the largest parameter to zero."""

    gamma: np.ndarray  # (k,) parameter at each turning point; gamma[0] is inf
    x: np.ndarray  # (k, N) the solution there
    nu: np.ndarray  # (k, m) the equality multipliers there; at gamma = inf, their slope nu / gamma


def cla(
    sigma: np.ndarray,
    mu: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    q: np.ndarray | None = None,
    start: np.ndarray | None = None,
    stop: int | None = None,
) -> Turns:
    r"""Trace :math:`\min \tfrac12 x^\top\Sigma x - (q+\gamma\mu)^\top x` from gamma = inf to 0.

    Subject to :math:`Ax=b` and :math:`l\le x\le u`. This is Markowitz's Critical Line
    Algorithm in its general form, with the benchmark term ``q`` (``q = Sigma @ x_bar``
    makes the objective the tracking variance against ``x_bar``; ``q = 0`` is the classical
    problem). On each segment the free block solves the bordered system

        [ Sigma_FF  A_F' ] [x_F]   [ q_F - Sigma_FB x_B ]       [ mu_F ]
        [ A_F       0    ] [nu ] = [ b   - A_B x_B      ] + gamma [ 0    ]

    so that x_F and nu are affine in gamma, and so is the gradient
    g = Sigma x - q - gamma mu + A' nu on the blocked coordinates. Lowering gamma, the
    segment ends where a free coordinate reaches a bound, or where a blocked coordinate's
    gradient changes sign and it wants to move off its bound.

    ``start`` is the solution at gamma = inf, the vertex maximising mu'x; when omitted it
    is computed by linear programming. Its free set must be exactly as large as A has
    rows, the nondegenerate case, or the bordered system is singular and the run stops.

    ``stop`` names a coordinate whose leaving its bound ends the run. It also wins a tie:
    when it fires at the same parameter as other events, to rounding, it is taken.
    """
    n = len(mu)
    q = np.zeros(n) if q is None else q
    m = a.shape[0]
    if start is None:
        lp = linprog(-mu, A_eq=a if m else None, b_eq=b if m else None,
                     bounds=list(zip(lower, np.where(np.isinf(upper), None, upper), strict=True)),
                     method="highs")
        assert lp.status == 0, lp.message
        start = lp.x
    x = start.copy()
    free = (x > lower + VERTEX_TOL) & (x < upper - VERTEX_TOL)
    x[~free] = np.where(np.abs(x[~free] - lower[~free]) <= np.abs(x[~free] - upper[~free]),
                        lower[~free], upper[~free])
    if free.sum() != m:
        msg = f"degenerate start: {free.sum()} free coordinates for {m} equality rows"
        raise ValueError(msg)

    gammas, xs, nus = [np.inf], [x.copy()], []
    gamma, last, floor = np.inf, -1, 0.0
    while True:
        f, blk = np.flatnonzero(free), np.flatnonzero(~free)
        k = np.block([[sigma[np.ix_(f, f)], a[:, f].T], [a[:, f], np.zeros((m, m))]])
        rhs0 = np.concatenate([q[f] - sigma[np.ix_(f, blk)] @ x[blk], b - a[:, blk] @ x[blk]])
        rhs1 = np.concatenate([mu[f], np.zeros(m)])
        sol = np.linalg.solve(k, np.column_stack([rhs0, rhs1])) if len(rhs0) else np.zeros((0, 2))
        alpha, delta = sol[: len(f), 0], sol[: len(f), 1]
        nu0, nu1 = sol[len(f) :, 0], sol[len(f) :, 1]
        if len(nus) < len(xs):  # the multipliers at the turning point just recorded
            nus.append(nu1 if np.isinf(gamma) else nu0 + gamma * nu1)

        # Gradient on the blocked coordinates, g0 + gamma g1.
        g0 = sigma[np.ix_(blk, f)] @ alpha + sigma[np.ix_(blk, blk)] @ x[blk] - q[blk] + a[:, blk].T @ nu0
        g1 = sigma[np.ix_(blk, f)] @ delta - mu[blk] + a[:, blk].T @ nu1

        cand: list[tuple[float, int]] = []
        with np.errstate(divide="ignore", invalid="ignore"):
            # A free coordinate moves by -delta per unit decrease of gamma.
            hit_lo = np.where(delta > 0, (lower[f] - alpha) / delta, -np.inf)
            hit_up = np.where(delta < 0, (upper[f] - alpha) / delta, -np.inf)
            # At a lower bound g >= 0 must hold, at an upper bound g <= 0.
            at_lo = np.isclose(x[blk], lower[blk])
            leave = np.where(at_lo, g1 > 0, g1 < 0)
            off = np.where(leave, -g0 / g1, -np.inf)
        cand += [(t, i) for t, i in zip(np.maximum(hit_lo, hit_up), f, strict=True)]
        cand += [(t, j) for t, j in zip(off, blk, strict=True)]
        ceiling = np.inf if np.isinf(gamma) else gamma * (1 - EVENT_TOL)
        cand = [(t, i) for t, i in cand if np.isfinite(t) and floor < t < ceiling and i != last]
        gamma, last = max(cand, default=(0.0, -1))
        mine = [t for t, i in cand if i == stop]
        if mine and mine[0] >= gamma * (1 - 1e-8):
            gamma, last = mine[0], stop
        # A coordinate whose twin is free has a gradient that vanishes only at gamma = 0, and
        # rounding can put that root at 1e-16 instead. Roots that small, relative to the
        # first event, are zero.
        floor = floor or EVENT_TOL * gamma

        x[f] = alpha + gamma * delta
        xs.append(x.copy())
        gammas.append(gamma)
        if last < 0 or last == stop:
            nus.append(nu0 + gamma * nu1)
            break
        if free[last]:  # a free coordinate reached a bound
            x[last] = lower[last] if delta[f == last][0] > 0 else upper[last]
        free[last] = not free[last]
    return Turns(gamma=np.array(gammas), x=np.array(xs), nu=np.array(nus).reshape(len(xs), m))


def cvxcla_cla(
    sigma: np.ndarray,
    mu: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    q: np.ndarray | None = None,
    stop: int | None = None,
) -> Turns:
    """The same trace by the released CLA of cvxcla, unchanged, in the shape ``cla`` returns.

    cvxcla takes, since 2.4.0, the benchmark term ``q`` and the early ``stop`` the two forms
    need, finds the starting vertex itself, and records the equality multipliers with
    the same convention as ``cla``: their slope at gamma = inf, their value below.
    """
    trace = CvxclaCLA(mean=mu, covariance=sigma, lower_bounds=lower, upper_bounds=upper, a=a, b=b, q=q,
                      stop=stop).turning_points
    return Turns(gamma=np.array([t.lamb for t in trace]), x=np.array([t.weights for t in trace]),
                 nu=np.array([t.nu for t in trace]).reshape(len(trace), a.shape[0]))


# The CLA a form runs on: this note's ``cla`` when None, or ``cvxcla_cla``.
Engine = Callable[..., Turns] | None


# --------------------------------------------------------------------------------------
# The constrained LASSO, and its two CLA forms
# --------------------------------------------------------------------------------------
@dataclass
class Lasso:
    r""":math:`\min \tfrac12\|y-X\beta\|^2 + \lambda\|\beta\|_1` s.t. :math:`A\beta=b,\ l\le\beta\le u`."""

    x: np.ndarray
    y: np.ndarray
    a: np.ndarray | None = None
    b: np.ndarray | None = None
    lower: np.ndarray | None = None  # each entry <= 0
    upper: np.ndarray | None = None  # each entry >= 0

    def __post_init__(self) -> None:
        """Fill in the absent constraints as the empty ones."""
        p = self.x.shape[1]
        self.a = np.zeros((0, p)) if self.a is None else np.atleast_2d(self.a)
        self.b = np.zeros(self.a.shape[0]) if self.b is None else np.atleast_1d(self.b).astype(float)
        self.lower = np.full(p, -np.inf) if self.lower is None else self.lower
        self.upper = np.full(p, np.inf) if self.upper is None else self.upper


@dataclass
class LassoPath:
    """Knots of the LASSO path, from the largest lambda down to zero."""

    lam: np.ndarray  # (k,)
    beta: np.ndarray  # (k, p)
    kappa: np.ndarray  # (k, m) multipliers of A beta = b
    turns: Turns  # the CLA run the knots were read off


def split(prob: Lasso) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive and negative parts: kept columns of [I, -I], and the upper bounds of each."""
    p = prob.x.shape[1]
    keep = np.concatenate([prob.upper > 0, prob.lower < 0])  # a sign the bounds forbid is dropped
    s = np.hstack([np.eye(p), -np.eye(p)])[:, keep]
    cap = np.concatenate([prob.upper, -prob.lower])[keep]
    return s, cap, keep


def penalty_form(prob: Lasso, engine: Engine = None) -> LassoPath:
    r"""The LASSO as a CLA whose parameter is lambda itself.

    With :math:`\beta = S z`, :math:`z\ge0` the positive and negative parts, the
    objective is :math:`\tfrac12 z^\top S^\top G S z - (S^\top c)^\top z + \lambda 1^\top z`
    for :math:`G=X^\top X`, :math:`c=X^\top y`: a CLA instance with covariance
    :math:`S^\top G S`, expected returns :math:`\mu = -1` and benchmark term
    :math:`q = S^\top c`, which lies in the range of the covariance because :math:`c` lies
    in that of :math:`G`. The rows are :math:`ASz=b`, the bounds :math:`0\le z\le` cap.
    """
    gram, c = prob.x.T @ prob.x, prob.x.T @ prob.y
    s, cap, _ = split(prob)
    turns = (engine or cla)(s.T @ gram @ s, -np.ones(s.shape[1]), prob.a @ s, prob.b,
                np.zeros(s.shape[1]), cap, q=s.T @ c)
    keep = np.isfinite(turns.gamma)  # the gamma = inf vertex is the gamma_1 knot again
    return LassoPath(lam=turns.gamma[keep], beta=turns.x[keep] @ s.T, kappa=turns.nu[keep], turns=turns)


def budget_form(prob: Lasso, engine: Engine = None) -> LassoPath:
    r"""The LASSO as a classical CLA: long only, fully invested, with a cash asset.

    Put :math:`\beta = t\,S w` with :math:`w\ge0` and :math:`1^\top w + w_0 = 1`, the
    riskless :math:`w_0` taking up whatever of the l1 budget :math:`t` is unused. Dividing
    the bound-form objective by :math:`t^2` leaves
    :math:`\tfrac12 w^\top S^\top G S w - \gamma (S^\top c)^\top w` with
    :math:`\gamma = 1/t`: covariance :math:`S^\top G S` (zero for cash), expected returns
    :math:`S^\top c` (zero for cash). Only homogeneous constraints survive the rescaling,
    so this form takes :math:`A\beta = 0` and sign constraints, and no finite box.
    """
    assert not np.any(prob.b), "the budget form needs homogeneous constraints, A beta = 0"
    assert np.all(np.isin(prob.lower, [0, -np.inf])) and np.all(np.isin(prob.upper, [0, np.inf]))
    gram, c = prob.x.T @ prob.x, prob.x.T @ prob.y
    s, _, _ = split(prob)
    k, m = s.shape[1], prob.a.shape[0]
    sigma = np.zeros((k + 1, k + 1))
    sigma[:k, :k] = s.T @ gram @ s
    mu = np.append(s.T @ c, 0.0)
    rows = np.vstack([np.ones(k + 1), np.hstack([prob.a @ s, np.zeros((m, 1))])])
    # Cash leaving its bound is the budget going slack, lambda = 0: the LASSO path is over,
    # and past it the twin coordinates' gradients vanish identically, so the run stops there.
    turns = (engine or cla)(sigma, mu, rows, np.append(1.0, np.zeros(m)), np.zeros(k + 1), np.full(k + 1, np.inf),
                stop=k)
    # beta = t S w = S w / gamma, which is 0 at gamma = inf (t = 0). Multiplying the CLA's
    # stationarity row by t: lambda is t times the budget multiplier, and at gamma = inf
    # the slope cla records there is exactly that limit.
    t = np.where(np.isinf(turns.gamma), 0.0, 1.0 / turns.gamma)
    beta = turns.x[:, :k] @ s.T * t[:, None]
    nu = np.where(np.isinf(turns.gamma)[:, None], turns.nu, turns.nu * t[:, None])
    return LassoPath(lam=nu[:, 0], beta=beta, kappa=nu[:, 1:], turns=turns)


def kkt_residual(prob: Lasso, lam: float, beta: np.ndarray, kappa: np.ndarray) -> float:
    r"""Largest violation of the LASSO's optimality conditions, relative to lambda_max.

    Read off the original problem, not either CLA form: with
    :math:`r = G\beta - c + A^\top\kappa`, a nonzero coordinate strictly inside its box
    needs :math:`r_j = -\lambda\,\mathrm{sign}\,\beta_j`, one at a cap :math:`u_j>0` needs
    :math:`r_j \le -\lambda` and one at a floor :math:`l_j<0` needs :math:`r_j\ge\lambda`.
    A zero coordinate needs :math:`r_j\ge-\lambda` and :math:`r_j\le\lambda`, less the
    side a sign constraint (:math:`l_j=0` or :math:`u_j=0`) takes away. Primal
    feasibility is included.
    """
    gram, c = prob.x.T @ prob.x, prob.x.T @ prob.y
    r = gram @ beta - c + prob.a.T @ kappa
    tol = 1e-9 * max(1.0, np.abs(beta).max())
    zero = np.abs(beta) <= tol
    at_up, at_lo = ~zero & (beta >= prob.upper - tol), ~zero & (beta <= prob.lower + tol)
    inner = ~zero & ~at_up & ~at_lo
    viol = np.concatenate([
        np.abs(r[inner] + lam * np.sign(beta[inner])),
        np.maximum(r[at_up] + lam, 0),
        np.maximum(lam - r[at_lo], 0),
        np.maximum(r[zero & (prob.lower < 0)] - lam, 0),
        np.maximum(-r[zero & (prob.upper > 0)] - lam, 0),
        np.abs(prob.a @ beta - prob.b),
        np.maximum(beta - prob.upper, 0),
        np.maximum(prob.lower - beta, 0),
    ])
    return float(viol.max(initial=0.0) / np.abs(c).max())


def reference(prob: Lasso, lam: float) -> np.ndarray:
    """The same LASSO at one lambda, from an interior-point solver."""
    beta = cp.Variable(prob.x.shape[1])
    cons = []
    if prob.a.shape[0]:
        cons.append(prob.a @ beta == prob.b)
    if np.isfinite(prob.upper).any():
        i = np.isfinite(prob.upper)
        cons.append(beta[i] <= prob.upper[i])
    if np.isfinite(prob.lower).any():
        i = np.isfinite(prob.lower)
        cons.append(beta[i] >= prob.lower[i])
    obj = 0.5 * cp.sum_squares(prob.y - prob.x @ beta) + lam * cp.norm1(beta)
    cp.Problem(cp.Minimize(obj), cons).solve(**CONIC)
    return beta.value


def midpoints(path: LassoPath) -> tuple[np.ndarray, np.ndarray]:
    """Lambda halfway along each segment, and the path's beta there (it is affine in lambda)."""
    return 0.5 * (path.lam[:-1] + path.lam[1:]), 0.5 * (path.beta[:-1] + path.beta[1:])


# --------------------------------------------------------------------------------------
# The test problems
# --------------------------------------------------------------------------------------
def diabetes() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The diabetes data of Efron et al. (2004): 442 patients, 10 baseline variables."""
    d = load_diabetes()
    return d.data, d.target - d.target.mean(), list(d.feature_names)


def wide(seed: int = SEED, n: int = 40, p: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """More predictors than observations, with a sparse truth."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:8] = rng.choice([-1, 1], 8) * rng.uniform(1, 3, 8)
    return x, x @ beta + 0.5 * rng.standard_normal(n)


COMPOSITION_TRUTH = np.zeros(30)
# The negative parts are odd-numbered, so the truth meets the signs of Zero-sum, signs:
# every even-numbered part has a nonnegative coefficient.
COMPOSITION_TRUTH[[0, 3, 7, 13, 21]] = [1.2, -0.8, 0.9, -0.6, -0.7]


def compositional(seed: int = SEED, n: int = 60, p: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Log-compositions as predictors, with a zero-sum truth (Lin et al., 2014)."""
    rng = np.random.default_rng(seed)
    raw = np.exp(rng.standard_normal((n, p)) + np.linspace(0, 2, p))
    z = np.log(raw / raw.sum(axis=1, keepdims=True))
    y = z @ COMPOSITION_TRUTH[:p] + 0.3 * rng.standard_normal(n)
    z -= z.mean(axis=0)
    return z, y - y.mean()


# The sizes of the scale test: observations and parts of a zero-sum problem.
SCALE = [(200, 100), (400, 200), (1000, 500)]


def large_composition(n: int, p: int, seed: int = SEED, k: int = 10) -> Lasso:
    """A zero-sum LASSO on log-compositions of ``p`` parts, ``k`` of them in a zero-sum truth."""
    rng = np.random.default_rng(seed)
    raw = np.exp(rng.standard_normal((n, p)) + np.linspace(0, 2, p))
    z = np.log(raw / raw.sum(axis=1, keepdims=True))
    beta = np.zeros(p)
    values = rng.uniform(0.5, 1.5, k) * rng.choice([-1, 1], k)
    beta[rng.choice(p, k, replace=False)] = values - values.mean()
    y = z @ beta + 0.3 * rng.standard_normal(n)
    z -= z.mean(axis=0)
    return Lasso(z, y - y.mean(), a=np.ones(p), b=np.zeros(1))


@dataclass
class ScaleRow:
    """One line of the scale table: the budget form on a large zero-sum problem."""

    n: int
    p: int
    knots: int
    cvxcla: float  # knot gap against cvxcla's Lasso
    kkt: float


def scale() -> list[ScaleRow]:
    """The budget form on growing zero-sum problems, checked without an interior-point solver.

    A midpoint solve per segment would cost more than everything else together at p = 500,
    so the checks here are the two that scale: the optimality conditions, and cvxcla.
    """
    rows = []
    for n, p in SCALE:
        prob = large_composition(n, p)
        path = budget_form(prob)
        rows.append(ScaleRow(n, p, len(path.lam), against_cvxcla(prob, path), certify(prob, path)))
    return rows


def scale_tex(rows: list[ScaleRow]) -> str:
    """The body of Table 3, for the note to input, with its closing rule."""
    return "".join(f"{r.n:>4} & {r.p:>3} & {r.knots:>3} & {sci(r.cvxcla)} & {sci(r.kkt)}\\\\\n"
                   for r in rows) + "\\bottomrule\n"


# Small counts as the note spells them out.
NUMBERS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def signed_parts(p: int) -> np.ndarray:
    """The parts of Zero-sum, signs whose coefficient must be nonnegative: the even-numbered ones."""
    return np.arange(p) % 2 == 0


def problems() -> dict[str, Lasso]:
    """Every problem the note reports, by the name its table uses."""
    xd, yd, _ = diabetes()
    xw, yw = wide()
    xc, yc = compositional()
    p = xd.shape[1]
    ols = np.linalg.lstsq(xd, yd, rcond=None)[0]
    cap = 0.5 * np.abs(ols).max()
    point = xd[0] / np.abs(xd[0]).max()  # pin the fit at the first patient, rescaled
    return {
        "plain": Lasso(xd, yd),
        "nonnegative": Lasso(xd, yd, lower=np.zeros(p)),
        "wide": Lasso(xw, yw),
        "box": Lasso(xd, yd, lower=np.full(p, -cap), upper=np.full(p, cap)),
        "pinned": Lasso(xd, yd, a=point, b=np.array([yd[0] / np.abs(xd[0]).max()])),
        "zero-sum": Lasso(xc, yc, a=np.ones(xc.shape[1]), b=np.zeros(1)),
        "zero-sum, signs": Lasso(xc, yc, a=np.ones(xc.shape[1]), b=np.zeros(1),
                                 lower=np.where(signed_parts(xc.shape[1]), 0.0, -np.inf)),
    } | ({"sCD14": scd14()[0]} if SCD14.exists() else {})


# --------------------------------------------------------------------------------------
# The opening example
# --------------------------------------------------------------------------------------
@dataclass
class Example:
    """The opening example: the zero-sum LASSO with and without the known signs.

    The signs are those of Zero-sum, signs: every even-numbered part has a nonnegative
    coefficient, which the truth meets. c-lasso can impose the sum but not the signs.
    """

    lam: float  # mid-segment, where the zero-sum LASSO selects exactly the true parts
    truth: np.ndarray  # indices of the parts with a nonzero true coefficient
    beta: np.ndarray  # (p,) the zero-sum LASSO there, from the CLA
    knots: int  # knots of the zero-sum path without the signs
    broken: int  # of those, the knots where some even-numbered part is negative
    wrong: int  # even-numbered parts that are negative somewhere on that path
    signed: int  # knots of the path with the signs


def support(beta: np.ndarray) -> np.ndarray:
    """The nonzero coefficients, to rounding."""
    return np.flatnonzero(np.abs(beta) > 1e-9 * np.abs(beta).max())


def example() -> Example:
    """The compositional problem of Section 1, at the middle of its five-part segment."""
    prob = problems()["zero-sum"]
    path = budget_form(prob)
    truth = np.flatnonzero(COMPOSITION_TRUTH)
    seg = next(k for k in range(len(path.lam) - 1)
               if np.array_equal(support(path.beta[k + 1]), truth) and np.array_equal(
                   support(0.5 * (path.beta[k] + path.beta[k + 1])), truth))
    lam = 0.5 * (path.lam[seg] + path.lam[seg + 1])
    beta = 0.5 * (path.beta[seg] + path.beta[seg + 1])
    negative = path.beta[:, signed_parts(prob.x.shape[1])] < -1e-12 * np.abs(path.beta).max()
    signed = budget_form(problems()["zero-sum, signs"])
    return Example(lam, truth, beta, len(path.lam), int(negative.any(axis=1).sum()),
                   int(negative.any(axis=0).sum()), len(signed.lam))


# The real-data example: 16S counts of 60 genera in 151 samples, and the inflammation
# marker sCD14 (Noguera-Julian et al., 2016, via coda4microbiome). The data are not ours
# and are not committed: fetch_scd14.py downloads them, and without them this is skipped.
SCD14 = OUT.parent / "data" / "scd14.csv"
SCD14_SHA256 = "3c6dcd6c18e10ea6f09bdd34329df3a8ce0a11dccd8026bf6c9a71f3c086d0da"


def scd14() -> tuple[Lasso, list[str]]:
    """The zero-sum LASSO of sCD14 (in thousands) on the log-shares of the genera.

    A count of zero has no logarithm, so every count gets a pseudocount of one first, a
    common choice for such data.
    """
    with SCD14.open() as f:
        header = f.readline().strip().split(",")
    table = np.loadtxt(SCD14, delimiter=",", skiprows=1, usecols=range(1, len(header)))
    counts, y = table[:, :-1] + 1.0, table[:, -1] / 1000
    z = np.log(counts / counts.sum(axis=1, keepdims=True))
    z -= z.mean(axis=0)
    p = z.shape[1]
    return Lasso(z, y - y.mean(), a=np.ones(p), b=np.zeros(1)), header[1:-1]


def print_scd14() -> None:
    """The real-data example, as the note quotes it."""
    prob, names = scd14()
    snapshot = hashlib.sha256(SCD14.read_bytes()).hexdigest() == SCD14_SHA256
    path = budget_form(prob)
    first = list(dict.fromkeys(names[j] for b in path.beta for j in support(b)))[:5]
    print(f"\nsCD14 ({'the note' if snapshot else 'NOT the note'}'s snapshot): n = {prob.x.shape[0]}, "
          f"p = {prob.x.shape[1]}, {len(path.lam)} knots, KKT {certify(prob, path):.1e}, "
          f"cvxcla {against_cvxcla(prob, path):.1e}")
    print(f"  first to enter: {', '.join(first)}")


def example_tex(ex: Example) -> str:
    """The example's numbers as LaTeX macros, for the note to input."""
    return "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in {
        "exParts": str(len(ex.beta)),
        "exKnots": str(ex.knots),
        "exBroken": str(ex.broken),
        "exWrong": NUMBERS[ex.wrong],
        "exSigned": str(ex.signed),
    }.items())


# --------------------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------------------
def against_lars(prob: Lasso, path: LassoPath, positive: bool = False) -> float:
    """Largest knot discrepancy against scikit-learn's LARS, relative to the coefficient scale."""
    alphas, _, coefs = lars_path(prob.x, prob.y, method="lasso", positive=positive)
    lam = alphas * prob.x.shape[0]
    if positive:
        # In positive mode scikit-learn's last knot is labelled alpha = 0 but holds the path
        # at the last point where a variable could still have entered; nothing changes
        # state below it, so the true endpoint is the nonnegative least-squares fit.
        coefs[:, -1] = nnls(prob.x, prob.y)[0]
    assert len(lam) == len(path.lam), (len(lam), len(path.lam))
    scale = np.abs(coefs).max()
    return float(max(np.abs(path.beta - coefs.T).max() / scale, np.abs(path.lam - lam).max() / lam[0]))


def against_solver(prob: Lasso, path: LassoPath) -> float:
    """Largest midpoint discrepancy against the interior-point solve, relative."""
    lam, beta = midpoints(path)
    keep = lam > 1e-6 * path.lam[0]
    gaps = [np.abs(reference(prob, l) - b).max() for l, b in zip(lam[keep], beta[keep], strict=True)]
    return float(max(gaps) / np.abs(path.beta).max())


def by_cvxcla(prob: Lasso) -> tuple[np.ndarray, np.ndarray] | None:
    """The same path from the cvxcla package's ``Lasso``, as (lambda, beta), where it applies.

    ``cvxcla.Lasso`` takes ``nonneg=True`` for beta >= 0, rows ``G beta <= h`` with h > 0
    (a finite box is two of them per coefficient), or homogeneous rows ``A beta = 0``, one
    kind at a time. It has no nonhomogeneous rows and no signs on some coefficients only.
    """
    p = prob.x.shape[1]
    unbounded = np.isinf(prob.lower).all() and np.isinf(prob.upper).all()
    rows = prob.a.shape[0] > 0
    if np.any(prob.b):
        return None
    if unbounded:
        kwargs = {"a": prob.a} if rows else {}
    elif not rows and (prob.lower == 0).all() and np.isinf(prob.upper).all():
        kwargs = {"nonneg": True}
    elif not rows and np.isfinite(prob.lower).all() and np.isfinite(prob.upper).all():
        kwargs = {"g": np.vstack([np.eye(p), -np.eye(p)]), "h": np.concatenate([prob.upper, -prob.lower])}
    else:
        return None
    path = CvxclaLasso(prob.x, prob.y, **kwargs).path
    return np.array([k.lam for k in path]), np.array([k.beta for k in path])


def against_cvxcla(prob: Lasso, path: LassoPath) -> float | None:
    """Largest knot discrepancy against cvxcla's ``Lasso``, relative; None where it does not apply."""
    theirs = by_cvxcla(prob)
    if theirs is None:
        return None
    lam, beta = theirs
    assert len(lam) == len(path.lam), (len(lam), len(path.lam))
    return float(max(np.abs(path.beta - beta).max() / np.abs(beta).max(),
                     np.abs(path.lam - lam).max() / lam[0]))


def certify(prob: Lasso, path: LassoPath) -> float:
    """Largest KKT violation over every knot and every segment midpoint."""
    lam, beta = midpoints(path)
    kap = 0.5 * (path.kappa[:-1] + path.kappa[1:])
    knots = [kkt_residual(prob, *z) for z in zip(path.lam, path.beta, path.kappa, strict=True)]
    mids = [kkt_residual(prob, *z) for z in zip(lam, beta, kap, strict=True)]
    return max(knots + mids)


@dataclass
class Row:
    """One line of the note's table."""

    name: str
    form: str
    n: int
    p: int
    knots: int
    reference: str
    gap: float
    engine: float  # knot gap against the same form run on cvxcla's CLA
    cvxcla: float | None  # knot gap against cvxcla's Lasso, None where it takes no such problem
    kkt: float


def table() -> list[Row]:
    """Every problem, through every form that takes it."""
    probs = problems()
    rows = []
    for name, prob in probs.items():
        forms = [("penalty", penalty_form)]
        homogeneous = not np.any(prob.b) and np.isin(prob.lower, [0, -np.inf]).all() \
            and np.isin(prob.upper, [0, np.inf]).all()
        if homogeneous:
            forms.append(("budget", budget_form))
        if prob.a.shape[0] and not np.any(prob.b):
            forms = forms[1:]  # b = 0 makes the penalty form's start degenerate
        for form, fn in forms:
            path = fn(prob)
            if name in ("plain", "wide", "nonnegative"):
                ref, gap = "LARS", against_lars(prob, path, positive=name == "nonnegative")
            else:
                ref, gap = "IPM", against_solver(prob, path)
            theirs = fn(prob, engine=cvxcla_cla)
            assert len(theirs.lam) == len(path.lam), (name, form)
            engine = float(np.abs(theirs.beta - path.beta).max() / np.abs(path.beta).max())
            rows.append(Row(name, form, *prob.x.shape, len(path.lam), ref, gap, engine,
                            against_cvxcla(prob, path), certify(prob, path)))
    return rows


# --------------------------------------------------------------------------------------
# The figures
# --------------------------------------------------------------------------------------
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#d9d9d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def style() -> None:
    """Thin marks, recessive axes, the note's type size."""
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK, "legend.fontsize": 7, "legend.frameon": False,
        "pdf.fonttype": 42,
    })


def fig_path(out: Path) -> None:
    """The diabetes LASSO path from the CLA, with the LARS knots on top."""
    x, y, names = diabetes()
    prob = Lasso(x, y)
    path = penalty_form(prob)
    alphas, _, coefs = lars_path(x, y, method="lasso")
    norm = np.abs(path.beta).sum(axis=1)
    norm_l = np.abs(coefs).sum(axis=0)

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for v in norm:
        ax.axvline(v, color=GRID, lw=0.5, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.plot(norm, path.beta, "-", color=BLUE, lw=1.2, zorder=2)
    ax.plot(norm_l, coefs.T, "o", ms=4.5, mfc="none", mec=ORANGE, mew=0.9, zorder=3)
    # End labels, pushed apart where the paths finish too close to read.
    ends = path.beta[-1]
    gap = 0.045 * np.ptp(ends)
    order = np.argsort(ends)
    placed = ends[order].copy()
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    placed -= (placed - ends[order]).mean()
    for j, y_label in zip(order, placed, strict=True):
        ax.annotate(names[j], (norm[-1], ends[j]), xytext=(norm[-1] * 1.03, y_label), textcoords="data",
                    va="center", fontsize=6.5, color=INK,
                    arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.5, "shrinkA": 0, "shrinkB": 0})
    ax.plot([], [], "-", color=BLUE, lw=1.2, label="CLA, penalty form")
    ax.plot([], [], "o", ms=4.5, mfc="none", mec=ORANGE, mew=0.9, label="LARS knots")
    ax.legend(loc="upper left")
    ax.set_xlabel(r"$\|\beta\|_1$")
    ax.set_ylabel(r"coefficient $\beta_j$")
    ax.set_xlim(0, norm[-1] * 1.12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_portfolio(out: Path) -> None:
    """The same path as the budget form sees it: long, short and cash weights against 1/t."""
    x, y, _ = diabetes()
    path = budget_form(Lasso(x, y))
    t = path.turns
    keep = np.isfinite(t.gamma) & (t.gamma > 0)
    budget = 1.0 / t.gamma[keep]
    w = t.x[keep]
    p = x.shape[1]
    long_, short, cash = w[:, :p].sum(axis=1), w[:, p : 2 * p].sum(axis=1), w[:, -1]
    # Past the last knot the weights move along a ray; one more point shows it.
    tail = 1.6 * budget[-1]
    beta_end = path.beta[-1]
    budget = np.append(budget, tail)
    long_ = np.append(long_, np.clip(beta_end, 0, None).sum() / tail)
    short = np.append(short, np.clip(-beta_end, 0, None).sum() / tail)
    cash = np.append(cash, 1 - np.abs(beta_end).sum() / tail)

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.stackplot(budget, long_, short, cash, colors=[BLUE, ORANGE, GRID], edgecolor="white", lw=0.6,
                 labels=["long book", "short book", "cash"])
    ax.axvline(budget[-2], color=MUTED, lw=0.6, ls="--")
    ax.annotate("cash enters:\nend of the LASSO path", (budget[-2], 0.92), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="top", fontsize=6.5, color=INK)
    ax.set_xlim(0, tail)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\ell_1$ budget $t = 1/\gamma$")
    ax.set_ylabel("share of the budget")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def path_panel(ax: plt.Axes, prob: Lasso, path: LassoPath) -> None:
    """A CLA path against lambda, with interior-point solves at the segment midpoints."""
    lam, _ = midpoints(path)
    lam = lam[lam > 1e-6 * path.lam[0]]
    ref = np.array([reference(prob, v) for v in lam])
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.plot(path.lam, path.beta, "-", color=BLUE, lw=1.1, zorder=2)
    ax.plot(lam, ref, "o", ms=3.8, mfc="none", mec=ORANGE, mew=0.8, zorder=3)
    ax.set_xscale("symlog", linthresh=path.lam[0] / 50)
    ax.set_xlim(path.lam[0] * 1.05, 0)
    ax.set_xlabel(r"penalty $\lambda$ (decreasing)")
    ax.set_ylabel(r"coefficient $\beta_j$")


def path_legend(fig: plt.Figure) -> None:
    """The legend shared by the path panels."""
    handles = [
        plt.Line2D([], [], color=BLUE, lw=1.1, label="CLA path"),
        plt.Line2D([], [], ls="none", marker="o", ms=3.8, mfc="none", mec=ORANGE, mew=0.8,
                   label="interior-point solve, mid-segment"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=2)


def fig_scd14(out: Path) -> None:
    """The real-data example: the zero-sum path of sCD14 on 60 genera."""
    prob, _ = scd14()
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    path_panel(ax, prob, budget_form(prob))
    ax.set_title("sCD14 on 60 genera, zero-sum LASSO by the CLA", fontsize=8, color=INK)
    path_legend(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def scd14_tex() -> str:
    """The real-data example's numbers as LaTeX macros, for the note to input."""
    prob, _ = scd14()
    path = budget_form(prob)
    return "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in {
        "rdSamples": str(prob.x.shape[0]),
        "rdGenera": str(prob.x.shape[1]),
        "rdKnots": str(len(path.lam)),
    }.items())


def fig_example(out: Path) -> None:
    """The opening example, as the known signs see it: the even-numbered parts, near zero.

    Left, the zero-sum LASSO alone, the model c-lasso fits; right, with the signs. The
    shaded half-plane is what the signs rule out, and the parts the left path pushes into
    it are drawn in orange. Part 0, the one even part in the truth, leaves the window.
    """
    probs = problems()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, name, title in ((axes[0], "zero-sum", r"$\sum_j\beta_j=0$ only: what c-lasso fits"),
                            (axes[1], "zero-sum, signs", r"and $\beta_j\geq0$ for even $j$: only the CLA")):
        prob = probs[name]
        path = budget_form(prob)
        even = signed_parts(prob.x.shape[1])
        wrong = even & (path.beta < -1e-12 * np.abs(path.beta).max()).any(axis=0)
        lam, _ = midpoints(path)
        lam = lam[lam > 1e-6 * path.lam[0]]
        ref = np.array([reference(prob, v) for v in lam])
        ax.axhspan(-1.0, 0.0, color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.plot(path.lam, path.beta[:, even & ~wrong], "-", color=BLUE, lw=1.1, zorder=2)
        ax.plot(path.lam, path.beta[:, wrong], "-", color=ORANGE, lw=1.4, zorder=3)
        ax.plot(lam, ref[:, even], "o", ms=3.4, mfc="none", mec=INK, mew=0.6, zorder=4)
        ax.set_xscale("symlog", linthresh=path.lam[0] / 50)
        ax.set_xlim(path.lam[0] * 1.05, 0)
        ax.set_ylim(-0.08, 0.25)
        ax.set_xlabel(r"penalty $\lambda$ (decreasing)")
        ax.set_title(title, fontsize=8, color=INK)
    axes[0].set_ylabel(r"even-numbered $\beta_j$")
    axes[0].text(0.03, 0.06, "ruled out by the signs", transform=axes[0].transAxes, fontsize=6.5, color=MUTED)
    handles = [
        plt.Line2D([], [], color=BLUE, lw=1.1, label="CLA path"),
        plt.Line2D([], [], color=ORANGE, lw=1.4, label="negative, against the known sign"),
        plt.Line2D([], [], ls="none", marker="o", ms=3.4, mfc="none", mec=INK, mew=0.6,
                   label="interior-point solve, mid-segment"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def sci(x: float) -> str:
    """A number as the note prints it, ``$8.6\\times10^{-14}$``."""
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def table_tex(rows: list[Row]) -> str:
    """The body of Table 2, one LaTeX row per problem and form, for the note to input.

    The closing rule comes with it: after an ``\\input`` inside a tabular, LaTeX no longer
    accepts a ``\\bottomrule`` of the note's own.
    """
    return "".join(
        f"{r.name:<15} & {r.form:<7} & {r.n:>3} & {r.p:>3} & {r.knots:>2} & {r.reference:<4} & "
        f"{sci(r.gap):<19} & {sci(r.engine):<19} & {'---' if r.cvxcla is None else sci(r.cvxcla):<19} & "
        f"{sci(r.kkt)}\\\\\n"
        for r in rows
    ) + "\\bottomrule\n"


def main() -> None:
    """Write the figures and Table 2, and print the table."""
    OUT.mkdir(exist_ok=True)
    style()
    fig_path(OUT / "path.pdf")
    fig_portfolio(OUT / "portfolio.pdf")
    fig_example(OUT / "example.pdf")
    ex = example()
    (OUT / "example.tex").write_text(example_tex(ex))
    rows = table()
    (OUT / "evidence.tex").write_text(table_tex(rows))
    big = scale()
    (OUT / "scale.tex").write_text(scale_tex(big))
    print("\nscale, budget form on zero-sum compositions:")
    for r in big:
        print(f"  n = {r.n:4d}, p = {r.p:3d}: {r.knots:3d} knots, cvxcla {r.cvxcla:.1e}, KKT {r.kkt:.1e}")
    print(f"{'problem':<16} {'form':<8} {'n':>4} {'p':>4} {'knots':>6} {'ref':>5} {'gap':>9} "
          f"{'CLA 2.4':>9} {'Lasso':>9} {'KKT':>9}")
    for r in rows:
        cvx = "---" if r.cvxcla is None else f"{r.cvxcla:.1e}"
        print(f"{r.name:<16} {r.form:<8} {r.n:>4} {r.p:>4} {r.knots:>6} {r.reference:>5} "
              f"{r.gap:9.1e} {r.engine:9.1e} {cvx:>9} {r.kkt:9.1e}")
    # The exact zero-sum paths, for classo_check.py, which runs c-lasso in its own environment.
    saved = {"example": problems()["zero-sum"]} | ({"scd14": scd14()[0]} if SCD14.exists() else {})
    arrays = {}
    for name, prob in saved.items():
        path = budget_form(prob)
        arrays |= {f"{name}.x": prob.x, f"{name}.y": prob.y, f"{name}.lam": path.lam, f"{name}.beta": path.beta}
    np.savez(OUT / "zero_sum_paths.npz", **arrays)
    # Every problem of Table 2 with its exact path, for julia/gaines_check.jl, which runs
    # ConstrainedLasso.jl on them. The path is from the first form that takes the problem.
    arrays = {}
    for name, prob in problems().items():
        path = (budget_form if prob.a.shape[0] and not np.any(prob.b) else penalty_form)(prob)
        key = name.replace(", ", "-")
        fields = {"x": prob.x, "y": prob.y, "a": prob.a, "b": prob.b, "lower": prob.lower,
                  "upper": prob.upper, "lam": path.lam, "beta": path.beta}
        # NPZ.jl cannot read an empty array, so a problem without rows has no "a" and "b"
        arrays |= {f"{key}.{k}": np.ascontiguousarray(v) for k, v in fields.items() if v.size}
    np.savez(OUT / "table_paths.npz", **arrays)
    if SCD14.exists():
        fig_scd14(OUT / "scd14.pdf")
        (OUT / "scd14.tex").write_text(scd14_tex())
        print_scd14()
    else:
        print(f"\nreal data skipped: {SCD14} not found (fetch it with `uv run python/fetch_scd14.py`)")


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rows() -> list[Row]:
    """The table, computed once."""
    return table()


def test_every_problem_reported(rows: list[Row]) -> None:
    """Every problem goes through at least one form."""
    assert {r.name for r in rows} == set(problems())


def test_lars_knots(rows: list[Row]) -> None:
    """Where LARS applies, the CLA returns its knots to rounding."""
    for r in rows:
        if r.reference == "LARS":
            assert r.gap < 1e-10, r


def test_solver_midpoints(rows: list[Row]) -> None:
    """Elsewhere the CLA agrees with an interior-point solve to the solver's own accuracy."""
    for r in rows:
        if r.reference == "IPM":
            assert r.gap < 1e-6, r


def test_kkt(rows: list[Row]) -> None:
    """Every knot and midpoint satisfies the LASSO's optimality conditions to rounding."""
    for r in rows:
        assert r.kkt < 1e-10, r


def test_forms_agree() -> None:
    """The two forms return the same knots wherever both apply."""
    for name, prob in problems().items():
        if name in ("plain", "nonnegative", "wide"):
            pa, bu = penalty_form(prob), budget_form(prob)
            assert len(pa.lam) == len(bu.lam)
            assert np.abs(pa.beta - bu.beta).max() < 1e-9 * np.abs(pa.beta).max()


def test_cvxcla_cla(rows: list[Row]) -> None:
    """Both forms run unchanged on cvxcla's released CLA, on every problem, to the same knots."""
    for r in rows:
        assert r.engine < 1e-10, r


def test_cvxcla(rows: list[Row]) -> None:
    """cvxcla's Lasso returns the same knots wherever it takes the problem."""
    expected = {"plain", "nonnegative", "wide", "box", "zero-sum"} | ({"sCD14"} if SCD14.exists() else set())
    assert {r.name for r in rows if r.cvxcla is not None} == expected
    for r in rows:
        if r.cvxcla is not None:
            assert r.cvxcla < 1e-10, r


def test_table_tex(rows: list[Row]) -> None:
    """Table 2 has one row per problem and form, with the numbers as the note prints them."""
    *tex, rule = table_tex(rows).splitlines()
    assert len(tex) == len(rows)
    assert rule == r"\bottomrule"
    assert sci(8.6e-14) == r"$8.6\times10^{-14}$"
    assert all(line.endswith(r"\\") and line.count("&") == 9 for line in tex)


def test_example() -> None:
    """The opening example: the CLA finds the true parts and keeps the sum at zero."""
    ex = example()
    assert np.array_equal(support(ex.beta), ex.truth)
    assert abs(ex.beta.sum()) < 1e-12 * np.abs(ex.beta).max()
    assert np.abs(reference(problems()["zero-sum"], ex.lam) - ex.beta).max() < 1e-6 * np.abs(ex.beta).max()
    # The truth meets the known signs; without them the path breaks them, with them it does not.
    even = signed_parts(len(COMPOSITION_TRUTH))
    assert (COMPOSITION_TRUTH[even] >= 0).all()
    assert ex.broken > ex.knots // 2 and ex.wrong >= 3
    signed = budget_form(problems()["zero-sum, signs"])
    assert (signed.beta[:, even] >= -1e-12 * np.abs(signed.beta).max()).all()


@pytest.mark.skipif(not SCD14.exists(), reason="no sCD14 data; fetch it with `uv run python/fetch_scd14.py`")
def test_scd14() -> None:
    """On the real data the CLA is exact."""
    assert hashlib.sha256(SCD14.read_bytes()).hexdigest() == SCD14_SHA256
    prob, _ = scd14()
    path = budget_form(prob)
    assert len(path.lam) == 60
    assert certify(prob, path) < 1e-10
    assert against_cvxcla(prob, path) < 1e-10


def test_scale() -> None:
    """At p = 500 the path is still exact, and cvxcla still returns the same knots."""
    for r in scale():
        assert r.kkt < 1e-10, r
        assert r.cvxcla < 1e-10, r


def test_degenerate_start_refused() -> None:
    """With b = 0 the penalty form starts on a degenerate vertex, and says so."""
    with pytest.raises(ValueError, match="degenerate start"):
        penalty_form(problems()["zero-sum"])


# --------------------------------------------------------------------------------------
# Timings: machine-dependent, so run by hand and quoted only as ratios
# --------------------------------------------------------------------------------------
TIMING_REPEATS = 5  # each timed step runs this often, and the fastest run counts
GRID_POINTS = 100  # points in the lambda grid of the interior-point baseline


def machine() -> str:
    """The CPU, thread count and Python of this run, which the timings depend on."""
    name = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        models = [line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                  if line.startswith("model name")]
        name = models[0] if models else name
    elif sys.platform == "darwin":
        brand = ["sysctl", "-n", "machdep.cpu.brand_string"]
        name = subprocess.run(brand, capture_output=True, text=True, check=False).stdout.strip() or name
    return f"{name or 'unknown CPU'}, {os.cpu_count()} threads, Python {platform.python_version()}"


def fastest(step: Callable[[], object], repeats: int = TIMING_REPEATS) -> float:
    """Seconds of the fastest of ``repeats`` runs of ``step``.

    Other work on the machine can only slow a run down, so the fastest run is the
    closest to what the step itself costs, and it varies far less than a single run.
    """
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        step()
        seconds.append(time.perf_counter() - start)
    return min(seconds)


def grid(prob: Lasso, lam_max: float) -> Callable[[], None]:
    """Solve the constrained LASSO at GRID_POINTS values of lambda, log-spaced down to lam_max / 1000.

    The problem is built once with lambda as a parameter, so CVXPY compiles it once and
    each point costs one Clarabel solve at its default tolerances: the fairest setup for
    the solver.
    """
    beta, lam = cp.Variable(prob.x.shape[1]), cp.Parameter(nonneg=True)
    cons = [prob.a @ beta == prob.b] if prob.a.shape[0] else []
    if np.isfinite(prob.upper).any():
        i = np.isfinite(prob.upper)
        cons.append(beta[i] <= prob.upper[i])
    if np.isfinite(prob.lower).any():
        i = np.isfinite(prob.lower)
        cons.append(beta[i] >= prob.lower[i])
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(prob.y - prob.x @ beta) + lam * cp.norm1(beta)), cons)

    def solve() -> None:
        for value in np.geomspace(lam_max, lam_max / 1000, GRID_POINTS):
            lam.value = value
            problem.solve(solver=cp.CLARABEL)
    return solve


def timings() -> None:
    """Print, for each constrained problem, what the path costs against cvxcla and a grid."""
    print(f"machine: {machine()}; each time the fastest of {TIMING_REPEATS} runs")
    forms = {"box": penalty_form, "zero-sum": budget_form, "sCD14": budget_form}
    for name, prob in problems().items():
        if name not in forms:
            continue
        form = forms[name]
        path = form(prob)
        cla_s = fastest(lambda prob=prob, form=form: form(prob))
        cvx_s = fastest(lambda prob=prob: by_cvxcla(prob))
        grid_s = fastest(grid(prob, path.lam[0]), repeats=3)  # the first run compiles; later ones reuse it
        print(f"  {name:<9} {len(path.lam):3d} knots: CLA {cla_s * 1e3:7.1f} ms, cvxcla {cvx_s / cla_s:5.1f}x, "
              f"{GRID_POINTS}-point Clarabel grid {grid_s / cla_s:6.0f}x the CLA")
    print("Table 3, the zero-sum problems at scale (the grid timed twice: the first run compiles):")
    for n, p in SCALE:
        prob = large_composition(n, p)
        path = budget_form(prob)
        cla_s = fastest(lambda prob=prob: budget_form(prob))
        cvx_s = fastest(lambda prob=prob: by_cvxcla(prob))
        grid_s = fastest(grid(prob, path.lam[0]), repeats=2)
        print(f"  n = {n:4d}, p = {p:3d}, {len(path.lam):3d} knots: CLA {cla_s:6.2f} s, cvxcla {cvx_s / cla_s:4.1f}x, "
              f"{GRID_POINTS}-point Clarabel grid {grid_s / cla_s:4.0f}x the CLA")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        sys.exit(pytest.main([__file__, "-q"]))
    if sys.argv[1:] == ["timings"]:
        sys.exit(timings())
    main()
