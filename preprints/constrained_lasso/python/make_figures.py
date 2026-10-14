# /// script
# requires-python = "==3.12.*"
# dependencies = [
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

    uv run python/make_figures.py        # the three figures, and every number the note quotes
    uv run python/make_figures.py test   # the same numbers as pytest checks, with tolerances

Seeded and offline: the diabetes data ship with scikit-learn.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pytest
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


def penalty_form(prob: Lasso) -> LassoPath:
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
    turns = cla(s.T @ gram @ s, -np.ones(s.shape[1]), prob.a @ s, prob.b,
                np.zeros(s.shape[1]), cap, q=s.T @ c)
    keep = np.isfinite(turns.gamma)  # the gamma = inf vertex is the gamma_1 knot again
    return LassoPath(lam=turns.gamma[keep], beta=turns.x[keep] @ s.T, kappa=turns.nu[keep], turns=turns)


def budget_form(prob: Lasso) -> LassoPath:
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
    turns = cla(sigma, mu, rows, np.append(1.0, np.zeros(m)), np.zeros(k + 1), np.full(k + 1, np.inf),
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
    :math:`r = G\beta - c + A^\top\kappa`, a coordinate strictly inside its box needs
    :math:`r_j = -\lambda\,\mathrm{sign}\,\beta_j` if nonzero and :math:`|r_j|\le\lambda`
    if zero; one at its upper bound needs :math:`r_j \le -\lambda`, at its lower bound
    :math:`r_j\ge\lambda`. Primal feasibility is included.
    """
    gram, c = prob.x.T @ prob.x, prob.x.T @ prob.y
    r = gram @ beta - c + prob.a.T @ kappa
    tol = 1e-9 * max(1.0, np.abs(beta).max())
    at_up, at_lo = beta >= prob.upper - tol, beta <= prob.lower + tol
    zero = (np.abs(beta) <= tol) & ~at_up & ~at_lo
    inner = ~zero & ~at_up & ~at_lo
    viol = np.concatenate([
        np.abs(r[inner] + lam * np.sign(beta[inner])),
        np.maximum(np.abs(r[zero]) - lam, 0),
        np.maximum(r[at_up & ~zero] + lam, 0),
        np.maximum(-r[at_lo & ~zero] - lam, 0),
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


def compositional(seed: int = SEED, n: int = 60, p: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Log-compositions as predictors, with a zero-sum truth (Lin et al., 2014)."""
    rng = np.random.default_rng(seed)
    raw = np.exp(rng.standard_normal((n, p)) + np.linspace(0, 2, p))
    z = np.log(raw / raw.sum(axis=1, keepdims=True))
    beta = np.zeros(p)
    beta[[0, 3, 7, 12, 20]] = [1.2, -0.8, 0.9, -0.6, -0.7]
    y = z @ beta + 0.3 * rng.standard_normal(n)
    z -= z.mean(axis=0)
    return z, y - y.mean()


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
                                 lower=np.where(np.arange(xc.shape[1]) % 2 == 0, 0.0, -np.inf)),
    }


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
        if name.startswith("zero-sum"):
            forms = forms[1:]  # b = 0 makes the penalty form's start degenerate
        for form, fn in forms:
            path = fn(prob)
            if name in ("plain", "wide", "nonnegative"):
                ref, gap = "LARS", against_lars(prob, path, positive=name == "nonnegative")
            else:
                ref, gap = "IPM", against_solver(prob, path)
            rows.append(Row(name, form, *prob.x.shape, len(path.lam), ref, gap, certify(prob, path)))
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

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
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

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
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


def fig_constrained(out: Path) -> None:
    """Left: the box-constrained diabetes path. Right: the zero-sum compositional path."""
    probs = problems()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, name, fn in ((left, "box", penalty_form), (right, "zero-sum", budget_form)):
        prob = probs[name]
        path = fn(prob)
        lam, _ = midpoints(path)
        lam = lam[lam > 1e-6 * path.lam[0]]
        ref = np.array([reference(prob, v) for v in lam])
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.plot(path.lam, path.beta, "-", color=BLUE, lw=1.1, zorder=2)
        ax.plot(lam, ref, "o", ms=3.8, mfc="none", mec=ORANGE, mew=0.8, zorder=3)
        ax.set_xscale("symlog", linthresh=path.lam[0] / 50)
        ax.set_xlim(path.lam[0] * 1.05, 0)
        ax.set_xlabel(r"penalty $\lambda$ (decreasing)")
    cap = probs["box"].upper[0]
    for v in (cap, -cap):
        left.axhline(v, color=MUTED, lw=0.6, ls="--")
    left.set_ylabel(r"coefficient $\beta_j$")
    left.set_title(r"diabetes, $|\beta_j|\leq u$ (penalty form)", fontsize=8, color=INK)
    right.set_title(r"log-compositions, $\sum_j\beta_j=0$ (budget form)", fontsize=8, color=INK)
    handles = [
        plt.Line2D([], [], color=BLUE, lw=1.1, label="CLA path"),
        plt.Line2D([], [], ls="none", marker="o", ms=3.8, mfc="none", mec=ORANGE, mew=0.8,
                   label="interior-point solve, mid-segment"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    """Write the figures and print every number the note quotes."""
    OUT.mkdir(exist_ok=True)
    style()
    fig_path(OUT / "path.pdf")
    fig_portfolio(OUT / "portfolio.pdf")
    fig_constrained(OUT / "constrained.pdf")
    print(f"{'problem':<16} {'form':<8} {'n':>4} {'p':>4} {'knots':>6} {'ref':>5} {'gap':>9} {'KKT':>9}")
    for r in table():
        print(f"{r.name:<16} {r.form:<8} {r.n:>4} {r.p:>4} {r.knots:>6} {r.reference:>5} "
              f"{r.gap:9.1e} {r.kkt:9.1e}")


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


def test_degenerate_start_refused() -> None:
    """With b = 0 the penalty form starts on a degenerate vertex, and says so."""
    with pytest.raises(ValueError, match="degenerate start"):
        penalty_form(problems()["zero-sum"])


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        sys.exit(pytest.main([__file__, "-q"]))
    main()
