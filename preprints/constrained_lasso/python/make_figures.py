# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "cvxcla==2.7.1",
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
Algorithm computes LASSO paths. Every path comes from one call, ``cvxcla.Lasso``, which
encodes the LASSO as a CLA in the penalty or the budget form the note describes and runs
Markowitz's algorithm unchanged. This script holds no CLA and no encoding: it states the
problems, calls ``cvxcla``, and checks what comes back without it, against LARS, an
interior-point solver and the LASSO's optimality conditions.

    uv run python/make_figures.py        # the figures, Table 2 as figures/evidence.tex,
                                         # the example's numbers as figures/example.tex,
                                         # Table 4 of hostile problems as figures/stress.tex
    uv run python/make_figures.py test   # the same numbers as pytest checks, with tolerances
    uv run python/make_figures.py timings  # the cost ratios the note quotes, and the machine
    uv run python/make_figures.py export   # only figures/zero_sum_paths.npz, for the checks that
                                         # run c-lasso and comlasso on it

Seeded and offline: the diabetes data ship with scikit-learn.
"""

from __future__ import annotations

import hashlib
import itertools
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
from cvxcla import Lasso as CvxclaLasso
from matplotlib.transforms import blended_transform_factory
from scipy.optimize import linprog, nnls
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lars_path

SEED = 7
OUT = Path(__file__).resolve().parent.parent / "figures"  # the preprint folder; this script lives in python/

# The baseline is an interior-point solve asked for everything it has; around 1e-8 relative
# is its floor, not the path's.
CONIC = {"solver": cp.CLARABEL, "tol_gap_abs": 1e-12, "tol_gap_rel": 1e-12, "tol_feas": 1e-12}


# --------------------------------------------------------------------------------------
# The constrained LASSO, and its path from cvxcla
# --------------------------------------------------------------------------------------
@dataclass
class Lasso:
    r""":math:`\min \tfrac12\|y-X\beta\|^2 + \lambda\|\beta\|_1` s.t. :math:`A\beta=b,\ C\beta\le h,\ l\le\beta\le u`."""

    x: np.ndarray
    y: np.ndarray
    a: np.ndarray | None = None
    b: np.ndarray | None = None
    lower: np.ndarray | None = None  # each entry <= 0
    upper: np.ndarray | None = None  # each entry >= 0
    c: np.ndarray | None = None  # inequality rows C beta <= h
    h: np.ndarray | None = None  # any sign; h_r < 0 excludes beta = 0

    def __post_init__(self) -> None:
        """Fill in the absent constraints as the empty ones."""
        p = self.x.shape[1]
        self.a = np.zeros((0, p)) if self.a is None else np.atleast_2d(self.a)
        self.b = np.zeros(self.a.shape[0]) if self.b is None else np.atleast_1d(self.b).astype(float)
        self.lower = np.full(p, -np.inf) if self.lower is None else self.lower
        self.upper = np.full(p, np.inf) if self.upper is None else self.upper
        self.c = np.zeros((0, p)) if self.c is None else np.atleast_2d(self.c)
        self.h = np.zeros(self.c.shape[0]) if self.h is None else np.atleast_1d(self.h).astype(float)


@dataclass
class LassoPath:
    """Knots of the LASSO path, from the largest lambda down to zero."""

    lam: np.ndarray  # (k,)
    beta: np.ndarray  # (k, p)


def trace(prob: Lasso) -> LassoPath:
    """The exact path from ``cvxcla.Lasso``, which picks the CLA encoding that takes the problem.

    The bounds are always passed, infinite ones included: without any constraint
    ``cvxcla`` would trace the plain LASSO by its homotopy instead, and the point here is
    that the CLA computes every path.
    """
    path = CvxclaLasso(
        prob.x, prob.y,
        a=prob.a if prob.a.shape[0] else None,
        b=prob.b if prob.a.shape[0] else None,
        lower=prob.lower,
        upper=prob.upper,
        g=prob.c if prob.c.shape[0] else None,
        h=prob.h if prob.c.shape[0] else None,
    ).path
    return LassoPath(lam=np.array([k.lam for k in path]), beta=np.array([k.beta for k in path]))


@dataclass
class Pattern:
    """Which coordinates are zero, inside their box or at a bound, and which rows bind."""

    zero: np.ndarray
    at_up: np.ndarray
    at_lo: np.ndarray
    inner: np.ndarray
    sign: np.ndarray
    active: np.ndarray


def pattern(prob: Lasso, beta: np.ndarray) -> Pattern:
    """The pattern of beta, to a tolerance relative to its size."""
    tol = 1e-9 * max(1.0, np.abs(beta).max())
    zero = np.abs(beta) <= tol
    at_up, at_lo = ~zero & (beta >= prob.upper - tol), ~zero & (beta <= prob.lower + tol)
    active = prob.c @ beta >= prob.h - tol * max(1.0, np.abs(prob.h).max(initial=1.0))
    return Pattern(zero, at_up, at_lo, ~zero & ~at_up & ~at_lo, np.sign(beta), active)


def conditions(prob: Lasso, pat: Pattern, lam: float, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""The optimality conditions under a pattern, as rows ``coef @ mult + const <= t``.

    The unknowns are the multipliers :math:`\kappa` of :math:`A\beta=b` and :math:`\eta`
    of the rows of :math:`C\beta\le h` that the pattern makes active, in that order.
    """
    rows = np.vstack([prob.a, prob.c[pat.active]])  # r = r0 + rows^T mult
    r0 = prob.x.T @ (prob.x @ beta - prob.y)
    # Each condition as  sigma * (r0_j + rows_j^T mult) + offset <= t.
    sigma, offset, coord = [], [], []
    def need(mask: np.ndarray, sign: float, off: np.ndarray) -> None:
        for j, o in zip(np.flatnonzero(mask), off[mask], strict=True):
            sigma.append(sign); offset.append(o); coord.append(j)
    s_inner = lam * pat.sign
    need(pat.inner, 1.0, s_inner)              #  r_j + lam s_j <= t
    need(pat.inner, -1.0, -s_inner)            # -(r_j + lam s_j) <= t
    need(pat.at_up, 1.0, np.full_like(beta, lam))
    need(pat.at_lo, -1.0, np.full_like(beta, lam))
    need(pat.zero & (prob.lower < 0), 1.0, np.full_like(beta, -lam))
    need(pat.zero & (prob.upper > 0), -1.0, np.full_like(beta, -lam))
    sigma, offset, coord = np.array(sigma), np.array(offset), np.array(coord, dtype=int)
    return sigma[:, None] * rows.T[coord], sigma * r0[coord] + offset


def feasibility(prob: Lasso, beta: np.ndarray) -> np.ndarray:
    """How far beta is outside each constraint."""
    return np.concatenate([
        np.abs(prob.a @ beta - prob.b),
        np.maximum(prob.c @ beta - prob.h, 0),
        np.maximum(beta - prob.upper, 0),
        np.maximum(prob.lower - beta, 0),
    ])


def best_multipliers(prob: Lasso, pat: Pattern, ends: list[tuple[float, np.ndarray]]) -> float:
    """The worst condition at the given points, under one pattern, at the best multipliers for each.

    A linear program picks the multipliers to make the worst violation as small as it can
    be. It only meets its rows to its own tolerance, so the violation is evaluated again,
    exactly, at the multipliers it found: an upper bound that rounding cannot hide.
    """
    m, k = prob.a.shape[0], prob.a.shape[0] + int(pat.active.sum())
    blocks = [conditions(prob, pat, lam, beta) for lam, beta in ends]
    if not sum(len(const) for _, const in blocks):
        return 0.0
    lhs = np.vstack([
        np.hstack([np.zeros((len(const), i * k)), coef, np.zeros((len(const), (len(ends) - 1 - i) * k)),
                   -np.ones((len(const), 1))])
        for i, (coef, const) in enumerate(blocks)
    ])
    rhs = -np.concatenate([const for _, const in blocks])
    bounds = ([(None, None)] * m + [(0.0, None)] * (k - m)) * len(ends) + [(0.0, None)]
    lp = linprog(np.append(np.zeros(k * len(ends)), 1.0), A_ub=lhs, b_ub=rhs, bounds=bounds, method="highs")
    worst = 0.0
    for i, (coef, const) in enumerate(blocks):
        mult = lp.x[i * k:(i + 1) * k].copy()
        mult[m:] = np.maximum(mult[m:], 0.0)
        worst = max(worst, float(np.max(coef @ mult + const, initial=0.0)))
    return worst


def kkt_residual(prob: Lasso, lam: float, beta: np.ndarray) -> float:
    r"""Largest violation of the LASSO's optimality conditions, relative to lambda_max.

    Read off the original problem, without ``cvxcla``. With
    :math:`r = G\beta - c + A^\top\kappa + C_R^\top\eta` over the active rows :math:`R` of
    :math:`C\beta\le h`, a nonzero coordinate strictly inside its box needs
    :math:`r_j = -\lambda\,\mathrm{sign}\,\beta_j`, one at a cap :math:`u_j>0` needs
    :math:`r_j \le -\lambda` and one at a floor :math:`l_j<0` needs :math:`r_j\ge\lambda`.
    A zero coordinate needs :math:`r_j\ge-\lambda` and :math:`r_j\le\lambda`, less the
    side a sign constraint (:math:`l_j=0` or :math:`u_j=0`) takes away. The multipliers
    are not taken from anywhere: a linear program picks :math:`\kappa` and
    :math:`\eta\ge0` to make the worst of these violations as small as it can be, which
    is zero exactly when beta is optimal. Primal feasibility is added on top.
    """
    worst = best_multipliers(prob, pattern(prob, beta), [(lam, beta)])
    return float(max(worst, feasibility(prob, beta).max(initial=0.0)) / np.abs(prob.x.T @ prob.y).max())


def segment_residual(prob: Lasso, lam0: float, beta0: np.ndarray, lam1: float, beta1: np.ndarray) -> float:
    """Largest violation of the optimality conditions anywhere on one segment, not only at sampled points.

    The pattern of the open segment is read off its midpoint. Under a fixed pattern every
    condition is affine in (lambda, beta, multipliers) jointly, and on the segment beta is
    affine in lambda; take the multipliers affine too, between a certificate at each end,
    and every condition is affine in lambda. It then holds on the whole segment exactly
    when it holds at both ends, under the segment's pattern rather than each end's own.
    Both ends must also be consistent with that pattern: zero coordinates zero, capped
    ones at the cap, active rows tight, and the support of the same sign as at the
    midpoint, so that nothing crosses zero or leaves a bound inside the segment without a
    knot. This is the check of Hobson, Butcher and Lentati (arXiv:2610.17024, Section 8.4).
    """
    pat = pattern(prob, 0.5 * (beta0 + beta1))
    worst = best_multipliers(prob, pat, [(lam0, beta0), (lam1, beta1)])
    for beta in (beta0, beta1):
        consistent = np.concatenate([
            np.abs(beta[pat.zero]),
            np.abs(beta - prob.upper)[pat.at_up],
            np.abs(beta - prob.lower)[pat.at_lo],
            np.maximum(-pat.sign * beta, 0)[~pat.zero],
            np.abs(prob.c @ beta - prob.h)[pat.active],
            feasibility(prob, beta),
        ])
        worst = max(worst, float(consistent.max(initial=0.0)))
    return float(worst / np.abs(prob.x.T @ prob.y).max())


def reference(prob: Lasso, lam: float) -> np.ndarray:
    """The same LASSO at one lambda, from an interior-point solver."""
    beta = cp.Variable(prob.x.shape[1])
    cons = []
    if prob.a.shape[0]:
        cons.append(prob.a @ beta == prob.b)
    if prob.c.shape[0]:
        cons.append(prob.c @ beta <= prob.h)
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
    """One line of the scale table: a large zero-sum problem, the worst over the seeds."""

    n: int
    p: int
    knots: tuple[int, int]  # fewest and most knots
    kkt: float


def scale() -> list[ScaleRow]:
    """Growing zero-sum problems, checked without an interior-point solver.

    A midpoint solve per segment would cost more than everything else together at p = 500,
    so the check here is the one that scales: the optimality conditions.
    """
    rows = []
    for n, p in SCALE:
        knots, kkt = [], 0.0
        for seed in SCALE_SEEDS:
            prob = large_composition(n, p, seed=seed)
            path = trace(prob)
            knots.append(len(path.lam))
            kkt = max(kkt, certify(prob, path))
        rows.append(ScaleRow(n, p, (min(knots), max(knots)), kkt))
    return rows


def knot_range(knots: tuple[int, int]) -> str:
    """Fewest and most knots over the seeds, as the tables print them: 43--45, or 42."""
    return f"{knots[0]}" if knots[0] == knots[1] else f"{knots[0]}--{knots[1]}"


def scale_tex(rows: list[ScaleRow]) -> str:
    """The body of Table 3, for the note to input, with its closing rule."""
    return "".join(f"{r.n:>4} & {r.p:>3} & {knot_range(r.knots):>7} & {sci(r.kkt)}\\\\\n"
                   for r in rows) + "\\bottomrule\n"


# The hostile problems of Table 4: how close to one another the columns are allowed to get.
STRESS_RHO = 0.99999  # pairwise correlation of the equicorrelated design
STRESS_DUPLICATE = (1e-4, 1e-6)  # noise, relative to its spread, on a copy of bmi
# Each hostile problem of Tables 4 and 5 is drawn from these seeds, and each size of Table 3
# from the first three: the tables report the worst case over them.
STRESS_SEEDS = tuple(range(SEED, SEED + 10))
SCALE_SEEDS = STRESS_SEEDS[:3]


def correlated(rho: float = STRESS_RHO, seed: int = SEED, n: int = 100, p: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """An equicorrelated Gaussian design: every pair of columns has correlation ``rho``."""
    rng = np.random.default_rng(seed)
    x = np.sqrt(rho) * rng.standard_normal((n, 1)) + np.sqrt(1 - rho) * rng.standard_normal((n, p))
    x -= x.mean(axis=0)
    beta = np.zeros(p)
    beta[:6] = rng.choice([-1, 1], 6) * rng.uniform(1, 3, 6)
    y = x @ beta + 0.5 * rng.standard_normal(n)
    return x, y - y.mean()


def near_duplicate(eps: float, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """The diabetes data with an eleventh column, bmi plus noise ``eps`` times its spread."""
    xd, yd, names = diabetes()
    bmi = xd[:, names.index("bmi")]
    rng = np.random.default_rng(seed)
    return np.column_stack([xd, bmi + eps * bmi.std() * rng.standard_normal(len(yd))]), yd


def stress_problems(seed: int = SEED) -> dict[str, tuple[Lasso, str]]:
    """The hostile problems for one seed, by the names Table 4 uses, each with its reference."""
    xs, ys = correlated(seed=seed)
    p = xs.shape[1]
    cap = 0.5 * np.abs(np.linalg.lstsq(xs, ys, rcond=None)[0]).max()
    xd, yd, _ = diabetes()
    box = 0.5 * np.abs(np.linalg.lstsq(xd, yd, rcond=None)[0]).max()  # the cap of Box
    xw, yw = wide(seed=seed)
    out = {
        "correlated": (Lasso(xs, ys), "LARS"),
        "correlated, box": (Lasso(xs, ys, lower=np.full(p, -cap), upper=np.full(p, cap)), "IPM"),
        "correlated, zero-sum": (Lasso(xs, ys, a=np.ones(p), b=np.zeros(1)), "IPM"),
    }
    for eps in STRESS_DUPLICATE:
        x, y = near_duplicate(eps, seed=seed)
        q = x.shape[1]
        out[f"duplicate {eps:.0e}"] = (Lasso(x, y), "LARS")
        out[f"duplicate {eps:.0e}, box"] = (Lasso(x, y, lower=np.full(q, -box), upper=np.full(q, box)), "IPM")
    out["wide, zero-sum"] = (Lasso(xw - xw.mean(axis=0), yw - yw.mean(), a=np.ones(xw.shape[1]),
                                   b=np.zeros(1)), "IPM")
    return out


def wide_problems(seed: int = SEED) -> dict[str, tuple[Lasso, str]]:
    """Wide, p > n, under constraints beyond a zero sum, by the names Table 5 uses, for one seed.

    The zero sum alone has an exact path for p > n elsewhere too (comlasso, Jeon et al.,
    2020); the general path algorithms add a ridge. These take what comlasso does not:
    bounds, signs, inequality rows and a nonhomogeneous row, alone or under the zero sum.
    Caps and the inequality rows are set as in Box and Cholesterol, from the least-squares
    fit, here the one of smallest norm.
    """
    x, y = wide(seed=seed)
    x, y = x - x.mean(axis=0), y - y.mean()
    p = x.shape[1]
    one = np.ones(p)
    fit = np.linalg.lstsq(x, y, rcond=None)[0]
    cap = 0.5 * np.abs(fit).max()
    net = np.zeros(p)
    net[:3] = 1.0  # the net effect of the first three predictors, as Cholesterol caps three
    zero_sum = {"a": one, "b": np.zeros(1)}
    probs = {
        "nonnegative": Lasso(x, y, lower=np.zeros(p)),
        "box": Lasso(x, y, lower=-cap * one, upper=cap * one),
        "inequality rows": Lasso(x, y, c=np.vstack([net, -net]), h=np.full(2, 0.5 * abs(net @ fit))),
        "pinned": Lasso(x, y, a=x[:1], b=y[:1]),
        "zero-sum, signs": Lasso(x, y, **zero_sum, lower=np.where(signed_parts(p), 0.0, -np.inf)),
        "zero-sum, caps": Lasso(x, y, **zero_sum, lower=-cap * one, upper=cap * one),
    }
    return {name: (prob, "IPM") for name, prob in probs.items()}


def objective(prob: Lasso, lam: float, beta: np.ndarray) -> float:
    """The LASSO objective at ``lam``."""
    return float(0.5 * np.sum((prob.y - prob.x @ beta) ** 2) + lam * np.abs(beta).sum())


@dataclass
class StressRow:
    """One line of the stress table: the worst over the seeds. Declined seeds have no checks."""

    name: str
    n: int
    p: int
    reference: str
    cond: float = 0.0  # largest condition number of X
    knots: tuple[int, int] | None = None  # fewest and most knots
    gap: float | None = None  # against the reference, as in Table 2
    kkt: float | None = None
    ref_kkt: float | None = None  # LARS's own KKT residual, where LARS is the reference
    ref_excess: float | None = None  # the least by which the IPM objective is worse, relative, at any midpoint
    declined: int = 0  # how many seeds cvxcla declines
    reason: str | None = None  # its reason, when it declines


def stress(
    drawn: Callable[[int], dict[str, tuple[Lasso, str]]] = stress_problems,
) -> list[StressRow]:
    """The hostile problems over every seed, through the checks of Table 2, and the reference's accuracy.

    Where the path and its reference part, the reference is checked as well: LARS by the
    same KKT residual, the interior-point solver by its objective, which beats the path's by
    no more than the solver's tolerance when the gap is the solver's. ``drawn`` gives the
    problems for one seed; Table 5 passes ``wide_problems``.
    """
    rows: dict[str, StressRow] = {}
    for seed in STRESS_SEEDS:
        for name, (prob, ref) in drawn(seed).items():
            row = rows.setdefault(name, StressRow(name, *prob.x.shape, ref))
            row.cond = max(row.cond, float(np.linalg.cond(prob.x)))
            try:
                path = trace(prob)
            except ValueError as err:  # cvxcla's DegenerateProblemError is a ValueError too
                row.declined += 1
                row.reason = str(err)
                continue
            k = len(path.lam)
            row.knots = (k, k) if row.knots is None else (min(row.knots[0], k), max(row.knots[1], k))
            row.kkt = max(row.kkt or 0.0, certify(prob, path))
            if ref == "LARS":
                row.gap = max(row.gap or 0.0, against_lars(prob, path))
                alphas, _, coefs = lars_path(prob.x, prob.y, method="lasso")
                own = max(kkt_residual(prob, a * prob.x.shape[0], b) for a, b in zip(alphas, coefs.T, strict=True))
                row.ref_kkt = max(row.ref_kkt or 0.0, own)
            else:
                row.gap = max(row.gap or 0.0, against_solver(prob, path))
                lam, beta = midpoints(path)
                keep = lam > 1e-6 * path.lam[0]
                excess = min((objective(prob, l, reference(prob, l)) - objective(prob, l, b))
                             / objective(prob, l, b) for l, b in zip(lam[keep], beta[keep], strict=True))
                row.ref_excess = excess if row.ref_excess is None else min(row.ref_excess, excess)
    return list(rows.values())


def stress_tex(rows: list[StressRow]) -> str:
    """The body of Table 4, with its closing rule."""
    def line(r: StressRow) -> str:
        name = r.name
        for eps in STRESS_DUPLICATE:
            name = name.replace(f"{eps:.0e}", f"$10^{{{round(np.log10(eps))}}}$")
        head = f"{name:<21} & {r.n:>3} & {r.p:>3} & {sci(r.cond):<19} & "
        if r.declined == len(STRESS_SEEDS):
            return head + r"\multicolumn{4}{c}{declined}\\" + "\n"
        knots = knot_range(r.knots)
        if r.declined:
            knots += f" ({r.declined} declined)"
        return head + f"{knots:>5} & {r.reference:<4} & {sci(r.gap):<19} & {sci(r.kkt)}\\\\\n"
    return "".join(line(r) for r in rows) + "\\bottomrule\n"


def stress_numbers(rows: list[StressRow]) -> str:
    """The numbers the text quotes about Table 4, as LaTeX macros."""
    traced = [r for r in rows if r.kkt is not None]
    return "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in {
        "stRho": f"{STRESS_RHO}",
        "stSeeds": NUMBERS[len(STRESS_SEEDS)],
        "stSeedFirst": str(STRESS_SEEDS[0]),
        "stSeedLast": str(STRESS_SEEDS[-1]),
        "scSeeds": NUMBERS[len(SCALE_SEEDS)],
        "stEpsHi": sci(max(STRESS_DUPLICATE)),
        "stEpsLo": sci(min(STRESS_DUPLICATE)),
        "stKKT": sci(max(r.kkt for r in traced)),
        "stLarsKKT": sci(max(r.ref_kkt for r in traced if r.ref_kkt is not None)),
        "stIpmGap": sci(max(r.gap for r in traced if r.reference == "IPM")),
        "stIpmBetter": sci(max(0.0, -min(r.ref_excess for r in traced if r.ref_excess is not None))),
        # Under the cap the near-duplicate is declined only where both twins become free first.
        "stDupBoxDeclined": NUMBERS[{r.name: r for r in rows}[f"duplicate {min(STRESS_DUPLICATE):.0e}, box"].declined],
    }.items())


def wide_tex(rows: list[StressRow]) -> str:
    """The body of Table 5, with its closing rule."""
    def line(r: StressRow) -> str:
        knots = knot_range(r.knots)
        return f"{r.name:<16} & {knots:>7} & {sci(r.gap):<19} & {sci(r.kkt)}\\\\\n"
    return "".join(line(r) for r in rows) + "\\bottomrule\n"


def wide_numbers(rows: list[StressRow]) -> str:
    """The numbers the text quotes about Table 5, as LaTeX macros."""
    return "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in {
        "wdN": f"{rows[0].n}",
        "wdP": f"{rows[0].p}",
        "wdKKT": sci(max(r.kkt for r in rows)),
        "wdGap": sci(max(r.gap for r in rows)),
        "wdIpmBetter": sci(max(0.0, -min(r.ref_excess for r in rows))),
        # these two bounds depend on the CPU (Section 5.5), so the text names the one that wrote them
        "wdMachine": r"\texttt{" + platform.machine().replace("_", r"\_") + "}",
    }.items())


# Small counts as the note spells them out.
NUMBERS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def signed_parts(p: int) -> np.ndarray:
    """The parts of Zero-sum, signs whose coefficient must be nonnegative: the even-numbered ones."""
    return np.arange(p) % 2 == 0


def donors(seed: int = SEED, t: int = 40, j: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """A unit's outcomes over t periods as a combination of j donors' outcomes, synthetic-control style.

    The donors share three factors; the truth weights three of them, 0.6, 0.5 and -0.1,
    which sum to one.
    """
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((t, 3))
    x = f @ rng.standard_normal((3, j)) + 0.5 * rng.standard_normal((t, j))
    w = np.zeros(j)
    w[[2, 5, 11]] = [0.6, 0.5, -0.1]
    return x, x @ w + 0.2 * rng.standard_normal(t)


def zero_sum_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The least-squares fit under sum(beta) = 0, from its bordered normal equations."""
    p = x.shape[1]
    kkt = np.block([[x.T @ x, np.ones((p, 1))], [np.ones((1, p)), np.zeros((1, 1))]])
    return np.linalg.solve(kkt, np.append(x.T @ y, 0.0))[:p]


# Each problem's constraint, as Table 2 prints it. The note's macros \trans and \one apply.
CONSTRAINTS = {
    "plain": "none",
    "nonnegative": r"$\beta\ge0$",
    "wide": r"none, $p>n$",
    "box": r"$|\beta_j|\le u$",
    "cholesterol": r"$|\beta_{\mathrm{s1}}+\beta_{\mathrm{s2}}+\beta_{\mathrm{s3}}|\le h$",
    "pinned": r"$x_1\trans\beta=y_1$",
    "zero-sum": r"$\one\trans\beta=0$",
    "zero-sum, signs": r"$\one\trans\beta=0$, $\beta_{2j}\ge0$",
    "zero-sum, caps": r"$\one\trans\beta=0$, $|\beta_j|\le u$",
    "monotone": r"$\beta_j\le\beta_{j+1}$",
    "sum to one": r"$\one\trans\beta=1$",
    "sCD14": r"$\one\trans\beta=0$",
}


def problems() -> dict[str, Lasso]:
    """Every problem the note reports, by the name its table uses."""
    xd, yd, _ = diabetes()
    xw, yw = wide()
    xc, yc = compositional()
    p = xd.shape[1]
    ols = np.linalg.lstsq(xd, yd, rcond=None)[0]
    cap = 0.5 * np.abs(ols).max()
    point = xd[0] / np.abs(xd[0]).max()  # pin the fit at the first patient, rescaled
    _, _, names = diabetes()
    chol = np.isin(names, ["s1", "s2", "s3"]).astype(float)
    net = 0.5 * abs(chol @ ols)  # half the least-squares net effect of the cholesterol measures
    zcap = 0.5 * np.abs(zero_sum_fit(xc, yc)).max()  # half the largest zero-sum least-squares coefficient
    q = xc.shape[1]
    xs, ys = donors()
    return {
        "plain": Lasso(xd, yd),
        "nonnegative": Lasso(xd, yd, lower=np.zeros(p)),
        "wide": Lasso(xw, yw),
        "box": Lasso(xd, yd, lower=np.full(p, -cap), upper=np.full(p, cap)),
        "cholesterol": Lasso(xd, yd, c=np.vstack([chol, -chol]), h=np.full(2, net)),
        "pinned": Lasso(xd, yd, a=point, b=np.array([yd[0] / np.abs(xd[0]).max()])),
        "zero-sum": Lasso(xc, yc, a=np.ones(xc.shape[1]), b=np.zeros(1)),
        "zero-sum, signs": Lasso(xc, yc, a=np.ones(xc.shape[1]), b=np.zeros(1),
                                 lower=np.where(signed_parts(xc.shape[1]), 0.0, -np.inf)),
        "zero-sum, caps": Lasso(xc, yc, a=np.ones(q), b=np.zeros(1), lower=np.full(q, -zcap), upper=np.full(q, zcap)),
        "monotone": monotone(0.0),
        "sum to one": Lasso(xs, ys, a=np.ones(xs.shape[1]), b=np.ones(1)),
    } | ({"sCD14": scd14()[0]} if SCD14.exists() else {})


# --------------------------------------------------------------------------------------
# The opening example
# --------------------------------------------------------------------------------------
@dataclass
class Example:
    """The opening example: the diabetes LASSO without and with a cap on every coefficient.

    The problem is Box. Least squares plays the collinear cholesterol measures off against
    each other and gives HDL a positive coefficient; the cap stops both.
    """

    cap: float  # u in |beta_j| <= u, half the largest least-squares coefficient
    ols: dict[str, float]  # the least-squares coefficients the text quotes, by name
    knots: int  # knots of the plain LASSO path
    over: int  # of those, the knots where some coefficient passes the cap
    capped: np.ndarray  # (k, p) the knots of the capped path
    keeps: int  # every whole cap from 1 to this keeps HDL nonpositive along the whole path


def support(beta: np.ndarray) -> np.ndarray:
    """The nonzero coefficients, to rounding."""
    return np.flatnonzero(np.abs(beta) > 1e-9 * np.abs(beta).max())


def example() -> Example:
    """The diabetes LASSO of Section 1, plain and capped, and the caps that keep HDL's sign."""
    x, y, names = diabetes()
    box = problems()["box"]
    ols = np.linalg.lstsq(x, y, rcond=None)[0]
    plain = trace(Lasso(x, y))
    over = (np.abs(plain.beta) > box.upper * (1 + 1e-12)).any(axis=1)
    hdl, p = names.index("s3"), x.shape[1]

    def keeps(u: int) -> bool:
        prob = Lasso(x, y, lower=np.full(p, -float(u)), upper=np.full(p, float(u)))
        path = trace(prob)
        assert certify(prob, path) < 1e-10, u  # a sign read off a path that is not exact means nothing
        return bool(path.beta[:, hdl].max() <= 1e-9 * u)

    hi = 1
    while keeps(hi + 1):
        hi += 1
    assert keeps(1)
    return Example(float(box.upper[0]), {n: float(ols[names.index(n)]) for n in ("s1", "s2", "s3")},
                   len(plain.lam), int(over.sum()), trace(box).beta, hi)


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
    path = trace(prob)
    first = list(dict.fromkeys(names[j] for b in path.beta for j in support(b)))[:5]
    print(f"\nsCD14 ({'the note' if snapshot else 'NOT the note'}'s snapshot): n = {prob.x.shape[0]}, "
          f"p = {prob.x.shape[1]}, {len(path.lam)} knots, KKT {certify(prob, path):.1e}")
    print(f"  first to enter: {', '.join(first)}")


def example_tex(ex: Example) -> str:
    """The example's numbers as LaTeX macros, for the note to input."""
    return "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in {
        "exCap": f"{ex.cap:.1f}",
        "exTC": f"{ex.ols['s1']:.0f}",
        "exLDL": f"{ex.ols['s2']:.0f}",
        "exHDL": f"{ex.ols['s3']:.0f}",
        "exKnots": str(ex.knots),
        "exOver": NUMBERS[ex.over],
        "exCapped": str(len(ex.capped)),
        "exKeepHi": str(ex.keeps),
    }.items())


# --------------------------------------------------------------------------------------
# The limit: a monotone fit, whose rows are homogeneous
# --------------------------------------------------------------------------------------
MONOTONE_TRUTH = np.array([-2.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 3.0])
MONOTONE_SLACK = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]  # the relaxations h = eps the note reports


def monotone(h: float) -> Lasso:
    """beta_1 <= ... <= beta_p as rows beta_j - beta_{j+1} <= h, on a Gaussian design with an ordered truth.

    With h = 0 the rows are homogeneous, all tight at beta = 0, and the start is
    degenerate in either form; h > 0 relaxes them so the penalty form can start.
    """
    rng = np.random.default_rng(SEED)
    n, p = 100, len(MONOTONE_TRUTH)
    x = rng.standard_normal((n, p))
    y = x @ MONOTONE_TRUTH + 0.5 * rng.standard_normal(n)
    d = np.eye(p - 1, p) - np.eye(p - 1, p, k=1)
    return Lasso(x, y - y.mean(), c=d, h=np.full(p - 1, h))


@dataclass
class MonotoneRow:
    """The relaxed monotone fit at one h: its knots, its own KKT residual, and its distance to h = 0."""

    h: float
    knots: int
    kkt: float  # of the relaxed problem, which the path solves exactly
    gap: float  # largest midpoint distance to an interior-point solve of the h = 0 problem, absolute


def monotone_rows() -> list[MonotoneRow]:
    """The relaxed monotone fit for each h of MONOTONE_SLACK."""
    exact = monotone(0.0)
    rows = []
    for h in MONOTONE_SLACK:
        prob = monotone(h)
        path = trace(prob)
        lam, beta = midpoints(path)
        keep = lam > 1e-6 * path.lam[0]
        gap = max(np.abs(reference(exact, v) - b).max() for v, b in zip(lam[keep], beta[keep], strict=True))
        rows.append(MonotoneRow(h, len(path.lam), certify(prob, path), float(gap)))
    return rows


def monotone_tex(rows: list[MonotoneRow]) -> str:
    """The monotone fit's numbers as LaTeX macros, for the note to input."""
    ratios = [r.gap / r.h for r in rows]
    return "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in {
        "moP": str(len(MONOTONE_TRUTH)),
        "moKnots": str(rows[0].knots),
        "moExact": str(len(trace(monotone(0.0)).lam)),  # the h = 0 path, from cvxcla's extrapolation
        "moRatioLo": f"{min(ratios):.2f}",
        "moRatioHi": f"{max(ratios):.2f}",
        "moKKT": sci(max(r.kkt for r in rows)),
        "moHMax": sci(max(MONOTONE_SLACK)),
        "moHMin": sci(min(MONOTONE_SLACK)),
    }.items())


# --------------------------------------------------------------------------------------
# A row that excludes beta = 0
# --------------------------------------------------------------------------------------
def floor() -> Lasso:
    """Diabetes with beta_bmi at least half its least-squares value: one row -beta_bmi <= -h, h > 0.

    The row excludes beta = 0, so the penalty form starts where the CLA's linear program
    puts it, at the feasible point of smallest l1 norm: bmi alone, on the floor.
    """
    xd, yd, names = diabetes()
    row = -(np.array(names) == "bmi").astype(float)
    ols = np.linalg.lstsq(xd, yd, rcond=None)[0]
    return Lasso(xd, yd, c=row, h=np.array([0.5 * (row @ ols)]))


@dataclass
class FloorRow:
    """The floor problem's path and its checks."""

    knots: int
    gap: float  # against the interior-point solver at every midpoint, relative, as in Table 2
    kkt: float
    start: float  # the row's slack at the first knot, which sits on the floor


def floor_row() -> FloorRow:
    """Trace the floor problem and check it."""
    prob = floor()
    path = trace(prob)
    slack = float((prob.h - prob.c @ path.beta[0])[0])
    return FloorRow(len(path.lam), against_solver(prob, path), certify(prob, path), slack)


def floor_tex(r: FloorRow) -> str:
    """The floor problem's numbers as LaTeX macros, for the note to input."""
    return "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in {
        "flKnots": str(r.knots),
        "flGap": sci(r.gap),
        "flKKT": sci(r.kkt),
    }.items())


# --------------------------------------------------------------------------------------
# One encoding or two: every zero sum through the penalty form alone
# --------------------------------------------------------------------------------------
@dataclass
class EncodingRow:
    """A zero-sum problem through both encodings: the budget form, and the penalty form relaxed."""

    name: str
    knots: int  # budget form
    penalty: int | None  # penalty form, relaxed and extrapolated; None where cvxcla declines
    gap: float | None  # largest knot discrepancy between the two, relative, as in Table 2
    kkt: float | None  # of the penalty form's path
    spacing: float  # the closest two knots of the budget form's path, relative to lambda_max
    reason: str | None = None  # cvxcla's reason, where it declines


def through_penalty(prob: Lasso) -> Lasso:
    """The same problem with caps so loose they never bind, which sends cvxcla to the penalty form.

    Under a zero sum the penalty form starts degenerate, so cvxcla relaxes the zero sum,
    traces twice, extrapolates and certifies (Section 6); the caps change nothing else.
    """
    big = 1e6 * max(1.0, float(np.abs(prob.x.T @ prob.y).max()))
    return Lasso(prob.x, prob.y, a=prob.a, b=prob.b, lower=np.maximum(prob.lower, -big),
                 upper=np.minimum(prob.upper, big), c=prob.c, h=prob.h)


def encodings() -> list[EncodingRow]:
    """Every zero-sum problem of Tables 2 to 4 through both encodings."""
    probs = problems()
    cases = {name: probs[name] for name in ("zero-sum", "zero-sum, signs", "sCD14") if name in probs}
    cases["wide, zero-sum"] = stress_problems()["wide, zero-sum"][0]
    n, p = SCALE[-1]
    cases[f"zero-sum, $p={p}$"] = large_composition(n, p)
    rows = []
    for name, prob in cases.items():
        budget = trace(prob)
        spacing = float(-np.diff(budget.lam).max() / budget.lam[0])  # lam falls, so the smallest step is -max
        try:
            path = trace(through_penalty(prob))
        except ValueError as err:  # cvxcla's DegenerateProblemError is a ValueError too
            rows.append(EncodingRow(name, len(budget.lam), None, None, None, spacing, str(err).split(":")[0]))
            continue
        assert np.abs(path.beta).max() < 1e-3 * through_penalty(prob).upper.max()  # the caps never bind
        gap = float("nan")
        if len(path.lam) == len(budget.lam):
            gap = max(float(np.abs(path.beta - budget.beta).max() / np.abs(budget.beta).max()),
                      float(np.abs(path.lam - budget.lam).max() / budget.lam[0]))
        rows.append(EncodingRow(name, len(budget.lam), len(path.lam), gap, certify(prob, path), spacing))
    return rows


def encodings_numbers(rows: list[EncodingRow]) -> str:
    """The numbers the text quotes about the largest zero sum, as LaTeX macros."""
    largest = rows[-1]
    return "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in {
        "encSpacing": sci(largest.spacing),
        "encLargestKnots": str(largest.penalty),
        "encLargestGap": sci(largest.gap),
    }.items())


def encodings_tex(rows: list[EncodingRow]) -> str:
    """The body of Table 6, with its closing rule."""
    def line(r: EncodingRow) -> str:
        head = f"{r.name:<22} & {r.knots:>3} & "
        if r.penalty is None:
            return head + r"\multicolumn{3}{c}{declined}\\" + "\n"
        return head + f"{r.penalty:>3} & {sci(r.gap)} & {sci(r.kkt)}\\\\\n"
    return "".join(line(r) for r in rows) + "\\bottomrule\n"


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


def lars_positive_end(prob: Lasso, path: LassoPath) -> tuple[float, float]:
    """Where ``lars_path``'s positive mode really ends, and its KKT violation at lambda = 0.

    scikit-learn labels its last point alpha = 0, but the point is the path at the lambda
    where nothing was left to enter: find that lambda on the CLA path.
    """
    _, _, coefs = lars_path(prob.x, prob.y, method="lasso", positive=True)
    end = coefs[:, -1]
    for (l0, b0), (l1, b1) in itertools.pairwise(zip(path.lam, path.beta, strict=True)):
        d = b1 - b0
        t = float(d @ (end - b0) / (d @ d))
        if 0 <= t <= 1 and np.abs(b0 + t * d - end).max() < 1e-9 * np.abs(end).max():
            return float(l0 + t * (l1 - l0)), kkt_residual(prob, 0.0, end)
    raise AssertionError("lars_path's last point is not on the CLA path")


@dataclass
class SegmentDemo:
    """A wrong path that a check at its knots and midpoints passes, and the whole-segment check fails."""

    sampled: float  # the largest KKT residual at the two ends and the midpoint
    whole: float  # the whole-segment check on the same path
    exact: float  # the whole-segment check on the exact path


def segment_demo() -> SegmentDemo:
    """With p = 1, y = 1 and a cap of 1/2, one straight segment from (5/4, 0) to (1/4, 1/2).

    The exact path is 0 down to lambda = 1, then 1 - lambda up to the cap at lambda = 1/2,
    then the cap. The one segment misses both knots, yet meets the path at its ends and at
    its midpoint, so a check at those three points alone passes it.
    """
    prob = Lasso(np.eye(1), np.ones(1), upper=np.full(1, 0.5))
    fake = LassoPath(lam=np.array([1.25, 0.25]), beta=np.array([[0.0], [0.5]]))
    lam, beta = midpoints(fake)
    sampled = max(kkt_residual(prob, *z) for z in zip([*fake.lam, *lam], [*fake.beta, *beta], strict=True))
    return SegmentDemo(sampled, certify(prob, fake), certify(prob, trace(prob)))


def evidence_numbers(rows: list[Row]) -> str:
    """The numbers the text of Section 5 and Figure 2 quote, as LaTeX macros."""
    by = {r.name: r for r in rows}
    probs = problems()
    lam_end, kkt_end = lars_positive_end(probs["nonnegative"], trace(probs["nonnegative"]))
    plain = trace(probs["plain"])
    demo = segment_demo()
    return "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in {
        "evWideKnots": str(by["wide"].knots),
        "evPlainKnots": str(by["plain"].knots),
        "evPlainGap": sci(by["plain"].gap),
        "evPlainNorm": f"{np.abs(plain.beta[-1]).sum():.0f}",  # ||beta(0)||_1, where cash enters
        "evPinnedHold": f"{trace(probs['pinned']).lam[0]:.0f}",  # the first knot, where Pinned starts to move
        "evIpmWorst": sci(max(r.gap for r in rows if r.reference == "IPM")),
        "evLarsEnd": f"{lam_end:.2f}",
        "evLarsKKT": sci(kkt_end),
        "evSegSampled": sci(demo.sampled) if demo.sampled else "0",
        "evSegWhole": f"{demo.whole:.2f}",
    }.items())


def against_solver(prob: Lasso, path: LassoPath) -> float:
    """Largest midpoint discrepancy against the interior-point solve, relative."""
    lam, beta = midpoints(path)
    keep = lam > 1e-6 * path.lam[0]
    gaps = [np.abs(reference(prob, l) - b).max() for l, b in zip(lam[keep], beta[keep], strict=True)]
    return float(max(gaps) / np.abs(path.beta).max())


def certify(prob: Lasso, path: LassoPath) -> float:
    """Largest KKT violation at every knot, and anywhere on every segment."""
    knots = [kkt_residual(prob, *z) for z in zip(path.lam, path.beta, strict=True)]
    segments = [segment_residual(prob, path.lam[i], path.beta[i], path.lam[i + 1], path.beta[i + 1])
                for i in range(len(path.lam) - 1)]
    return max(knots + segments)


@dataclass
class Row:
    """One line of the note's table."""

    name: str
    n: int
    p: int
    knots: int
    reference: str
    gap: float
    kkt: float


def table() -> list[Row]:
    """Every problem, its path from cvxcla, and the checks."""
    rows = []
    for name, prob in problems().items():
        path = trace(prob)
        if name in ("plain", "wide", "nonnegative"):
            ref, gap = "LARS", against_lars(prob, path, positive=name == "nonnegative")
        else:
            ref, gap = "IPM", against_solver(prob, path)
        rows.append(Row(name, *prob.x.shape, len(path.lam), ref, gap, certify(prob, path)))
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
    """The diabetes LASSO path from cvxcla, with the LARS knots on top."""
    x, y, names = diabetes()
    prob = Lasso(x, y)
    path = trace(prob)
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
    ax.plot([], [], "-", color=BLUE, lw=1.2, label="cvxcla.Lasso")
    ax.plot([], [], "o", ms=4.5, mfc="none", mec=ORANGE, mew=0.9, label="LARS knots")
    ax.legend(loc="upper left")
    ax.set_xlabel(r"$\|\beta\|_1$")
    ax.set_ylabel(r"coefficient $\beta_j$")
    ax.set_xlim(0, norm[-1] * 1.12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_portfolio(out: Path) -> None:
    """The diabetes path as the budget form sees it: long, short and cash shares against the budget t.

    Read off the path itself, not a CLA run: at budget t = ||beta||_1 the long and short
    books hold sum(beta+) / t and sum(beta-) / t, and cash holds nothing until the path
    ends at t = ||beta(0)||_1; past it the least-squares fit is diluted by cash.
    """
    x, y, _ = diabetes()
    path = trace(Lasso(x, y))
    beta = path.beta[1:]  # the first knot is beta = 0, at t = 0, where the shares are not defined
    budget = np.abs(beta).sum(axis=1)
    long_, short = np.clip(beta, 0, None).sum(axis=1) / budget, np.clip(-beta, 0, None).sum(axis=1) / budget
    cash = np.zeros_like(budget)
    # Past the last knot the weights move along a ray; one more point shows it.
    tail = 1.6 * budget[-1]
    budget = np.append(budget, tail)
    long_ = np.append(long_, np.clip(beta[-1], 0, None).sum() / tail)
    short = np.append(short, np.clip(-beta[-1], 0, None).sum() / tail)
    cash = np.append(cash, 1 - np.abs(beta[-1]).sum() / tail)

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


SCD14_SHOWN = 8  # Figure 3 draws the coefficients largest at lambda = 0 in colour, the rest in grey
SCD14_STRIDE = 3  # and marks the interior-point solve at every third segment midpoint


# The colourblind-safe qualitative palette of Okabe and Ito, for the eight genera of Figure 3.
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def path_panel(ax: plt.Axes, prob: Lasso, path: LassoPath, shown: int, stride: int,
               names: list[str] | None = None) -> None:
    """A CLA path against lambda, with interior-point solves at some segment midpoints.

    Sixty coefficients and a solve at every midpoint overplot into a solid band, so only
    the ``shown`` coefficients largest at the end are drawn in colour and marked, at every
    ``stride``-th midpoint. Table 2 checks every midpoint of every coefficient.
    """
    lam, _ = midpoints(path)
    lam = lam[lam > 1e-6 * path.lam[0]][::stride]
    ref = np.array([reference(prob, v) for v in lam])
    top = np.argsort(-np.abs(path.beta[-1]))[:shown]
    rest = np.setdiff1d(np.arange(path.beta.shape[1]), top)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.plot(path.lam, path.beta[:, rest], "-", color=GRID, lw=0.7, zorder=1)
    for colour, j in zip(OKABE_ITO, top, strict=True):
        ax.plot(path.lam, path.beta[:, j], "-", color=colour, lw=1.1, zorder=2)
        ax.plot(lam, ref[:, j], "o", ms=3.8, mfc="none", mec=INK, mew=0.6, zorder=3)
    if names is not None:
        # Each line is labelled at its right-hand end, lambda = 0, the labels pushed apart
        # just enough not to overlap, so the series read without relying on hue.
        ends = path.beta[-1, top]
        order = np.argsort(ends)
        lo, hi = ax.get_ylim()
        step = 0.055 * (hi - lo)
        placed = ends[order].copy()
        for i in range(1, len(placed)):
            placed[i] = max(placed[i], placed[i - 1] + step)
        placed -= max(0.0, placed[-1] - hi)
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - step)
        margin = blended_transform_factory(ax.transAxes, ax.transData)
        for k, i in enumerate(order):
            ax.plot([1.0, 1.03], [ends[i], placed[k]], color=OKABE_ITO[i], lw=0.6, transform=margin, clip_on=False)
            ax.text(1.04, placed[k], names[top[i]], transform=margin, fontsize=6, color=INK, va="center")
    ax.set_xscale("symlog", linthresh=path.lam[0] / 50)
    ax.set_xlim(path.lam[0] * 1.05, 0)
    ax.set_xlabel(r"penalty $\lambda$ (decreasing; symlog, linear below $\lambda_{\max}/50$)")
    ax.set_ylabel(r"coefficient $\beta_j$")


def path_legend(fig: plt.Figure) -> None:
    """The legend shared by the path panels."""
    handles = [
        plt.Line2D([], [], color=OKABE_ITO[0], lw=1.1, label=f"CLA path, {SCD14_SHOWN} largest (labelled)"),
        plt.Line2D([], [], color=GRID, lw=0.7, label="the other genera"),
        plt.Line2D([], [], ls="none", marker="o", ms=3.8, mfc="none", mec=INK, mew=0.6,
                   label="interior-point solve, mid-segment"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=3)


def genus(name: str) -> str:
    """A taxon as Figure 3 labels it: g_Prevotella as Prevotella, f_X_g_unclassified as X (unclassified)."""
    rank, _, rest = name.partition("_")
    if "_g_" in rest:
        family, _, tail = rest.partition("_g_")
        return f"{family} ({tail.replace('_', ' ').lower()})"
    return rest.replace("_", " ")


def fig_scd14(out: Path) -> None:
    """The real-data example: the zero-sum path of sCD14 on 60 genera."""
    prob, names = scd14()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    path_panel(ax, prob, trace(prob), SCD14_SHOWN, SCD14_STRIDE, [genus(n) for n in names])
    ax.set_title("sCD14 on 60 genera, zero-sum LASSO by the CLA", fontsize=8, color=INK)
    path_legend(fig)
    fig.tight_layout(rect=(0, 0.08, 0.80, 1))
    fig.savefig(out)
    plt.close(fig)


def scd14_tex() -> str:
    """The real-data example's numbers as LaTeX macros, for the note to input."""
    prob, _ = scd14()
    path = trace(prob)
    return "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in {
        "rdSamples": str(prob.x.shape[0]),
        "rdGenera": str(prob.x.shape[1]),
        "rdKnots": str(len(path.lam)),
    }.items())


def fig_example(out: Path) -> None:
    """The opening example: the diabetes LASSO path without and with the cap.

    Left, the plain LASSO, the path lars_path computes; right, every coefficient capped at
    u. The shaded bands are what the cap rules out, and the coefficients the left path
    takes into them are drawn in orange.
    """
    x, y, _ = diabetes()
    box = problems()["box"]
    u = box.upper[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, prob, title in ((axes[0], Lasso(x, y), "no cap: what lars_path fits"),
                            (axes[1], box, rf"$|\beta_j|\leq {u:.0f}$: the CLA")):
        path = trace(prob)
        over = (np.abs(path.beta) > u * (1 + 1e-12)).any(axis=0)
        lam, _ = midpoints(path)
        lam = lam[lam > 1e-6 * path.lam[0]]
        ref = np.array([reference(prob, v) for v in lam])
        for band in ((u, 900), (-900, -u)):
            ax.axhspan(*band, color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.plot(path.lam, path.beta[:, ~over], "-", color=BLUE, lw=1.1, zorder=2)
        ax.plot(path.lam, path.beta[:, over], "-", color=ORANGE, lw=1.4, zorder=3)
        ax.plot(lam, ref, "o", ms=3.4, mfc="none", mec=INK, mew=0.6, zorder=4)
        ax.set_xscale("symlog", linthresh=path.lam[0] / 50)
        ax.set_xlim(path.lam[0] * 1.05, 0)
        ax.set_ylim(-850, 850)
        ax.set_xlabel(r"penalty $\lambda$ (decreasing)")
        ax.set_title(title, fontsize=8, color=INK)
    axes[0].set_ylabel(r"coefficient $\beta_j$")
    axes[0].text(0.03, 0.94, "beyond the cap", transform=axes[0].transAxes, fontsize=6.5, color=MUTED, va="top")
    handles = [
        plt.Line2D([], [], color=BLUE, lw=1.1, label="exact path"),
        plt.Line2D([], [], color=ORANGE, lw=1.4, label="passes the cap"),
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
    """The body of Table 2, one LaTeX row per problem, for the note to input.

    The closing rule comes with it: after an ``\\input`` inside a tabular, LaTeX no longer
    accepts a ``\\bottomrule`` of the note's own.
    """
    return "".join(
        f"{r.name:<15} & {CONSTRAINTS[r.name]} & {r.n:>3} & {r.p:>3} & {r.knots:>2} & {r.reference:<4} & "
        f"{sci(r.gap):<19} & {sci(r.kkt)}\\\\\n"
        for r in rows
    ) + "\\bottomrule\n"


def export_zero_sum_paths() -> None:
    """Write the exact zero-sum paths to figures/zero_sum_paths.npz.

    For the checks that run other path algorithms in their own environments,
    classo_check.py (c-lasso) and R/comlasso_check.R (comlasso): Zero-sum, sCD14 and, for
    p > n, Wide, zero-sum for every seed of Table 4.
    """
    OUT.mkdir(exist_ok=True)
    saved = {"zero-sum": problems()["zero-sum"]} | ({"scd14": scd14()[0]} if SCD14.exists() else {})
    saved |= {f"wide-{seed}": stress_problems(seed)["wide, zero-sum"][0] for seed in STRESS_SEEDS}
    arrays = {}
    for name, prob in saved.items():
        path = trace(prob)
        arrays |= {f"{name}.x": prob.x, f"{name}.y": prob.y, f"{name}.lam": path.lam, f"{name}.beta": path.beta}
    np.savez(OUT / "zero_sum_paths.npz", **arrays)


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
    (OUT / "evidence_numbers.tex").write_text(evidence_numbers(rows))
    big = scale()
    (OUT / "scale.tex").write_text(scale_tex(big))
    hostile = stress()
    (OUT / "stress.tex").write_text(stress_tex(hostile))
    (OUT / "stress_numbers.tex").write_text(stress_numbers(hostile))
    print("\nstress, hostile problems:")
    for r in hostile:
        if r.declined:
            print(f"  {r.name:<22} cond {r.cond:.1e}: declined: {r.declined}")
        else:
            own = f"LARS's own KKT {r.ref_kkt:.1e}" if r.ref_kkt is not None else f"IPM objective worse by {r.ref_excess:.1e}"
            print(f"  {r.name:<22} cond {r.cond:.1e}: {r.knots} knots, {r.reference} gap {r.gap:.1e}, KKT {r.kkt:.1e}, {own}")
    wide_rows = stress(wide_problems)
    (OUT / "wide.tex").write_text(wide_tex(wide_rows))
    (OUT / "wide_numbers.tex").write_text(wide_numbers(wide_rows))
    print("\nwide, p > n, under constraints beyond a zero sum:")
    for r in wide_rows:
        print(f"  {r.name:<16} {r.knots} knots, IPM gap {r.gap:.1e}, KKT {r.kkt:.1e}, "
              f"IPM objective worse by {r.ref_excess:.1e}")
    mono = monotone_rows()
    (OUT / "monotone.tex").write_text(monotone_tex(mono))
    fl = floor_row()
    (OUT / "floor.tex").write_text(floor_tex(fl))
    print(f"\nfloor, a row that excludes beta = 0: {fl.knots} knots, IPM gap {fl.gap:.1e}, KKT {fl.kkt:.1e}, "
          f"slack at the start {fl.start:.1e}")
    enc = encodings()
    (OUT / "encodings.tex").write_text(encodings_tex(enc))
    (OUT / "encodings_numbers.tex").write_text(encodings_numbers(enc))
    print("\nzero sums, budget form against the penalty form relaxed:")
    for r in enc:
        rest = f"declined ({r.reason})" if r.penalty is None else f"{r.penalty} knots, gap {r.gap:.1e}, KKT {r.kkt:.1e}"
        print(f"  {r.name:<22} budget {r.knots:3d} knots; penalty {rest}")
    print("\nmonotone fit, rows relaxed to h:")
    for r in mono:
        print(f"  h = {r.h:.0e}: {r.knots} knots, KKT {r.kkt:.1e}, gap to h = 0 {r.gap:.1e} ({r.gap / r.h:.2f} h)")
    print("\nscale, zero-sum compositions:")
    for r in big:
        print(f"  n = {r.n:4d}, p = {r.p:3d}: {knot_range(r.knots):>7} knots, KKT {r.kkt:.1e}")
    print(f"{'problem':<16} {'n':>4} {'p':>4} {'knots':>6} {'ref':>5} {'gap':>9} {'KKT':>9}")
    for r in rows:
        print(f"{r.name:<16} {r.n:>4} {r.p:>4} {r.knots:>6} {r.reference:>5} {r.gap:9.1e} {r.kkt:9.1e}")
    export_zero_sum_paths()
    # Every problem of Table 2 with its exact path, for julia/gaines_check.jl, which runs
    # ConstrainedLasso.jl on them.
    arrays = {}
    for name, prob in problems().items():
        path = trace(prob)
        key = name.replace(", ", "-")
        fields = {"x": prob.x, "y": prob.y, "a": prob.a, "b": prob.b, "c": prob.c, "h": prob.h,
                  "lower": prob.lower, "upper": prob.upper, "lam": path.lam, "beta": path.beta}
        # NPZ.jl cannot read an empty array, so a problem without rows has no "a" and "b" (or "c" and "h")
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
    """Every knot, and every segment throughout, satisfies the LASSO's optimality conditions to rounding."""
    for r in rows:
        assert r.kkt < 1e-10, r


def test_segment_check_sees_between_samples() -> None:
    """A wrong segment that is right at both ends and at its midpoint fails the whole-segment check."""
    demo = segment_demo()
    assert demo.sampled < 1e-12
    assert demo.whole > 0.2
    assert demo.exact < 1e-12


def test_deterministic() -> None:
    """On one machine the discrete results repeat exactly; only residuals near rounding may move.

    Every random design is drawn from a recorded seed, so a second run over the same seeds
    must give the same knot counts, decline the same problems, and pass the same checks.
    """
    def discrete(rows: list[StressRow]) -> list[tuple]:
        return [(r.name, r.knots, r.declined, r.kkt is not None and r.kkt < 1e-10) for r in rows]
    assert discrete(stress()) == discrete(stress())
    assert [(r.knots, r.kkt < 1e-10) for r in scale()] == [(r.knots, r.kkt < 1e-10) for r in scale()]


def test_table_tex(rows: list[Row]) -> None:
    """Table 2 has one row per problem, with the numbers as the note prints them."""
    *tex, rule = table_tex(rows).splitlines()
    assert len(tex) == len(rows)
    assert rule == r"\bottomrule"
    assert sci(8.6e-14) == r"$8.6\times10^{-14}$"
    assert all(line.endswith(r"\\") and line.count("&") == 7 for line in tex)
    assert set(CONSTRAINTS) >= {r.name for r in rows}  # every problem has its constraint


def test_example() -> None:
    """The opening example: the plain path passes the cap and ends with HDL positive; capped, neither."""
    ex = example()
    _, _, names = diabetes()
    hdl = names.index("s3")
    assert ex.over > ex.knots // 2
    assert ex.ols["s1"] < -ex.cap and ex.ols["s2"] > ex.cap and ex.ols["s3"] > 0
    assert (np.abs(ex.capped) <= ex.cap * (1 + 1e-12)).all()
    assert (ex.capped[:, hdl] <= 1e-9 * ex.cap).all()
    # The sign is not an accident of u: every cap up to well past it keeps it, though not a loose one.
    assert 1.5 * ex.cap < ex.keeps < abs(ex.ols["s1"])


@pytest.mark.skipif(not SCD14.exists(), reason="no sCD14 data; fetch it with `uv run python/fetch_scd14.py`")
def test_scd14() -> None:
    """On the real data the path is exact."""
    assert hashlib.sha256(SCD14.read_bytes()).hexdigest() == SCD14_SHA256
    prob, _ = scd14()
    path = trace(prob)
    assert len(path.lam) == 60
    assert certify(prob, path) < 1e-10


def test_scale() -> None:
    """At p = 500 the path is still exact."""
    for r in scale():
        assert r.kkt < 1e-10, r


def test_stress() -> None:
    """On the hostile problems the path is exact or declined, never wrong.

    The near-duplicate at 1e-6 without a cap is declined for every seed. Under the cap
    it is declined only on the seeds where both twins become free before the cap binds,
    and then for the same reason, a free block singular to working precision. Nothing
    else is declined. Wide, zero-sum, which needs a singular Gram solved on the zero
    sum's null space, is traced (since cvxcla 2.6.0).

    Against LARS, two exact paths can part by as much as the conditioning allows, of the
    order of cond(X)^2 times the machine epsilon, and both meet the optimality conditions to
    rounding. Against the interior-point solver, the solver's objective is never better
    than the path's by more than its tolerance, so a gap there is the solver's.
    """
    rows = {r.name: r for r in stress()}
    assert rows["duplicate 1e-06"].declined == len(STRESS_SEEDS)
    assert {n for n, r in rows.items() if r.declined} <= {"duplicate 1e-06", "duplicate 1e-06, box"}
    if rows["duplicate 1e-06, box"].declined:
        assert "numerically singular" in rows["duplicate 1e-06, box"].reason, rows
    for r in rows.values():
        if r.kkt is None:
            continue
        # Wide, zero-sum has cond(X) near 1e16, and its residuals reach about 1e-11.
        assert r.kkt < 1e-10, r
        if r.reference == "LARS":
            # cond(X)^2 eps is the scale of the gap, not a bound: on aarch64 Linux, Duplicate 1e-04 is 1.6 times it
            assert r.gap < 10 * r.cond**2 * np.finfo(float).eps, r
            assert r.ref_kkt < 1e-12, r
        else:
            assert r.ref_excess > -1e-12, r  # the solver beats the path by no more than its tolerance


def test_wide() -> None:
    """With p > n the path is exact under bounds, signs and rows, with no ridge, for every seed.

    None is declined, every path meets the optimality conditions to rounding along whole
    segments, and where the interior-point solver parts from it, its objective is never
    better by more than rounding. That bound is looser than Table 4's: near the end of a
    path with p > n the fit almost interpolates, the objective falls to 1e-4 of its value
    at beta = 0, and a difference at the rounding of the sum of squares is a few 1e-12 of
    it. The inequality rows bind, so they are tested.

    The residuals are looser too. At the end of a path with p > n, near lambda = 1e-4
    lambda_max, the fit almost interpolates and cond(X) is near 1e16, and on one seed of
    Zero-sum, signs the residual there reaches 1e-9. Everywhere else it stays at rounding.

    Both bounds depend on the CPU at that one end of a path. On x86_64 and Apple silicon
    the solver's objective is lower by at most about 1e-11, relative, on aarch64 Linux by
    1.1e-10, and the residual reaches 1.3e-9 and 1.5e-9; the tolerances allow for all three.
    """
    rows = stress(wide_problems)
    for r in rows:
        assert r.declined == 0, r
        assert r.kkt < 1e-8, r
        assert r.ref_excess > -1e-9, r
    for seed in STRESS_SEEDS:
        prob = wide_problems(seed)["inequality rows"][0]
        end = trace(prob).beta[-1]
        assert np.isclose(np.abs(prob.c[0] @ end), prob.h[0])


def test_monotone() -> None:
    """The h = 0 path is exact; each relaxed path is exact for its h, keeps its knots, and lies h from h = 0.

    The second half is why the first works: cvxcla traces h = eps and h = 2 eps and
    extrapolates, which is exact when the knots move linearly with h.
    """
    prob = monotone(0.0)
    path = trace(prob)
    assert certify(prob, path) < 1e-12
    assert (prob.c @ path.beta.T <= 1e-12).all()
    rows = monotone_rows()
    assert len({r.knots for r in rows}) == 1
    for r in rows:
        assert r.kkt < 1e-10, r
        assert abs(r.gap / r.h - 1) < 0.005, r  # the note says: h itself, to two digits


def test_encodings() -> None:
    """The penalty form, relaxed, reproduces the budget form on every zero sum, the largest included."""
    rows = encodings()
    for r in rows:
        assert r.penalty == r.knots, r
        assert r.gap < 1e-12 and r.kkt < 1e-12, r
    assert rows[-1].spacing < 1e-6, rows[-1]  # the note says: knots closer than a millionth of lambda_max


def test_floor() -> None:
    """A row that excludes beta = 0 is traced exactly, from a start on the row."""
    r = floor_row()
    assert r.knots > 2, r
    assert r.gap < 1e-7 and r.kkt < 1e-12, r
    assert abs(r.start) < 1e-12, r  # the path starts on the floor, not at beta = 0



def test_lars_positive_end() -> None:
    """lars_path's positive mode stops short of lambda = 0, as the note says, and its end is on the CLA path."""
    prob = problems()["nonnegative"]
    lam, kkt = lars_positive_end(prob, trace(prob))
    assert lam > 1.0 and kkt > 1e-3, (lam, kkt)


def test_cholesterol_row_binds() -> None:
    """The inequality rows of Cholesterol are active at the end of the path, so they are tested."""
    prob = problems()["cholesterol"]
    end = trace(prob).beta[-1]
    assert np.isclose(np.abs(prob.c[0] @ end), prob.h[0])


# --------------------------------------------------------------------------------------
# Timings: machine-dependent, so run by hand and quoted only as ratios
# --------------------------------------------------------------------------------------
TIMING_REPEATS = 11  # each timed step runs this often after one warm-up; the median counts
SCALE_REPEATS = 3  # at p = 500 a grid takes minutes, so Table 3's sizes are timed this often
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


def timed(step: Callable[[], object], repeats: int = TIMING_REPEATS) -> np.ndarray:
    """Seconds of each of ``repeats`` runs of ``step``, after one untimed warm-up run.

    The warm-up takes the one-off costs out: CVXPY compiles the grid's problem on its
    first solve, and later solves reuse it.
    """
    step()
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        step()
        seconds.append(time.perf_counter() - start)
    return np.array(seconds)


def ratio(grid_s: np.ndarray, path_s: np.ndarray) -> str:
    """The grid's cost over the path's: the ratio of the medians, and the range of the ratio over the runs."""
    runs = grid_s / path_s
    return f"{np.median(grid_s) / np.median(path_s):5.0f}x the path (runs {runs.min():.0f}x to {runs.max():.0f}x)"


def grid(prob: Lasso, lam_max: float) -> Callable[[], None]:
    """Solve the constrained LASSO at GRID_POINTS values of lambda, log-spaced down to lam_max / 1000.

    The problem is built once with lambda as a parameter, so CVXPY compiles it once and
    each point costs one Clarabel solve at its default tolerances: the fairest setup for
    the solver.
    """
    beta, lam = cp.Variable(prob.x.shape[1]), cp.Parameter(nonneg=True)
    cons = [prob.a @ beta == prob.b] if prob.a.shape[0] else []
    if prob.c.shape[0]:
        cons.append(prob.c @ beta <= prob.h)
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
    """Print, for each constrained problem, what the exact path costs against a grid."""
    print(f"machine: {machine()}; each step timed {TIMING_REPEATS} times after a warm-up, "
          f"runs paired in order; the median and the range of the ratio")
    for name, prob in problems().items():
        if name not in ("box", "cholesterol", "zero-sum", "sCD14"):
            continue
        path = trace(prob)
        path_s = timed(lambda prob=prob: trace(prob))
        grid_s = timed(grid(prob, path.lam[0]))
        print(f"  {name:<11} {len(path.lam):3d} knots: path {np.median(path_s) * 1e3:7.1f} ms, "
              f"{GRID_POINTS}-point Clarabel grid {ratio(grid_s, path_s)}")
    print(f"Table 3, the zero-sum problems at scale, each step timed {SCALE_REPEATS} times after a warm-up:")
    for n, p in SCALE:
        prob = large_composition(n, p)
        path = trace(prob)
        path_s = timed(lambda prob=prob: trace(prob), SCALE_REPEATS)
        grid_s = timed(grid(prob, path.lam[0]), SCALE_REPEATS)
        print(f"  n = {n:4d}, p = {p:3d}, {len(path.lam):3d} knots: path {np.median(path_s):6.2f} s, "
              f"{GRID_POINTS}-point Clarabel grid {ratio(grid_s, path_s)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        sys.exit(pytest.main([__file__, "-q"]))
    if sys.argv[1:] == ["timings"]:
        sys.exit(timings())
    if sys.argv[1:] == ["export"]:
        sys.exit(export_zero_sum_paths())
    main()
