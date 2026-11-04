# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "casadi==3.8.1",
#     "clarabel==0.11.1",
#     "cvx-linalg==1.1.2",
#     "cvxcla==2.7.1",
#     "matplotlib==3.11.0",
#     "numpy==2.4.6",
#     "osqp==1.1.3",
#     "pandas==3.0.3",
#     "pyarrow==24.0.0",
#     "scipy==1.17.1",
#     "typer==0.26.7",
# ]
# ///
r"""Figures and numerical checks for the CLA paper.

One self-contained, seeded entry point that reproduces every figure and every
numerical check in the cvxcla paper. A small Typer CLI selects which
artefact to build; with no argument it asks interactively. The S&P 500 input is
the frozen snapshot ``sp500_pct_returns.parquet`` in ``data/`` beside this script
(``figures/data/`` in the paper's repository), so the data-driven steps reproduce
deterministically and offline.

This is the experiment script of cvxcla 2.7.1 (``experiments/make_figures.py`` at tag
v2.7.1), less the PyPortfolioOpt timing baseline, which the paper does not report, and
extended for the paper: the worked example of Section 6.4 (``worked-example``), the
factor-form grids, the comparison with the R package CLA (``r-cla``), the drift of the
incremental backend (``validate-incremental``), the single-thread timings (``threads``),
the KKT certificate of the largest traces, and the extended tolerance study.

    uv run make_figures.py            # choose interactively
    uv run make_figures.py all         # every figure + every check
    uv run make_figures.py all --quick # skip the four slow steps
    uv run make_figures.py frontier    # just one figure

Targets (artefact in parentheses; sections, figures and tables are the paper's):

  * ``frontier``        -- Figure 1: the 20-asset factor-model efficient frontier
    (frontier.pdf).
  * ``scaling``         -- Figure 4 and Sections 7.2-7.6: runtime vs problem size, dense
    vs factor (Woodbury) backend, with the incremental backend, the grids and qpOASES,
    and the memory figures (scaling.pdf).  SLOW: the dense backend at n=2560 dominates.
  * ``rank-scaling``    -- Figure 5, Section 7.4: runtime vs factor rank at fixed n
    (rank_scaling.pdf).  SLOW.
  * ``validate-exact``  -- Section 7.1: the frontier against a reference QP (no figure).
  * ``validate-constraints`` -- Section 7.1: the same under general constraints (no figure).
  * ``real-frontier``   -- Section 7.7: the S&P 500 frontier, dense and rank-20 (no figure).
  * ``worked-example``  -- Section 6.4, Figure 3: a 130/30 book on the S&P 500 with a
    group cap, run as printed in the paper (worked_example.pdf).
  * ``degeneracy``      -- Appendix B, Figure 6: the degeneracy boundary (degeneracy.pdf).
  * ``tie-degeneracy``  -- Appendix B: the tie-heavy stress families (no figure).
  * ``osqp``            -- Sections 7.2 and 7.5: S&P 500 trace vs the QP grids (no figure).
  * ``validate-kkt``    -- Section 7.1, Table 5: the KKT certificate (no figure).
  * ``validate-scaling`` -- Appendix A.1: invariance under a change of units, and the
    sensitivity of the trace to the slope floor (no figure).
  * ``validate-projection`` -- Appendix B: how often the feasibility projection fires
    and how large its corrections are (no figure).
  * ``validate-factor`` -- Section 5: the factor backend against the dense trace on
    ill-conditioned and degenerate factor models (no figure).
  * ``validate-conditioning`` -- Appendix B: problems of prescribed condition number
    across the singularity guard, each trace and its reference QP certified (no figure).
  * ``validate-incremental`` -- Section 5: round-off drift of the incremental backend
    against the dense one on the scaling problems (no figure).
  * ``r-cla``           -- Section 7.1: cvxcla against the R package CLA on the scaling
    problems; needs R and the CRAN package CLA, and is skipped without them (no figure).
  * ``threads``         -- Section 7: cvxcla's backends on one BLAS thread (no figure).
    SLOW.
  * ``parallel-grid``   -- Section 7.5: the Clarabel factor-form grid split over a warm
    process pool, against the serial grid and the factor trace (no figure).
  * ``blas``            -- Section 7.2: the n=2560 dense and factor traces, with the BLAS
    they ran on; run under Linux for the OpenBLAS comparison (no figure).  SLOW.
  * ``figures``         -- all five figures.
  * ``checks``          -- all fifteen numerical checks.
  * ``all``             -- every figure and every check.

Beyond ``cvxcla`` itself the steps need a few third-party packages: ``matplotlib`` (all
figures), ``pandas``/``pyarrow`` (the S&P 500 data), ``osqp`` (the reference QP and the
warm-started grid baseline), ``casadi`` (the qpOASES parametric QP solver, an external
baseline in ``scaling``), ``scipy`` (sparse
matrices). All are pinned in this
script's inline (PEP 723) metadata, so ``uv run`` provisions them automatically. A step
whose optional dependency is missing, or which raises, is reported and skipped rather
than aborting the whole run.

Pass ``--quick`` to ``all``/``figures``/``checks`` to skip the four slow steps
(``scaling``, ``rank-scaling``, ``threads`` and ``blas``); selecting one explicitly always runs it.
"""

from __future__ import annotations

import contextlib
import enum
import io
import itertools
import os
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import numpy as np
import typer

mpl.use("Agg")
import matplotlib.pyplot as plt

import cvxcla.cla as cla_module
from cvxcla import CLA, FactorCovariance, IncrementalDenseCovariance

# Default output directory: ``experiments/figures/``, so artefacts land there
# regardless of the working directory the command is run from.
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "figures"

# The frozen S&P 500 daily-return matrix (used by real-frontier and validate-exact):
# ``data/`` next to this script in cvxcla, ``figures/data/`` in the paper's repository.
_HERE = Path(__file__).resolve().parent
DATA = (
    next(
        (p for p in (_HERE / "data", _HERE / "figures" / "data") if (p / "sp500_pct_returns.parquet").exists()),
        _HERE / "data",
    )
    / "sp500_pct_returns.parquet"
)

# Samples per segment when drawing a frontier curve between turning points.
_CURVE_POINTS = 50


def _along_segments(weights: np.ndarray, num: int = _CURVE_POINTS) -> np.ndarray:
    """Return weights at ``num`` evenly spaced points along each segment between turning points.

    The weights move linearly between adjacent turning points, so their volatility traces
    a hyperbola there, not the chord a line plot through the corners draws. Every turning
    point is included exactly once, in order.
    """
    t = np.linspace(0.0, 1.0, num)[:-1, None]
    dense = [w0 + t * (w1 - w0) for w0, w1 in pairwise(weights)]
    return np.vstack([*dense, weights[-1:]])


# ======================================================================================
# OSQP: the reference QP; the grid tolerance
# ======================================================================================
_OSQP_EPS = 1e-6  # the tolerance of the factor-form OSQP grids


def _osqp_qp(
    mean: np.ndarray,
    cov: np.ndarray,
    lam: float,
    a: np.ndarray,
    b: np.ndarray,
    g: np.ndarray | None = None,
    h: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    leverage: float | None = None,
) -> np.ndarray:
    """Solve min 1/2 w'Sigma w - lam mu'w s.t. A w = b, G w <= h, lower <= w <= upper with OSQP.

    The box defaults to ``0 <= w <= 1``. A gross-exposure cap ``||w||_1 <= leverage`` is
    modelled with auxiliary variables ``t >= |w|`` and ``sum(t) <= leverage``. A cold,
    polished solve at tight tolerances, so the comparison probes the CLA's accuracy
    rather than the QP solver's stopping criteria.
    """
    import osqp
    from scipy import sparse

    n = mean.shape[0]
    lower = np.zeros(n) if lower is None else lower
    upper = np.ones(n) if upper is None else upper
    k = 0 if leverage is None else n  # number of auxiliary variables t

    def wide(block: object) -> sparse.csc_matrix:
        """Extend a row block over w with zero columns for t."""
        m = sparse.csc_matrix(block)
        return m if k == 0 else sparse.hstack([m, sparse.csc_matrix((m.shape[0], k))], format="csc")

    rows = [wide(a), wide(sparse.eye(n))]
    lo = [b, lower]
    up = [b, upper]
    if g is not None and g.shape[0] > 0:
        rows.append(wide(g))
        lo.append(np.full(g.shape[0], -np.inf))
        up.append(h)
    if leverage is not None:
        eye = sparse.eye(n)
        rows.append(sparse.hstack([eye, -eye], format="csc"))  # w - t <= 0
        rows.append(sparse.hstack([-eye, -eye], format="csc"))  # -w - t <= 0
        rows.append(sparse.hstack([sparse.csc_matrix((1, n)), sparse.csc_matrix(np.ones((1, n)))], format="csc"))
        lo += [np.full(2 * n, -np.inf), np.array([-np.inf])]
        up += [np.zeros(2 * n), np.array([leverage])]
    p_block = sparse.csc_matrix(np.triu(cov))
    if k:
        p_block = sparse.block_diag([p_block, sparse.csc_matrix((k, k))], format="csc")
    solver = osqp.OSQP()
    solver.setup(
        P=p_block,
        q=np.r_[-lam * mean, np.zeros(k)],
        A=sparse.vstack(rows, format="csc"),
        l=np.concatenate(lo),
        u=np.concatenate(up),
        eps_abs=1e-10,
        eps_rel=1e-10,
        max_iter=1_000_000,
        polishing=True,
        verbose=False,
    )
    return np.asarray(solver.solve(raise_error=True).x[:n], dtype=float)


def _cla_weight_at(cla: CLA, lam: float) -> np.ndarray:
    """Exact frontier weights ``w(lam)`` by linear interpolation between turning points.

    The frontier is affine in ``lambda`` on each segment and the turning points are
    the segment endpoints, so linear interpolation in ``lambda`` is exact (not an
    approximation). ``lam`` is clamped to the finite turning-point range.
    """
    pts = sorted((tp for tp in cla.turning_points if np.isfinite(tp.lamb)), key=lambda t: t.lamb)
    if lam <= pts[0].lamb:
        return pts[0].weights
    if lam >= pts[-1].lamb:
        return pts[-1].weights
    for lo, hi in pairwise(pts):
        if lo.lamb <= lam <= hi.lamb:
            t = (lam - lo.lamb) / (hi.lamb - lo.lamb)
            return (1.0 - t) * lo.weights + t * hi.weights
    msg = "lam within range but no bracketing segment found"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover


# ======================================================================================
# Figure: frontier (from frontier_20x50.py)  ->  frontier.pdf
# ======================================================================================
_FR20_N_ASSETS = 20
_FR20_N_DAYS = 50
_FR20_N_FACTORS = 5
_FR20_SEED = 42
_FR20_REPEATS = 5


@dataclass(frozen=True)
class _FactorModel:
    """The parameters of ``Sigma = diag(d) + U diag(delta) U^T``.

    Kept here rather than read back off the ``FactorCovariance`` operator, which
    since cvxcla 2.0 is a ``cvx.linalg.FactorOperator`` with no public ``d``/``u``/``delta``.
    """

    d: np.ndarray
    u: np.ndarray
    delta: np.ndarray

    def operator(self) -> object:
        """The covariance as a cvxcla factor backend."""
        return FactorCovariance(d=self.d, u=self.u, delta=self.delta)


def _fr20_build_model(rng: np.random.Generator) -> tuple[_FactorModel, np.ndarray]:
    """Return the ground-truth factor model and the per-asset expected returns."""
    u = rng.standard_normal((_FR20_N_ASSETS, _FR20_N_FACTORS)) / np.sqrt(_FR20_N_ASSETS)
    delta = rng.uniform(0.5, 2.0, _FR20_N_FACTORS) * _FR20_N_ASSETS
    d = rng.uniform(0.5, 2.0, _FR20_N_ASSETS)
    expected = rng.uniform(0.0, 1.0, _FR20_N_ASSETS)  # dispersed expected returns
    return _FactorModel(d=d, u=u, delta=delta), expected


def _fr20_simulate_returns(rng: np.random.Generator, factor: _FactorModel, expected: np.ndarray) -> np.ndarray:
    """Simulate (N_DAYS, N_ASSETS) returns whose population covariance is ``factor``."""
    factor_returns = rng.standard_normal((_FR20_N_DAYS, _FR20_N_FACTORS)) * np.sqrt(factor.delta)
    idiosyncratic = rng.standard_normal((_FR20_N_DAYS, _FR20_N_ASSETS)) * np.sqrt(factor.d)
    return expected + factor_returns @ factor.u.T + idiosyncratic


def _fr20_trace(problem: dict, covariance: object) -> tuple[object, float]:
    """Trace the frontier ``REPEATS`` times and return (last CLA, median seconds)."""
    times = []
    cla = None
    for _ in range(_FR20_REPEATS):
        start = time.perf_counter()
        cla = CLA(covariance=covariance, **problem)
        times.append(time.perf_counter() - start)
    return cla, float(np.median(times))


def figure_frontier(out_dir: Path) -> None:
    """Run the 20x50 experiment, print statistics, and write the frontier figure."""
    rng = np.random.default_rng(_FR20_SEED)
    factor, expected = _fr20_build_model(rng)
    returns = _fr20_simulate_returns(rng, factor, expected)

    mean = returns.mean(axis=0)
    dense_cov = np.diag(factor.d) + (factor.u * factor.delta) @ factor.u.T

    problem = {
        "mean": mean,
        "lower_bounds": np.zeros(_FR20_N_ASSETS),
        "upper_bounds": np.ones(_FR20_N_ASSETS),
        "a": np.ones((1, _FR20_N_ASSETS)),
        "b": np.ones(1),
    }

    cla, dense_time = _fr20_trace(problem, dense_cov)
    factor_cla, factor_time = _fr20_trace(problem, factor.operator())
    if len(cla) != len(factor_cla):
        msg = f"backends disagree: {len(cla)} vs {len(factor_cla)}"
        raise RuntimeError(msg)

    frontier = cla.frontier
    returns_f = frontier.returns
    vol_f = frontier.volatility
    max_sharpe, _ = frontier.max_sharpe
    cond = float(np.linalg.cond(dense_cov))

    print(f"problem                 : {_FR20_N_ASSETS} assets x {_FR20_N_DAYS} days, {_FR20_N_FACTORS}-factor model")
    print(f"covariance condition no.: {cond:,.1f}")
    distinct = len({tuple(np.round(tp.weights, 12)) for tp in cla.turning_points})
    print(f"turning points          : {len(cla)}, {distinct} distinct  (dense and factor agree)")
    print(f"dense trace  (median)   : {dense_time * 1e3:.1f} ms  ({dense_time / len(cla) * 1e3:.2f} ms/point)")
    print(f"factor trace (median)   : {factor_time * 1e3:.1f} ms  ({factor_time / len(cla) * 1e3:.2f} ms/point)")
    print(f"factor speedup          : {dense_time / factor_time:.2f}x")
    print(f"expected-return range   : [{returns_f.min():.4f}, {returns_f.max():.4f}]")
    print(f"volatility range        : [{vol_f.min():.4f}, {vol_f.max():.4f}]")
    print(f"max Sharpe (model units): {max_sharpe:.4f}")

    var_f = frontier.variance

    # Draw the exact curve between turning points and mark only the turning points: the
    # variance is quadratic along each segment, so each piece is a parabolic arc.
    curve_w = _along_segments(frontier.weights)
    curve_var = np.einsum("ij,jk,ik->i", curve_w, dense_cov, curve_w)
    curve_ret = curve_w @ mean
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ax.plot(curve_var, curve_ret, "-", lw=1.0, color="#1f4e79", label="Efficient frontier")
    ax.plot(var_f, returns_f, "o", ms=2.5, color="#1f4e79", label="Turning points")
    ax.scatter(var_f[[0, -1]], returns_f[[0, -1]], color="#c00000", zorder=5, s=18)
    ax.annotate(
        "max return",
        (var_f[0], returns_f[0]),
        textcoords="offset points",
        xytext=(0, -7),
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.annotate("min variance", (var_f[-1], returns_f[-1]), textcoords="offset points", xytext=(8, -2), fontsize=8)
    ax.set_xlabel(r"Variance $w^\top \Sigma w$ (model units)")
    ax.set_ylabel("Expected return (model units)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "frontier.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ======================================================================================
# Figure: scaling (from runtime_scaling.py)  ->  scaling.pdf  (SLOW)
# ======================================================================================
_SCALE_SIZES = [20, 40, 80, 160, 320, 640, 1280, 2560]
# qpOASES follows the same parametric path as the CLA, so it is the closest external
# comparison and is timed further, until it too takes minutes per trace.
_SCALE_QPOASES_MAX_N = 2560
_SCALE_N_FACTORS = 10
_SCALE_SEED = 7
_SCALE_REPEATS = 3
# Short traces are noisy (up to 40% between runs below 10 ms) and cheap, so the small
# sizes are timed more often.
_SCALE_REPEATS_SMALL = 11
_SCALE_SMALL_REPEAT_MAX_N = 160


def _scale_repeats(n: int) -> int:
    """How many times to time a method at size n."""
    return _SCALE_REPEATS_SMALL if n <= _SCALE_SMALL_REPEAT_MAX_N else _SCALE_REPEATS
_SCALE_GRID = 100  # fixed lambda-grid size for the factor-form grids, the same at every n


def _scale_make_problem(rng: np.random.Generator, n: int, k: int) -> tuple[np.ndarray, FactorCovariance, dict]:
    """Return (dense Sigma, FactorCovariance, problem dict) for a size-n factor model.

    The two covariance representations are mathematically identical, so both
    backends trace the same frontier; only the per-solve linear algebra differs.
    """
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    factor = FactorCovariance(d=d, u=u, delta=delta)
    dense = np.diag(d) + (u * delta) @ u.T
    problem = {
        "mean": rng.uniform(0.0, 1.0, n),  # dispersed expected returns
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }
    return dense, factor, problem


def _scale_stats(times: list[float]) -> tuple[float, float, float]:
    """Return (median, min, max) seconds, the centre and min--max band over repetitions."""
    return float(np.median(times)), float(np.min(times)), float(np.max(times))


def _scale_median_trace(covariance: object, problem: dict) -> tuple[int, float, float, float]:
    """Return (turning points, median, min, max) trace seconds over REPEATS repetitions."""
    cla = CLA(covariance=covariance, **problem)
    times = []
    for _ in range(_scale_repeats(len(problem["mean"]))):
        start = time.perf_counter()
        cla = CLA(covariance=covariance, **problem)
        times.append(time.perf_counter() - start)
    return (len(cla), *_scale_stats(times))


def _scale_median_trace_inverse(dense: np.ndarray, problem: dict) -> tuple[int, float, float, float]:
    """Time cvxcla's opt-in IncrementalDenseCovariance backend over REPEATS repetitions.

    The same loop and event logic as the dense backend, differing only in the
    linear algebra: a maintained free-block inverse, updated by a rank-one border or
    deletion at each turning point, in place of a fresh factorisation. The operator
    is rebuilt per repetition so each trace starts with no cached inverse. Returns
    (turning points, median, min, max) seconds.
    """
    cla = CLA(covariance=IncrementalDenseCovariance(dense), **problem)
    times = []
    for _ in range(_scale_repeats(len(problem["mean"]))):
        covariance = IncrementalDenseCovariance(dense)
        start = time.perf_counter()
        cla = CLA(covariance=covariance, **problem)
        times.append(time.perf_counter() - start)
    return (len(cla), *_scale_stats(times))


def _scale_qpoases_path(dense: np.ndarray, problem: dict) -> tuple[int, float, float, float, float] | None:
    """Time qpOASES following the whole frontier in one hot-started call, if installed.

    qpOASES (shipped with CasADi) is a parametric active-set solver: a hot start from
    one QP to the next follows the homotopy between them, changing the working set at
    every breakpoint on the way. Solving at the maximum-return end, above the first
    finite turning point, and hot-starting to lambda = 0 therefore walks the same path
    as the CLA, but returns only its endpoint, the minimum-variance portfolio. Each
    repetition starts from a fresh solver (a cold start), and the two calls are timed
    together; one call per trace keeps CasADi's per-call overhead out of the figure.
    Returns (working-set changes, median, min, max seconds, endpoint gap to the CLA) or
    None when CasADi is absent.
    """
    try:
        import casadi as ca
    except ImportError:
        return None

    cla = CLA(covariance=dense, **problem)
    lam_top = 2.0 * max(tp.lamb for tp in cla.turning_points if np.isfinite(tp.lamb))
    n = len(problem["mean"])
    h, a = ca.DM(dense), ca.DM(problem["a"])
    bounds = {"lba": problem["b"], "uba": problem["b"], "lbx": problem["lower_bounds"], "ubx": problem["upper_bounds"]}
    times, changes, endpoint = [], 0, np.zeros(n)
    # qpOASES prints its licence banner through CasADi, which writes to Python's
    # sys.stdout; keep it out of the sweep's log.
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(_scale_repeats(len(problem["mean"]))):
            solver = ca.conic("S", "qpoases", {"h": h.sparsity(), "a": a.sparsity()}, {"printLevel": "none"})
            start = time.perf_counter()
            solver(h=h, g=-lam_top * problem["mean"], a=a, **bounds)
            result = solver(h=h, g=np.zeros(n), a=a, **bounds)
            times.append(time.perf_counter() - start)
            changes = int(solver.stats()["iter_count"])
            endpoint = np.asarray(result["x"]).ravel()
    gap = float(np.max(np.abs(endpoint - cla.turning_points[-1].weights)))
    return (changes, *_scale_stats(times), gap)


def _scale_free_sizes(covariance: object, problem: dict) -> tuple[int, float]:
    """Return (max, median) free-set size |F| over the turning points of one trace.

    The dense step costs O(|F|^3), so the free-set size, not n, sets the per-step cost.
    """
    sizes = [int(np.sum(tp.free)) for tp in CLA(covariance=covariance, **problem).turning_points]
    return max(sizes), float(np.median(sizes))


def _scale_raw_factors(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The seed-7 size-n factor model as raw (d, U, delta, mean), drawn as in _scale_make_problem."""
    rng = np.random.default_rng(_SCALE_SEED)
    u = rng.standard_normal((n, _SCALE_N_FACTORS)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, _SCALE_N_FACTORS) * n
    d = rng.uniform(0.5, 2.0, n)
    return d, u, delta, rng.uniform(0.0, 1.0, n)


def _factor_form(d: np.ndarray, u: np.ndarray, delta: np.ndarray) -> tuple:
    """The factor form of the long-only budget QP in x = [w; y], y = U'w.

    The risk w'(D + U diag(delta) U')w becomes w'Dw + y'diag(delta)y, so the Hessian is
    diagonal and the constraint matrix sparse: what a general QP solver needs to exploit
    the factor structure. Returns (P, budget row, bounds block, factor rows), all sparse.
    """
    from scipy import sparse

    n, k = u.shape
    p_mat = sparse.diags(np.r_[d, delta], format="csc")
    budget = sparse.hstack([sparse.csc_matrix(np.ones((1, n))), sparse.csc_matrix((1, k))])
    box = sparse.hstack([sparse.eye(n), sparse.csc_matrix((n, k))])
    link = sparse.hstack([sparse.csc_matrix(u.T), -sparse.eye(k)])
    return p_mat, budget, box, link


def _osqp_factor_grid(d: np.ndarray, u: np.ndarray, delta: np.ndarray, mean: np.ndarray, lams: np.ndarray) -> tuple:
    """A warm-started OSQP grid on the factor form.

    OSQP is set up once, so its factorisation is reused; each lambda changes only the
    linear term, and each solve starts from the previous solution. Polishing recovers
    the active-set solution. Returns (weights, seconds, total iterations over the grid).
    """
    import osqp
    from scipy import sparse

    n, k = u.shape
    start = time.perf_counter()
    p_mat, budget, box, link = _factor_form(d, u, delta)
    solver = osqp.OSQP()
    solver.setup(
        P=p_mat,
        q=np.r_[-lams[0] * mean, np.zeros(k)],
        A=sparse.vstack([budget, box, link], format="csc"),
        l=np.r_[1.0, np.zeros(n), np.zeros(k)],
        u=np.r_[1.0, np.ones(n), np.zeros(k)],
        eps_abs=_OSQP_EPS,
        eps_rel=_OSQP_EPS,
        polishing=True,
        warm_starting=True,
        max_iter=40_000,
        verbose=False,
    )
    weights, iterations = [], 0
    for lam in lams:
        solver.update(q=np.r_[-lam * mean, np.zeros(k)])
        result = solver.solve(raise_error=True)
        weights.append(result.x[:n])
        iterations += int(result.info.iter)
    return np.array(weights), time.perf_counter() - start, iterations


def _clarabel_factor_grid(
    d: np.ndarray, u: np.ndarray, delta: np.ndarray, mean: np.ndarray, lams: np.ndarray, tol: float | None = None
) -> tuple:
    """A Clarabel grid on the factor form, at its default tolerances unless ``tol`` is given.

    Clarabel is an interior-point solver: it has no warm start, so each lambda is a fresh
    solve, and no active-set polish, so its weights are accurate only to its tolerance.
    Returns (weights, seconds, total iterations over the grid).
    """
    import clarabel
    from scipy import sparse

    n, k = u.shape
    start = time.perf_counter()
    p_mat, budget, box, link = _factor_form(d, u, delta)
    a_mat = sparse.vstack([budget, link, -box, box], format="csc")
    b_vec = np.r_[1.0, np.zeros(k), np.zeros(n), np.ones(n)]
    cones = [clarabel.ZeroConeT(1 + k), clarabel.NonnegativeConeT(2 * n)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    if tol is not None:
        for name in ("tol_gap_abs", "tol_gap_rel", "tol_feas", "tol_ktratio"):
            setattr(settings, name, tol)
    weights, iterations = [], 0
    for lam in lams:
        solution = clarabel.DefaultSolver(p_mat, np.r_[-lam * mean, np.zeros(k)], a_mat, b_vec, cones, settings).solve()
        if "Solved" not in str(solution.status):
            msg = f"Clarabel did not solve the factor-form QP at lambda={lam}: {solution.status}"
            raise RuntimeError(msg)
        weights.append(np.array(solution.x)[:n])
        iterations += int(solution.iterations)
    return np.array(weights), time.perf_counter() - start, iterations


def _factor_grid_run(
    solver: str,
    d: np.ndarray,
    u: np.ndarray,
    delta: np.ndarray,
    mean: np.ndarray,
    repeats: int,
    tol: float | None = None,
) -> tuple:
    """Time a factor-form grid of GRID lambdas against the exact long-only trace of the same model.

    Each grid point is compared with the exact frontier at the same lambda, in its weights
    and in what a user reads off a frontier portfolio: its expected return and its
    variance, relative to the exact ones. Returns (grid points, median, min, max seconds,
    largest weight error, largest relative return error, largest relative variance error,
    iterations per grid point).
    """
    n = len(mean)
    cla = CLA(
        mean=mean,
        covariance=FactorCovariance(d=d, u=u, delta=delta),
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
        a=np.ones((1, n)),
        b=np.ones(1),
    )
    lam_max = max(tp.lamb for tp in cla.turning_points if np.isfinite(tp.lamb))
    lams = np.linspace(lam_max, 0.0, _SCALE_GRID)
    times = []
    for _ in range(repeats):
        if solver == "osqp":
            weights, seconds, iterations = _osqp_factor_grid(d, u, delta, mean, lams)
        else:
            weights, seconds, iterations = _clarabel_factor_grid(d, u, delta, mean, lams, tol)
        times.append(seconds)

    def variance(w: np.ndarray) -> float:
        y = u.T @ w
        return float(w @ (d * w) + y @ (delta * y))  # delta is the diagonal of the factor covariance

    err_w = err_ret = err_var = 0.0
    for w, lam in zip(weights, lams, strict=True):
        exact = _cla_weight_at(cla, lam)
        err_w = max(err_w, float(np.abs(w - exact).max()))
        err_ret = max(err_ret, abs(float(mean @ (w - exact))) / abs(float(mean @ exact)))
        err_var = max(err_var, abs(variance(w) - variance(exact)) / variance(exact))
    return (len(lams), *_scale_stats(times), err_w, err_ret, err_var, iterations / len(lams))


def _scale_factor_grid(solver: str, n: int, tol: float | None = None) -> tuple:
    """The factor-form grid of _factor_grid_run on the seed-7 size-n factor model of the sweep."""
    d, u, delta, mean = _scale_raw_factors(n)
    return _factor_grid_run(solver, d, u, delta, mean, _scale_repeats(n), tol)


def _scale_memory(covariance: object, problem: dict, cov_bytes: int) -> tuple[int, int]:
    """Return (covariance storage, trace working memory) in bytes for one trace.

    The stored frontier is n weights per turning point, O(n^2) for any backend, so the
    working memory is the tracemalloc peak during the trace less what the finished
    trace retains: the transient allocations of the loop itself.
    """
    import tracemalloc

    tracemalloc.start()
    cla = CLA(covariance=covariance, **problem)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del cla
    return cov_bytes, peak - current


def _scale_measure(method: str, n: int) -> object:
    """Rebuild the size-n problem and time one method on it (run in a fresh process)."""
    rng = np.random.default_rng(_SCALE_SEED)
    dense, factor, problem = _scale_make_problem(rng, n, _SCALE_N_FACTORS)
    if method == "dense":
        return _scale_median_trace(dense, problem)
    if method == "factor":
        return _scale_median_trace(factor, problem)
    if method == "inverse":
        return _scale_median_trace_inverse(dense, problem)
    if method == "qpoases":
        return _scale_qpoases_path(dense, problem)
    if method == "osqp-factor":
        return _scale_factor_grid("osqp", n)
    if method == "clarabel-factor":
        return _scale_factor_grid("clarabel", n)
    if method == "free":
        return _scale_free_sizes(factor, problem)
    if method == "memory-dense":
        return _scale_memory(dense, problem, dense.nbytes)
    if method == "memory-factor":
        return _scale_memory(factor, problem, 8 * (n * (_SCALE_N_FACTORS + 1) + _SCALE_N_FACTORS))
    msg = f"unknown scaling method {method!r}"
    raise ValueError(msg)


# Environment variables that cap the BLAS thread pool (Accelerate, OpenBLAS, MKL, OpenMP).
_THREAD_VARS = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")


def _scale_fresh(method: str, n: int, single_thread: bool = False) -> object:
    """Time one method in its own freshly spawned interpreter.

    Run in one process, the external baselines (OSQP, qpOASES) leave the
    interpreter in a state that slows the cvxcla timings taken after them: the dense
    trace at n=640 took 1.16 s inside the sweep against 0.62 s alone. A fresh process
    per (method, n) keeps every measurement independent of what ran before it.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    if single_thread:
        # A spawned interpreter loads its BLAS afresh and reads these on loading.
        os.environ.update(dict.fromkeys(_THREAD_VARS, "1"))
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            return pool.submit(_scale_measure, method, n).result()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _clarabel_chunk(d: np.ndarray, u: np.ndarray, delta: np.ndarray, mean: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """Solve one worker's share of a Clarabel factor-form grid (run in a pool process)."""
    return _clarabel_factor_grid(d, u, delta, mean, lams)[0]


def _parallel_grid(
    d: np.ndarray, u: np.ndarray, delta: np.ndarray, mean: np.ndarray, repeats: int
) -> tuple[float, float, float, int]:
    """Time the 100-point Clarabel factor-form grid serially and split over a warm process pool.

    The pool is started, and every worker has imported Clarabel and solved one point, before
    the clock starts, so the parallel time is what a user with a running pool would see.
    Returns (median serial seconds, median parallel seconds, median factor trace seconds,
    workers).
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    n = len(mean)
    problem = {"lower_bounds": np.zeros(n), "upper_bounds": np.ones(n), "a": np.ones((1, n)), "b": np.ones(1)}
    covariance = FactorCovariance(d=d, u=u, delta=delta)
    trace_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        cla = CLA(mean=mean, covariance=covariance, **problem)
        trace_times.append(time.perf_counter() - start)
    lam_max = max(tp.lamb for tp in cla.turning_points if np.isfinite(tp.lamb))
    lams = np.linspace(lam_max, 0.0, _SCALE_GRID)
    serial = [_clarabel_factor_grid(d, u, delta, mean, lams)[1] for _ in range(repeats)]
    workers = os.cpu_count() or 1
    chunks = np.array_split(lams, workers)
    parallel = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(_clarabel_chunk, *zip(*[(d, u, delta, mean, c[:1]) for c in chunks], strict=True)))
        for _ in range(repeats):
            start = time.perf_counter()
            parts = list(pool.map(_clarabel_chunk, *zip(*[(d, u, delta, mean, c) for c in chunks], strict=True)))
            parallel.append(time.perf_counter() - start)
    if sum(len(part) for part in parts) != len(lams):
        msg = "the parallel grid lost grid points"
        raise RuntimeError(msg)
    return float(np.median(serial)), float(np.median(parallel)), float(np.median(trace_times)), workers


def check_parallel_grid(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Time the Clarabel factor-form grid in parallel, against the serial grid and the factor trace."""
    import pandas as pd

    r = pd.read_parquet(DATA).to_numpy()
    d_sp, u_sp, delta_sp = _real_factor_parts(np.cov(r, rowvar=False), _REAL_N_FACTORS)
    d_syn, u_syn, delta_syn, mean_syn = _scale_raw_factors(_SCALE_SIZES[-1])
    markets = [
        (f"synthetic factor market, n={_SCALE_SIZES[-1]}", d_syn, u_syn, delta_syn, mean_syn),
        (f"S&P 500, rank-{_REAL_N_FACTORS} factor model", d_sp, u_sp, delta_sp, r.mean(axis=0)),
    ]
    for label, d, u, delta, mean in markets:
        serial, parallel, trace, workers = _parallel_grid(d, u, delta, mean, _REAL_REPEATS)
        print(
            f"{label:<40}: factor trace {trace * 1e3:8.1f} ms; Clarabel grid serial {serial * 1e3:8.1f} ms, "
            f"on {workers} processes {parallel * 1e3:8.1f} ms ({serial / parallel:.1f}x; "
            f"{parallel / trace:.2f}x the trace)"
        )


def check_blas(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Time the n = 2560 dense and factor traces and name the BLAS they ran on.

    Run under Linux with NumPy's bundled OpenBLAS (Computational details) to see whether
    the factor backend's advantage depends on the BLAS of the main machine.
    """
    n = _SCALE_SIZES[-1]
    dense, factor, problem = _scale_make_problem(np.random.default_rng(_SCALE_SEED), n, _SCALE_N_FACTORS)
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    print(f"numpy {np.__version__}, BLAS {blas['name']} {blas.get('version', '')}")
    medians = {}
    for label, covariance in (("dense", dense), ("factor", factor)):
        n_pts, t_med, t_lo, t_hi = _scale_median_trace(covariance, problem)
        medians[label] = t_med
        print(f"{label:<6} n={n}: {n_pts} turning points, median {t_med:.3f} s ({t_lo:.3f}-{t_hi:.3f})")
    print(f"factor backend {medians['dense'] / medians['factor']:.0f}x faster")


# Sizes at which the threads check reruns cvxcla's backends on one BLAS thread.
_THREADS_SIZES = (640, 2560)


def check_threads(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Time cvxcla's backends with the default BLAS threads and with one, as the grids run.

    OSQP and Clarabel solve on one thread; the dense backends call a multithreaded BLAS.
    The ratios against the grids are fair to the grid only with cvxcla on one thread too.
    """
    print(f"{'method':<10}{'n':>6}{'default':>12}{'1 thread':>12}{'ratio':>8}")
    for n in _THREADS_SIZES:
        for method in ("dense", "inverse", "factor"):
            default = _scale_fresh(method, n)
            single = _scale_fresh(method, n, single_thread=True)
            print(
                f"{method:<10}{n:>6}{default[1] * 1e3:>10.1f}ms{single[1] * 1e3:>10.1f}ms"
                f"{single[1] / default[1]:>8.2f}"
            )


def figure_scaling(out_dir: Path) -> None:
    """Run the scaling sweep, print a table, and write the figure."""
    ns, dense_times, factor_times, inv_times, points = [], [], [], [], []
    dense_band, factor_band, inv_band = [], [], []
    qpo_times, qpo_band = [], []
    osf_times, osf_band, clf_times, clf_band = [], [], [], []
    memory = []
    for n in _SCALE_SIZES:
        n_pts, t_dense, d_lo, d_hi = _scale_fresh("dense", n)
        n_pts_f, t_factor, f_lo, f_hi = _scale_fresh("factor", n)
        if n_pts != n_pts_f:
            msg = f"backends disagree at n={n}: {n_pts} vs {n_pts_f}"
            raise RuntimeError(msg)
        inv = _scale_fresh("inverse", n)
        if inv and inv[0] != n_pts:
            # A maintained inverse accumulates round-off over the trace, so on a
            # near-tied size it can split or merge a turning point. Drop its point
            # rather than abort, so the figure still completes.
            print(f"  [note] incremental backend disagrees at n={n}: {inv[0]} vs {n_pts}; skipping its point")
            inv = None
        qpo = _scale_fresh("qpoases", n) if n <= _SCALE_QPOASES_MAX_N else None
        # The factor-form grids exploit the same structure as the factor backend and are
        # cheap enough to time at every size.
        osf = _scale_fresh("osqp-factor", n)
        clf = _scale_fresh("clarabel-factor", n)
        free_max, free_med = _scale_fresh("free", n)
        memory.append((n, *_scale_fresh("memory-dense", n), *_scale_fresh("memory-factor", n)))
        ns.append(n)
        points.append(n_pts)
        dense_times.append(t_dense)
        factor_times.append(t_factor)
        inv_times.append(inv[1] if inv else None)
        dense_band.append((d_lo, d_hi))
        factor_band.append((f_lo, f_hi))
        inv_band.append((inv[2], inv[3]) if inv else None)
        qpo_times.append(qpo[1] if qpo else None)
        qpo_band.append((qpo[2], qpo[3]) if qpo else None)
        osf_times.append(osf[1])
        osf_band.append((osf[2], osf[3]))
        clf_times.append(clf[1])
        clf_band.append((clf[2], clf[3]))
        inv_str = f"  inverse={inv[1] * 1e3:9.1f} ms" if inv else "  inverse=n/a"
        qpo_str = (
            f"  qpoases={qpo[1] * 1e3:9.1f} ms ({qpo[0]} working-set changes, endpoint gap {qpo[4]:.1e})"
            if qpo
            else "  qpoases=n/a"
        )
        print(
            f"n={n:4d}  points={n_pts:4d}  dense={t_dense * 1e3:8.1f} ms  "
            f"factor={t_factor * 1e3:8.1f} ms  speedup={t_dense / t_factor:5.2f}x  "
            f"|F| max={free_max:4d} median={free_med:6.1f}{inv_str}{qpo_str}"
        )
        print(
            f"        factor form: osqp={osf[1] * 1e3:9.1f} ms (max |dw| {osf[4]:.1e})  "
            f"clarabel={clf[1] * 1e3:9.1f} ms (max |dw| {clf[4]:.1e})  "
            f"vs cvxcla factor: osqp {osf[1] / t_factor:.1f}x, clarabel {clf[1] / t_factor:.1f}x"
        )
        print(
            f"        factor form, relative return / variance error: osqp {osf[5]:.1e} / {osf[6]:.1e}, "
            f"clarabel {clf[5]:.1e} / {clf[6]:.1e};  iterations per grid point: "
            f"osqp {osf[7]:.0f}, clarabel {clf[7]:.0f}"
        )

        def spread(lo_hi: tuple[float, float] | None, mid: float | None) -> str:
            return "n/a" if lo_hi is None or mid is None else f"{(lo_hi[1] - lo_hi[0]) / mid:.0%}"

        print(
            f"        spread (max-min)/median: dense {spread((d_lo, d_hi), t_dense)}, "
            f"factor {spread((f_lo, f_hi), t_factor)}, "
            f"inverse {spread(inv[2:4] if inv else None, inv[1] if inv else None)}, "
            f"qpoases {spread(qpo[2:4] if qpo else None, qpo[1] if qpo else None)}, "
            f"osqp-f {spread(osf[2:4], osf[1])}, clarabel-f {spread(clf[2:4], clf[1])}"
        )

    # The clean comparison is the incremental-inverse baseline vs the dense
    # backend: same vectorised event logic, differing only in the linear-algebra
    # strategy (maintained inverse vs fresh block-eliminated solve).
    if inv_times[-1] is not None:
        n_last = ns[-1]
        strategy = dense_times[-1] / inv_times[-1]  # solve cost / incremental-inverse cost
        line = (
            f"\nat n={n_last}: incremental inverse is {strategy:.2f}x the speed of the "
            f"dense fresh-solve backend; factor/dense = {dense_times[-1] / factor_times[-1]:.1f}x"
        )
        print(line)

    # Empirical log-log slope (runtime ~ n^p) over the largest four sizes a series
    # was timed at, plus the local slope of its last doubling.
    def slope(times: list[float | None]) -> tuple[float, float]:
        pairs = [(n, t) for n, t in zip(ns, times, strict=True) if t is not None][-4:]
        x = np.log(np.array([p[0] for p in pairs], dtype=float))
        y = np.log(np.array([p[1] for p in pairs]))
        return float(np.polyfit(x, y, 1)[0]), float((y[-1] - y[-2]) / (x[-1] - x[-2]))

    print()
    for name, series in [
        ("dense", dense_times),
        ("factor", factor_times),
        ("inverse", inv_times),
        ("qpoases", qpo_times),
        ("osqp-f", osf_times),
        ("clar-f", clf_times),
    ]:
        if sum(t is not None for t in series) >= 4:
            fit, last = slope(series)
            print(f"{name:7s} exponent p (time ~ n^p): {fit:.2f} over the largest four sizes, {last:.2f} last doubling")
    for n_q, t_q in zip(ns, qpo_times, strict=True):
        if t_q is not None:
            i_q = ns.index(n_q)
            print(
                f"at n={n_q}: qpoases/dense = {t_q / dense_times[i_q]:.1f}x, "
                f"qpoases/factor = {t_q / factor_times[i_q]:.1f}x"
            )

    # Clarabel's weights are accurate only to its tolerance: the cost of tightening it.
    print("\nClarabel factor-form grid against its tolerance (time; largest weight, return, variance error):")
    for n_t in (640, 2560):
        cells = []
        for tol in (None, 1e-10, 1e-12, 1e-14):
            name = "default" if tol is None else f"{tol:g}"
            try:
                _, t_med, _, _, err, err_r, err_v, _ = _scale_factor_grid("clarabel", n_t, tol)
            except RuntimeError as exc:
                cells.append(f"{name}: {str(exc).rsplit(': ', 1)[-1]}")
                continue
            cells.append(f"{name}: {t_med * 1e3:7.1f} ms, {err:.1e}, {err_r:.1e}, {err_v:.1e}")
        print(f"n={n_t:4d}  " + "   ".join(cells))

    mib = 2.0**20
    print("\nmemory: covariance storage and trace working memory (beyond the stored frontier), MiB")
    for n, d_cov, d_work, f_cov, f_work in memory:
        print(
            f"n={n:4d}  dense: storage={d_cov / mib:8.2f} working={d_work / mib:8.2f}   "
            f"factor: storage={f_cov / mib:6.3f} working={f_work / mib:7.3f}"
        )

    from matplotlib.ticker import NullFormatter, ScalarFormatter

    series = [
        # (times, bands, marker, colour, label); external baselines first, cvxcla last
        (qpo_times, qpo_band, "-P", "#9467bd", "qpOASES, hot-started path"),
        (osf_times, osf_band, "--v", "#ff7f0e", f"OSQP, {_SCALE_GRID}-point $\\lambda$-grid, factor form"),
        (clf_times, clf_band, "--^", "#8c564b", f"Clarabel, {_SCALE_GRID}-point $\\lambda$-grid, factor form"),
        (inv_times, inv_band, "-D", "#2ca02c", "cvxcla, incremental dense"),
        (dense_times, dense_band, "-o", "#c00000", "cvxcla, dense"),
        (factor_times, factor_band, "-s", "#1f4e79", f"cvxcla, factor ($K={_SCALE_N_FACTORS}$)"),
    ]

    def draw(n_max: int, name: str, title: str, headroom: float) -> None:
        """Plot every series over the sizes n <= n_max, with min--max bands, to ``name``."""
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        sizes = [n for n in ns if n <= n_max]
        for times, bands, marker, colour, label in series:
            keep = [i for i, n in enumerate(ns) if n <= n_max and times[i] is not None]
            if not keep:
                continue
            xs = [ns[i] for i in keep]
            spans = [bands[i] for i in keep if bands[i] is not None]
            if spans:
                ax.fill_between(xs, [b[0] for b in spans], [b[1] for b in spans], color=colour, alpha=0.18, linewidth=0)
            ax.loglog(xs, [times[i] for i in keep], marker, ms=4, color=colour, label=label)
        ax.set_xlabel("Number of assets $n$")
        ax.set_ylabel("Frontier trace time [s]")
        # Label the x-axis at the actual problem sizes as plain integers, not the
        # default powers of ten (which never coincide with 20, 40, ..., 640 and
        # leave cluttered minor-tick labels on a log axis).
        ax.set_xticks(sizes)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(sizes[0] * 0.85, sizes[-1] * 1.18)
        ax.tick_params(axis="x", labelsize=7)
        # Headroom above the slowest series keeps the legend clear of every curve.
        slowest = max(t for times, *_ in series for i, t in enumerate(times) if t is not None and ns[i] <= n_max)
        ax.set_ylim(top=slowest * headroom)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper left")
        fig.tight_layout()
        out = out_dir / name
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")

    draw(ns[-1], "scaling.pdf", "CLA runtime vs problem size", 300.0)


# ======================================================================================
# Figure: rank-scaling (from rank_scaling.py)  ->  rank_scaling.pdf  (SLOW)
# ======================================================================================
_RANK_N_ASSETS = 320
_RANK_RANKS = [20, 40, 80, 160, 320]
_RANK_SEED = 11
_RANK_REPEATS = 3


def _rank_make_problem(rng: np.random.Generator, n: int, k: int) -> tuple[np.ndarray, FactorCovariance, dict]:
    """Return (dense Sigma, FactorCovariance, problem dict) for an n-asset, K-factor model."""
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    factor = FactorCovariance(d=d, u=u, delta=delta)
    dense = np.diag(d) + (u * delta) @ u.T
    problem = {
        "mean": rng.uniform(0.0, 1.0, n),  # dispersed expected returns
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }
    return dense, factor, problem


def _rank_median_trace(covariance: object, problem: dict) -> tuple[int, float, float, float]:
    """Return (turning points, median, min, max) trace seconds over REPEATS repetitions."""
    cla = CLA(covariance=covariance, **problem)
    times = []
    for _ in range(_RANK_REPEATS):
        start = time.perf_counter()
        cla = CLA(covariance=covariance, **problem)
        times.append(time.perf_counter() - start)
    return len(cla), float(np.median(times)), float(np.min(times)), float(np.max(times))


def figure_rank_scaling(out_dir: Path) -> None:
    """Run the rank sweep, print a table, and write the figure."""
    ks, dense_times, factor_times, points = [], [], [], []
    dense_band, factor_band = [], []
    for k in _RANK_RANKS:
        rng = np.random.default_rng(_RANK_SEED)
        dense, factor, problem = _rank_make_problem(rng, _RANK_N_ASSETS, k)
        n_pts, t_dense, d_lo, d_hi = _rank_median_trace(dense, problem)
        n_pts_f, t_factor, f_lo, f_hi = _rank_median_trace(factor, problem)
        if n_pts != n_pts_f:
            msg = f"backends disagree at K={k}: {n_pts} vs {n_pts_f}"
            raise RuntimeError(msg)
        ks.append(k)
        points.append(n_pts)
        dense_times.append(t_dense)
        factor_times.append(t_factor)
        dense_band.append((d_lo, d_hi))
        factor_band.append((f_lo, f_hi))
        print(
            f"K={k:4d}  points={n_pts:4d}  dense={t_dense * 1e3:8.1f} ms  "
            f"factor={t_factor * 1e3:8.1f} ms  speedup={t_dense / t_factor:5.2f}x"
        )

    from matplotlib.ticker import NullFormatter, ScalarFormatter

    def band(bands: list[tuple[float, float]], color: str) -> None:
        """Shade the min--max range across repetitions for one series."""
        ax.fill_between(ks, [b[0] for b in bands], [b[1] for b in bands], color=color, alpha=0.18, linewidth=0)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    band(dense_band, "#c00000")
    ax.loglog(ks, dense_times, "-o", ms=4, color="#c00000", label="cvxcla, dense backend")
    band(factor_band, "#1f4e79")
    ax.loglog(ks, factor_times, "-s", ms=4, color="#1f4e79", label="cvxcla, factor backend (Woodbury)")
    ax.set_xlabel(f"Number of factors $K$ (fixed $n={_RANK_N_ASSETS}$)")
    ax.set_ylabel("Frontier trace time [s]")
    ax.set_xticks(ks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(ks[0] * 0.85, ks[-1] * 1.18)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_dir / "rank_scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ======================================================================================
# Check: real-frontier  (Section 7.7: the S&P 500 frontier)
# ======================================================================================
_REAL_SHORT_WINDOW = 120  # trading days < N -> rank-deficient sample covariance
_REAL_N_FACTORS = 20
# The Gram backend on a window shorter than the universe (T < N = 494), without and with
# a ridge (Sections 5 and 7.7): certified in validate-kkt, timed in real-frontier.
_GRAM_WINDOW = 300
_GRAM_RIDGE = 1e-5
_REAL_REPEATS = 5


def _real_problem(n: int) -> dict:
    """Long-only, fully-invested box-constrained problem of size n."""
    return {
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }


def _real_median_trace(mean: np.ndarray, covariance: object, n: int) -> tuple[object, float]:
    """Return (CLA, median trace seconds) over REPEATS repetitions."""
    cla = CLA(mean=mean, covariance=covariance, **_real_problem(n))
    times = []
    for _ in range(_REAL_REPEATS):
        start = time.perf_counter()
        cla = CLA(mean=mean, covariance=covariance, **_real_problem(n))
        times.append(time.perf_counter() - start)
    return cla, float(np.median(times))


def _real_factor_parts(cov: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(d, U, delta) of the diagonal-plus-low-rank estimate from the top-k eigenpairs of ``cov``."""
    evals, evecs = np.linalg.eigh(cov)
    top = np.argsort(evals)[::-1][:k]
    u = evecs[:, top]
    delta = np.clip(evals[top], 1e-12, None)
    d = np.clip(np.diag(cov) - np.einsum("ij,j,ij->i", u, delta, u), 1e-10, None)
    return d, u, delta


def _real_factor_estimate(cov: np.ndarray, k: int) -> FactorCovariance:
    """Diagonal-plus-low-rank estimate from the top-k eigenpairs of ``cov``."""
    d, u, delta = _real_factor_parts(cov, k)
    return FactorCovariance(d=d, u=u, delta=delta)


def check_real_frontier(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Run the empirical S&P 500 experiment and print its statistics (Section 7.7)."""
    import pandas as pd

    returns = pd.read_parquet(DATA)
    r = returns.to_numpy()
    t_days, n = r.shape
    mean = r.mean(axis=0)
    sample_cov = np.cov(r, rowvar=False)

    span = f"{returns.index[0].date()} -> {returns.index[-1].date()}"
    print(f"universe                : {n} assets x {t_days} days ({span})")
    print(f"covariance condition no.: {np.linalg.cond(sample_cov):,.0f}")

    # 1. Full-history frontier (dense) + Woodbury approximation.
    cla, dense_time = _real_median_trace(mean, sample_cov, n)
    factor_cla, factor_time = _real_median_trace(mean, _real_factor_estimate(sample_cov, _REAL_N_FACTORS), n)
    print(f"dense frontier          : {len(cla)} turning points in {dense_time * 1e3:.1f} ms (median)")
    print(
        f"factor (K={_REAL_N_FACTORS}) approx    : {len(factor_cla)} turning points in "
        f"{factor_time * 1e3:.1f} ms (median)"
    )
    # The backend comparison on one matrix: the rank-20 model, traced densely and by Woodbury.
    d20, u20, delta20 = _real_factor_parts(sample_cov, _REAL_N_FACTORS)
    dense20_cla, dense20_time = _real_median_trace(mean, np.diag(d20) + (u20 * delta20) @ u20.T, n)
    if len(dense20_cla) != len(factor_cla):
        msg = f"rank-{_REAL_N_FACTORS} backends disagree: {len(dense20_cla)} vs {len(factor_cla)}"
        raise RuntimeError(msg)
    print(
        f"same rank-{_REAL_N_FACTORS} matrix, dense: {len(dense20_cla)} turning points in "
        f"{dense20_time * 1e3:.1f} ms (median); factor backend {dense20_time / factor_time:.2f}x faster"
    )

    # The Gram backend against the dense matrix on a window shorter than the universe,
    # without a ridge (a singular sample covariance) and with one (Woodbury in the
    # T-dimensional observation space).
    from cvxcla import GramCovariance

    window = r[-_GRAM_WINDOW:]
    w_mean, w_cov = window.mean(axis=0), np.cov(window, rowvar=False)
    for label, ridge in (("no ridge", 0.0), (f"ridge {_GRAM_RIDGE:g}", _GRAM_RIDGE)):
        dense_w, t_dense_w = _real_median_trace(w_mean, w_cov + ridge * np.eye(n), n)
        gram_w, t_gram_w = _real_median_trace(w_mean, GramCovariance(window, ridge=ridge), n)
        gap = max(
            float(np.max(np.abs(a.weights - b.weights)))
            for a, b in zip(dense_w.turning_points, gram_w.turning_points, strict=False)
        )
        print(
            f"last {_GRAM_WINDOW} days, {label:<12}: dense {len(dense_w)} points in {t_dense_w * 1e3:.1f} ms, "
            f"Gram {len(gram_w)} points in {t_gram_w * 1e3:.1f} ms (median), max |dw| {gap:.1e}"
        )

    frontier = cla.frontier
    ann = 252.0  # annualise daily moments for readable axes
    ret = frontier.returns * ann
    vol = frontier.volatility * np.sqrt(ann)
    sharpe, _ = frontier.max_sharpe
    print(f"annualised return range : [{ret.min():.3f}, {ret.max():.3f}]")
    print(f"annualised vol range    : [{vol.min():.3f}, {vol.max():.3f}]")
    print(f"max Sharpe (annualised) : {sharpe * np.sqrt(ann):.3f}")

    # 2. A short (rank-deficient) estimation window. Since cvxcla 2.6.0 its frontier is
    # traced: the budget row holds the free assets on its null space, where the singular
    # free block is still positive definite. Earlier releases declined it.
    short = r[-_REAL_SHORT_WINDOW:]
    short_cov = np.cov(short, rowvar=False)
    rank = int(np.linalg.matrix_rank(short_cov))
    try:
        short_cla = CLA(mean=short.mean(axis=0), covariance=short_cov, **_real_problem(n))
        print(
            f"short window W={_REAL_SHORT_WINDOW}     : rank(S)={rank} < N={n}, traced: {len(short_cla)} turning points"
        )
    except Exception as exc:  # noqa: BLE001 - a release before 2.6.0 declines it
        print(f"short window W={_REAL_SHORT_WINDOW}     : rank(S)={rank} < N={n}, declined: {type(exc).__name__}")


# ======================================================================================
# Figure: worked example (Section 6.4: a 130/30 book on the S&P 500)  ->  worked_example.pdf
# ======================================================================================
_WORKED_TECH = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "ORCL", "ADBE", "CRM"]


def figure_worked_example(out_dir: Path) -> None:
    """Run the worked example of Section 6.4 as printed, and draw its two frontiers.

    A 130/30 book on the S&P 500 snapshot: per-stock bounds [-3%, 5%], the budget, a 6%
    cap on ten large technology names and a 130% gross-exposure cap, set up with the
    ``CLA.problem`` builder. The figure sets its frontier against the long-only book with
    the same 5% per-stock cap.
    """
    import pandas as pd

    from cvxcla import InfeasibleProblemError

    returns = pd.read_parquet(DATA)
    mu = 252 * returns.mean().to_numpy()  # annualized moments
    sigma = 252 * returns.cov().to_numpy()
    in_tech = returns.columns.isin(_WORKED_TECH).astype(float)

    book = (
        CLA.problem(mu, sigma)
        .bounds(-0.03, 0.05)
        .budget()
        .inequality(in_tech[None, :], 0.06)
        .leverage(1.3)
        .trace()
    )
    binds = sum(tp.active_ineq[0] for tp in book.turning_points)
    gross = sum(abs(tp.weights).sum() > 1.3 - 1e-9 for tp in book.turning_points)
    print(len(book), "turning points; the cap binds at", binds, "and the gross cap at", gross)
    _, w = book.frontier.max_sharpe
    print(
        f"max Sharpe: gross {abs(w).sum():.2f}, the ten names {in_tech @ w:.3f},",
        (w > 1e-9).sum(),
        "long,",
        (w < -1e-9).sum(),
        "short",
    )
    fine = book.frontier.interpolate(20)
    print(len(fine.weights), "portfolios on the exact frontier")
    try:
        CLA.problem(mu, sigma).bounds(0.0, 0.001).budget().trace()
    except InfeasibleProblemError as err:
        print(type(err).__name__ + ":", err)

    long_only = CLA.problem(mu, sigma).bounds(0.0, 0.05).budget().trace()
    print(len(long_only), "turning points for the long-only book with the same 5% cap")

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for cla, colour, label in (
        (long_only, "#7f7f7f", "long-only, 5% per stock"),
        (book, "#1f4e79", "130/30, 6% cap on ten names"),
    ):
        curve = cla.frontier.interpolate(20)
        ax.plot(curve.volatility, curve.returns, "-", lw=1.0, color=colour, label=label)
    tp = book.frontier
    active = np.array([bool(t.active_ineq[0]) for t in book.turning_points])
    ax.plot(tp.volatility[active], tp.returns[active], "x", ms=3.5, mew=0.9, color="#c00000", label="cap binding")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = out_dir / "worked_example.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ======================================================================================
# Figure: degeneracy (from degeneracy_boundary.py)  ->  degeneracy.pdf
# ======================================================================================
_DEGEN_N_ASSETS = 120
_DEGEN_WINDOWS = [240, 180, 150, 130, 120, 100, 90, 75, 60, 45, 30, 20, 15]
_DEGEN_SEED = 0
_DEGEN_GUARD = 1e12  # the singularity guard: a reduced free block with cond above this is declined


@dataclass
class _DegenSweepResult:
    """Outcome of one trace attempt at a given number of observations T."""

    t_obs: int
    rank: int
    completed: bool
    n_points: int
    worst_cond: float  # worst candidate free-block condition number, Sigma_FF
    obj_gap: float  # worst objective gap of a completed turning point vs QP
    reduced_cond: float = 0.0  # the same, on the budget's null space, Z^T Sigma_FF Z: what the guard reads


def _degen_max_objective_gap(cla: CLA, mean: np.ndarray, cov: np.ndarray) -> float:
    """Worst (obj_cla - obj_qp) over turning points; ~0 means the frontier is optimal."""
    n = len(mean)
    gap = 0.0
    for tp in cla.turning_points:
        lam = tp.lamb
        if not np.isfinite(lam) or lam <= 0:
            continue
        wq = _osqp_qp(mean, cov, lam, np.ones((1, n)), np.ones(1))
        wc = tp.weights
        obj_cla = 0.5 * wc @ cov @ wc - lam * mean @ wc
        obj_qp = 0.5 * wq @ cov @ wq - lam * mean @ wq
        gap = max(gap, float(obj_cla - obj_qp))
    return gap


def _degen_trace(t_obs: int) -> _DegenSweepResult:
    """Trace the frontier for a size-``t_obs`` sample, recording the diagnostics."""
    rng = np.random.default_rng(_DEGEN_SEED)
    returns = rng.standard_normal((t_obs, _DEGEN_N_ASSETS)) * 0.01 + rng.uniform(0.0, 1e-3, _DEGEN_N_ASSETS)
    cov = np.cov(returns, rowvar=False)
    mean = returns.mean(axis=0)
    kwargs = {
        "lower_bounds": np.zeros(_DEGEN_N_ASSETS),
        "upper_bounds": np.ones(_DEGEN_N_ASSETS),
        "a": np.ones((1, _DEGEN_N_ASSETS)),
        "b": np.ones(1),
    }

    from scipy.linalg import null_space

    worst, reduced = [0.0], [0.0]
    original_emit = CLA._emit

    def recording_emit(
        self: CLA, lamb: float, weights: np.ndarray, free: np.ndarray, *args: object, **kwargs: object
    ) -> None:
        # _emit's trailing arguments vary across releases (2.4.0 added the multipliers,
        # 2.6.0 the guard flag); pass them through untouched.
        if np.any(free):
            block = cov[np.ix_(free, free)]
            worst[0] = max(worst[0], float(np.linalg.cond(block)))
            if block.shape[0] > 1:  # the budget row holds the free assets on its null space
                z = null_space(np.ones((1, block.shape[0])))
                reduced[0] = max(reduced[0], float(np.linalg.cond(z.T @ block @ z)))
        original_emit(self, lamb, weights, free, *args, **kwargs)

    cla_module.CLA._emit = recording_emit
    try:
        cla = CLA(mean=mean, covariance=cov, **kwargs)
        completed, n_points = True, len(cla)
        obj_gap = _degen_max_objective_gap(cla, mean, cov)
    except ValueError:
        completed, n_points, obj_gap = False, 0, 0.0
    finally:
        cla_module.CLA._emit = original_emit

    return _DegenSweepResult(
        t_obs=t_obs,
        rank=int(np.linalg.matrix_rank(cov)),
        completed=completed,
        n_points=n_points,
        worst_cond=worst[0],
        obj_gap=obj_gap,
        reduced_cond=reduced[0],
    )


def figure_degeneracy(out_dir: Path) -> None:
    """Run the degeneracy sweep, print a table, and write the figure."""
    results = [_degen_trace(t) for t in _DEGEN_WINDOWS]

    print(f"universe n = {_DEGEN_N_ASSETS}; singularity guard (cond) = {_DEGEN_GUARD:g}\n")
    print(f"{'T':>5} {'rank':>5} {'status':>9} {'pts':>5} {'worst cond':>11} {'reduced':>11} {'obj gap':>11}")
    for r in results:
        status = "completed" if r.completed else "declined"
        print(
            f"{r.t_obs:>5} {r.rank:>5} {status:>9} {r.n_points:>5} {r.worst_cond:>11.2e} "
            f"{r.reduced_cond:>11.2e} {r.obj_gap:>11.2e}"
        )

    from matplotlib.lines import Line2D

    ts = [r.t_obs for r in results]
    cond = [max(r.worst_cond, 1.0) for r in results]
    red = [max(r.reduced_cond, 1.0) for r in results]
    done = [r.completed for r in results]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.scatter(ts, cond, s=30, zorder=2, facecolors="none", edgecolors="#999999", marker="o")
    for t, c, ok in zip(ts, red, done, strict=True):
        ax.scatter(t, c, s=36, zorder=3, color="#1f4e79" if ok else "#c00000", marker="o" if ok else "X")
    ax.axhline(_DEGEN_GUARD, ls="--", lw=1.0, color="#555555")
    ax.text(ts[0], _DEGEN_GUARD * 1.6, "singularity guard", ha="right", va="bottom", fontsize=7.5, color="#555555")
    ax.axvline(_DEGEN_N_ASSETS, ls=":", lw=1.0, color="#999999")
    ax.text(
        _DEGEN_N_ASSETS * 1.03,
        min(cond) * 1.5,
        f"$T=n={_DEGEN_N_ASSETS}$",
        rotation=90,
        va="bottom",
        fontsize=7.5,
        color="#777777",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observations $T$ (sample covariance rank $\\approx \\min(T-1, n)$)")
    ax.set_ylabel("Worst free-block cond. number")

    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1f4e79",
            ms=7,
            label="on the null space: completed (optimal)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#999999",
            ms=7,
            label="raw free block $\\Sigma_{FF}$",
        ),
    ]
    if not all(done):  # a declined marker only when some trace is declined
        legend.insert(
            1, Line2D([0], [0], marker="X", color="w", markerfacecolor="#c00000", ms=7, label="declined (guard)")
        )
    ax.legend(handles=legend, fontsize=7.5, loc="center left")
    fig.tight_layout()
    out = out_dir / "degeneracy.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")


# ======================================================================================
# Check: validate-exact (from validate_exact.py)
# ======================================================================================
_VEXACT_N_ASSETS = 40


def _vexact_qp_solution(mean: np.ndarray, cov: np.ndarray, lam: float) -> np.ndarray:
    """Solve min 1/2 w'Sigma w - lam mu'w s.t. 1'w=1, 0<=w<=1 with the reference QP."""
    n = mean.shape[0]
    return _osqp_qp(mean, cov, lam, np.ones((1, n)), np.ones(1))


def check_validate_exact(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Trace with the CLA, re-solve each segment midpoint as a QP, report error."""
    import pandas as pd

    returns = pd.read_parquet(DATA).iloc[:, :_VEXACT_N_ASSETS]
    mean = returns.mean(axis=0).to_numpy()
    cov = np.cov(returns.to_numpy(), rowvar=False)

    cla = CLA(
        mean=mean,
        covariance=cov,
        lower_bounds=np.zeros(_VEXACT_N_ASSETS),
        upper_bounds=np.ones(_VEXACT_N_ASSETS),
        a=np.ones((1, _VEXACT_N_ASSETS)),
        b=np.ones(1),
    )
    tps = cla.turning_points
    print(f"subproblem              : {_VEXACT_N_ASSETS} assets, cond(S)={np.linalg.cond(cov):,.0f}")
    print(f"turning points          : {len(tps)}")

    errors = []
    # Segments between consecutive turning points with finite lambda.
    for hi, lo in pairwise(tps):
        lam_hi, lam_lo = hi.lamb, lo.lamb
        if not np.isfinite(lam_hi):  # skip the lambda = inf endpoint segment
            continue
        lam = 0.5 * (lam_hi + lam_lo)
        frac = (lam - lam_lo) / (lam_hi - lam_lo)  # affine in lambda on the segment
        w_cla = lo.weights + frac * (hi.weights - lo.weights)
        w_qp = _vexact_qp_solution(mean, cov, lam)
        errors.append(float(np.max(np.abs(w_cla - w_qp))))

    errors = np.array(errors)
    print(f"segment midpoints tested: {errors.size}")
    print(f"median |w_CLA - w_QP|   : {np.median(errors):.2e}")
    print(f"max    |w_CLA - w_QP|   : {errors.max():.2e}")
    verdict = "EXACT (matches QP to solver tolerance)" if errors.max() < 1e-4 else "MISMATCH"
    print(f"verdict                 : {verdict}")


# ======================================================================================
# Check: validate-constraints (from validate_constraints.py)
# ======================================================================================
_VCON_N_ASSETS = 30
_VCON_N_FACTORS = 4
_VCON_N_SECTORS = 3
_VCON_SEED = 11
_VCON_SECTOR_CAP = 0.45
_VCON_SHORT = 0.3  # scenario 3: per-asset short limit
_VCON_LEVERAGE = 1.3  # scenario 3: gross-exposure cap


def _vcon_make_market(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dense Sigma, mean, sector id) for a deterministic factor market.

    Sector 0 carries a return premium, so the maximum-return end of the frontier
    wants to overweight it and the sector-exposure cap binds (scenario 2).
    """
    u = rng.standard_normal((_VCON_N_ASSETS, _VCON_N_FACTORS)) / np.sqrt(_VCON_N_ASSETS)
    delta = rng.uniform(0.5, 2.0, _VCON_N_FACTORS) * _VCON_N_ASSETS
    d = rng.uniform(0.5, 2.0, _VCON_N_ASSETS)
    cov = np.diag(d) + (u * delta) @ u.T
    sector = np.arange(_VCON_N_ASSETS) % _VCON_N_SECTORS  # interleaved sector membership
    mean = rng.uniform(0.0, 1.0, _VCON_N_ASSETS)
    mean[sector == 0] += 1.0  # a clear premium for sector 0 -> its cap will bind
    return cov, mean, sector


def _vcon_qp_solution(
    mean: np.ndarray,
    cov: np.ndarray,
    lam: float,
    a: np.ndarray,
    b: np.ndarray,
    g: np.ndarray | None,
    h: np.ndarray | None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    leverage: float | None = None,
) -> np.ndarray:
    """Solve the return-parametrised QP under A w = b, G w <= h, the box and an optional cap."""
    return _osqp_qp(mean, cov, lam, a, b, g, h, lower, upper, leverage)


def _vcon_validate(
    label: str,
    mean: np.ndarray,
    cov: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    g: np.ndarray | None = None,
    h: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    leverage: float | None = None,
) -> float:
    """Trace one constrained problem and check every segment midpoint against a QP.

    Returns the maximum weight discrepancy over the tested segment midpoints, and
    reports how many turning points hold an inequality row, or the gross-exposure cap,
    active (binding).
    """
    lower = np.zeros(_VCON_N_ASSETS) if lower is None else lower
    upper = np.ones(_VCON_N_ASSETS) if upper is None else upper
    cla = CLA(
        mean=mean,
        covariance=cov,
        lower_bounds=lower,
        upper_bounds=upper,
        a=a,
        b=b,
        g=g,
        h=h,
        leverage=leverage,
    )
    tps = cla.turning_points
    active = 0
    if g is not None and g.shape[0] > 0:
        active = sum(int(np.any(np.abs(g @ tp.weights - h) <= 1e-7)) for tp in tps)
    capped = 0
    if leverage is not None:
        capped = sum(int(abs(np.abs(tp.weights).sum() - leverage) <= 1e-7) for tp in tps)

    errors = []
    ties = 0  # zero-length segments: several events at one lambda, nothing to test
    for hi, lo in pairwise(tps):
        if not np.isfinite(hi.lamb):  # skip the lambda = inf endpoint segment
            continue
        if hi.lamb == lo.lamb:
            ties += 1
            continue
        lam = 0.5 * (hi.lamb + lo.lamb)
        frac = (lam - lo.lamb) / (hi.lamb - lo.lamb)  # affine in lambda on the segment
        w_cla = lo.weights + frac * (hi.weights - lo.weights)
        w_qp = _vcon_qp_solution(mean, cov, lam, a, b, g, h, lower, upper, leverage)
        errors.append(float(np.max(np.abs(w_cla - w_qp))))

    err = np.array(errors)
    print(f"\n{label}")
    print(f"  equality rows m         : {a.shape[0]}")
    if g is not None and g.shape[0] > 0:
        print(f"  inequality rows p       : {g.shape[0]}")
        print(f"  turning pts w/ active G : {active} / {len(tps)}")
    if leverage is not None:
        print(f"  gross-exposure cap c    : {leverage}")
        print(f"  turning pts w/ cap tight: {capped} / {len(tps)}")
    print(f"  turning points          : {len(tps)}")
    print(f"  segment midpoints tested: {err.size}")
    if ties:
        print(f"  zero-length segments    : {ties}")
    print(f"  median |w_CLA - w_QP|   : {np.median(err):.2e}")
    print(f"  max    |w_CLA - w_QP|   : {err.max():.2e}")
    return err.max()


def check_validate_constraints(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Run the multi-row-equality, inequality-cap and gross-exposure-cap scenarios and report exactness."""
    rng = np.random.default_rng(_VCON_SEED)
    cov, mean, sector = _vcon_make_market(rng)
    ones = np.ones((1, _VCON_N_ASSETS))

    # Scenario 1: budget + a characteristic-neutrality row (m = 2). The neutrality
    # target is reachable (equal weight satisfies it), so the problem is feasible
    # and the first vertex is the Section 3 linear program, not the greedy fill.
    char = rng.standard_normal(_VCON_N_ASSETS)
    target = float(char.mean())  # c' (1/n 1) = mean(char): equal weight is feasible
    a2 = np.vstack([ones, char[None, :]])
    b2 = np.array([1.0, target])
    err1 = _vcon_validate("Scenario 1 -- multi-row equality (budget + characteristic neutrality)", mean, cov, a2, b2)

    # Scenario 2: budget + per-sector exposure caps G w <= h. Sector 0 carries a
    # premium, so the high-return end wants to pile into it and the cap binds.
    g = np.array([(sector == s).astype(float) for s in range(_VCON_N_SECTORS)])
    h = np.full(_VCON_N_SECTORS, _VCON_SECTOR_CAP)
    err2 = _vcon_validate("Scenario 2 -- inequality sector-exposure caps (G w <= h)", mean, cov, ones, np.ones(1), g, h)

    # Scenario 3: a 130/30 book -- budget, shorts down to -30% per asset, and a
    # gross-exposure cap ||w||_1 <= 1.3 traced through the long/short lift.
    lower = np.full(_VCON_N_ASSETS, -_VCON_SHORT)
    upper = np.ones(_VCON_N_ASSETS)
    err3 = _vcon_validate(
        "Scenario 3 -- gross-exposure cap (130/30: ||w||_1 <= 1.3)",
        mean,
        cov,
        ones,
        np.ones(1),
        lower=lower,
        upper=upper,
        leverage=_VCON_LEVERAGE,
    )

    worst = max(err1, err2, err3)
    verdict = "EXACT (matches QP to solver tolerance)" if worst < 1e-4 else "MISMATCH"
    print(f"\nworst discrepancy across scenarios: {worst:.2e}  ->  {verdict}")


# ======================================================================================
# Check: tie-degeneracy (from tie_degeneracy.py)
# ======================================================================================
_TIE_SIZES = [20, 60, 120, 240]
_TIE_SEEDS = range(8)
_TIE_N_FACTORS = 5


def _tie_factor_cov(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """A well-conditioned positive-definite K-factor covariance."""
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    return np.diag(d) + (u * delta) @ u.T


def _tie_long_only(mean: np.ndarray, cov: np.ndarray, g: np.ndarray | None = None, h: np.ndarray | None = None) -> dict:
    """Assemble a long-only, fully-invested problem dict for the CLA constructor."""
    n = len(mean)
    return {
        "mean": mean,
        "covariance": cov,
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
        "g": g,
        "h": h,
    }


def _tie_tied_means(rng: np.random.Generator, n: int) -> dict:
    """Means fall into 4 exactly-tied groups (events tie at and along the trace)."""
    base = rng.uniform(0.0, 1.0, 4)
    mean = np.resize(np.repeat(base, int(np.ceil(n / 4))), n)
    return _tie_long_only(mean, _tie_factor_cov(rng, n, _TIE_N_FACTORS))


def _tie_duplicated_assets(rng: np.random.Generator, n: int) -> dict:
    """The second half copies the first: exchangeable pairs with tied means and events.

    Each pair shares its mean and its covariance row, but the idiosyncratic diagonal
    keeps the covariance positive definite, so the pair's events tie exactly.
    """
    half = n // 2
    u = rng.standard_normal((half, _TIE_N_FACTORS)) / np.sqrt(half)
    delta = rng.uniform(0.5, 2.0, _TIE_N_FACTORS) * half
    d = rng.uniform(0.5, 2.0, half)
    u_full = np.vstack([u, u])[:n]
    d_full = np.concatenate([d, d])[:n]
    cov = np.diag(d_full) + (u_full * delta) @ u_full.T  # positive definite: d_full > 0
    mean_half = rng.uniform(0.0, 1.0, half)
    mean = np.concatenate([mean_half, mean_half])[:n]
    return _tie_long_only(mean, cov)


def _tie_group_caps(rng: np.random.Generator, n: int) -> dict:
    """Non-overlapping group-exposure caps (p ~ n/3), tightened so many bind at once."""
    cov = _tie_factor_cov(rng, n, _TIE_N_FACTORS)
    mean = rng.uniform(0.0, 1.0, n)
    groups = np.arange(n) // 3  # groups of 3 -> p = ceil(n/3)
    p = int(groups.max()) + 1
    g = np.array([(groups == j).astype(float) for j in range(p)])
    h = np.full(p, 1.5 * 3 / n)  # 1.5x the equal-weight group mass: binds, stays feasible
    return _tie_long_only(mean, cov, g, h)


def _tie_overlapping_caps(rng: np.random.Generator, n: int) -> dict:
    """Overlapping windowed caps (p = n), tight enough to force a degenerate first vertex."""
    cov = _tie_factor_cov(rng, n, _TIE_N_FACTORS)
    mean = rng.uniform(0.0, 1.0, n)
    width = max(2, n // 10)
    g = np.array([np.isin(np.arange(n), np.arange(i, i + width) % n).astype(float) for i in range(n)])
    h = np.full(n, width / n)  # tight: the concentrated max-return vertex over-activates rows
    return _tie_long_only(mean, cov, g, h)


def _tie_points(cla: CLA, kwargs: dict) -> tuple[int, int]:
    """Return (tie points, tie points without linearly independent active constraints).

    A tie point is a turning point at the same lambda as the one before it (within the
    relative window of the event selection), skipping the first vertex, which the CLA
    lists twice. Proposition 1 of the paper needs the gradients of the constraints that
    hold with equality there -- the rows of A, the binding rows of G, the bounds a weight
    sits on -- to be linearly independent; the second count is the points where they are not.
    """
    n = len(kwargs["mean"])
    a = np.atleast_2d(kwargs["a"])
    g, h = kwargs.get("g"), kwargs.get("h")
    lower, upper = kwargs["lower_bounds"], kwargs["upper_bounds"]
    tps = cla.turning_points[1:]
    ties = dependent = 0
    for prev, tp in itertools.pairwise(tps):
        if not (np.isfinite(prev.lamb) and abs(tp.lamb - prev.lamb) <= 1e-10 * max(1.0, abs(prev.lamb))):
            continue
        ties += 1
        w = tp.weights
        at_bound = np.flatnonzero((np.abs(w - lower) <= 1e-9) | (np.abs(w - upper) <= 1e-9))
        rows = [a, np.eye(n)[at_bound]]
        if g is not None:
            rows.append(np.atleast_2d(g)[np.abs(np.atleast_2d(g) @ w - h) <= 1e-7])
        grads = np.vstack(rows)
        dependent += int(np.linalg.matrix_rank(grads) < grads.shape[0])
    return ties, dependent


# A completed trace counts only if it is optimal: its worst KKT residual, the certificate
# of Table 5, must not exceed the threshold that check uses. Counting completions alone
# would report traces that end but leave the efficient frontier, as check_tie_degeneracy
# shows by starting the tied families at the solver's vertex.
_TIE_CERTIFIED = 1e-8


def _tie_outcome(kwargs: dict) -> tuple[str, int, int, int, int, int]:
    """Trace one instance; return (status, turning_points, cap, max_active_rows, ties, dependent ties).

    status in {certified, off, declined, cap_hit}: a completed trace is certified when every
    turning point passes the KKT certificate, and off the frontier otherwise. The CLA
    traces in its constructor, so a decline (ValueError) or a cap hit (RuntimeError)
    surfaces here.
    """
    n = len(kwargs["mean"])
    g = kwargs.get("g")
    p = 0 if g is None else np.atleast_2d(g).shape[0]
    cap = 100 * (n + p + 1)
    try:
        cla = CLA(**kwargs)
    except ValueError:
        return "declined", 0, cap, 0, 0, 0
    except RuntimeError:
        return "cap_hit", 0, cap, 0, 0, 0
    active = 0
    if g is not None:
        h = kwargs["h"]
        active = max(int(np.sum(np.abs(g @ tp.weights - h) <= 1e-7)) for tp in cla.turning_points)
    res = _kkt_check(cla, np.asarray(kwargs["mean"]), np.asarray(kwargs["covariance"]), kwargs)
    worst = max(res.primal, res.stationarity, res.dual, res.complementarity)
    status = "certified" if worst <= _TIE_CERTIFIED else "off"
    return status, len(cla.turning_points), cap, active, *_tie_points(cla, kwargs)


def check_tie_degeneracy(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Run every family over the size/seed grid and report the robustness envelope."""
    families = {
        "tied means": _tie_tied_means,
        "exchangeable pairs": _tie_duplicated_assets,
        "group caps (p~n/3)": _tie_group_caps,
        "overlapping caps (p=n)": _tie_overlapping_caps,
    }
    grid = [(n, s) for n in _TIE_SIZES for s in _TIE_SEEDS]
    print(f"{len(grid)} instances per family; sizes {_TIE_SIZES}, seeds {_TIE_SEEDS.start}..{_TIE_SEEDS.stop - 1}\n")
    head = (
        f"{'family':<24}{'certified':>10}{'off':>5}{'declined':>9}{'cap hits':>9}{'max tps':>8}{'tps/cap':>9}"
        f"{'max active':>11}{'ties':>7}{'ties dep.':>10}"
    )
    print(head)
    for name, build in families.items():
        completed = off = declined = cap_hit = 0
        max_tps = max_active = ties = dependent = 0
        worst_ratio = 0.0
        for n, seed in grid:
            status, tps, cap, active, n_ties, n_dep = _tie_outcome(build(np.random.default_rng(seed), n))
            ties += n_ties
            dependent += n_dep
            if status in ("certified", "off"):
                completed += status == "certified"
                off += status == "off"
                max_tps = max(max_tps, tps)
                max_active = max(max_active, active)
                worst_ratio = max(worst_ratio, tps / cap)
            elif status == "declined":
                declined += 1
            else:
                cap_hit += 1
        print(
            f"{name:<24}{completed:>10}{off:>5}{declined:>9}{cap_hit:>9}{max_tps:>8}{worst_ratio:>8.1%}"
            f"{max_active:>11}{ties:>7}{dependent:>10}"
        )

    print(f"\n'certified': completed, with every turning point passing the KKT certificate to {_TIE_CERTIFIED:g};")
    print("'off': completed but failing it, a frontier that ends but is wrong. Every completing")
    print("trace stays far below the iteration cap 100(n+p+1).")
    print("'ties' counts turning points at the lambda of the one before (a tie); 'ties dep.'")
    print("those where the active constraint gradients are dependent, outside Proposition 1.")

    # Why completion alone proves nothing: start the tied families at the vertex the
    # solver returns instead of the point of the tied face the path leaves from (patch
    # CLA._face_minimiser to return its argument), and certify those traces too.
    print("\nstarted at the solver's vertex of the tied face instead (CLA._face_minimiser bypassed):")
    original = CLA._face_minimiser
    CLA._face_minimiser = lambda self, first: first  # type: ignore[method-assign]
    try:
        for name, build in (("tied means", _tie_tied_means), ("exchangeable pairs", _tie_duplicated_assets)):
            counts = dict.fromkeys(("certified", "off", "declined", "cap_hit"), 0)
            for n, seed in grid:
                counts[_tie_outcome(build(np.random.default_rng(seed), n))[0]] += 1
            print(
                f"{name:<24}{counts['certified']:>10}{counts['off']:>5}{counts['declined']:>9}{counts['cap_hit']:>9}"
            )
    finally:
        CLA._face_minimiser = original  # type: ignore[method-assign]


# ======================================================================================
# Check: osqp (Sections 7.2 and 7.5: the S&P 500 trace against the factor-form grids)
# ======================================================================================

def check_osqp(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Time the exact S&P 500 trace against OSQP and Clarabel grids on the factor form.

    Also reports the free-set sizes and the dense-against-factor timings of Sections 7.2
    and 7.7. The synthetic n-point grids are part of the ``scaling`` sweep.
    """
    import pandas as pd

    r = pd.read_parquet(DATA).to_numpy()
    t_days, n = r.shape
    mean, sigma = r.mean(axis=0), np.cov(r, rowvar=False)
    cla, t_cla = _real_median_trace(mean, sigma, n)

    print(f"universe                : {n} assets x {t_days} days")
    print(f"cvxcla exact trace      : {len(cla)} turning points in {t_cla * 1e3:.1f} ms (median)")

    # Free-set sizes set the cost of a dense step: on real data they stay far below n.
    def free_sizes(trace: CLA) -> str:
        sizes = [int(np.sum(tp.free)) for tp in trace.turning_points]
        return f"|F| max {max(sizes)}, median {np.median(sizes):.0f}"

    d, u, delta = _real_factor_parts(sigma, _REAL_N_FACTORS)
    factor_cla, t_factor = _real_median_trace(mean, FactorCovariance(d=d, u=u, delta=delta), n)
    print(f"free sets, sample covariance  : {free_sizes(cla)} of n = {n}")
    print(
        f"rank-{_REAL_N_FACTORS} factor model (Section 7.7): {len(factor_cla)} turning points in "
        f"{t_factor * 1e3:.1f} ms (median), {free_sizes(factor_cla)}"
    )
    dense20_cla, t_dense20 = _real_median_trace(mean, np.diag(d) + (u * delta) @ u.T, n)
    print(
        f"same rank-{_REAL_N_FACTORS} matrix, dense backend: {len(dense20_cla)} turning points in "
        f"{t_dense20 * 1e3:.1f} ms (median); factor backend {t_dense20 / t_factor:.2f}x faster"
    )
    # The worked example of Section 6.4 (5% caps, shorts, a group cap and a gross cap) on
    # the same universe: its free sets, and dense against factor on its rank-20 model.
    returns = pd.read_parquet(DATA)
    mu_w, sigma_w = 252 * returns.mean().to_numpy(), 252 * returns.cov().to_numpy()
    worked = {
        "lower_bounds": np.full(n, -0.03),
        "upper_bounds": np.full(n, 0.05),
        "a": np.ones((1, n)),
        "b": np.ones(1),
        "g": returns.columns.isin(_WORKED_TECH).astype(float)[None, :],
        "h": np.array([0.06]),
        "leverage": 1.3,
    }
    print(f"130/30 book, sample covariance: {free_sizes(CLA(mean=mu_w, covariance=sigma_w, **worked))} of n = {n}")
    dw, uw, deltaw = _real_factor_parts(sigma_w, _REAL_N_FACTORS)
    for label, covariance in (
        ("dense ", np.diag(dw) + (uw * deltaw) @ uw.T),
        ("factor", FactorCovariance(d=dw, u=uw, delta=deltaw)),
    ):
        CLA(mean=mu_w, covariance=covariance, **worked)
        times = []
        for _ in range(_REAL_REPEATS):
            start = time.perf_counter()
            book = CLA(mean=mu_w, covariance=covariance, **worked)
            times.append(time.perf_counter() - start)
        print(
            f"130/30 book, rank-{_REAL_N_FACTORS} model, {label}: {len(book)} turning points in "
            f"{np.median(times) * 1e3:.1f} ms (median), {free_sizes(book)}"
        )
    # The grids of the scaling sweep on that factor model: the solver gets the factor form.
    for solver in ("osqp", "clarabel"):
        pts, t_med, t_lo, t_hi, err_w, err_r, err_v, iters = _factor_grid_run(
            solver, d, u, delta, mean, _REAL_REPEATS
        )
        print(
            f"{solver:8s} grid, factor form : {pts} solves in {t_med * 1e3:.1f} ms (median), "
            f"{t_med / t_factor:.1f}x the factor trace; max|w - w*| = {err_w:.1e}, "
            f"relative return / variance error {err_r:.1e} / {err_v:.1e}, {iters:.0f} iterations per point"
        )


# ======================================================================================
# Check: validate-kkt  (Section 7.1, Table 5: the KKT certificate)
# ======================================================================================
_KKT_SHORT_WINDOW = 600  # trading days: more than the 494 assets, but ill-conditioned
_KKT_TOL = 1e-9  # a weight within this of a bound, or a row within it of h, is active
_KKT_FRACTIONS = (0.25, 0.5, 0.75)  # interior points tested on every segment
_KKT_ABOVE = (2.0, 4.0)  # multiples of the first finite lambda tested above it
_KKT_PASS = 1e-8  # relative residual below which a condition counts as satisfied


@dataclass
class _KKTResiduals:
    """The largest residual of each KKT condition over the points tested on one trace."""

    points: int = 0
    vertices: int = 0  # points whose free set does not determine the multipliers
    primal: float = 0.0
    stationarity: float = 0.0
    dual: float = 0.0
    complementarity: float = 0.0

    def update(self, other: tuple[float, float, float, float, bool]) -> None:
        """Fold in the four residuals of one test point (and whether it was a vertex)."""
        self.points += 1
        self.vertices += int(other[4])
        self.primal = max(self.primal, other[0])
        self.stationarity = max(self.stationarity, other[1])
        self.dual = max(self.dual, other[2])
        self.complementarity = max(self.complementarity, other[3])


def _lstsq(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Least squares by NumPy's SVD driver, or by SciPy's QR driver if that SVD does not converge.

    LAPACK's divide-and-conquer SVD occasionally fails on a nearly rank-deficient matrix,
    such as hundreds of active rows over few free weights; the QR route solves the same
    problem without an SVD.
    """
    try:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        import scipy.linalg

        return scipy.linalg.lstsq(matrix, rhs, lapack_driver="gelsy")[0]


def _kkt_point(
    w: np.ndarray,
    lam: float,
    mean: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
) -> tuple[float, float, float, float, bool]:
    """KKT residuals of ``min 1/2 w'Sw - lam mu'w`` s.t. ``Aw = b, Gw <= h, l <= w <= u`` at w.

    The certificate does not use the CLA's own multipliers or partition. The active
    set is read off w (a weight within _KKT_TOL of a bound is blocked, a row within it
    of h is active). When the free coordinates determine the equality and active-row
    multipliers (their rows have full rank there), these solve stationarity on the
    free coordinates by least squares and the bound multipliers are the remaining
    gradient on the blocked ones, so dual feasibility is tested on its own. At a
    vertex they are not unique (on the maximum-return vertex no weight is free), and
    the test asks instead whether sign-feasible multipliers exist: stationarity on the
    free coordinates fixes them up to a small null space, over which a linear program
    picks the choice with the largest sign margin on the blocked coordinates and rows.

    Returns (primal infeasibility, stationarity, dual infeasibility, complementarity,
    vertex); the middle three are relative to the scale max(1, |S w|, lam |mu|).
    """
    at_lower = np.abs(w - lower) <= _KKT_TOL
    at_upper = (np.abs(w - upper) <= _KKT_TOL) & ~at_lower
    free = ~(at_lower | at_upper)
    slack = g @ w - h
    active = np.abs(slack) <= _KKT_TOL

    grad = cov @ w - lam * mean
    scale = max(1.0, float(np.max(np.abs(cov @ w))), lam * float(np.max(np.abs(mean))))
    rows = np.vstack([a, g[active]])
    m = a.shape[0]
    vertex = rows.shape[0] > 0 and np.linalg.matrix_rank(rows[:, free]) < rows.shape[0]
    if not vertex:
        mult = _lstsq(rows[:, free].T, -grad[free]) if rows.shape[0] else np.zeros(0)
        full = grad + rows.T @ mult  # gradient of the Lagrangian without the bound terms
        nu_lower = full[at_lower]  # multiplier of l <= w: must be >= 0
        nu_upper = -full[at_upper]  # multiplier of w <= u: must be >= 0
        eta = mult[m:]  # multipliers of the active inequality rows: >= 0
        stationarity = float(np.max(np.abs(full[free]), initial=0.0)) / scale
    else:
        # The free coordinates leave some multipliers undetermined. Stationarity on them
        # fixes the multipliers up to the null space of rows[:, free]^T; over that small
        # space a linear program picks the multipliers with the largest sign margin on the
        # blocked coordinates and the active rows, so dual feasibility is tested for the
        # best choice and a failure means no sign-feasible multipliers exist.
        from scipy.linalg import null_space
        from scipy.optimize import linprog

        k = rows.shape[0]
        if free.any():
            base_mult = _lstsq(rows[:, free].T, -grad[free])
            basis = null_space(rows[:, free].T)
        else:
            base_mult, basis = np.zeros(k), np.eye(k)
        blocked = at_lower | at_upper
        sign = np.where(at_lower, 1.0, -1.0)[blocked]  # the sign the bound multiplier needs
        offset = np.concatenate([sign * (grad + rows.T @ base_mult)[blocked], base_mult[m:]])
        slope = np.vstack([sign[:, None] * (rows[:, blocked].T @ basis), basis[m:]])
        q = basis.shape[1]
        z = np.zeros(q)
        if q and len(offset):
            # maximise t subject to offset + slope z >= t, with t at most the scale
            lp = linprog(
                c=np.r_[np.zeros(q), -1.0],
                A_ub=np.hstack([-slope, np.ones((len(offset), 1))]),
                b_ub=offset,
                bounds=[(None, None)] * q + [(None, scale)],
                method="highs",
            )
            if lp.status == 0:
                z = lp.x[:q]
        mult = base_mult + basis @ z
        full = grad + rows.T @ mult
        nu_lower = full[at_lower]
        nu_upper = -full[at_upper]
        eta = mult[m:]
        stationarity = float(np.max(np.abs(full[free]), initial=0.0)) / scale

    primal = max(
        float(np.max(np.maximum(lower - w, 0.0), initial=0.0)),
        float(np.max(np.maximum(w - upper, 0.0), initial=0.0)),
        float(np.max(np.abs(a @ w - b), initial=0.0)),
        float(np.max(np.maximum(slack, 0.0), initial=0.0)),
    )
    dual = (
        max(
            float(np.max(np.maximum(-nu_lower, 0.0), initial=0.0)),
            float(np.max(np.maximum(-nu_upper, 0.0), initial=0.0)),
            float(np.max(np.maximum(-eta, 0.0), initial=0.0)),
        )
        / scale
    )
    complementarity = (
        max(
            float(np.max(np.abs(nu_lower * (w - lower)[at_lower]), initial=0.0)),
            float(np.max(np.abs(nu_upper * (upper - w)[at_upper]), initial=0.0)),
            float(np.max(np.abs(eta * slack[active]), initial=0.0)),
        )
        / scale
    )
    return primal, stationarity, dual, complementarity, vertex


def _kkt_trace_points(cla: CLA) -> list[tuple[float, np.ndarray]]:
    """Return (lambda, w) at interior points of every segment of a trace.

    Between consecutive turning points the weights are affine in lambda, so the
    interior points are interpolations; above the first finite turning point the
    portfolio is the constant maximum-return vertex.
    """
    points = []
    tps = cla.turning_points
    finite = [tp.lamb for tp in tps if np.isfinite(tp.lamb)]
    for hi, lo in pairwise(tps):
        if not np.isfinite(hi.lamb):
            points += [(c * finite[0], hi.weights) for c in _KKT_ABOVE]
            continue
        if hi.lamb - lo.lamb <= 1e-12 * max(1.0, abs(hi.lamb)):
            continue  # a zero-length step at a tie
        for frac in _KKT_FRACTIONS:
            points.append((lo.lamb + frac * (hi.lamb - lo.lamb), lo.weights + frac * (hi.weights - lo.weights)))
    return points


def _kkt_check(cla: CLA, mean: np.ndarray, cov: np.ndarray, kwargs: dict) -> _KKTResiduals:
    """KKT residuals over a trace of the problem with box, equality and inequality rows."""
    n = len(mean)
    g = np.zeros((0, n)) if kwargs.get("g") is None else np.atleast_2d(kwargs["g"])
    h = np.zeros(0) if kwargs.get("h") is None else np.asarray(kwargs["h"], dtype=float)
    res = _KKTResiduals()
    for lam, w in _kkt_trace_points(cla):
        res.update(
            _kkt_point(
                w, lam, mean, cov, kwargs["lower_bounds"], kwargs["upper_bounds"], kwargs["a"], kwargs["b"], g, h
            )
        )
    return res


def _kkt_check_leverage(cla: CLA, mean: np.ndarray, cov: np.ndarray, kwargs: dict, cap: float) -> _KKTResiduals:
    """KKT residuals of the lifted long/short program for a gross-exposure cap.

    Every asset whose box straddles zero is split into legs x+ = max(w, 0) and
    x- = max(-w, 0), so ``||w||_1 <= cap`` becomes the linear row ``1'x <= cap`` and
    the residuals are those of the lifted problem, built here independently of the
    CLA's own lift.
    """
    lower, upper = kwargs["lower_bounds"], kwargs["upper_bounds"]
    n = len(mean)
    legs = [(i, 1.0) for i in range(n) if upper[i] > 0] + [(i, -1.0) for i in range(n) if lower[i] < 0]
    lift = np.zeros((n, len(legs)))
    for j, (i, sign) in enumerate(legs):
        lift[i, j] = sign
    leg_upper = np.array([upper[i] if sign > 0 else -lower[i] for i, sign in legs])
    leg_asset = np.array([i for i, _ in legs])
    leg_sign = np.array([sign for _, sign in legs])
    a_l = kwargs["a"] @ lift
    # The problem's own rows of G w <= h carry over as G P x <= h; the cap is one more row.
    g = np.zeros((0, n)) if kwargs.get("g") is None else np.atleast_2d(kwargs["g"])
    h = np.zeros(0) if kwargs.get("h") is None else np.asarray(kwargs["h"], dtype=float)
    g_l = np.vstack([g @ lift, np.ones((1, len(legs)))])
    h_l = np.concatenate([h, [cap]])
    mean_l, cov_l = lift.T @ mean, lift.T @ cov @ lift
    res = _KKTResiduals()
    for lam, w in _kkt_trace_points(cla):
        x = np.maximum(leg_sign * w[leg_asset], 0.0)
        res.update(_kkt_point(x, lam, mean_l, cov_l, np.zeros(len(legs)), leg_upper, a_l, kwargs["b"], g_l, h_l))
    return res


def check_validate_kkt(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Report the largest KKT residual of each condition over every segment of several traces."""
    import pandas as pd

    from cvxcla import GramCovariance, IncrementalDenseCovariance

    returns = pd.read_parquet(DATA)
    cases = []

    def long_only(n: int) -> dict:
        return {"lower_bounds": np.zeros(n), "upper_bounds": np.ones(n), "a": np.ones((1, n)), "b": np.ones(1)}

    sub = returns.iloc[:, :_VEXACT_N_ASSETS]
    mean40, cov40 = sub.mean(axis=0).to_numpy(), np.cov(sub.to_numpy(), rowvar=False)
    cases.append(("S&P 500, 40 assets", mean40, cov40, cov40, long_only(_VEXACT_N_ASSETS), None))

    full_mean, full_cov = returns.mean(axis=0).to_numpy(), np.cov(returns.to_numpy(), rowvar=False)
    n_full = len(full_mean)
    cases.append(("S&P 500, 494 assets", full_mean, full_cov, full_cov, long_only(n_full), None))
    cases.append(
        (
            "S&P 500, 494 assets, incremental backend",
            full_mean,
            full_cov,
            IncrementalDenseCovariance(full_cov),
            long_only(n_full),
            None,
        )
    )
    short = returns.iloc[-_KKT_SHORT_WINDOW:]
    short_mean, short_cov = short.mean(axis=0).to_numpy(), np.cov(short.to_numpy(), rowvar=False)
    cases.append(
        (
            f"S&P 500, 494 assets, last {_KKT_SHORT_WINDOW} days",
            short_mean,
            short_cov,
            short_cov,
            long_only(n_full),
            None,
        )
    )

    window = returns.iloc[-_GRAM_WINDOW:].to_numpy()
    gram_mean, gram_cov = window.mean(axis=0), np.cov(window, rowvar=False)
    cases.append(
        (
            f"S&P 500, last {_GRAM_WINDOW} days, Gram backend",
            gram_mean,
            gram_cov,
            GramCovariance(window),
            long_only(n_full),
            None,
        )
    )
    cases.append(
        (
            f"  same, Gram backend with ridge {_GRAM_RIDGE:g}",
            gram_mean,
            gram_cov + _GRAM_RIDGE * np.eye(n_full),
            GramCovariance(window, ridge=_GRAM_RIDGE),
            long_only(n_full),
            None,
        )
    )

    dense, factor, problem = _scale_make_problem(np.random.default_rng(_SCALE_SEED), 320, _SCALE_N_FACTORS)
    factor_kwargs = {k: v for k, v in problem.items() if k != "mean"}
    cases.append(("factor market, 320 assets, factor backend", problem["mean"], dense, factor, factor_kwargs, None))

    rng = np.random.default_rng(_VCON_SEED)
    cov30, mean30, sector = _vcon_make_market(rng)
    ones = np.ones((1, _VCON_N_ASSETS))
    char = rng.standard_normal(_VCON_N_ASSETS)
    neutral = long_only(_VCON_N_ASSETS) | {"a": np.vstack([ones, char[None, :]]), "b": np.array([1.0, char.mean()])}
    cases.append(("30 assets, budget + neutrality row", mean30, cov30, cov30, neutral, None))
    caps = long_only(_VCON_N_ASSETS) | {
        "g": np.array([(sector == s).astype(float) for s in range(_VCON_N_SECTORS)]),
        "h": np.full(_VCON_N_SECTORS, _VCON_SECTOR_CAP),
    }
    cases.append(("30 assets, budget + sector caps", mean30, cov30, cov30, caps, None))
    book = {
        "lower_bounds": np.full(_VCON_N_ASSETS, -_VCON_SHORT),
        "upper_bounds": np.ones(_VCON_N_ASSETS),
        "a": ones,
        "b": np.ones(1),
    }
    cases.append(("30 assets, 130/30 gross-exposure cap", mean30, cov30, cov30, book, _VCON_LEVERAGE))

    # The largest traces of the paper: the n = 2560 market of the scaling sweep, traced by
    # both backends, and the 130/30 worked example of Section 6.4 on the S&P 500.
    dense_big, factor_big, problem_big = _scale_make_problem(
        np.random.default_rng(_SCALE_SEED), _SCALE_SIZES[-1], _SCALE_N_FACTORS
    )
    big_kwargs = {k: v for k, v in problem_big.items() if k != "mean"}
    n_big = len(problem_big["mean"])
    cases.append(
        (f"factor market, {n_big} assets, factor backend", problem_big["mean"], dense_big, factor_big, big_kwargs, None)
    )
    cases.append(
        (f"factor market, {n_big} assets, dense backend", problem_big["mean"], dense_big, dense_big, big_kwargs, None)
    )
    mu_w, sigma_w = 252 * returns.mean().to_numpy(), 252 * returns.cov().to_numpy()
    worked = {
        "lower_bounds": np.full(n_full, -0.03),
        "upper_bounds": np.full(n_full, 0.05),
        "a": np.ones((1, n_full)),
        "b": np.ones(1),
        "g": returns.columns.isin(_WORKED_TECH).astype(float)[None, :],
        "h": np.array([0.06]),
    }
    cases.append(("S&P 500 130/30 book (Section 6.4)", mu_w, sigma_w, sigma_w, worked, 1.3))

    print(
        f"{'problem':<42}{'cond':>9}{'points':>8}{'vertex':>7}{'primal':>10}{'station.':>10}{'dual':>10}{'compl.':>10}"
    )
    worst = 0.0
    for label, mean, cov, covariance, kwargs, cap in cases:
        cla = CLA(mean=mean, covariance=covariance, leverage=cap, **kwargs)
        res = _kkt_check(cla, mean, cov, kwargs) if cap is None else _kkt_check_leverage(cla, mean, cov, kwargs, cap)
        worst = max(worst, res.primal, res.stationarity, res.dual, res.complementarity)
        print(
            f"{label:<42}{np.linalg.cond(cov):>9.1e}{res.points:>8d}{res.vertices:>7d}{res.primal:>10.1e}"
            f"{res.stationarity:>10.1e}{res.dual:>10.1e}{res.complementarity:>10.1e}"
        )
    # The certificate must also detect errors: feed it wrong points from the first trace.
    mean, cov, kwargs = cases[0][1], cases[0][2], cases[0][4]
    points = _kkt_trace_points(CLA(mean=mean, covariance=cov, **kwargs))
    lam, w = points[len(points) // 2]
    no_rows = (np.zeros((0, len(w))), np.zeros(0))
    bounds = (kwargs["lower_bounds"], kwargs["upper_bounds"], kwargs["a"], kwargs["b"], *no_rows)
    inside = np.flatnonzero((w > 1e-6) & (w < 1.0 - 1e-6))
    shift = np.zeros_like(w)
    shift[inside[0]], shift[inside[1]] = 1e-3, -1e-3  # stays on the budget
    wrong = {
        "lambda of another segment": (points[len(points) // 4][0], w),
        "weights shifted by 1e-3": (lam, w + shift),
    }
    print("\nsensitivity (largest relative residual of the four conditions):")
    for name, (lam_x, w_x) in wrong.items():
        print(f"  {name:<28}{max(_kkt_point(w_x, lam_x, mean, cov, *bounds)[:4]):.1e}")
    print(
        f"\nprimal: absolute; stationarity, dual and complementarity: relative to max(1, |Sw|, lam|mu|)."
        f"\nvertex: points whose free set does not determine the multipliers; there the undetermined"
        f"\npart is chosen by a linear program for the largest sign margin, so a dual violation means"
        f"\nthat no sign-feasible multipliers exist."
        f"\nworst residual {worst:.1e}  ->  {'PASS' if worst < _KKT_PASS else 'FAIL'} (threshold {_KKT_PASS:.0e})"
    )


# ======================================================================================
# Check: validate-scaling  (Appendix A.1: units and the slope floor)
# ======================================================================================
_VSCALE_FACTORS = (1e-6, 1e-3, 1.0, 1e3, 1e6)  # rescalings of mu and of Sigma
_VSCALE_FLOORS = (1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6)  # multiples of the slope floors
_VSCALE_TOLS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10)  # values of the user tolerance


def _vscale_weights(cla: CLA) -> np.ndarray:
    """The turning-point weights of a trace, stacked row by row."""
    return np.array([tp.weights for tp in cla.turning_points])


def _vscale_floor_trace(mean: np.ndarray, cov: np.ndarray, kwargs: dict, factor: float) -> np.ndarray | str:
    """Trace with both slope floors multiplied by ``factor``; return the weights or the error.

    The floors are ``sqrt(eps) / lambda_scale`` for weight slopes and ``sqrt(eps) *
    mu_scale`` for multiplier slopes; passing ``lambda_scale / factor`` and
    ``mu_scale * factor`` to the event scan multiplies both by ``factor`` and leaves
    the event-ordering window alone.
    """
    original = cla_module.segment_events

    def scaled(segment, lower, upper, g, h, lam_scale=1.0, mu_scale=1.0):
        return original(segment, lower, upper, g, h, lam_scale / factor, mu_scale * factor)

    cla_module.segment_events = scaled
    try:
        return _vscale_weights(CLA(mean=mean, covariance=cov, **kwargs))
    except (RuntimeError, ValueError) as exc:
        return type(exc).__name__
    finally:
        cla_module.segment_events = original


def check_validate_scaling(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Invariance under a change of units, and the sensitivity of the trace to the slope floor."""
    import pandas as pd

    if not hasattr(CLA, "lambda_scale"):
        print("needs a cvxcla whose event tests are scale-aware (newer than 2.1.0); skipped")
        return
    returns = pd.read_parquet(DATA)

    def long_only(n: int) -> dict:
        return {"lower_bounds": np.zeros(n), "upper_bounds": np.ones(n), "a": np.ones((1, n)), "b": np.ones(1)}

    sub = returns.iloc[:, :_VEXACT_N_ASSETS]
    mean40, cov40 = sub.mean(axis=0).to_numpy(), np.cov(sub.to_numpy(), rowvar=False)
    kwargs40 = long_only(_VEXACT_N_ASSETS)
    reference = CLA(mean=mean40, covariance=cov40, **kwargs40)
    ref_w = _vscale_weights(reference)
    ref_lam = np.array([tp.lamb for tp in reference.turning_points])
    finite = np.isfinite(ref_lam) & (ref_lam > 0)

    print(f"rescaling mu by c and Sigma by s ({_VEXACT_N_ASSETS} S&P 500 assets, {len(ref_w)} turning points):")
    worst_w = worst_lam = 0.0
    failures = []
    for c in _VSCALE_FACTORS:
        for s in _VSCALE_FACTORS:
            try:
                cla = CLA(mean=c * mean40, covariance=s * cov40, **kwargs40)
            except (RuntimeError, ValueError) as exc:
                failures.append(f"c={c:.0e}, s={s:.0e}: {type(exc).__name__}")
                continue
            w = _vscale_weights(cla)
            lam = np.array([tp.lamb for tp in cla.turning_points])
            if w.shape != ref_w.shape:
                failures.append(f"c={c:.0e}, s={s:.0e}: {len(w)} turning points")
                continue
            worst_w = max(worst_w, float(np.max(np.abs(w - ref_w))))
            worst_lam = max(worst_lam, float(np.max(np.abs(lam[finite] * c / s - ref_lam[finite]) / ref_lam[finite])))
    print(
        f"  {len(_VSCALE_FACTORS) ** 2} rescalings, c and s from {_VSCALE_FACTORS[0]:.0e} to {_VSCALE_FACTORS[-1]:.0e}"
    )
    print(f"  failures: {', '.join(failures) if failures else 'none'}")
    print(f"  max |w - w_ref| = {worst_w:.1e}, max relative error of lambda * c / s = {worst_lam:.1e}")

    full_mean, full_cov = returns.mean(axis=0).to_numpy(), np.cov(returns.to_numpy(), rowvar=False)
    short = returns.iloc[-_KKT_SHORT_WINDOW:]
    dense, _, problem = _scale_make_problem(np.random.default_rng(_SCALE_SEED), 320, _SCALE_N_FACTORS)
    cases = [
        (f"S&P 500, {_VEXACT_N_ASSETS} assets", mean40, cov40, kwargs40),
        ("S&P 500, 494 assets", full_mean, full_cov, long_only(len(full_mean))),
        (
            f"S&P 500, 494 assets, last {_KKT_SHORT_WINDOW} days",
            short.mean(axis=0).to_numpy(),
            np.cov(short.to_numpy(), rowvar=False),
            long_only(len(full_mean)),
        ),
        ("factor market, 320 assets", problem["mean"], dense, {k: v for k, v in problem.items() if k != "mean"}),
    ]
    print("\nslope floors multiplied by k: max |w - w(k=1)|, or the changed turning-point count")
    print(f"{'problem':<42}" + "".join(f"{k:>9.0e}" for k in _VSCALE_FLOORS))
    for label, mean, cov, kwargs in cases:
        ref = _vscale_floor_trace(mean, cov, kwargs, 1.0)
        cells = []
        for factor in _VSCALE_FLOORS:
            w = _vscale_floor_trace(mean, cov, kwargs, factor)
            if isinstance(w, str):
                cells.append(w[:8])
            elif w.shape != ref.shape:
                cells.append(f"{len(w)} pts")
            else:
                cells.append(f"{np.max(np.abs(w - ref)):.0e}")
        print(f"{label + f' ({len(ref)})':<42}" + "".join(f"{c:>9}" for c in cells))

    # The user tolerance ``tol`` (default 1e-5) decides that a weight sits on a bound and
    # validates each turning point; it plays no part in ordering events.
    # tol also classifies the linear-programming first vertex (scenario 1), starts a trace on
    # a face of tied maximum returns, and validates the lifted gross-exposure cap; these
    # problems exercise those paths.
    rng = np.random.default_rng(_VCON_SEED)
    vcov, vmean, sector = _vcon_make_market(rng)
    ones = np.ones((1, _VCON_N_ASSETS))
    char = rng.standard_normal(_VCON_N_ASSETS)
    box = {"lower_bounds": np.zeros(_VCON_N_ASSETS), "upper_bounds": np.ones(_VCON_N_ASSETS)}
    tol_cases = [
        *cases,
        (
            "30 assets, budget + neutrality row",
            vmean,
            vcov,
            {**box, "a": np.vstack([ones, char[None, :]]), "b": np.array([1.0, float(char.mean())])},
        ),
        (
            "30 assets, budget + sector caps",
            vmean,
            vcov,
            {
                **box,
                "a": ones,
                "b": np.ones(1),
                "g": np.array([(sector == k).astype(float) for k in range(_VCON_N_SECTORS)]),
                "h": np.full(_VCON_N_SECTORS, _VCON_SECTOR_CAP),
            },
        ),
        (
            "30 assets, 130/30 gross-exposure cap",
            vmean,
            vcov,
            {
                "lower_bounds": np.full(_VCON_N_ASSETS, -_VCON_SHORT),
                "upper_bounds": np.ones(_VCON_N_ASSETS),
                "a": ones,
                "b": np.ones(1),
                "leverage": _VCON_LEVERAGE,
            },
        ),
    ]
    for size in (20, 60, 120):
        tied = _tie_tied_means(np.random.default_rng(0), size)
        tied_mean, tied_cov = tied.pop("mean"), tied.pop("covariance")
        tol_cases.append((f"tied means, {size} assets", tied_mean, tied_cov, tied))

    print("\nuser tolerance tol: max |w - w(tol=1e-5)|, or the changed turning-point count")
    print(f"{'problem':<42}" + "".join(f"{t:>9.0e}" for t in _VSCALE_TOLS))
    for label, mean, cov, kwargs in tol_cases:
        ref = _vscale_weights(CLA(mean=mean, covariance=cov, **kwargs))
        cells = []
        for tol in _VSCALE_TOLS:
            try:
                w = _vscale_weights(CLA(mean=mean, covariance=cov, tol=tol, **kwargs))
            except Exception as exc:  # noqa: BLE001 - report a declined trace in the table
                cells.append(type(exc).__name__[:8])
                continue
            cells.append(f"{len(w)} pts" if w.shape != ref.shape else f"{np.max(np.abs(w - ref)):.0e}")
        print(f"{label + f' ({len(ref)})':<42}" + "".join(f"{c:>9}" for c in cells))


# ======================================================================================
# Check: validate-incremental  (Section 5: round-off drift of the maintained inverse)
# ======================================================================================
def check_validate_incremental(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """How far the incremental backend drifts from the dense one over the scaling problems.

    IncrementalDenseCovariance maintains the inverse of the free block by rank-one updates,
    so round-off carries from step to step. On the seed-7 factor models of the scaling
    sweep, both backends trace the same dense covariance; reported are the turning-point
    counts, the largest weight difference between the two traces, and the largest KKT
    residual of each (any condition, as in Table 4).
    """
    from cvxcla import IncrementalDenseCovariance

    print(f"{'n':>5} {'points':>7} {'max |w_inc - w_dense|':>22} {'KKT dense':>10} {'KKT incr.':>10}")
    for n in [80, 160, 320, 640, 1280, 2560]:
        dense, _, problem = _scale_make_problem(np.random.default_rng(_SCALE_SEED), n, _SCALE_N_FACTORS)
        kwargs = {k: v for k, v in problem.items() if k != "mean"}
        ref = CLA(covariance=dense, **problem)
        inc = CLA(covariance=IncrementalDenseCovariance(dense), **problem)
        if len(inc) != len(ref):
            print(f"{n:5d} {len(ref):4d}/{len(inc):<4d} counts differ")
            continue
        gap = max(float(np.abs(a.weights - b.weights).max()) for a, b in zip(ref.turning_points, inc.turning_points, strict=True))
        worst = []
        for cla in (ref, inc):
            r = _kkt_check(cla, problem["mean"], dense, kwargs)
            worst.append(max(r.primal, r.stationarity, r.dual, r.complementarity))
        print(f"{n:5d} {len(ref):7d} {gap:22.1e} {worst[0]:10.1e} {worst[1]:10.1e}")


# ======================================================================================
# Check: r-cla  (Section 1: cvxcla against the R package CLA on the scaling problems)
# ======================================================================================
_RCLA_SCRIPT = r"""
suppressPackageStartupMessages(library(CLA))
args <- commandArgs(trailingOnly = TRUE)
mu <- scan(args[1], quiet = TRUE)
S <- as.matrix(read.csv(args[2], header = FALSE))
n <- length(mu)
start <- proc.time()[["elapsed"]]
r <- CLA(mu, S, rep(0, n), rep(1, n))
cat(proc.time()[["elapsed"]] - start, "\n", file = args[4])
write.table(t(r$weights_set), args[3], sep = ",", row.names = FALSE, col.names = FALSE)
cat(as.character(packageVersion("CLA")), "\n", file = args[4], append = TRUE)
"""


def _rcla_turning_points(mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Run R's CLA on (mean, cov) long-only with a budget; return (weights, seconds, version)."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        script = tmp_path / "run.R"
        script.write_text(_RCLA_SCRIPT)
        np.savetxt(tmp_path / "mu.txt", mean, fmt="%.17g")
        np.savetxt(tmp_path / "cov.csv", cov, fmt="%.17g", delimiter=",")
        subprocess.run(
            ["Rscript", str(script), str(tmp_path / "mu.txt"), str(tmp_path / "cov.csv"),
             str(tmp_path / "w.csv"), str(tmp_path / "meta.txt")],
            check=True,
            capture_output=True,
        )
        weights = np.loadtxt(tmp_path / "w.csv", delimiter=",", ndmin=2)
        seconds, version = (tmp_path / "meta.txt").read_text().split()
    return weights, float(seconds), version


def _set_distance(a: np.ndarray, b: np.ndarray) -> float:
    """The largest distance from a point of one set of portfolios to the nearest of the other."""
    gaps = np.abs(a[:, None, :] - b[None, :, :]).max(axis=2)
    return float(max(gaps.min(axis=1).max(), gaps.min(axis=0).max()))


def check_r_cla(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Compare cvxcla's turning points with those of the R package CLA on the scaling problems.

    Both trace the long-only, fully-invested frontier of the seed-7 factor models of the
    scaling sweep, given the dense covariance. The turning points are compared in weight
    space (the two packages scale lambda differently): their number, and the largest
    distance from a turning point of either to the nearest of the other. cvxcla lists the
    maximum-return vertex twice (Section 6.1), so it is counted once here.
    """
    import shutil

    if shutil.which("Rscript") is None:
        print("Rscript not found; skipped")
        return
    print(f"{'n':>5} {'cvxcla':>7} {'R CLA':>7} {'max distance':>13} {'R seconds':>10}")
    version = "?"
    for n in [20, 40, 80, 160, 320, 640]:
        dense, _, problem = _scale_make_problem(np.random.default_rng(_SCALE_SEED), n, _SCALE_N_FACTORS)
        cla = CLA(covariance=dense, **problem)
        mine = np.array([tp.weights for tp in cla.turning_points[1:]])  # the vertex once
        try:
            theirs, seconds, version = _rcla_turning_points(problem["mean"], dense)
        except Exception as exc:  # noqa: BLE001 - R or its CLA package missing
            print(f"R CLA failed ({type(exc).__name__}); is the CRAN package CLA installed? skipped")
            return
        print(f"{n:5d} {len(mine):7d} {len(theirs):7d} {_set_distance(mine, theirs):13.1e} {seconds:10.2f}")
    print(f"R package CLA {version}")


# ======================================================================================
# Check: validate-projection  (Appendix A feasibility corrections)
# ======================================================================================
def _vproj_trace(mean: np.ndarray, cov: np.ndarray) -> tuple[str, list[tuple[float, float, float]]]:
    """Trace a long-only problem, recording every feasibility projection that changes a point.

    Returns the outcome and, per correction, (max |w' - w|, relative change of the
    objective 1/2 w'Sw - lam mu'w, largest constraint residual of w').
    """
    n = len(mean)
    kwargs = {"lower_bounds": np.zeros(n), "upper_bounds": np.ones(n), "a": np.ones((1, n)), "b": np.ones(1)}
    records: list[tuple[float, float, float]] = []
    current = [0.0]
    original_project, original_emit = cla_module.project_feasible, cla_module.CLA._emit

    def objective(w: np.ndarray) -> float:
        return 0.5 * float(w @ cov @ w) - current[0] * float(mean @ w)

    def recording_project(weights, lower, upper, a, b, g, h, active_ineq):
        out = original_project(weights, lower, upper, a, b, g, h, active_ineq)
        if not np.array_equal(out, weights):
            change = abs(objective(out) - objective(weights)) / max(1.0, abs(objective(weights)))
            residual = max(
                float(np.max(np.maximum(lower - out, 0.0))),
                float(np.max(np.maximum(out - upper, 0.0))),
                float(np.max(np.abs(a @ out - b))),
            )
            records.append((float(np.max(np.abs(out - weights))), change, residual))
        return out

    def recording_emit(self, lamb, weights, free, *args, **kwargs):
        current[0] = 0.0 if not np.isfinite(lamb) else float(lamb)
        original_emit(self, lamb, weights, free, *args, **kwargs)

    cla_module.project_feasible, cla_module.CLA._emit = recording_project, recording_emit
    try:
        outcome = f"{len(CLA(mean=mean, covariance=cov, **kwargs))} points"
    except (RuntimeError, ValueError) as exc:
        outcome = f"declined ({type(exc).__name__})"
    finally:
        cla_module.project_feasible, cla_module.CLA._emit = original_project, original_emit
    return outcome, records


# Sample-covariance windows of the S&P 500 (the last T days), down to T < n = 494.
_SP500_WINDOWS = [1213, 900, 750, 600, 500, 400, 300, 250, 180, 120]


def check_validate_projection(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Report how often the feasibility projection fires and how large its corrections are."""
    import pandas as pd

    problems = []
    for t_obs in _DEGEN_WINDOWS:
        rng = np.random.default_rng(_DEGEN_SEED)
        returns = rng.standard_normal((t_obs, _DEGEN_N_ASSETS)) * 0.01 + rng.uniform(0.0, 1e-3, _DEGEN_N_ASSETS)
        problems.append(
            (f"synthetic n={_DEGEN_N_ASSETS}, T={t_obs}", returns.mean(axis=0), np.cov(returns, rowvar=False))
        )
    sp500 = pd.read_parquet(DATA)
    for t_obs in _SP500_WINDOWS:
        window = sp500.iloc[-t_obs:]
        problems.append(
            (f"S&P 500 n=494, T={t_obs}", window.mean(axis=0).to_numpy(), np.cov(window.to_numpy(), rowvar=False))
        )

    print(f"{'problem':<28}{'outcome':>22}{'proj.':>7}{'max |dw|':>11}{'max dobj':>11}{'residual':>11}")
    worst = (0.0, 0.0, 0.0)
    for label, mean, cov in problems:
        outcome, records = _vproj_trace(mean, cov)
        if records:
            stats = tuple(max(r[i] for r in records) for i in range(3))
            worst = tuple(max(a, b) for a, b in zip(worst, stats, strict=True))
            cells = "".join(f"{x:>11.1e}" for x in stats)
        else:
            cells = f"{'-':>11}" * 3
        print(f"{label:<28}{outcome:>22}{len(records):>7d}{cells}")
    print(
        f"\nlargest correction {worst[0]:.1e}, objective change {worst[1]:.1e} (relative), "
        f"constraint residual after projection {worst[2]:.1e}"
    )


# ======================================================================================
# Check: validate-factor  (Section 5 factor-model conditioning)
# ======================================================================================
_VFACTOR_N = 120
_VFACTOR_K = 10
_VFACTOR_SEED = 1


def _vfactor_compare(d: np.ndarray, u: np.ndarray, delta: np.ndarray, mean: np.ndarray) -> str:
    """Trace with the factor backend and with the dense Sigma it represents; summarise the gap."""
    n = len(d)
    dense = np.diag(d) + (u * delta) @ u.T
    kwargs = {"lower_bounds": np.zeros(n), "upper_bounds": np.ones(n), "a": np.ones((1, n)), "b": np.ones(1)}
    factor = CLA(mean=mean, covariance=FactorCovariance(d=d, u=u, delta=delta), **kwargs)
    res = _kkt_check(factor, mean, dense, kwargs)
    kkt = max(res.primal, res.stationarity, res.dual, res.complementarity)
    w_factor = np.array([tp.weights for tp in factor.turning_points])
    w_dense = np.array([tp.weights for tp in CLA(mean=mean, covariance=dense, **kwargs).turning_points])
    gap = f"{np.max(np.abs(w_factor - w_dense)):9.1e}" if w_factor.shape == w_dense.shape else f"{len(w_dense):>6} pts"
    return f"{len(w_factor):>7d}{len(w_dense):>7d}{gap}{kkt:>10.1e}{np.linalg.cond(dense):>11.1e}"


def check_validate_factor(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Factor backend against the dense trace on ill-conditioned and degenerate factor models.

    The Woodbury solve inverts Delta and the capacitance matrix Delta^{-1} + U_F' D_F^{-1} U_F.
    With d > 0 the latter is positive definite, so neither an ill-conditioned Delta, nor nearly
    collinear loadings, nor a full rank K = n should cost accuracy; this checks it.
    """
    rng = np.random.default_rng(_VFACTOR_SEED)
    n, k = _VFACTOR_N, _VFACTOR_K
    d = rng.uniform(0.5, 2.0, n)
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    mean = rng.uniform(0.0, 1.0, n)
    cases = [(f"cond(Delta) = {c:.0e}", d, u, np.geomspace(float(n), n / c, k)) for c in (1e0, 1e4, 1e8, 1e12, 1e14)]
    for gap in (1e-2, 1e-6, 1e-10):
        collinear = u.copy()
        collinear[:, 1] = collinear[:, 0] + gap * rng.standard_normal(n)
        cases.append((f"collinear loadings, gap {gap:.0e}", d, collinear, np.full(k, float(n))))
    for rank in (n // 2, n):
        cases.append((f"rank K = {rank}", d, rng.standard_normal((n, rank)) / np.sqrt(n), np.full(rank, float(n))))

    print(f"n = {n}; columns: turning points (factor, dense), max |w_factor - w_dense|, KKT residual, cond(Sigma)")
    print(f"{'factor model':<34}{'factor':>7}{'dense':>7}{'max|dw|':>9}{'KKT':>10}{'cond':>11}")
    for label, d_i, u_i, delta in cases:
        print(f"{label:<34}{_vfactor_compare(d_i, u_i, delta, mean)}")

    print("\nnot positive definite (refused at construction by a cvxcla newer than 2.1.0):")
    for label, delta in (
        ("singular", np.ones((k, k))),
        ("indefinite", np.r_[1.0, -0.5, np.ones(k - 2)]),
    ):
        try:
            FactorCovariance(d=d, u=u, delta=delta)
            outcome = "accepted"
        except ValueError as exc:
            outcome = f"ValueError: {exc}"
        print(f"  {label:<12}{outcome}")


# ======================================================================================
# Check: validate-conditioning  (Appendix A conditioning study)
# ======================================================================================
_VCOND_N = 60
_VCOND_SEED = 5
_VCOND_KAPPAS = (1e2, 1e4, 1e6, 1e8, 1e10, 1e11, 3e11, 1e12, 3e12, 1e13, 1e14, 1e16)
_VCOND_POINTS = 6  # segment midpoints compared with the reference QP per trace
_VCOND_CERTIFIED = 1e-8  # KKT residual at or below which a solution counts as certified


def _vcond_problem(kappa: float, rotated: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """A long-only, fully-invested problem whose covariance has condition number ``kappa``.

    The spectrum is prescribed, ``Sigma = Q diag(lambda) Q'`` with eigenvalues spread
    geometrically from 1 down to 1/kappa, so conditioning is varied on its own,
    independently of the sample size that drives the rank-deficiency sweep of the
    degeneracy target. With ``rotated`` Q is a random orthogonal matrix and the free
    blocks see only part of the spectrum; otherwise Q = I, Sigma is diagonal, and a free
    block inherits the full spread once its assets are free together.
    """
    rng = np.random.default_rng(_VCOND_SEED)
    q, _ = np.linalg.qr(rng.standard_normal((_VCOND_N, _VCOND_N)))
    if not rotated:
        q = np.eye(_VCOND_N)
    cov = (q * np.geomspace(1.0, 1.0 / kappa, _VCOND_N)) @ q.T
    cov = 0.5 * (cov + cov.T)
    mean = rng.uniform(0.0, 1.0, _VCOND_N)
    kwargs = {
        "lower_bounds": np.zeros(_VCOND_N),
        "upper_bounds": np.ones(_VCOND_N),
        "a": np.ones((1, _VCOND_N)),
        "b": np.ones(1),
    }
    return mean, cov, kwargs


def _vcond_reference(cla: CLA, mean: np.ndarray, cov: np.ndarray, kwargs: dict) -> tuple[float, float, int]:
    """Compare the trace with OSQP at segment midpoints; certify OSQP's own solutions.

    Returns (max |w_CLA - w_QP|, largest KKT residual of the QP solutions, QP failures).
    The QP solutions are judged by the same KKT certificate as the trace, so a reference
    that struggles shows up in its own residual, not only as a disagreement.
    """
    tps = cla.turning_points
    segments = [(hi, lo) for hi, lo in pairwise(tps) if np.isfinite(hi.lamb) and hi.lamb > lo.lamb]
    picks = np.unique(np.linspace(0, len(segments) - 1, _VCOND_POINTS).round().astype(int)) if segments else []
    gap, ref_kkt, failures = 0.0, 0.0, 0
    zero_g, zero_h = np.zeros((0, _VCOND_N)), np.zeros(0)
    for i in picks:
        hi, lo = segments[i]
        lam = 0.5 * (hi.lamb + lo.lamb)
        w_cla = 0.5 * (hi.weights + lo.weights)
        try:
            w_qp = _vexact_qp_solution(mean, cov, lam)
        except Exception:  # noqa: BLE001 - any solver failure is recorded, not raised
            failures += 1
            continue
        gap = max(gap, float(np.max(np.abs(w_cla - w_qp))))
        lower, upper, a, b = kwargs["lower_bounds"], kwargs["upper_bounds"], kwargs["a"], kwargs["b"]
        ref_kkt = max(ref_kkt, max(_kkt_point(w_qp, lam, mean, cov, lower, upper, a, b, zero_g, zero_h)[:4]))
    return gap, ref_kkt, failures


def check_validate_conditioning(out_dir: Path) -> None:  # noqa: ARG001 - the shared runner signature
    """Trace problems of prescribed condition number across the singularity guard."""
    print(f"n = {_VCOND_N}, long-only budget; Sigma with eigenvalues spread from 1 to 1/kappa")
    for rotated in (True, False):
        print(f"\n{'random eigenvectors' if rotated else 'eigenvectors aligned with the assets (Sigma diagonal)'}:")
        _vcond_family(rotated)
    print(
        f"\ncertified: every KKT residual of the trace <= {_VCOND_CERTIFIED:.0e} (see validate-kkt);"
        "\nmax|dw|: largest gap to OSQP at segment midpoints; QP KKT: the same certificate applied"
        "\nto OSQP's own solutions, so a large value means the reference, not the trace, is unreliable."
    )


def _vcond_family(rotated: bool) -> None:
    """Print one row per condition number for one eigenvector family."""
    print(
        f"{'kappa':>8}{'outcome':>26}{'points':>8}{'cond(S_FF)':>12}{'KKT':>10}"
        f"{'max|dw|':>10}{'QP KKT':>10}{'QP fails':>9}"
    )
    for kappa in _VCOND_KAPPAS:
        mean, cov, kwargs = _vcond_problem(kappa, rotated)
        try:
            cla = CLA(mean=mean, covariance=cov, **kwargs)
        except (ValueError, RuntimeError) as exc:
            print(f"{kappa:>8.0e}{'declined: ' + type(exc).__name__:>26}")
            continue
        res = _kkt_check(cla, mean, cov, kwargs)
        kkt = max(res.primal, res.stationarity, res.dual, res.complementarity)
        worst = max(float(np.linalg.cond(cov[np.ix_(tp.free, tp.free)])) for tp in cla.turning_points if tp.free.any())
        gap, ref_kkt, failures = _vcond_reference(cla, mean, cov, kwargs)
        outcome = "completed, certified" if kkt <= _VCOND_CERTIFIED else "completed, not certified"
        print(
            f"{kappa:>8.0e}{outcome:>26}{len(cla):>8d}{worst:>12.1e}{kkt:>10.1e}"
            f"{gap:>10.1e}{ref_kkt:>10.1e}{failures:>9d}"
        )


# ======================================================================================
# Orchestration
# ======================================================================================
@dataclass
class Step:
    """One reproducible step: a target name, its runner, and whether it is slow."""

    target: Target
    artefact: str
    runner: object  # Callable[[Path], None]
    slow: bool


class Target(enum.StrEnum):
    """A buildable artefact, or a group of them."""

    frontier = "frontier"
    scaling = "scaling"
    rank_scaling = "rank-scaling"
    validate_exact = "validate-exact"
    validate_constraints = "validate-constraints"
    real_frontier = "real-frontier"
    degeneracy = "degeneracy"
    tie_degeneracy = "tie-degeneracy"
    osqp = "osqp"
    validate_kkt = "validate-kkt"
    validate_scaling = "validate-scaling"
    validate_projection = "validate-projection"
    validate_factor = "validate-factor"
    validate_conditioning = "validate-conditioning"
    r_cla = "r-cla"
    validate_incremental = "validate-incremental"
    threads = "threads"
    parallel_grid = "parallel-grid"
    blas = "blas"
    worked_example = "worked-example"
    figures = "figures"
    checks = "checks"
    all = "all"


# The individual steps, in the order the script runs them.
STEPS: list[Step] = [
    Step(Target.frontier, "Figure 1 (frontier.pdf)", figure_frontier, False),
    Step(Target.scaling, "Figure 4 (scaling.pdf), Sections 7.2-7.6", figure_scaling, True),
    Step(Target.rank_scaling, "Figure 5 (rank_scaling.pdf), Section 7.4", figure_rank_scaling, True),
    Step(Target.validate_exact, "Section 7.1: the frontier against a reference QP", check_validate_exact, False),
    Step(Target.validate_constraints, "Section 7.1: general constraints", check_validate_constraints, False),
    Step(Target.real_frontier, "Section 7.7: the S&P 500 frontier", check_real_frontier, False),
    Step(Target.worked_example, "Section 6.4, Figure 3 (worked_example.pdf)", figure_worked_example, False),
    Step(Target.degeneracy, "Appendix B, Figure 6 (degeneracy.pdf)", figure_degeneracy, False),
    Step(Target.tie_degeneracy, "Appendix B: tie-heavy stress families", check_tie_degeneracy, False),
    Step(Target.osqp, "Sections 7.2 and 7.5: the S&P 500 trace against QP grids", check_osqp, False),
    Step(Target.validate_kkt, "Section 7.1, Table 5: the KKT certificate", check_validate_kkt, False),
    Step(Target.validate_scaling, "Appendix A.1: units and the slope floor", check_validate_scaling, False),
    Step(Target.validate_projection, "Appendix B: feasibility corrections", check_validate_projection, False),
    Step(Target.validate_factor, "Section 5: factor-model conditioning", check_validate_factor, False),
    Step(Target.validate_conditioning, "Appendix B: conditioning study", check_validate_conditioning, False),
    Step(Target.r_cla, "Section 7.1: cvxcla against the R package CLA", check_r_cla, False),
    Step(Target.validate_incremental, "Section 5: drift of the incremental backend", check_validate_incremental, False),
    Step(Target.threads, "Section 7: cvxcla's backends on one BLAS thread", check_threads, True),
    Step(Target.parallel_grid, "Section 7.5: the Clarabel grid on a process pool", check_parallel_grid, False),
    Step(Target.blas, "Section 7.2: the n=2560 traces and the BLAS they ran on", check_blas, True),
]

_STEP_BY_TARGET: dict[Target, Step] = {s.target: s for s in STEPS}

# Which individual steps each group target expands to.
_FIGURE_TARGETS = [
    Target.frontier,
    Target.scaling,
    Target.rank_scaling,
    Target.worked_example,
    Target.degeneracy,
]
_CHECK_TARGETS = [
    Target.validate_exact,
    Target.validate_constraints,
    Target.real_frontier,
    Target.tie_degeneracy,
    Target.osqp,
    Target.validate_kkt,
    Target.validate_scaling,
    Target.validate_projection,
    Target.validate_factor,
    Target.validate_conditioning,
    Target.r_cla,
    Target.validate_incremental,
    Target.threads,
    Target.parallel_grid,
    Target.blas,
]


def _run_step(step: Step, out_dir: Path) -> bool:
    """Run one step, reporting (but not propagating) a missing dependency or failure.

    Returns True if it ran cleanly; False if an optional dependency is missing or
    the run raised (the error is reported, not propagated, so one step's failure
    does not abort the rest).
    """
    print(f"\n=== {step.target.value}  ->  {step.artefact} ===", flush=True)
    try:
        step.runner(out_dir)
    except ImportError as exc:
        print(f"[skip] {step.target.value}: missing optional dependency ({exc})", flush=True)
        return False
    except Exception as exc:  # noqa: BLE001 - one step must not abort the rest
        print(f"[FAILED] {step.target.value}: {type(exc).__name__}: {exc}", flush=True)
        return False
    print(f"[ok] {step.target.value}", flush=True)
    return True


def _expand(target: Target, quick: bool) -> list[Step]:
    """Resolve a target into the ordered list of individual steps to run."""
    if target is Target.all:
        wanted = [s.target for s in STEPS]
    elif target is Target.figures:
        wanted = list(_FIGURE_TARGETS)
    elif target is Target.checks:
        wanted = list(_CHECK_TARGETS)
    else:
        wanted = [target]
    # --quick only skips the slow sweeps for the group targets; an explicit slow
    # target always runs.
    is_group = target in (Target.all, Target.figures, Target.checks)
    steps = [_STEP_BY_TARGET[t] for t in wanted]
    if quick and is_group:
        steps = [s for s in steps if not s.slow]
    # Preserve the canonical STEPS order.
    return [s for s in STEPS if s in steps]


# ======================================================================================
# Typer CLI
# ======================================================================================
_DESCRIPTIONS: dict[Target, str] = {
    **{step.target: step.artefact for step in STEPS},
    Target.figures: "all five figures",
    Target.checks: "all fifteen numerical checks",
    Target.all: "every figure and every check",
}


def _choose_target() -> Target:
    """Interactively ask which artefact to build when none was given on the CLI."""
    options = list(Target)
    typer.echo("Which artefact to generate?\n")
    for i, t in enumerate(options, 1):
        typer.echo(f"  {i:>2}. {t.value:<26} {_DESCRIPTIONS[t]}")
    typer.echo("")
    while True:
        raw = typer.prompt("Enter a number or name", default="all").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        try:
            return Target(raw)
        except ValueError:
            typer.echo(f"  '{raw}' is not a valid choice; try again.")


def _run_target(target: Target, out_dir: Path, quick: bool) -> int:
    """Build the requested artefact(s); return a process exit code."""
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = _expand(target, quick)
    is_group = target in (Target.all, Target.figures, Target.checks)
    if quick and is_group:
        print("[--quick] skipping the four slow steps", flush=True)

    results = {step.target: _run_step(step, out_dir) for step in steps}

    if is_group:
        print("\n=== summary ===", flush=True)
        for tgt, ok in results.items():
            print(f"  {'ok  ' if ok else 'FAIL'}  {tgt.value}", flush=True)
    return 0 if all(results.values()) else 1


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    target: Annotated[
        Target | None,
        typer.Argument(help="Which artefact to build. Omit to choose interactively."),
    ] = None,
    out_dir: Annotated[
        Path,
        typer.Option(help="Directory to write artefacts into."),
    ] = DEFAULT_OUT_DIR,
    quick: Annotated[
        bool,
        typer.Option(help="For 'all'/'figures', skip the slow scaling sweeps (scaling, rank-scaling)."),
    ] = False,
) -> None:
    """Build a CLA-paper figure (or all of them, or the check suite)."""
    if target is None:
        target = _choose_target()
    raise typer.Exit(_run_target(target, out_dir, quick))


if __name__ == "__main__":
    app()
