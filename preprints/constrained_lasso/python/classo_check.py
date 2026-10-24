# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "c-lasso==1.0.11",
#     "matplotlib==3.11.2",
#     "numpy==1.26.4",
#     "pandas==2.3.3",
# ]
# ///
# run: after figures
# The check lives in this script, so assert is the idiom here too.
# ruff: noqa: S101
"""Compare c-lasso's path algorithm with the exact zero-sum paths of the note.

c-lasso (Simpson, Combettes and Mueller, 2021) is the Python package built for the
LASSO under ``C beta = 0``. Its last release, 1.0.11, uses ``np.infty``, which NumPy 2
removed, so it runs here in its own environment, pinned to NumPy 1.26, and talks to
``make_figures.py`` through a file: that script writes the exact paths of the Zero-sum
problem, of the sCD14 data and of Wide, zero-sum (p > n, one per seed) to
``figures/zero_sum_paths.npz``, and this one runs
c-lasso's ``Path-Alg`` on the same problems, measures how far its answers lie from the
exact path, and writes the numbers the note quotes to ``figures/classo.tex``. c-lasso
also imports pandas and matplotlib without declaring them, hence the two pins above.

Path-Alg computes the breakpoints of a slightly regularised problem: it adds
``eps_L2 * I``, with ``eps_L2 = 1e-3``, to ``X^T X``, a default of ``parameters_for_update``
that no public setting reaches. That ridge term is all of its distance to the exact path,
so the script runs it twice, as shipped and with the ridge lowered to ``RIDGE`` by hand.
With n >= p the ridge moves the path only slightly. With p > n it decides where the path
ends: the ridged path heads for a different interpolating fit. There c-lasso's grid is
carried down to ``WIDE_LAMIN`` times its largest lambda, so that it reaches the end.

The line ``# run: after figures`` above makes ``make figures`` run this script after
``make_figures.py``, here and in CI.

    uv run python/classo_check.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from classo import classo_problem
from classo import path_alg

FIGURES = Path(__file__).resolve().parent.parent / "figures"
POINTS = 100  # the grid c-lasso reports its path on
REPEATS = 5  # each run timed this often, and the fastest counts
RIDGE = 1e-12  # eps_L2 for the second run, in place of the shipped 1e-3
WIDE_LAMIN = 1e-9  # c-lasso's smallest lambda, relative, for p > n; the shipped 1e-3 stops short of the end


def classo_path(x: np.ndarray, y: np.ndarray, lamin: float = 1e-3) -> tuple[np.ndarray, np.ndarray, float]:
    """c-lasso's Path-Alg on the zero-sum LASSO: lambda in the note's scaling, beta, seconds.

    c-lasso's R1 problem is ``||y - X b||^2 + lam ||b||_1`` under ``C b = 0``, without
    the note's half in front, so its lambda is twice the note's.
    """
    best, out = np.inf, None
    for _ in range(REPEATS):
        problem = classo_problem(x, y, np.ones((1, x.shape[1])))
        problem.formulation.concomitant = False
        problem.formulation.huber = False
        problem.formulation.intercept = False
        selection = problem.model_selection
        selection.PATH, selection.CV, selection.StabSel, selection.LAMfixed = True, False, False, False
        selection.PATHparameters.numerical_method = "Path-Alg"
        selection.PATHparameters.Nlam = POINTS
        selection.PATHparameters.lamin = lamin
        start = time.perf_counter()
        problem.solve()
        best = min(best, time.perf_counter() - start)
        out = problem.solution.PATH
    return np.asarray(out.LAMBDAS) / 2, np.asarray(out.BETAS), best


def gap(lam: np.ndarray, beta: np.ndarray, knots: np.ndarray, exact: np.ndarray) -> float:
    """Largest distance from c-lasso's answers to the exact path, relative to its largest coefficient.

    The exact path is affine between its knots, so at any lambda it is the interpolation
    of the two knots around it; above the first knot it is zero.
    """
    order = np.argsort(knots)
    ref = np.array([[np.interp(v, knots[order], exact[order, j], right=0.0) for j in range(exact.shape[1])]
                    for v in lam])
    return float(np.abs(beta - ref).max() / np.abs(exact).max())


def with_ridge(eps: float) -> None:
    """Set the ridge term Path-Alg adds to X^T X, which c-lasso fixes as a default argument."""
    init = path_alg.parameters_for_update.__init__
    init.__defaults__ = (eps, *init.__defaults__[1:])


def sci(x: float) -> str:
    """A number as the note prints it, ``$6.2\\times10^{-5}$``."""
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def main() -> None:
    """Run c-lasso on every path make_figures.py saved, as shipped and without its ridge."""
    data = np.load(FIGURES / "zero_sum_paths.npz")
    shipped, unridged = {}, {}
    for name in sorted({key.split(".")[0] for key in data.files}):
        x, y, knots, exact = data[f"{name}.x"], data[f"{name}.y"], data[f"{name}.lam"], data[f"{name}.beta"]
        lamin = WIDE_LAMIN if name.startswith("wide") else 1e-3
        lam, beta, seconds = classo_path(x, y, lamin)
        shipped[name] = gap(lam, beta, knots, exact)
        with_ridge(RIDGE)
        lam, beta, _ = classo_path(x, y, lamin)
        unridged[name] = gap(lam, beta, knots, exact)
        with_ridge(1e-3)
        print(f"{name}: c-lasso Path-Alg, {len(lam)} grid points in {seconds * 1e3:.1f} ms, largest gap to "
              f"the exact path {shipped[name]:.1e}, and {unridged[name]:.1e} with the ridge at {RIDGE:.0e}")
        # As shipped c-lasso is close to the path but off it by its ridge, and with p > n
        # far off it; without the ridge it is on it. Flag it if any of that ever changes.
        if name.startswith("wide"):
            assert shipped[name] > 1e-1, shipped[name]
        else:
            assert 1e-9 < shipped[name] < 1e-2, shipped[name]
        assert unridged[name] < 1e-10, unridged[name]
    square = [name for name in shipped if not name.startswith("wide")]
    wide = [name for name in shipped if name.startswith("wide")]
    (FIGURES / "classo.tex").write_text(
        rf"\newcommand{{\clPoints}}{{{POINTS}}}" + "\n"
        + rf"\newcommand{{\clGap}}{{{sci(max(shipped[n] for n in square))}}}" + "\n"
        + rf"\newcommand{{\clGapUnridged}}{{{sci(max(unridged[n] for n in square))}}}" + "\n"
        + rf"\newcommand{{\clWideGapLo}}{{{min(shipped[n] for n in wide):.2f}}}" + "\n"
        + rf"\newcommand{{\clWideGapHi}}{{{max(shipped[n] for n in wide):.2f}}}" + "\n"
        + rf"\newcommand{{\clWideGapUnridged}}{{{sci(max(unridged[n] for n in wide))}}}" + "\n"
        + rf"\newcommand{{\clWideLamin}}{{{sci(WIDE_LAMIN)}}}" + "\n"
    )


if __name__ == "__main__":
    sys.exit(main())
