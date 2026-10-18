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
``make_figures.py`` through a file: that script writes the exact paths of the opening
example and of the sCD14 data to ``figures/zero_sum_paths.npz``, and this one runs
c-lasso's ``Path-Alg`` on the same problems, measures how far its answers lie from the
exact path, and writes the numbers the note quotes to ``figures/classo.tex``. c-lasso
also imports pandas and matplotlib without declaring them, hence the two pins above.

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

FIGURES = Path(__file__).resolve().parent.parent / "figures"
POINTS = 100  # the grid c-lasso reports its path on
REPEATS = 5  # each run timed this often, and the fastest counts


def classo_path(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
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


def main() -> None:
    """Run c-lasso on every path make_figures.py saved, and write the note's numbers."""
    data = np.load(FIGURES / "zero_sum_paths.npz")
    gaps = {}
    for name in sorted({key.split(".")[0] for key in data.files}):
        x, y = data[f"{name}.x"], data[f"{name}.y"]
        lam, beta, seconds = classo_path(x, y)
        gaps[name] = gap(lam, beta, data[f"{name}.lam"], data[f"{name}.beta"])
        print(f"{name}: c-lasso Path-Alg, {len(lam)} grid points in {seconds * 1e3:.1f} ms, "
              f"largest gap to the exact path {gaps[name]:.1e}")
        # c-lasso is close to the path but not exact; flag it if that ever changes.
        assert 1e-9 < gaps[name] < 1e-2, gaps[name]
    worst = max(gaps.values())
    mantissa, exponent = f"{worst:.1e}".split("e")
    (FIGURES / "classo.tex").write_text(
        rf"\newcommand{{\clPoints}}{{{POINTS}}}" + "\n"
        + rf"\newcommand{{\clGap}}{{${mantissa}\times10^{{{int(exponent)}}}$}}" + "\n"
    )


if __name__ == "__main__":
    sys.exit(main())
