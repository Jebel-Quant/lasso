# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cvxcla==2.7.1",
#     "numpy==2.4.6",
# ]
# ///
r"""Run the KKT listing of Appendix C exactly as printed, and check what the text says of it.

Figure 5 (``fig:kktcla`` in ``sections/appendix-bailey.tex``) is a complete CLA for box
bounds and one budget row. This script cuts the code out of the ``lstlisting`` block in
that figure, runs it, and checks the claims the appendix makes about it:

1. On the scaling problems of Section 7.2 (``_scale_make_problem`` in
   ``make_figures.py``, seed 7) it returns cvxcla's turning points: the same number,
   and paths ``w(lambda)`` within 1e-12. The counts must also be the ones the appendix
   quotes for cvxcla.
2. It agrees with cvxcla on a few other problems: long-only and shorting factor models,
   upper bounds that bind exactly (``1/4``, whose four caps sum to exactly 1, and
   ``1/5``, where rounding blurs that), so that several events fall at one lambda, and
   an ill-conditioned covariance (cond 1e10).
3. It refuses, by raising, infeasible bounds, a tied largest expected return and a
   numerically singular free block, rather than returning a wrong frontier.

cvxcla is the reference because its own test suite certifies every segment with the
KKT conditions; this script checks the listing, not cvxcla. It exits non-zero on the
first failed check, so CI fails if an edit to the listing breaks it.

Usage (from the repository root or from cla/):
    uv run cla/check_listing.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from cvxcla import CLA

HERE = Path(__file__).resolve().parent
APPENDIX = HERE / "sections" / "appendix-bailey.tex"
SIZES = [20, 40, 80, 160, 320, 640]
TOL = 1e-12


def listing() -> str:
    """Return the code of the lstlisting block in the figure labelled fig:kktcla."""
    figures = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", APPENDIX.read_text(), re.DOTALL)
    blocks = [f for f in figures if r"\label{fig:kktcla}" in f]
    if len(blocks) != 1:
        sys.exit("expected exactly one figure labelled fig:kktcla")
    (block,) = blocks
    match = re.search(r"\\begin\{lstlisting\}(?:\[[^\]]*\])?\n(.*?)\\end\{lstlisting\}", block, re.DOTALL)
    if match is None:
        sys.exit("no lstlisting block in the figure labelled fig:kktcla")
    return match.group(1)


def path_distance(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    """Exact sup-norm distance between two piecewise-linear paths w(lambda).

    Each path is linear in lambda between turning points and constant above its first
    finite one, so the distance is attained at a turning point of one of the two.
    """

    def at(weights: np.ndarray, lambdas: np.ndarray, grid: np.ndarray) -> np.ndarray:
        finite = np.isfinite(lambdas)
        order = np.argsort(lambdas[finite], kind="stable")
        lam, w = lambdas[finite][order], weights[finite][order]
        return np.stack([np.interp(grid, lam, w[:, j]) for j in range(w.shape[1])], axis=1)

    (wa, la), (wb, lb) = a, b
    grid = np.union1d(la[np.isfinite(la)], lb[np.isfinite(lb)])
    return max(float(np.abs(wa[0] - wb[0]).max()), float(np.abs(at(wa, la, grid) - at(wb, lb, grid)).max()))


def both(cla, mu: np.ndarray, cov: np.ndarray, lo: np.ndarray, up: np.ndarray) -> tuple[tuple, tuple]:
    """Trace one problem with the listing and with cvxcla; return both paths."""
    pts = cla(mu, cov, lo, up)
    mine = (np.array([w for _, w in pts]), np.array([lam for lam, _ in pts]))
    n = len(mu)
    ref = CLA(mean=mu, covariance=cov, lower_bounds=lo, upper_bounds=up, a=np.ones((1, n)), b=np.ones(1))
    theirs = (np.array([tp.weights for tp in ref.turning_points]), np.array([tp.lamb for tp in ref.turning_points]))
    return mine, theirs


def factor_cov(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """The K-factor covariance of ``_scale_make_problem`` (same draws, same order)."""
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    return np.diag(d) + (u * delta) @ u.T


def scaling_problem(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The seed-7 scaling problem of size n as (mu, cov, lo, up), as in ``make_figures.py``."""
    rng = np.random.default_rng(7)
    cov = factor_cov(rng, n, 10)
    return rng.uniform(0.0, 1.0, n), cov, np.zeros(n), np.ones(n)


def main() -> None:
    """Run every check; exit non-zero on the first failure."""
    code = listing()
    namespace: dict[str, object] = {}
    exec(compile(code, f"{APPENDIX.name}:fig:kktcla", "exec"), namespace)  # noqa: S102  # nosec B102
    cla = namespace["cla"]
    print(f"listing: {len(code.splitlines())} lines from {APPENDIX.relative_to(HERE.parent)}")
    failures: list[str] = []

    def check(ok: bool, label: str, detail: str) -> None:
        print(f"  {'ok  ' if ok else 'FAIL'} {label:34s} {detail}")
        if not ok:
            failures.append(label)

    print("1. scaling problems of Section 7.2 (seed 7)")
    counts = []
    for n in SIZES:
        try:
            mine, theirs = both(cla, *scaling_problem(n))
        except Exception as exc:  # noqa: BLE001 -- any exception on a valid problem is a failure
            check(False, f"n={n}", f"raised {type(exc).__name__}: {exc}")
            continue
        dist = path_distance(mine, theirs)
        same = len(mine[0]) == len(theirs[0]) and dist <= TOL
        check(same, f"n={n}", f"points {len(mine[0])} vs cvxcla {len(theirs[0])}, path distance {dist:.1e}")
        counts.append(len(theirs[0]))
    quoted = re.search(r"against \$([0-9, ]+)\$\)", APPENDIX.read_text())
    check(
        quoted is not None and [int(x) for x in quoted.group(1).split(",")] == counts,
        "counts quoted in the appendix",
        f"{quoted.group(1) if quoted else 'not found'} vs {counts}",
    )

    print("2. other problems")
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    ill = (q * np.geomspace(1.0, 1e-10, 20)) @ q.T
    others = [
        ("long-only n=50", factor_cov(rng, 50, 5), 50, 0.0, 1.0, TOL),
        ("shorting n=50", factor_cov(rng, 50, 5), 50, -1.0, 1.0, TOL),
        ("caps 1/4 bind exactly, n=20", factor_cov(rng, 20, 5), 20, 0.0, 1 / 4, TOL),
        ("caps 1/5 bind, n=20", factor_cov(rng, 20, 5), 20, 0.0, 1 / 5, TOL),
        ("cond(S) = 1e10, n=20", 0.5 * (ill + ill.T), 20, 0.0, 1.0, 1e-8),
    ]
    for label, cov, n, lo, up, tol in others:
        try:
            mine, theirs = both(cla, rng.uniform(0.0, 1.0, n), cov, np.full(n, lo), np.full(n, up))
        except Exception as exc:  # noqa: BLE001 -- any exception on a valid problem is a failure
            check(False, label, f"raised {type(exc).__name__}: {exc}")
            continue
        dist = path_distance(mine, theirs)
        check(dist <= tol, label, f"path distance {dist:.1e}")

    print("3. refusals")
    n = 20
    cov, mu = factor_cov(rng, n, 5), rng.uniform(0.0, 1.0, n)
    x = rng.standard_normal((8, n))
    tied = mu.copy()
    tied[np.argsort(mu)[-2]] = mu.max()
    for label, args, error in (
        ("infeasible bounds", (mu, cov, np.zeros(n), np.full(n, 0.9 / n)), ValueError),
        ("tied largest return", (tied, cov, np.zeros(n), np.ones(n)), ValueError),
        (
            "singular free block (T=8 < n)",
            (mu, np.cov(x, rowvar=False), np.zeros(n), np.ones(n)),
            np.linalg.LinAlgError,
        ),
    ):
        try:
            cla(*args)
        except error as exc:
            check(True, label, f"raised {type(exc).__name__}: {exc}")
        else:
            check(False, label, f"returned a frontier instead of raising {error.__name__}")

    if failures:
        sys.exit(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    print("all checks passed")


if __name__ == "__main__":
    main()
