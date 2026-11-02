# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cvxcla==2.7.1",
#     "numpy==2.4.6",
#     "scipy==1.17.1",
# ]
# ///
"""The block-diagonal backend of Section 5.2, complete and runnable.

The paper prints the two members that carry the structure, ``solve_free`` and
``rcond_free``; this is the whole class, from the package documentation
(docs/custom-backend.md at tag v2.7.1). On four groups of five assets it traces the
same 19 turning points as the dense matrix.

    uv run custom_backend.py
"""

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

from cvxcla import CLA, QuadraticForm


class BlockDiagonalCovariance(QuadraticForm):
    """Sigma = blockdiag(S_1, ..., S_m) with symmetric positive-definite blocks."""

    def __init__(self, blocks):
        self._blocks = [np.asarray(b, dtype=float) for b in blocks]
        sizes = [len(b) for b in self._blocks]
        self._starts = np.cumsum([0, *sizes])
        self._block_of = np.repeat(np.arange(len(sizes)), sizes)

    @property
    def n(self):
        return int(self._starts[-1])

    def matvec(self, x):
        return np.concatenate([b @ x[s:e] for b, s, e in zip(self._blocks, self._starts[:-1], self._starts[1:])])

    def block_matvec(self, rows, cols, v):
        x = np.zeros((self.n, *np.shape(v)[1:]))
        x[np.asarray(cols)] = v
        return self.matvec(x)[np.asarray(rows)]

    def _free_blocks(self, free):
        free = np.asarray(free)
        for k in np.unique(self._block_of[free]):
            pos = np.flatnonzero(self._block_of[free] == k)
            local = free[pos] - self._starts[k]
            yield pos, self._blocks[k][np.ix_(local, local)]

    def solve_free(self, free, rhs):
        out = np.empty_like(np.asarray(rhs, dtype=float))
        for pos, sub in self._free_blocks(free):
            out[pos] = cho_solve(cho_factor(sub), rhs[pos])
        return out

    def rcond_free(self, free):
        eig = np.concatenate([np.linalg.eigvalsh(sub) for _, sub in self._free_blocks(free)])
        return float(eig.min() / eig.max())


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    blocks = [m @ m.T / 5 + 0.2 * np.eye(5) for m in rng.standard_normal((4, 5, 5))]
    mean = rng.uniform(0.0, 1.0, 20)
    problem = dict(lower_bounds=np.zeros(20), upper_bounds=np.ones(20), a=np.ones((1, 20)), b=np.ones(1))

    custom = CLA(mean=mean, covariance=BlockDiagonalCovariance(blocks), **problem)
    dense = CLA(mean=mean, covariance=block_diag(*blocks), **problem)
    same = np.allclose([tp.weights for tp in custom.turning_points], [tp.weights for tp in dense.turning_points])
    print(f"{len(custom)} turning points; same as the dense matrix: {same}")
    if not same:
        raise SystemExit(1)
