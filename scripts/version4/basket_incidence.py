"""Binary basket incidence and matrix-free off-diagonal pair operators."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator


def incidence(baskets, products: int) -> sparse.csr_matrix:
    rows = [np.unique(np.asarray(row, dtype=np.int32)) for row in baskets]
    pointer = np.r_[0, np.cumsum([len(row) for row in rows])]
    columns = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    return sparse.csr_matrix((np.ones(len(columns)), columns, pointer),
                             shape=(len(rows), products))


def off_diagonal_operator(x: sparse.csr_matrix) -> LinearOperator:
    """(X.T X - diag(X.T 1))/M, without materializing any product pairs."""
    if x.shape[0] == 0 or not np.all(x.data == 1):
        raise ValueError("a nonempty binary basket-incidence matrix is required")
    mass = np.asarray(x.sum(axis=0)).ravel()

    def multiply(v):
        diagonal = mass * v if v.ndim == 1 else mass[:, None] * v
        return (x.T @ (x @ v) - diagonal) / x.shape[0]

    return LinearOperator((x.shape[1], x.shape[1]), matvec=multiply,
                          rmatvec=multiply, matmat=multiply, dtype=np.float64)

