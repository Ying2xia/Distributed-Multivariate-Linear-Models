"""Small linear-algebra helpers used by the DMLM-S implementation."""

from __future__ import annotations

import numpy as np


def ar1_cov(dim: int, rho: float) -> np.ndarray:
    idx = np.arange(dim)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def make_pos_def(matrix: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    matrix = symmetrize(matrix)
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.maximum(vals, min_eig)
    return symmetrize((vecs * vals) @ vecs.T)


def safe_solve(
    lhs: np.ndarray,
    rhs: np.ndarray,
    ridge: float = 1e-8,
    max_tries: int = 8,
) -> np.ndarray:
    lhs = symmetrize(lhs) if lhs.ndim == 2 and lhs.shape[0] == lhs.shape[1] else lhs
    eye = np.eye(lhs.shape[0], dtype=lhs.dtype)
    penalty = 0.0
    last_error: Exception | None = None
    for _ in range(max_tries):
        try:
            return np.linalg.solve(lhs + penalty * eye, rhs)
        except np.linalg.LinAlgError as err:
            last_error = err
            penalty = ridge if penalty == 0 else penalty * 10.0
    raise np.linalg.LinAlgError(f"safe_solve failed after ridge escalation: {last_error}")


def safe_inverse(lhs: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    return safe_solve(lhs, np.eye(lhs.shape[0]), ridge=ridge)


def residuals(X: np.ndarray, Y: np.ndarray, B: np.ndarray) -> np.ndarray:
    return Y - X @ B


def residual_covariance(X: np.ndarray, Y: np.ndarray, B: np.ndarray) -> np.ndarray:
    res = residuals(X, Y, B)
    return make_pos_def(res.T @ res / max(len(Y), 1))


def vech(matrix: np.ndarray) -> np.ndarray:
    rows, cols = np.triu_indices(matrix.shape[0])
    return matrix[rows, cols]


def symmetric_size(dim: int) -> int:
    return dim * (dim + 1) // 2


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale == 0] = 1.0
    X_train_std = (X_train - mean) / scale
    X_test_std = None if X_test is None else (X_test - mean) / scale
    return X_train_std, X_test_std, mean, scale

