"""Inference utilities for selected linear functionals of B."""

from __future__ import annotations

import numpy as np

from .estimators import WorkerData
from .linalg import safe_inverse


def covariance_vec_B(workers: list[WorkerData], Sigma: np.ndarray, *, ridge: float = 1e-8) -> np.ndarray:
    X = np.vstack([w.X for w in workers])
    xtx_inv = safe_inverse(X.T @ X, ridge=ridge)
    return np.kron(Sigma, xtx_inv)


def functional_value(B: np.ndarray, A: np.ndarray) -> float:
    return float(np.sum(A * B))


def functional_se(cov_vec_B: np.ndarray, A: np.ndarray) -> float:
    a = A.reshape(-1, order="F")
    var = float(a @ cov_vec_B @ a)
    return float(np.sqrt(max(var, 0.0)))


def default_functionals(p: int, m: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    A1 = np.zeros((p, m))
    A1[0, 0] = 1.0

    A2 = np.zeros((p, m))
    A2[0, :] = 1.0 / m

    A3 = np.zeros((p, m))
    rows = rng.choice(p, size=min(3, p), replace=False)
    cols = rng.choice(m, size=min(2, m), replace=False)
    for r in rows:
        for c in cols:
            A3[r, c] = rng.choice([-1.0, 1.0]) / max(len(rows) * len(cols), 1)

    return {
        "psi1_B11": A1,
        "psi2_row1_avg": A2,
        "psi3_sparse_trace": A3,
    }

