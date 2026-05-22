"""Simulation data generation and distributed storage strategies."""

from __future__ import annotations

import numpy as np

from .estimators import WorkerData
from .linalg import ar1_cov


def coefficient_matrix(
    p: int,
    m: int,
    structure: str,
    rng: np.random.Generator,
    *,
    s_B: int | None = None,
    r_B: int = 2,
) -> np.ndarray:
    structure = structure.upper()
    if structure == "S1":
        rows = np.arange(1, p + 1)[:, None]
        cols = np.arange(1, m + 1)[None, :]
        return 0.6 * ((-1.0) ** (rows + cols)) * np.exp(-np.abs(rows - cols) / m)
    if structure == "S2":
        if s_B is None:
            s_B = max(1, min(p, p // 4))
        B = np.zeros((p, m))
        rows = np.arange(1, s_B + 1)[:, None]
        cols = np.arange(1, m + 1)[None, :]
        B[:s_B] = ((-1.0) ** (rows + cols)) / (1.0 + rows / p)
        return B
    if structure == "S3":
        A = rng.normal(size=(p, r_B))
        C = rng.normal(size=(m, r_B))
        B = (A @ C.T) / np.sqrt(max(r_B, 1))
        n_sparse = max(1, int(round(0.05 * p * m)))
        flat_idx = rng.choice(p * m, size=n_sparse, replace=False)
        perturb = np.zeros(p * m)
        perturb[flat_idx] = rng.choice([-1.0, 1.0], size=n_sparse) * rng.uniform(0.3, 0.8, size=n_sparse)
        return B + perturb.reshape(p, m)
    raise ValueError(f"Unknown coefficient structure: {structure}")


def generate_mlm_arrays(
    n: int,
    p: int,
    m: int,
    B0: np.ndarray,
    Sigma0: np.ndarray,
    rng: np.random.Generator,
    *,
    rho_x: float = 0.5,
    mu_x: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    Omega = ar1_cov(p, rho_x)
    mean = np.zeros(p) if mu_x is None else mu_x
    X = rng.multivariate_normal(mean=mean, cov=Omega, size=n)
    E = rng.multivariate_normal(mean=np.zeros(m), cov=Sigma0, size=n)
    Y = X @ B0 + E
    return X, Y


def _split_sizes(n: int, k: int) -> list[int]:
    base = n // k
    rem = n % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def make_workers_from_arrays(
    X: np.ndarray,
    Y: np.ndarray,
    K: int,
    *,
    strategy: str,
    rng: np.random.Generator,
    B0: np.ndarray | None = None,
    sorted_score: str = "linear_signal",
) -> list[WorkerData]:
    n, p = X.shape
    strategy = strategy.upper()
    if strategy == "R":
        order = rng.permutation(n)
    elif strategy == "C":
        if sorted_score == "linear_signal" and B0 is not None:
            z = (X @ B0).sum(axis=1) / np.sqrt(B0.shape[1])
        elif sorted_score == "weighted_covariates":
            weights = np.zeros(p)
            base = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
            weights[: min(p, len(base))] = base[: min(p, len(base))]
            z = X @ weights / np.sqrt(max(np.sum(weights * weights), 1.0))
        else:
            z = X.sum(axis=1) / np.sqrt(p)
        order = np.argsort(z)
    else:
        raise ValueError("make_workers_from_arrays only supports strategies R and C.")

    sizes = _split_sizes(n, K)
    workers = []
    start = 0
    for worker_id, size in enumerate(sizes):
        idx = order[start : start + size]
        workers.append(WorkerData(X=X[idx], Y=Y[idx], indices=idx.copy(), worker_id=worker_id))
        start += size
    return workers


def generate_distributed_simulation(
    *,
    N: int,
    K: int,
    p: int,
    m: int,
    structure: str,
    strategy: str,
    rho_x: float,
    rho_y: float,
    rng: np.random.Generator,
    B0: np.ndarray | None = None,
    sorted_score: str = "linear_signal",
    heterogeneity_scale: float = 3.0,
    heterogeneity_rho_low: float = 0.05,
    heterogeneity_rho_high: float = 0.95,
) -> tuple[list[WorkerData], np.ndarray, np.ndarray]:
    Sigma0 = ar1_cov(m, rho_y)
    if B0 is None:
        B0 = coefficient_matrix(p, m, structure, rng)
    strategy = strategy.upper()

    if strategy in {"R", "C"}:
        X, Y = generate_mlm_arrays(N, p, m, B0, Sigma0, rng, rho_x=rho_x)
        return make_workers_from_arrays(
            X,
            Y,
            K,
            strategy=strategy,
            rng=rng,
            B0=B0,
            sorted_score=sorted_score,
        ), B0, Sigma0

    if strategy == "H":
        workers = []
        sizes = _split_sizes(N, K)
        start = 0
        for k, size in enumerate(sizes):
            if K == 1:
                rho_k = rho_x
            else:
                rho_k = heterogeneity_rho_low + (heterogeneity_rho_high - heterogeneity_rho_low) * k / (K - 1)
            centered_rank = (k - (K - 1) / 2) / max(K - 1, 1)
            mu_k = heterogeneity_scale * centered_rank * np.ones(p)
            Xk, Yk = generate_mlm_arrays(size, p, m, B0, Sigma0, rng, rho_x=rho_k, mu_x=mu_k)
            idx = np.arange(start, start + size)
            workers.append(WorkerData(X=Xk, Y=Yk, indices=idx, worker_id=k))
            start += size
        return workers, B0, Sigma0

    raise ValueError(f"Unknown storage strategy: {strategy}")


def generate_test_set(
    *,
    n_test: int,
    p: int,
    m: int,
    B0: np.ndarray,
    Sigma0: np.ndarray,
    rho_x: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    return generate_mlm_arrays(n_test, p, m, B0, Sigma0, rng, rho_x=rho_x)
