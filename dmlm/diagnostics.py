"""Local influence diagnostics for DMLM-S."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .estimators import WorkerData
from .linalg import safe_inverse
from .metrics import rank_auc


def observation_influence(
    workers: list[WorkerData],
    B: np.ndarray,
    *,
    hessian_x: np.ndarray | None = None,
    ridge: float = 1e-8,
) -> pd.DataFrame:
    if hessian_x is None:
        total_n = sum(w.n for w in workers)
        hessian_x = sum(w.X.T @ w.X for w in workers) / total_n
    H_inv = safe_inverse(hessian_x, ridge=ridge)

    rows = []
    for worker in workers:
        res = worker.Y - worker.X @ B
        for local_i in range(worker.n):
            delta = np.outer(worker.X[local_i], res[local_i])
            adjusted = H_inv @ delta
            rows.append(
                {
                    "global_index": int(worker.indices[local_i]),
                    "worker": worker.worker_id,
                    "influence": float(2.0 * np.sum(adjusted * adjusted)),
                }
            )
    return pd.DataFrame(rows).sort_values("influence", ascending=False).reset_index(drop=True)


def worker_influence(
    workers: list[WorkerData],
    B: np.ndarray,
    *,
    hessian_x: np.ndarray | None = None,
    ridge: float = 1e-8,
) -> pd.DataFrame:
    if hessian_x is None:
        total_n = sum(w.n for w in workers)
        hessian_x = sum(w.X.T @ w.X for w in workers) / total_n
    H_inv = safe_inverse(hessian_x, ridge=ridge)

    rows = []
    for worker in workers:
        res = worker.Y - worker.X @ B
        block_delta = worker.X.T @ res / np.sqrt(max(worker.n, 1))
        adjusted = H_inv @ block_delta
        rows.append(
            {
                "worker": worker.worker_id,
                "influence": float(2.0 * np.sum(adjusted * adjusted)),
                "n": worker.n,
            }
        )
    return pd.DataFrame(rows).sort_values("influence", ascending=False).reset_index(drop=True)


def contaminate_response(
    workers: list[WorkerData],
    contaminated_indices: set[int],
    shift: float,
) -> list[WorkerData]:
    out = []
    for worker in workers:
        Y = worker.Y.copy()
        mask = np.array([idx in contaminated_indices for idx in worker.indices])
        Y[mask] += shift
        out.append(WorkerData(X=worker.X.copy(), Y=Y, indices=worker.indices.copy(), worker_id=worker.worker_id))
    return out


def contaminate_leverage(
    workers: list[WorkerData],
    contaminated_indices: set[int],
    shift: float,
    direction: np.ndarray,
) -> list[WorkerData]:
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    out = []
    for worker in workers:
        X = worker.X.copy()
        mask = np.array([idx in contaminated_indices for idx in worker.indices])
        X[mask] += shift * direction
        out.append(WorkerData(X=X, Y=worker.Y.copy(), indices=worker.indices.copy(), worker_id=worker.worker_id))
    return out


def contaminate_worker_response(
    workers: list[WorkerData],
    worker_id: int,
    proportion: float,
    shift: float,
    rng: np.random.Generator,
) -> tuple[list[WorkerData], set[int]]:
    out = []
    contaminated: set[int] = set()
    for worker in workers:
        Y = worker.Y.copy()
        if worker.worker_id == worker_id:
            count = max(1, int(round(worker.n * proportion)))
            pos = rng.choice(worker.n, size=count, replace=False)
            Y[pos] += shift
            contaminated.update(int(i) for i in worker.indices[pos])
        out.append(WorkerData(X=worker.X.copy(), Y=Y, indices=worker.indices.copy(), worker_id=worker.worker_id))
    return out, contaminated


def detection_summary(
    obs_df: pd.DataFrame,
    worker_df: pd.DataFrame,
    contaminated_indices: set[int],
    contaminated_worker: int | None = None,
    *,
    q_values: tuple[int, ...] = (5, 10),
) -> dict:
    labels = obs_df["global_index"].map(lambda x: int(x) in contaminated_indices).to_numpy(dtype=bool)
    row = {"AUC": rank_auc(obs_df["influence"].to_numpy(), labels)}
    for q in q_values:
        top = set(int(i) for i in obs_df.head(q)["global_index"])
        row[f"Hit@{q}"] = len(top & contaminated_indices) / max(len(contaminated_indices), 1)
    if contaminated_worker is not None:
        ranks = {int(w): rank + 1 for rank, w in enumerate(worker_df["worker"])}
        row["Rank_contaminated_worker"] = ranks.get(int(contaminated_worker), np.nan)
    return row

