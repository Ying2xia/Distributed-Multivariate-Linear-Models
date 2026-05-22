"""Evaluation metrics for DMLM-S experiments."""

from __future__ import annotations

import numpy as np


def frobenius_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth, ord="fro"))


def relative_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    denom = np.linalg.norm(reference, ord="fro")
    if denom == 0:
        return float(np.linalg.norm(estimate - reference, ord="fro"))
    return float(np.linalg.norm(estimate - reference, ord="fro") / denom)


def prediction_error(X: np.ndarray, Y: np.ndarray, B: np.ndarray) -> float:
    res = Y - X @ B
    return float(np.mean(res * res))


def summarize_estimator(
    result,
    *,
    B0: np.ndarray | None = None,
    Sigma0: np.ndarray | None = None,
    gmlm_B: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    Y_test: np.ndarray | None = None,
) -> dict:
    row = {
        "Method": result.name,
        "Time": result.time_seconds,
        "Comm": result.communication_scalars,
    }
    if B0 is not None:
        row["RMSE_B"] = frobenius_error(result.B, B0)
    if Sigma0 is not None:
        row["RMSE_Sigma"] = frobenius_error(result.Sigma, Sigma0)
    if gmlm_B is not None:
        row["RE_GMLM"] = relative_error(result.B, gmlm_B)
    if X_test is not None and Y_test is not None:
        row["PE"] = prediction_error(X_test, Y_test, result.B)
    return row


def aggregate_rows(rows: list[dict], group_cols: list[str]) -> "pd.DataFrame":
    import pandas as pd

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in group_cols]
    return df.groupby(group_cols, as_index=False)[metric_cols].mean(numeric_only=True)


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end

    pos_rank_sum = ranks[labels].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

