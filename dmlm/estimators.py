"""Estimators for distributed multivariate linear models."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .linalg import make_pos_def, residual_covariance, safe_inverse, safe_solve, symmetric_size


@dataclass
class WorkerData:
    X: np.ndarray
    Y: np.ndarray
    indices: np.ndarray
    worker_id: int

    @property
    def n(self) -> int:
        return self.X.shape[0]


@dataclass
class EstimationResult:
    name: str
    B: np.ndarray
    Sigma: np.ndarray
    time_seconds: float = 0.0
    communication_scalars: int = 0
    extra: dict = field(default_factory=dict)


def stack_workers(workers: list[WorkerData]) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([w.X for w in workers])
    Y = np.vstack([w.Y for w in workers])
    return X, Y


def fit_full_mlm(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    name: str = "GMLM",
    ridge: float = 1e-8,
) -> EstimationResult:
    start = perf_counter()
    n = X.shape[0]
    xtx = X.T @ X / n
    xty = X.T @ Y / n
    B = safe_solve(xtx, xty, ridge=ridge)
    Sigma = residual_covariance(X, Y, B)
    return EstimationResult(name=name, B=B, Sigma=Sigma, time_seconds=perf_counter() - start)


def fit_workers_full(
    workers: list[WorkerData],
    *,
    name: str = "GMLM",
    ridge: float = 1e-8,
) -> EstimationResult:
    X, Y = stack_workers(workers)
    return fit_full_mlm(X, Y, name=name, ridge=ridge)


def fit_local_average(
    workers: list[WorkerData],
    *,
    ridge: float = 1e-8,
    weighted: bool = False,
) -> EstimationResult:
    start = perf_counter()
    local_results = [fit_full_mlm(w.X, w.Y, name=f"local-{w.worker_id}", ridge=ridge) for w in workers]
    local_serial_time = sum(result.time_seconds for result in local_results)
    local_parallel_time = max((result.time_seconds for result in local_results), default=0.0)
    coordinator_start = perf_counter()
    if weighted:
        weights = np.array([w.n for w in workers], dtype=float)
        weights /= weights.sum()
    else:
        weights = np.full(len(workers), 1.0 / len(workers))
    B = sum(weight * result.B for weight, result in zip(weights, local_results))
    Sigma = make_pos_def(sum(weight * result.Sigma for weight, result in zip(weights, local_results)))
    p, m = B.shape
    comm = len(workers) * (p * m + symmetric_size(m))
    coordinator_time = perf_counter() - coordinator_start
    serial_time = perf_counter() - start
    loop_overhead_time = max(0.0, serial_time - local_serial_time - coordinator_time)
    return EstimationResult(
        name="AMLM",
        B=B,
        Sigma=Sigma,
        time_seconds=coordinator_time + local_parallel_time,
        communication_scalars=comm,
        extra={
            "weighted": weighted,
            "serial_time_seconds": serial_time,
            "local_serial_time_seconds": local_serial_time,
            "local_parallel_time_seconds": local_parallel_time,
            "coordinator_time_seconds": coordinator_time,
            "single_machine_loop_overhead_seconds": loop_overhead_time,
        },
    )


def fit_first_worker(workers: list[WorkerData], *, ridge: float = 1e-8) -> EstimationResult:
    result = fit_full_mlm(workers[0].X, workers[0].Y, name="MLM-F", ridge=ridge)
    p, m = result.B.shape
    result.communication_scalars = workers[0].n * (p + m)
    return result


def _pilot_counts(
    workers: list[WorkerData],
    pilot_proportion: float | None,
    pilot_size: int | None,
    min_per_worker: int,
) -> list[int]:
    total_n = sum(w.n for w in workers)
    if pilot_size is None:
        if pilot_proportion is None:
            raise ValueError("Either pilot_proportion or pilot_size must be provided.")
        pilot_size = max(1, int(round(total_n * pilot_proportion)))
    raw = np.array([pilot_size * w.n / total_n for w in workers])
    counts = np.floor(raw).astype(int)
    if min_per_worker > 0 and pilot_size >= len(workers):
        counts = np.maximum(counts, min_per_worker)
    while counts.sum() > pilot_size:
        candidates = np.where(counts > min_per_worker)[0]
        if len(candidates) == 0:
            break
        j = candidates[np.argmax(counts[candidates])]
        counts[j] -= 1
    remainders = raw - np.floor(raw)
    for j in np.argsort(-remainders):
        if counts.sum() >= pilot_size:
            break
        if counts[j] < workers[j].n:
            counts[j] += 1
    return [int(min(c, workers[i].n)) for i, c in enumerate(counts)]


def sample_pilot(
    workers: list[WorkerData],
    *,
    rng: np.random.Generator,
    pilot_proportion: float | None = 0.01,
    pilot_size: int | None = None,
    min_per_worker: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = _pilot_counts(workers, pilot_proportion, pilot_size, min_per_worker)
    X_parts: list[np.ndarray] = []
    Y_parts: list[np.ndarray] = []
    idx_parts: list[np.ndarray] = []
    for worker, count in zip(workers, counts):
        if count <= 0:
            continue
        local_pos = rng.choice(worker.n, size=count, replace=False)
        X_parts.append(worker.X[local_pos])
        Y_parts.append(worker.Y[local_pos])
        idx_parts.append(worker.indices[local_pos])
    if not X_parts:
        raise ValueError("Pilot sample is empty.")
    return np.vstack(X_parts), np.vstack(Y_parts), np.concatenate(idx_parts)


def fit_pilot_only(
    workers: list[WorkerData],
    *,
    rng: np.random.Generator,
    pilot_proportion: float | None = 0.01,
    pilot_size: int | None = None,
    ridge: float = 1e-8,
) -> EstimationResult:
    start = perf_counter()
    Xp, Yp, pilot_indices = sample_pilot(
        workers,
        rng=rng,
        pilot_proportion=pilot_proportion,
        pilot_size=pilot_size,
    )
    result = fit_full_mlm(Xp, Yp, name="MLM-S", ridge=ridge)
    p, m = result.B.shape
    result.time_seconds = perf_counter() - start
    result.communication_scalars = Xp.shape[0] * (p + m)
    result.extra = {
        "pilot_indices": pilot_indices,
        "pilot_size": Xp.shape[0],
        "pilot_xtx": Xp.T @ Xp,
    }
    return result


def global_score_B(workers: list[WorkerData], B: np.ndarray) -> np.ndarray:
    score, _, _ = _timed_global_score_B(workers, B)
    return score


def _timed_global_score_B(workers: list[WorkerData], B: np.ndarray) -> tuple[np.ndarray, float, float]:
    total_n = sum(w.n for w in workers)
    score = np.zeros_like(B)
    worker_times = []
    for worker in workers:
        start = perf_counter()
        score += worker.X.T @ (worker.Y - worker.X @ B)
        worker_times.append(perf_counter() - start)
    return score / total_n, float(sum(worker_times)), float(max(worker_times, default=0.0))


def global_residual_covariance(workers: list[WorkerData], B: np.ndarray) -> np.ndarray:
    Sigma, _, _ = _timed_global_residual_covariance(workers, B)
    return Sigma


def _timed_global_residual_covariance(workers: list[WorkerData], B: np.ndarray) -> tuple[np.ndarray, float, float]:
    total_n = sum(w.n for w in workers)
    m = workers[0].Y.shape[1]
    moment = np.zeros((m, m))
    worker_times = []
    for worker in workers:
        start = perf_counter()
        res = worker.Y - worker.X @ B
        moment += res.T @ res
        worker_times.append(perf_counter() - start)
    return make_pos_def(moment / total_n), float(sum(worker_times)), float(max(worker_times, default=0.0))


def global_squared_loss(workers: list[WorkerData], B: np.ndarray) -> float:
    loss, _, _ = _timed_global_squared_loss(workers, B)
    return loss


def _timed_global_squared_loss(workers: list[WorkerData], B: np.ndarray) -> tuple[float, float, float]:
    total_n = sum(w.n for w in workers)
    total = 0.0
    worker_times = []
    for worker in workers:
        start = perf_counter()
        res = worker.Y - worker.X @ B
        total += float(np.sum(res * res))
        worker_times.append(perf_counter() - start)
    return 0.5 * total / total_n, float(sum(worker_times)), float(max(worker_times, default=0.0))


def fit_dmlm_s(
    workers: list[WorkerData],
    *,
    rng: np.random.Generator,
    pilot_proportion: float | None = 0.01,
    pilot_size: int | None = None,
    rounds: int = 2,
    ridge: float = 1e-8,
    backtracking: bool = True,
    min_step: float = 1e-4,
) -> EstimationResult:
    start = perf_counter()
    worker_serial_time = 0.0
    worker_parallel_time = 0.0
    coordinator_time = 0.0
    coordinator_start = perf_counter()
    Xp, Yp, pilot_indices = sample_pilot(
        workers,
        rng=rng,
        pilot_proportion=pilot_proportion,
        pilot_size=pilot_size,
    )
    pilot = fit_full_mlm(Xp, Yp, name="MLM-S", ridge=ridge)
    B = pilot.B.copy()
    n_pilot = Xp.shape[0]
    H_pilot = Xp.T @ Xp / n_pilot
    H_pilot_inv = safe_inverse(H_pilot, ridge=ridge)
    coordinator_time += perf_counter() - coordinator_start

    p, m = B.shape
    comm = n_pilot * (p + m)
    per_round_comm = len(workers) * (p * m)
    final_covariance_comm = len(workers) * symmetric_size(m)
    if backtracking:
        initial_loss, serial_part, parallel_part = _timed_global_squared_loss(workers, B)
        worker_serial_time += serial_part
        worker_parallel_time += parallel_part
        objective_trace = [initial_loss]
    else:
        objective_trace = []
    step_trace = []

    for _ in range(rounds):
        score, serial_part, parallel_part = _timed_global_score_B(workers, B)
        worker_serial_time += serial_part
        worker_parallel_time += parallel_part
        coordinator_start = perf_counter()
        direction = H_pilot_inv @ score
        coordinator_time += perf_counter() - coordinator_start
        step = 1.0
        if backtracking:
            current_loss = objective_trace[-1]
            while step >= min_step:
                coordinator_start = perf_counter()
                candidate = B + step * direction
                coordinator_time += perf_counter() - coordinator_start
                candidate_loss, serial_part, parallel_part = _timed_global_squared_loss(workers, candidate)
                worker_serial_time += serial_part
                worker_parallel_time += parallel_part
                if candidate_loss <= current_loss:
                    break
                coordinator_start = perf_counter()
                step *= 0.5
                coordinator_time += perf_counter() - coordinator_start
            if step < min_step:
                candidate = B
                candidate_loss = current_loss
                step = 0.0
        else:
            coordinator_start = perf_counter()
            candidate = B + direction
            coordinator_time += perf_counter() - coordinator_start
            candidate_loss = np.nan
        B = candidate
        if backtracking:
            objective_trace.append(candidate_loss)
        step_trace.append(step)
        comm += per_round_comm

    Sigma, serial_part, parallel_part = _timed_global_residual_covariance(workers, B)
    worker_serial_time += serial_part
    worker_parallel_time += parallel_part
    comm += final_covariance_comm
    serial_time = perf_counter() - start
    single_machine_loop_overhead = max(0.0, serial_time - worker_serial_time - coordinator_time)
    return EstimationResult(
        name="DMLM-S",
        B=B,
        Sigma=Sigma,
        time_seconds=coordinator_time + worker_parallel_time,
        communication_scalars=comm,
        extra={
            "pilot_indices": pilot_indices,
            "pilot_size": n_pilot,
            "pilot_xtx": Xp.T @ Xp,
            "rounds": rounds,
            "pilot_B": pilot.B,
            "pilot_Sigma": pilot.Sigma,
            "pilot_hessian": H_pilot,
            "objective_trace": objective_trace,
            "step_trace": step_trace,
            "backtracking": backtracking,
            "serial_time_seconds": serial_time,
            "worker_serial_time_seconds": worker_serial_time,
            "worker_parallel_time_seconds": worker_parallel_time,
            "coordinator_time_seconds": coordinator_time,
            "single_machine_loop_overhead_seconds": single_machine_loop_overhead,
        },
    )
