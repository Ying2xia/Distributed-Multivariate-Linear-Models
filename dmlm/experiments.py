"""Simulation experiments matching the DMLM-S paper."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    tqdm = None

from .data import (
    coefficient_matrix,
    generate_distributed_simulation,
    generate_mlm_arrays,
    generate_test_set,
)
from .diagnostics import (
    contaminate_leverage,
    contaminate_response,
    contaminate_worker_response,
    detection_summary,
    observation_influence,
    worker_influence,
)
from .estimators import (
    EstimationResult,
    WorkerData,
    fit_dmlm_s,
    fit_first_worker,
    fit_local_average,
    fit_pilot_only,
    fit_workers_full,
)
from .inference import default_functionals, functional_se, functional_value
from .linalg import ar1_cov, safe_inverse
from .metrics import prediction_error, relative_error


@dataclass(frozen=True)
class SimulationConfig:
    N: int = 50_000
    K: int = 200
    p: int = 50
    m: int = 10
    rho_x: float = 0.7
    rho_y: float = 0.6
    pilot_proportion: float = 0.02
    replications: int = 200
    dmlm_rounds: int = 3
    n_test: int = 5_000
    seed: int = 20260509
    structures: tuple[str, ...] = ("S1", "S2", "S3")
    strategies: tuple[str, ...] = ("R", "C", "H")
    show_progress: bool = True
    sorted_score: str = "linear_signal"
    heterogeneity_scale: float = 3.0
    heterogeneity_rho_low: float = 0.05
    heterogeneity_rho_high: float = 0.95


class _NullProgress:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix_str(self, s: str) -> None:
        return None


def _progress(total: int, desc: str, config: SimulationConfig):
    if not config.show_progress or tqdm is None:
        return _NullProgress()
    return tqdm(total=total, desc=desc, unit="run", dynamic_ncols=True)


def quick_config(replications: int = 5, seed: int = 20260509) -> SimulationConfig:
    return SimulationConfig(
        N=1_500,
        K=8,
        p=8,
        m=3,
        rho_x=0.5,
        rho_y=0.6,
        pilot_proportion=0.08,
        replications=replications,
        dmlm_rounds=3,
        n_test=600,
        seed=seed,
        structures=("S1", "S2", "S3"),
        strategies=("R", "C", "H"),
    )


def paper_config(replications: int = 200, seed: int = 20260509) -> SimulationConfig:
    return replace(SimulationConfig(), replications=replications, seed=seed)


def _sim_kwargs(config: SimulationConfig) -> dict:
    return {
        "sorted_score": config.sorted_score,
        "heterogeneity_scale": config.heterogeneity_scale,
        "heterogeneity_rho_low": config.heterogeneity_rho_low,
        "heterogeneity_rho_high": config.heterogeneity_rho_high,
    }


def _method_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.int32).max))


def _fit_methods(
    workers: list[WorkerData],
    *,
    pilot_proportion: float,
    rounds: int,
    rng: np.random.Generator,
    include_gmlm: bool = True,
    include_amlm: bool = True,
    include_pilot: bool = True,
    include_first: bool = True,
    include_dmlm: bool = True,
) -> list[EstimationResult]:
    results: list[EstimationResult] = []
    shared_pilot_seed = _method_seed(rng)
    if include_gmlm:
        results.append(fit_workers_full(workers, name="GMLM"))
    if include_amlm:
        results.append(fit_local_average(workers))
    if include_pilot:
        results.append(
            fit_pilot_only(
                workers,
                rng=np.random.default_rng(shared_pilot_seed),
                pilot_proportion=pilot_proportion,
            )
        )
    if include_first:
        results.append(fit_first_worker(workers))
    if include_dmlm:
        results.append(
            fit_dmlm_s(
                workers,
                rng=np.random.default_rng(shared_pilot_seed),
                pilot_proportion=pilot_proportion,
                rounds=rounds,
            )
        )
    return results


def _result_row(
    result: EstimationResult,
    *,
    B0: np.ndarray,
    Sigma0: np.ndarray,
    gmlm_B: np.ndarray | None,
    X_test: np.ndarray | None,
    Y_test: np.ndarray | None,
    extra: dict,
) -> dict:
    row = dict(extra)
    row.update(
        {
            "Method": result.name,
            "SE_B": float(np.linalg.norm(result.B - B0, ord="fro") ** 2),
            "SE_Sigma": float(np.linalg.norm(result.Sigma - Sigma0, ord="fro") ** 2),
            "Time": result.time_seconds,
            "Comm": result.communication_scalars,
        }
    )
    if gmlm_B is not None:
        row["RE_GMLM"] = relative_error(result.B, gmlm_B)
    if X_test is not None and Y_test is not None:
        row["PE"] = prediction_error(X_test, Y_test, result.B)
    return row


def summarize_estimation_rows(rows: list[dict], group_cols: list[str]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    grouped = raw.groupby(group_cols, as_index=False)
    aggregations = {
        "RMSE_B": ("SE_B", lambda x: float(np.sqrt(np.mean(x)))),
        "RMSE_Sigma": ("SE_Sigma", lambda x: float(np.sqrt(np.mean(x)))),
        "Time": ("Time", "mean"),
        "Comm": ("Comm", "mean"),
    }
    if "RE_GMLM" in raw.columns:
        aggregations["RE_GMLM"] = ("RE_GMLM", "mean")
    if "PE" in raw.columns:
        aggregations["PE"] = ("PE", "mean")
    out = grouped.agg(**aggregations)
    return out


def experiment1(config: SimulationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed + 101)
    rows = []
    total = len(config.structures) * len(config.strategies) * config.replications
    with _progress(total, "Experiment 1", config) as bar:
        for structure in config.structures:
            B0 = coefficient_matrix(config.p, config.m, structure, rng)
            for strategy in config.strategies:
                for rep in range(config.replications):
                    bar.set_postfix_str(f"{structure}, {strategy}, rep {rep + 1}/{config.replications}")
                    rep_rng = np.random.default_rng(_method_seed(rng))
                    workers, _, Sigma0 = generate_distributed_simulation(
                        N=config.N,
                        K=config.K,
                        p=config.p,
                        m=config.m,
                        structure=structure,
                        strategy=strategy,
                        rho_x=config.rho_x,
                        rho_y=config.rho_y,
                        rng=rep_rng,
                        B0=B0,
                        **_sim_kwargs(config),
                    )
                    X_test, Y_test = generate_test_set(
                        n_test=config.n_test,
                        p=config.p,
                        m=config.m,
                        B0=B0,
                        Sigma0=Sigma0,
                        rho_x=config.rho_x,
                        rng=rep_rng,
                    )
                    results = _fit_methods(
                        workers,
                        pilot_proportion=config.pilot_proportion,
                        rounds=config.dmlm_rounds,
                        rng=rep_rng,
                    )
                    gmlm_B = next(r.B for r in results if r.name == "GMLM")
                    for result in results:
                        rows.append(
                            _result_row(
                                result,
                                B0=B0,
                                Sigma0=Sigma0,
                                gmlm_B=gmlm_B,
                                X_test=X_test,
                                Y_test=Y_test,
                                extra={"Experiment": 1, "Structure": structure, "Strategy": strategy, "Replication": rep},
                            )
                        )
                    bar.update(1)
    summary = summarize_estimation_rows(rows, ["Experiment", "Structure", "Strategy", "Method"])
    return pd.DataFrame(rows), summary


def experiment2(config: SimulationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed + 202)
    pilot_grid = (0.002, 0.005, 0.01, 0.02, 0.05)
    if config.N < 10_000:
        pilot_grid = (0.02, 0.05, 0.08, 0.12)
    rows = []
    B0 = coefficient_matrix(config.p, config.m, "S1", rng)
    total = len(pilot_grid) * config.replications
    with _progress(total, "Experiment 2", config) as bar:
        for rho_r in pilot_grid:
            for rep in range(config.replications):
                bar.set_postfix_str(f"rho={rho_r}, rep {rep + 1}/{config.replications}")
                rep_rng = np.random.default_rng(_method_seed(rng))
                workers, _, Sigma0 = generate_distributed_simulation(
                    N=config.N,
                    K=config.K,
                    p=config.p,
                    m=config.m,
                    structure="S1",
                    strategy="C",
                    rho_x=config.rho_x,
                    rho_y=config.rho_y,
                    rng=rep_rng,
                    B0=B0,
                    **_sim_kwargs(config),
                )
                X_test, Y_test = generate_test_set(
                    n_test=config.n_test,
                    p=config.p,
                    m=config.m,
                    B0=B0,
                    Sigma0=Sigma0,
                    rho_x=config.rho_x,
                    rng=rep_rng,
                )
                seed = _method_seed(rep_rng)
                pilot = fit_pilot_only(workers, rng=np.random.default_rng(seed), pilot_proportion=rho_r)
                dmlm = fit_dmlm_s(
                    workers,
                    rng=np.random.default_rng(seed),
                    pilot_proportion=rho_r,
                    rounds=config.dmlm_rounds,
                )
                gmlm = fit_workers_full(workers, name="GMLM")
                for result in (pilot, dmlm):
                    rows.append(
                        _result_row(
                            result,
                            B0=B0,
                            Sigma0=Sigma0,
                            gmlm_B=gmlm.B,
                            X_test=X_test,
                            Y_test=Y_test,
                            extra={
                                "Experiment": 2,
                                "Pilot_proportion": rho_r,
                                "Pilot_size": result.extra.get("pilot_size"),
                                "Replication": rep,
                            },
                        )
                    )
                bar.update(1)
    summary = summarize_estimation_rows(rows, ["Experiment", "Pilot_proportion", "Method"])
    summary["Pilot_size"] = (summary["Pilot_proportion"] * config.N).round().astype(int)
    return pd.DataFrame(rows), summary


def experiment3(config: SimulationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed + 303)
    K_grid = (10, 20, 50, 100, 200)
    if config.N < 10_000:
        K_grid = (4, 8, 16)
    rows = []
    B0 = coefficient_matrix(config.p, config.m, "S1", rng)
    total = 2 * len(K_grid) * config.replications
    with _progress(total, "Experiment 3", config) as bar:
        for strategy in ("C", "H"):
            for K in K_grid:
                for rep in range(config.replications):
                    bar.set_postfix_str(f"{strategy}, K={K}, rep {rep + 1}/{config.replications}")
                    rep_rng = np.random.default_rng(_method_seed(rng))
                    workers, _, Sigma0 = generate_distributed_simulation(
                        N=config.N,
                        K=K,
                        p=config.p,
                        m=config.m,
                        structure="S1",
                        strategy=strategy,
                        rho_x=config.rho_x,
                        rho_y=config.rho_y,
                        rng=rep_rng,
                        B0=B0,
                        **_sim_kwargs(config),
                    )
                    results = _fit_methods(
                        workers,
                        pilot_proportion=config.pilot_proportion,
                        rounds=config.dmlm_rounds,
                        rng=rep_rng,
                    )
                    gmlm_B = next(r.B for r in results if r.name == "GMLM")
                    for result in results:
                        rows.append(
                            _result_row(
                                result,
                                B0=B0,
                                Sigma0=Sigma0,
                                gmlm_B=gmlm_B,
                                X_test=None,
                                Y_test=None,
                                extra={"Experiment": 3, "Strategy": strategy, "K": K, "Replication": rep},
                            )
                        )
                    bar.update(1)
    summary = summarize_estimation_rows(rows, ["Experiment", "Strategy", "K", "Method"])
    return pd.DataFrame(rows), summary


def experiment4(config: SimulationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed + 404)
    rounds_grid = (1, 2, 3)
    rows = []
    B0 = coefficient_matrix(config.p, config.m, "S1", rng)
    with _progress(config.replications, "Experiment 4", config) as bar:
        for rep in range(config.replications):
            bar.set_postfix_str(f"rep {rep + 1}/{config.replications}")
            rep_rng = np.random.default_rng(_method_seed(rng))
            workers, _, Sigma0 = generate_distributed_simulation(
                N=config.N,
                K=config.K,
                p=config.p,
                m=config.m,
                structure="S1",
                strategy="C",
                rho_x=config.rho_x,
                rho_y=config.rho_y,
                rng=rep_rng,
                B0=B0,
                **_sim_kwargs(config),
            )
            gmlm = fit_workers_full(workers, name="GMLM")
            seed = _method_seed(rep_rng)
            pilot = fit_pilot_only(workers, rng=np.random.default_rng(seed), pilot_proportion=config.pilot_proportion)
            methods = [pilot]
            for rounds in rounds_grid:
                result = fit_dmlm_s(
                    workers,
                    rng=np.random.default_rng(seed),
                    pilot_proportion=config.pilot_proportion,
                    rounds=rounds,
                )
                result.name = f"DMLM-S({rounds})"
                methods.append(result)
            methods.append(gmlm)
            for result in methods:
                rows.append(
                    _result_row(
                        result,
                        B0=B0,
                        Sigma0=Sigma0,
                        gmlm_B=gmlm.B,
                        X_test=None,
                        Y_test=None,
                        extra={"Experiment": 4, "Replication": rep},
                    )
                )
            bar.update(1)
    summary = summarize_estimation_rows(rows, ["Experiment", "Method"])
    return pd.DataFrame(rows), summary


def _cov_vec_from_xtx(Sigma: np.ndarray, xtx: np.ndarray) -> np.ndarray:
    return np.kron(Sigma, safe_inverse(xtx))


def experiment5(config: SimulationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed + 505)
    rows = []
    B0 = coefficient_matrix(config.p, config.m, "S1", rng)
    functionals = default_functionals(config.p, config.m, rng)
    with _progress(config.replications, "Experiment 5", config) as bar:
        for rep in range(config.replications):
            bar.set_postfix_str(f"rep {rep + 1}/{config.replications}")
            rep_rng = np.random.default_rng(_method_seed(rng))
            workers, _, _ = generate_distributed_simulation(
                N=config.N,
                K=config.K,
                p=config.p,
                m=config.m,
                structure="S1",
                strategy="C",
                rho_x=config.rho_x,
                rho_y=config.rho_y,
                rng=rep_rng,
                B0=B0,
                **_sim_kwargs(config),
            )
            seed = _method_seed(rep_rng)
            methods = [
                fit_pilot_only(workers, rng=np.random.default_rng(seed), pilot_proportion=config.pilot_proportion),
                fit_first_worker(workers),
                fit_dmlm_s(
                    workers,
                    rng=np.random.default_rng(seed),
                    pilot_proportion=config.pilot_proportion,
                    rounds=config.dmlm_rounds,
                ),
            ]
            for result in methods:
                if result.name == "MLM-S":
                    cov = _cov_vec_from_xtx(result.Sigma, result.extra["pilot_xtx"])
                elif result.name == "MLM-F":
                    cov = _cov_vec_from_xtx(result.Sigma, workers[0].X.T @ workers[0].X)
                else:
                    xtx_full = sum(w.X.T @ w.X for w in workers)
                    cov = _cov_vec_from_xtx(result.Sigma, xtx_full)
                for functional_name, A in functionals.items():
                    psi_hat = functional_value(result.B, A)
                    psi_true = functional_value(B0, A)
                    se = functional_se(cov, A)
                    z = (psi_hat - psi_true) / se if se > 0 else np.nan
                    rows.append(
                        {
                            "Experiment": 5,
                            "Replication": rep,
                            "Functional": functional_name,
                            "Method": result.name,
                            "Estimate": psi_hat,
                            "Truth": psi_true,
                            "SE": se,
                            "T": z,
                            "Covered_95": float(abs(z) <= 1.96) if np.isfinite(z) else np.nan,
                            "Length_95": 2 * 1.96 * se,
                            "Bias": psi_hat - psi_true,
                        }
                    )
            bar.update(1)
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["Experiment", "Functional", "Method"], as_index=False).agg(
        Coverage=("Covered_95", "mean"),
        Avg_length=("Length_95", "mean"),
        Bias=("Bias", "mean"),
        SD=("Estimate", "std"),
        T_mean=("T", "mean"),
        T_sd=("T", "std"),
    )
    return raw, summary


def experiment6(config: SimulationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed + 606)
    rows = []
    contamination_indices = {100, 240, 520}
    total = len(config.structures) * config.replications * 3
    with _progress(total, "Experiment 6", config) as bar:
        for structure in config.structures:
            B0 = coefficient_matrix(config.p, config.m, structure, rng)
            for rep in range(config.replications):
                rep_rng = np.random.default_rng(_method_seed(rng))
                workers, _, _ = generate_distributed_simulation(
                    N=config.N,
                    K=config.K,
                    p=config.p,
                    m=config.m,
                    structure=structure,
                    strategy="C",
                    rho_x=config.rho_x,
                    rho_y=config.rho_y,
                    rng=rep_rng,
                    B0=B0,
                    **_sim_kwargs(config),
                )
                valid_indices = set(range(config.N))
                selected = set(i for i in contamination_indices if i in valid_indices)
                if len(selected) < 3:
                    selected = set(int(i) for i in rep_rng.choice(config.N, size=min(3, config.N), replace=False))

                scenarios: list[tuple[str, list[WorkerData], set[int], int | None]] = []
                scenarios.append(("Response-shift observations", contaminate_response(workers, selected, 8.0), selected, None))
                direction = rep_rng.normal(size=config.p)
                scenarios.append(("Leverage-point observations", contaminate_leverage(workers, selected, 8.0, direction), selected, None))
                contaminated_worker = min(config.K - 1, max(0, config.K // 2))
                worker_contaminated, worker_indices = contaminate_worker_response(
                    workers,
                    contaminated_worker,
                    proportion=0.05,
                    shift=8.0,
                    rng=rep_rng,
                )
                scenarios.append(("Worker-level contamination", worker_contaminated, worker_indices, contaminated_worker))

                for contamination_type, contaminated_workers, contaminated_set, worker_id in scenarios:
                    bar.set_postfix_str(f"{structure}, rep {rep + 1}/{config.replications}, {contamination_type}")
                    result = fit_dmlm_s(
                        contaminated_workers,
                        rng=np.random.default_rng(_method_seed(rep_rng)),
                        pilot_proportion=config.pilot_proportion,
                        rounds=config.dmlm_rounds,
                    )
                    hessian = result.extra["pilot_hessian"]
                    obs_df = observation_influence(contaminated_workers, result.B, hessian_x=hessian)
                    worker_df = worker_influence(contaminated_workers, result.B, hessian_x=hessian)
                    row = detection_summary(obs_df, worker_df, contaminated_set, worker_id)
                    row.update(
                        {
                            "Experiment": 6,
                            "Structure": structure,
                            "Contamination_type": contamination_type,
                            "Replication": rep,
                        }
                    )
                    rows.append(row)
                    bar.update(1)
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["Experiment", "Structure", "Contamination_type"], as_index=False).mean(numeric_only=True)
    return raw, summary


def write_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    output_dir: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / f"{stem}_raw.csv"
    summary_path = output / f"{stem}_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    return raw_path, summary_path
