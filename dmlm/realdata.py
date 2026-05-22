"""Real-data helpers for the DMLM-S paper applications."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import make_workers_from_arrays
from .diagnostics import observation_influence, worker_influence
from .estimators import (
    WorkerData,
    fit_dmlm_s,
    fit_first_worker,
    fit_full_mlm,
    fit_local_average,
    fit_pilot_only,
)
from .linalg import add_intercept, standardize_train_test
from .metrics import prediction_error


DEFAULT_BEIJING_PILOTS = (0.002, 0.005, 0.01, 0.02)
DEFAULT_SARCOS_PILOTS = (0.005, 0.01, 0.02)
DEFAULT_SARCOS_KS = (20, 50)
DEFAULT_SARCOS_STORAGES = ("random", "motion")


def _with_median_time(factory, *, repeats: int = 3):
    results = [factory() for _ in range(max(1, repeats))]
    times = np.array([result.time_seconds for result in results], dtype=float)
    result = results[int(np.argsort(times)[len(times) // 2])]
    result.time_seconds = float(np.median(times))
    result.extra = {
        **result.extra,
        "timing_repeats": int(max(1, repeats)),
        "timing_seconds": times.tolist(),
    }
    return result


def _read_csv_path(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        frames = []
        required_any = {"PM2.5", "PM10", "SO2", "NO2", "CO", "O3"}
        for csv_path in sorted(path.rglob("*.csv")):
            frame = pd.read_csv(csv_path)
            frame.columns = [str(col).strip() for col in frame.columns]
            if not required_any.issubset(set(frame.columns)):
                continue
            if "station" not in frame.columns:
                frame["station"] = csv_path.stem
            frames.append(frame)
        if not frames:
            raise FileNotFoundError(f"No Beijing air-quality CSV files found in {path}")
        return pd.concat(frames, ignore_index=True)
    frame = pd.read_csv(path)
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def _chronological_sort(df: pd.DataFrame) -> pd.DataFrame:
    time_cols = [c for c in ["year", "month", "day", "hour"] if c in df.columns]
    if len(time_cols) == 4:
        return df.sort_values(time_cols).reset_index(drop=True)
    for col in ["datetime", "date", "time"]:
        if col in df.columns:
            return df.assign(_time=pd.to_datetime(df[col], errors="coerce")).sort_values("_time").drop(columns="_time").reset_index(drop=True)
    return df.reset_index(drop=True)


def _standardize_matrix_train_test(
    train_values: np.ndarray,
    test_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale[scale == 0] = 1.0
    return (train_values - mean) / scale, (test_values - mean) / scale


def prepare_beijing(path: str | Path) -> tuple[list[WorkerData], np.ndarray, np.ndarray, pd.DataFrame, list[str], list[str]]:
    df = _chronological_sort(_read_csv_path(path))
    response_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    missing = [c for c in response_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Beijing response columns: {missing}")
    if "station" not in df.columns:
        raise ValueError("Beijing data need a 'station' column or a directory of station CSV files.")

    df = df.dropna(subset=response_cols).reset_index(drop=True)
    split = int(0.8 * len(df))
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    numeric_candidates = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "year", "month", "day", "hour"]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    for col in numeric_cols:
        medians = train_df.groupby("station")[col].transform("median")
        train_df[col] = train_df[col].fillna(medians).fillna(train_df[col].median())
        station_medians = train_df.groupby("station")[col].median()
        test_df[col] = test_df[col].fillna(test_df["station"].map(station_medians)).fillna(train_df[col].median())

    design_full = pd.concat(
        [
            pd.concat([train_df[numeric_cols], test_df[numeric_cols]], ignore_index=True),
            pd.get_dummies(pd.concat([train_df.get("wd", pd.Series(["missing"] * len(train_df))), test_df.get("wd", pd.Series(["missing"] * len(test_df)))], ignore_index=True), prefix="wd"),
            pd.get_dummies(pd.concat([train_df["station"], test_df["station"]], ignore_index=True), prefix="station"),
        ],
        axis=1,
    )
    feature_names = list(design_full.columns)
    X_train_raw = design_full.iloc[: len(train_df)].to_numpy(dtype=float)
    X_test_raw = design_full.iloc[len(train_df) :].to_numpy(dtype=float)
    X_train_std, X_test_std, _, _ = standardize_train_test(X_train_raw, X_test_raw)
    X_train = add_intercept(X_train_std)
    X_test = add_intercept(X_test_std)

    Y_train, Y_test = _standardize_matrix_train_test(
        train_df[response_cols].to_numpy(dtype=float),
        test_df[response_cols].to_numpy(dtype=float),
    )

    workers = []
    for worker_id, (_, station_df) in enumerate(train_df.groupby("station", sort=True)):
        pos = station_df.index.to_numpy()
        workers.append(WorkerData(X=X_train[pos], Y=Y_train[pos], indices=pos.copy(), worker_id=worker_id))
    return workers, X_test, Y_test, train_df, response_cols, ["intercept"] + feature_names


def _format_beijing_time(row: pd.Series) -> str:
    if {"year", "month", "day", "hour"}.issubset(row.index):
        return f"{int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d} {int(row['hour']):02d}:00"
    for col in ["datetime", "date", "time"]:
        if col in row.index:
            return str(row[col])
    return str(int(row.name))


def _main_feature(row: pd.Series, cols: list[str]) -> str:
    available = [col for col in cols if col in row.index and pd.notna(row[col])]
    if not available:
        return ""
    col = max(available, key=lambda name: abs(float(row[name])))
    return f"{col}={float(row[col]):.3g}"


def enrich_beijing_observation_influence(
    obs: pd.DataFrame,
    train_df: pd.DataFrame,
    response_cols: list[str],
) -> pd.DataFrame:
    meteo_cols = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    rows = []
    for rank, record in enumerate(obs.to_dict("records"), start=1):
        idx = int(record["global_index"])
        row = train_df.loc[idx]
        rows.append(
            {
                "Rank": rank,
                "Station": row.get("station", ""),
                "Time": _format_beijing_time(row),
                "C_i": record["influence"],
                "Main pollutant feature": _main_feature(row, response_cols),
                "Meteorological feature": _main_feature(row, meteo_cols),
                "global_index": idx,
                "worker": int(record["worker"]),
            }
        )
    return pd.DataFrame(rows)


def fit_real_methods(
    workers: list[WorkerData],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    *,
    pilot_proportion: float,
    rounds: int,
    seed: int,
    timing_repeats: int = 3,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shared_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    X_train = np.vstack([w.X for w in workers])
    Y_train = np.vstack([w.Y for w in workers])
    methods = [
        _with_median_time(lambda: fit_full_mlm(X_train, Y_train, name="GMLM"), repeats=timing_repeats),
        _with_median_time(lambda: fit_local_average(workers), repeats=timing_repeats),
        _with_median_time(
            lambda: fit_pilot_only(workers, rng=np.random.default_rng(shared_seed), pilot_proportion=pilot_proportion),
            repeats=timing_repeats,
        ),
        _with_median_time(lambda: fit_first_worker(workers), repeats=timing_repeats),
        _with_median_time(
            lambda: fit_dmlm_s(
                workers,
                rng=np.random.default_rng(shared_seed),
                pilot_proportion=pilot_proportion,
                rounds=rounds,
            ),
            repeats=timing_repeats,
        ),
    ]

    rows = []
    for result in methods:
        row = {
            "Method": result.name,
            "MSPE": prediction_error(X_test, Y_test, result.B),
            "Time": result.time_seconds,
            "Comm": result.communication_scalars,
        }
        res = Y_test - X_test @ result.B
        for j in range(Y_test.shape[1]):
            row[f"RMSE_{j + 1}"] = float(np.sqrt(np.mean(res[:, j] ** 2)))
        rows.append(row)
    return pd.DataFrame(rows)


def run_beijing(
    path: str | Path,
    *,
    pilot_proportions: tuple[float, ...] = DEFAULT_BEIJING_PILOTS,
    primary_pilot: float = 0.01,
    rounds: int = 3,
    seed: int = 20260509,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workers, X_test, Y_test, train_df, response_cols, _ = prepare_beijing(path)
    performance_parts = []
    for pilot_proportion in pilot_proportions:
        part = fit_real_methods(
            workers,
            X_test,
            Y_test,
            pilot_proportion=pilot_proportion,
            rounds=rounds,
            seed=seed,
        )
        part.insert(0, "Pilot_proportion", pilot_proportion)
        performance_parts.append(part)
    performance = pd.concat(performance_parts, ignore_index=True)
    performance = performance.rename(columns={f"RMSE_{j + 1}": col for j, col in enumerate(response_cols)})

    primary_mask = np.isclose(performance["Pilot_proportion"].to_numpy(dtype=float), primary_pilot)
    main_results = performance.loc[primary_mask].reset_index(drop=True)
    pilot_sensitivity = performance.loc[performance["Method"] == "DMLM-S"].reset_index(drop=True)

    result = fit_dmlm_s(workers, rng=np.random.default_rng(seed), pilot_proportion=primary_pilot, rounds=rounds)
    obs_raw = observation_influence(workers, result.B, hessian_x=result.extra["pilot_hessian"]).head(50)
    obs = enrich_beijing_observation_influence(obs_raw, train_df, response_cols)
    wk = worker_influence(workers, result.B, hessian_x=result.extra["pilot_hessian"])
    return performance, main_results, pilot_sensitivity, obs, wk


def _read_numeric_table(path: str | Path) -> np.ndarray:
    path = Path(path)
    sep = "," if path.suffix.lower() == ".csv" else r"\s+|,"
    frame = pd.read_csv(path, header=None, sep=sep, engine="python")
    return frame.to_numpy(dtype=float)


def prepare_sarcos(
    train_path: str | Path,
    test_path: str | Path,
    *,
    K: int,
    storage: str,
    seed: int,
) -> tuple[list[WorkerData], np.ndarray, np.ndarray]:
    train = _read_numeric_table(train_path)
    test = _read_numeric_table(test_path)
    if train.shape[1] < 28 or test.shape[1] < 28:
        raise ValueError("SARCOS files must contain at least 28 columns: 21 inputs followed by 7 outputs.")

    X_train_raw, Y_train_raw = train[:, :21], train[:, 21:28]
    X_test_raw, Y_test_raw = test[:, :21], test[:, 21:28]
    X_train_std, X_test_std, _, _ = standardize_train_test(X_train_raw, X_test_raw)
    Y_train, Y_test = _standardize_matrix_train_test(Y_train_raw, Y_test_raw)
    X_train = add_intercept(X_train_std)
    X_test = add_intercept(X_test_std)

    rng = np.random.default_rng(seed)
    if storage.lower() in {"random", "r"}:
        workers = make_workers_from_arrays(X_train, Y_train, K, strategy="R", rng=rng)
    elif storage.lower() in {"motion", "motion-regime", "c"}:
        z = X_train_std[:, 0] + X_train_std[:, 7] + X_train_std[:, 14]
        order = np.argsort(z)
        sizes = [len(order) // K + (1 if i < len(order) % K else 0) for i in range(K)]
        workers = []
        start = 0
        for worker_id, size in enumerate(sizes):
            idx = order[start : start + size]
            workers.append(WorkerData(X=X_train[idx], Y=Y_train[idx], indices=idx.copy(), worker_id=worker_id))
            start += size
    else:
        raise ValueError("storage must be 'random' or 'motion'.")
    return workers, X_test, Y_test


def run_sarcos(
    train_path: str | Path,
    test_path: str | Path,
    *,
    K_grid: tuple[int, ...] = DEFAULT_SARCOS_KS,
    storage_grid: tuple[str, ...] = DEFAULT_SARCOS_STORAGES,
    pilot_proportions: tuple[float, ...] = DEFAULT_SARCOS_PILOTS,
    primary_pilot: float = 0.01,
    rounds: int = 3,
    seed: int = 20260509,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance_parts = []
    worker_parts = []
    for storage in storage_grid:
        for K in K_grid:
            workers, X_test, Y_test = prepare_sarcos(train_path, test_path, K=K, storage=storage, seed=seed)
            for pilot_proportion in pilot_proportions:
                part = fit_real_methods(
                    workers,
                    X_test,
                    Y_test,
                    pilot_proportion=pilot_proportion,
                    rounds=rounds,
                    seed=seed,
                )
                part = part.rename(columns={f"RMSE_{j + 1}": f"Torque_{j + 1}" for j in range(7)})
                part.insert(0, "Pilot_proportion", pilot_proportion)
                part.insert(0, "K", K)
                part.insert(0, "Storage", storage)
                performance_parts.append(part)

            if storage.lower() in {"motion", "motion-regime", "c"}:
                result = fit_dmlm_s(
                    workers,
                    rng=np.random.default_rng(seed),
                    pilot_proportion=primary_pilot,
                    rounds=rounds,
                )
                wk = worker_influence(workers, result.B, hessian_x=result.extra["pilot_hessian"])
                wk.insert(0, "Pilot_proportion", primary_pilot)
                wk.insert(0, "K", K)
                wk.insert(0, "Storage", storage)
                worker_parts.append(wk)

    performance = pd.concat(performance_parts, ignore_index=True)
    primary_mask = np.isclose(performance["Pilot_proportion"].to_numpy(dtype=float), primary_pilot)
    main_results = performance.loc[primary_mask].reset_index(drop=True)
    pilot_sensitivity = performance.loc[performance["Method"] == "DMLM-S"].reset_index(drop=True)
    worker_influence_df = pd.concat(worker_parts, ignore_index=True) if worker_parts else pd.DataFrame()
    return performance, main_results, pilot_sensitivity, worker_influence_df
