#!/usr/bin/env python3
"""Generate Chapter 5 real-data figures from DMLM-S CSV results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dmlm_matplotlib"))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_FIGURES_DIR = Path("/Users/chenxiaoxia/Desktop/DMLM论文/figures/ch5")
REAL_DATA_FILES = [
    "beijing_main_results.csv",
    "beijing_pilot_sensitivity.csv",
    "beijing_observation_influence.csv",
    "beijing_worker_influence.csv",
    "sarcos_main_results.csv",
    "sarcos_pilot_sensitivity.csv",
    "sarcos_motion_worker_influence.csv",
]

METHOD_ORDER = ["GMLM", "AMLM", "MLM-S", "MLM-F", "DMLM-S"]
METHOD_COLORS = {
    "GMLM": "#222222",
    "AMLM": "#0072B2",
    "MLM-S": "#E69F00",
    "MLM-F": "#CC79A7",
    "DMLM-S": "#009E73",
}
STORAGE_LABELS = {
    "random": "Random",
    "motion": "Motion-regime",
    "motion-regime": "Motion-regime",
}
METRIC_STEMS = {
    "MSPE": "mspe",
    "Time": "cpu_time",
    "Comm": "communication_cost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing the real-data CSV outputs. If omitted, results/ is used when available.",
    )
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--format", choices=["pdf", "png", "both"], default="pdf")
    parser.add_argument(
        "--beijing-data",
        default="data/beijing",
        help="Beijing data directory, used only to infer station names when worker labels are missing.",
    )
    parser.add_argument(
        "--top-observations",
        type=int,
        default=10,
        help="Number of top Beijing station-hour observations to show.",
    )
    parser.add_argument(
        "--sarcos-worker-k",
        type=int,
        default=20,
        help="Worker count used for the SARCOS worker influence plot.",
    )
    return parser.parse_args()


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is not installed. Install it with `python3 -m pip install matplotlib`, "
            "then rerun scripts/make_ch5_plots.py."
        ) from exc

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    return plt


def _resolve_results_dir(user_value: str | None) -> Path:
    if user_value:
        return Path(user_value)
    candidates = [ROOT / "results", ROOT / "scripts" / "results"]
    scored: list[tuple[int, float, Path]] = []
    for candidate in candidates:
        present = sum(1 for name in REAL_DATA_FILES if (candidate / name).exists())
        if present == 0:
            continue
        score = present
        obs_path = candidate / "beijing_observation_influence.csv"
        if obs_path.exists():
            try:
                obs_cols = set(pd.read_csv(obs_path, nrows=1).columns)
            except Exception:
                obs_cols = set()
            if {"Rank", "Station", "Time", "C_i"}.issubset(obs_cols):
                score += 5
        newest = max((candidate / name).stat().st_mtime for name in REAL_DATA_FILES if (candidate / name).exists())
        scored.append((score, newest, candidate))
    if scored:
        return max(scored, key=lambda item: (item[0], item[1]))[2]
    return ROOT / "results"


def _read(results_dir: Path, name: str) -> pd.DataFrame | None:
    path = results_dir / name
    if not path.exists():
        print(f"skip: missing {path}")
        return None
    return pd.read_csv(path)


def _save(fig, figures_dir: Path, stem: str, fmt: str) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    formats = ["pdf", "png"] if fmt == "both" else [fmt]
    outputs: list[Path] = []
    for suffix in formats:
        path = figures_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)
    return outputs


def _ordered_methods(methods: pd.Series | list[str]) -> list[str]:
    seen = list(dict.fromkeys([str(method) for method in methods]))
    known = [method for method in METHOD_ORDER if method in seen]
    extra = sorted(method for method in seen if method not in METHOD_ORDER)
    return known + extra


def _storage_label(value: str) -> str:
    return STORAGE_LABELS.get(str(value).lower(), str(value))


def _storage_stem(value: str) -> str:
    return str(value).lower().replace("-", "_").replace(" ", "_")


def _safe_ci_column(df: pd.DataFrame) -> str:
    if "C_i" in df.columns:
        return "C_i"
    if "influence" in df.columns:
        return "influence"
    raise ValueError("Observation influence CSV must contain C_i or influence.")


def _safe_ck_column(df: pd.DataFrame) -> str:
    if "C_k" in df.columns:
        return "C_k"
    if "influence" in df.columns:
        return "influence"
    raise ValueError("Worker influence CSV must contain C_k or influence.")


def _metric_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return [col for col in df.columns if col.startswith(prefix)]


def _format_pilot_ticks(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    x = values.to_numpy(dtype=float)
    labels = [f"{value:g}" for value in x]
    return x, labels


def _read_beijing_station_map(data_path: str | Path) -> dict[int, str]:
    path = Path(data_path)
    if not path.exists():
        return {}

    frames: list[pd.DataFrame] = []
    required = {"PM2.5", "PM10", "SO2", "NO2", "CO", "O3"}
    csv_paths = sorted(path.rglob("*.csv")) if path.is_dir() else [path]
    for csv_path in csv_paths:
        try:
            frame = pd.read_csv(csv_path, nrows=5)
        except Exception:
            continue
        frame.columns = [str(col).strip() for col in frame.columns]
        if not required.issubset(set(frame.columns)):
            continue
        if "station" not in frame.columns:
            frame["station"] = csv_path.stem
        frames.append(frame[["station"]])

    if not frames:
        return {}
    stations = sorted(pd.concat(frames, ignore_index=True)["station"].dropna().astype(str).unique())
    return {idx: station for idx, station in enumerate(stations)}


def _worker_labels(df: pd.DataFrame, *, station_map: dict[int, str] | None = None) -> list[str]:
    if "Station" in df.columns:
        return df["Station"].astype(str).tolist()
    if "station" in df.columns:
        return df["station"].astype(str).tolist()
    station_map = station_map or {}
    labels = []
    for worker in df["worker"].astype(int):
        labels.append(station_map.get(worker, f"Worker {worker}"))
    return labels


def plot_beijing_pilot_sensitivity(results_dir: Path, figures_dir: Path, fmt: str) -> list[Path]:
    plt = _load_matplotlib()
    df = _read(results_dir, "beijing_pilot_sensitivity.csv")
    if df is None:
        return []

    df = df.sort_values("Pilot_proportion")
    x, tick_labels = _format_pilot_ticks(df["Pilot_proportion"])
    metrics = [
        ("MSPE", r"$\operatorname{MSPE}$"),
        ("Time", "CPU time"),
        ("Comm", "Communication cost"),
    ]
    outputs: list[Path] = []
    for col, ylabel in metrics:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(4.8, 3.5), constrained_layout=True)
        ax.plot(x, df[col], marker="o", color=METHOD_COLORS["DMLM-S"], linewidth=1.9)
        ax.set_xlabel("Pilot proportion")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        if col == "Comm":
            ax.set_yscale("log")
        outputs.extend(_save(fig, figures_dir, f"beijing_pilot_sensitivity_{METRIC_STEMS[col]}", fmt))
        plt.close(fig)
    return outputs


def plot_beijing_observation_influence(
    results_dir: Path,
    figures_dir: Path,
    fmt: str,
    *,
    top_n: int,
) -> list[Path]:
    plt = _load_matplotlib()
    df = _read(results_dir, "beijing_observation_influence.csv")
    if df is None:
        return []

    ci_col = _safe_ci_column(df)
    part = df.sort_values(ci_col, ascending=False).head(top_n).copy()
    part = part.iloc[::-1]
    labels = []
    for _, row in part.iterrows():
        station = str(row.get("Station", row.get("station", f"Worker {int(row.get('worker', -1))}")))
        time = str(row.get("Time", row.get("global_index", "")))
        labels.append(f"{station}\n{time}")

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    y = np.arange(len(part))
    ax.barh(y, part[ci_col].to_numpy(dtype=float), color="#009E73")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"Observation-level influence $C_i$")
    if part[ci_col].min() > 0:
        ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.22)
    ax.grid(axis="y", visible=False)
    return _save(fig, figures_dir, "beijing_local_influence_observations", fmt)


def plot_beijing_worker_influence(
    results_dir: Path,
    figures_dir: Path,
    fmt: str,
    *,
    station_map: dict[int, str],
) -> list[Path]:
    plt = _load_matplotlib()
    df = _read(results_dir, "beijing_worker_influence.csv")
    if df is None:
        return []

    ck_col = _safe_ck_column(df)
    part = df.sort_values(ck_col, ascending=False).copy()
    labels = _worker_labels(part, station_map=station_map)
    x = np.arange(len(part))

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    bars = ax.bar(x, part[ck_col].to_numpy(dtype=float), color="#56B4E9", edgecolor="white", linewidth=0.6)
    for idx in range(min(3, len(bars))):
        bars[idx].set_color("#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(r"Worker-level influence $C_k$")
    if part[ck_col].min() > 0:
        ax.set_yscale("log")
    return _save(fig, figures_dir, "beijing_worker_influence", fmt)


def plot_sarcos_prediction_comparison(results_dir: Path, figures_dir: Path, fmt: str) -> list[Path]:
    plt = _load_matplotlib()
    df = _read(results_dir, "sarcos_main_results.csv")
    if df is None:
        return []

    metrics = ["MSPE", *_metric_columns(df, "Torque_")]
    storage_order = [s for s in ["random", "motion"] if s in set(df["Storage"].astype(str).str.lower())]
    if not storage_order:
        storage_order = list(dict.fromkeys(df["Storage"].astype(str)))
    k_order = sorted(df["K"].dropna().unique().astype(int).tolist())

    panels = []
    for storage in storage_order:
        storage_mask = df["Storage"].astype(str).str.lower() == storage
        for k in k_order:
            part = df[storage_mask & (df["K"].astype(int) == k)]
            if not part.empty:
                panels.append((storage, k, part))

    outputs: list[Path] = []
    x = np.arange(len(metrics))
    metric_labels = ["MSPE", *[f"T{idx}" for idx in range(1, len(metrics))]]
    for storage, k, part in panels:
        fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
        for method in _ordered_methods(part["Method"]):
            method_part = part[part["Method"] == method]
            if method_part.empty:
                continue
            values = method_part[metrics].iloc[0].to_numpy(dtype=float)
            ax.plot(
                x,
                values,
                marker="o",
                linewidth=1.65,
                markersize=4.5,
                color=METHOD_COLORS.get(method),
                label=method,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=0)
        ax.set_ylabel("Prediction error")
        ax.legend(loc="best", ncol=2)
        outputs.extend(
            _save(
                fig,
                figures_dir,
                f"sarcos_prediction_comparison_storage_{_storage_stem(storage)}_K{int(k)}",
                fmt,
            )
        )
        plt.close(fig)
    return outputs


def plot_sarcos_scalability(results_dir: Path, figures_dir: Path, fmt: str) -> list[Path]:
    plt = _load_matplotlib()
    df = _read(results_dir, "sarcos_main_results.csv")
    if df is None:
        return []

    part = df[df["Method"] == "DMLM-S"].copy()
    if part.empty:
        return []
    metrics = [
        ("MSPE", r"$\operatorname{MSPE}$"),
        ("Time", "CPU time"),
        ("Comm", "Communication cost"),
    ]

    outputs: list[Path] = []
    storage_colors = {"random": "#0072B2", "motion": "#D55E00", "motion-regime": "#D55E00"}
    for col, ylabel in metrics:
        if col not in part.columns:
            continue
        fig, ax = plt.subplots(figsize=(4.8, 3.5), constrained_layout=True)
        for storage, storage_part in part.groupby("Storage"):
            storage_part = storage_part.sort_values("K")
            key = str(storage).lower()
            ax.plot(
                storage_part["K"],
                storage_part[col],
                marker="o",
                linewidth=1.9,
                color=storage_colors.get(key, None),
                label=_storage_label(str(storage)),
            )
        ax.set_xlabel("Number of workers K")
        ax.set_ylabel(ylabel)
        if col == "Comm":
            ax.set_yscale("log")
        ax.legend(loc="best")
        outputs.extend(_save(fig, figures_dir, f"sarcos_scalability_{METRIC_STEMS[col]}", fmt))
        plt.close(fig)
    return outputs


def plot_sarcos_worker_influence(
    results_dir: Path,
    figures_dir: Path,
    fmt: str,
    *,
    worker_k: int,
) -> list[Path]:
    plt = _load_matplotlib()
    df = _read(results_dir, "sarcos_motion_worker_influence.csv")
    if df is None:
        return []

    ck_col = _safe_ck_column(df)
    if "K" in df.columns and worker_k in set(df["K"].astype(int)):
        part = df[df["K"].astype(int) == worker_k].copy()
    else:
        part = df.copy()
        if "K" in part.columns:
            worker_k = int(part["K"].iloc[0])
    part = part.sort_values("worker")
    top_workers = set(part.nlargest(min(5, len(part)), ck_col)["worker"].astype(int))
    x = np.arange(len(part))
    colors = ["#D55E00" if int(worker) in top_workers else "#56B4E9" for worker in part["worker"]]

    fig, ax = plt.subplots(figsize=(8.2, 4.0), constrained_layout=True)
    ax.bar(x, part[ck_col].to_numpy(dtype=float), color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(part["worker"].astype(int).astype(str), rotation=0)
    ax.set_xlabel("Worker")
    ax.set_ylabel(r"Worker-level influence $C_k$")
    if part[ck_col].min() > 0:
        ax.set_yscale("log")
    return _save(fig, figures_dir, "sarcos_worker_influence", fmt)


def main() -> None:
    args = parse_args()
    results_dir = _resolve_results_dir(args.results_dir)
    figures_dir = Path(args.figures_dir)
    station_map = _read_beijing_station_map(args.beijing_data)

    print(f"Reading real-data results from: {results_dir}")
    print(f"Writing Chapter 5 figures to: {figures_dir}")

    outputs: list[Path] = []
    outputs.extend(plot_beijing_pilot_sensitivity(results_dir, figures_dir, args.format))
    outputs.extend(
        plot_beijing_observation_influence(
            results_dir,
            figures_dir,
            args.format,
            top_n=args.top_observations,
        )
    )
    outputs.extend(
        plot_beijing_worker_influence(
            results_dir,
            figures_dir,
            args.format,
            station_map=station_map,
        )
    )
    outputs.extend(plot_sarcos_prediction_comparison(results_dir, figures_dir, args.format))
    outputs.extend(plot_sarcos_scalability(results_dir, figures_dir, args.format))
    outputs.extend(
        plot_sarcos_worker_influence(
            results_dir,
            figures_dir,
            args.format,
            worker_k=args.sarcos_worker_k,
        )
    )

    if outputs:
        print("Generated Chapter 5 figures:")
        for path in outputs:
            print(f"  {path}")
    else:
        print("No Chapter 5 figures were generated. Check that the real-data CSV files exist.")


if __name__ == "__main__":
    main()
