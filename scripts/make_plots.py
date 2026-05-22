#!/usr/bin/env python3
"""Generate Chapter 4 simulation figures from DMLM-S CSV results."""

from __future__ import annotations

import argparse
from math import pi, sqrt
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_FIGURES_DIR = Path("/Users/chenxiaoxia/Desktop/DMLM论文/figures/ch4")
METHOD_ORDER = ["GMLM", "AMLM", "MLM-S", "MLM-F", "DMLM-S", "DMLM-S(1)", "DMLM-S(2)", "DMLM-S(3)"]
METHOD_COLORS = {
    "GMLM": "#2f2f2f",
    "AMLM": "#4c78a8",
    "MLM-S": "#f58518",
    "MLM-F": "#b279a2",
    "DMLM-S": "#54a24b",
    "DMLM-S(1)": "#9ecae9",
    "DMLM-S(2)": "#74c476",
    "DMLM-S(3)": "#238b45",
}
MARKERS = {
    "GMLM": "o",
    "AMLM": "s",
    "MLM-S": "^",
    "MLM-F": "v",
    "DMLM-S": "D",
    "DMLM-S(1)": "o",
    "DMLM-S(2)": "s",
    "DMLM-S(3)": "D",
}
METRIC_LABELS = {
    "RMSE_B": r"$\operatorname{RMSE}_{B}$",
    "RMSE_Sigma": r"$\operatorname{RMSE}_{\Sigma}$",
    "RE_GMLM": r"$\operatorname{RE}_{\mathrm{GMLM}}$",
    "PE": r"$\operatorname{PE}$",
    "Time": "CPU time",
    "Comm": "Communication cost",
    "Coverage": "Coverage probability",
    "Avg_length": "Average length",
    "AUC": "AUC",
    "Hit@5": "Hit@5",
    "Hit@10": "Hit@10",
}
FUNCTIONAL_LABELS = {
    "psi1_B11": r"$\psi_1$",
    "psi2_row1_avg": r"$\psi_2$",
    "psi3_sparse_trace": r"$\psi_3$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing experiment*_summary.csv and experiment*_raw.csv. "
        "If omitted, the script tries results/ and scripts/results/.",
    )
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--format", choices=["pdf", "png", "both"], default="pdf")
    return parser.parse_args()


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is not installed. Install it with `python3 -m pip install matplotlib`, "
            "then rerun scripts/make_plots.py."
        ) from exc

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )
    return plt


def _resolve_results_dir(user_value: str | None) -> Path:
    if user_value:
        return Path(user_value)
    candidates = [ROOT / "results", ROOT / "scripts" / "results"]
    for candidate in candidates:
        if (candidate / "experiment1_summary.csv").exists() or any(candidate.glob("experiment*_summary.csv")):
            return candidate
    return ROOT / "results"


def _read(results_dir: Path, name: str) -> pd.DataFrame | None:
    path = results_dir / name
    if not path.exists():
        print(f"skip: missing {path}")
        return None
    return pd.read_csv(path)


def _ordered_methods(methods) -> list[str]:
    methods = list(dict.fromkeys(methods))
    known = [m for m in METHOD_ORDER if m in methods]
    extra = sorted(m for m in methods if m not in METHOD_ORDER)
    return known + extra


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def _functional_label(functional: str) -> str:
    return FUNCTIONAL_LABELS.get(functional, functional)


def _save(fig, figures_dir: Path, stem: str, fmt: str) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    formats = ["pdf", "png"] if fmt == "both" else [fmt]
    for suffix in formats:
        path = figures_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)
    return outputs


def plot_accuracy_communication_tradeoff(results_dir: Path, figures_dir: Path, fmt: str) -> list[Path]:
    """Experiment 1 summary: strategy-specific accuracy versus communication cost."""
    plt = _load_matplotlib()
    df = _read(results_dir, "experiment1_summary.csv")
    if df is None:
        return []

    metrics = [m for m in ["RMSE_B", "RMSE_Sigma", "RE_GMLM", "PE"] if m in df.columns]
    if not metrics:
        return []

    outputs: list[Path] = []
    for strategy, part_strategy in df.groupby("Strategy"):
        agg = part_strategy.groupby("Method", as_index=False)[["Comm", *metrics]].mean(numeric_only=True)
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(4.6, 3.5), constrained_layout=True)
            for method in _ordered_methods(agg["Method"]):
                part = agg[agg["Method"] == method]
                x = part["Comm"].to_numpy(dtype=float) + 1.0
                y = part[metric].to_numpy(dtype=float)
                ax.scatter(
                    x,
                    y,
                    s=72,
                    marker=MARKERS.get(method, "o"),
                    color=METHOD_COLORS.get(method, None),
                    label=method,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
                ax.annotate(method, (x[0], y[0]), xytext=(4, 3), textcoords="offset points", fontsize=8)
            ax.set_xscale("log")
            ax.set_xlabel("Communication cost")
            ax.set_ylabel(_metric_label(metric))
            metric_name = metric.lower().replace("rmse_", "rmse_").replace("re_gmlm", "re_gmlm")
            outputs.extend(_save(fig, figures_dir, f"accuracy_communication_strategy_{strategy}_{metric_name}", fmt))
            plt.close(fig)
    return outputs


def plot_pilot_sensitivity(results_dir: Path, figures_dir: Path, fmt: str) -> list[Path]:
    plt = _load_matplotlib()
    df = _read(results_dir, "experiment2_summary.csv")
    if df is None:
        return []
    metrics = [m for m in ["RMSE_B", "RMSE_Sigma", "PE", "Time"] if m in df.columns]
    outputs: list[Path] = []
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(4.8, 3.5), constrained_layout=True)
        for method in _ordered_methods(df["Method"]):
            part = df[df["Method"] == method].sort_values("Pilot_proportion")
            ax.plot(
                part["Pilot_proportion"],
                part[metric],
                marker=MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method, None),
                label=method,
                linewidth=1.8,
            )
        ax.set_xlabel("Pilot proportion")
        ax.set_ylabel(_metric_label(metric))
        ax.legend(loc="best")
        outputs.extend(_save(fig, figures_dir, f"pilot_sensitivity_{metric.lower()}", fmt))
        plt.close(fig)
    return outputs


def plot_worker_scalability(results_dir: Path, figures_dir: Path, fmt: str) -> list[Path]:
    plt = _load_matplotlib()
    df = _read(results_dir, "experiment3_summary.csv")
    if df is None:
        return []
    metrics = [m for m in ["RMSE_B", "RE_GMLM", "Time", "Comm"] if m in df.columns]
    outputs: list[Path] = []
    for strategy, part_strategy in df.groupby("Strategy"):
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(4.8, 3.5), constrained_layout=True)
            for method in _ordered_methods(part_strategy["Method"]):
                part = part_strategy[part_strategy["Method"] == method].sort_values("K")
                ax.plot(
                    part["K"],
                    part[metric],
                    marker=MARKERS.get(method, "o"),
                    color=METHOD_COLORS.get(method, None),
                    label=method,
                    linewidth=1.8,
                )
            ax.set_xlabel("Number of workers K")
            ax.set_ylabel(_metric_label(metric))
            ax.legend(loc="best")
            outputs.extend(_save(fig, figures_dir, f"worker_scalability_strategy_{strategy}_{metric.lower()}", fmt))
            plt.close(fig)
    return outputs


def plot_iteration_effect(results_dir: Path, figures_dir: Path, fmt: str) -> list[Path]:
    plt = _load_matplotlib()
    df = _read(results_dir, "experiment4_summary.csv")
    if df is None:
        return []
    metrics = [m for m in ["RMSE_B", "RMSE_Sigma", "RE_GMLM", "Time"] if m in df.columns]
    method_order = [m for m in ["MLM-S", "DMLM-S(1)", "DMLM-S(2)", "DMLM-S(3)", "GMLM"] if m in set(df["Method"])]
    x = np.arange(len(method_order))
    outputs: list[Path] = []
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(4.8, 3.5), constrained_layout=True)
        values = [float(df.loc[df["Method"] == m, metric].iloc[0]) for m in method_order]
        ax.plot(x, values, marker="o", color="#4c78a8", linewidth=1.8)
        ax.set_xticks(x)
        ax.set_xticklabels(method_order, rotation=25, ha="right")
        ax.set_ylabel(_metric_label(metric))
        outputs.extend(_save(fig, figures_dir, f"iteration_effect_{metric.lower()}", fmt))
        plt.close(fig)
    return outputs


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / sqrt(2.0 * pi)


def plot_asymptotic_normality(results_dir: Path, figures_dir: Path, fmt: str) -> list[Path]:
    plt = _load_matplotlib()
    raw = _read(results_dir, "experiment5_raw.csv")
    summary = _read(results_dir, "experiment5_summary.csv")
    outputs: list[Path] = []
    # Okabe-Ito / Wong-style colorblind-safe colors, commonly used for journal figures.
    hist_color = "#56B4E9"
    density_color = "#D55E00"

    if raw is not None and {"Functional", "Method", "T"}.issubset(raw.columns):
        part = raw[raw["Method"] == "DMLM-S"].copy()
        if part.empty:
            part = raw.copy()
        functionals = list(dict.fromkeys(part["Functional"]))[:3]
        xs = np.linspace(-4, 4, 300)
        for functional in functionals:
            fig, ax = plt.subplots(figsize=(4.8, 3.5), constrained_layout=True)
            vals = part.loc[part["Functional"] == functional, "T"].replace([np.inf, -np.inf], np.nan).dropna()
            ax.hist(
                vals,
                bins=min(24, max(8, int(np.sqrt(max(len(vals), 1))))),
                density=True,
                alpha=0.78,
                color=hist_color,
                edgecolor="white",
                linewidth=0.7,
            )
            ax.plot(xs, _normal_pdf(xs), color=density_color, linewidth=2.0, label=r"$N(0,1)$")
            ax.set_title(_functional_label(functional))
            ax.set_xlabel("Standardized statistic")
            ax.set_ylabel("Density")
            ax.legend(loc="upper right")
            outputs.extend(_save(fig, figures_dir, f"asymptotic_normality_{functional}", fmt))
            plt.close(fig)

    if summary is not None and {"Functional", "Method", "Coverage"}.issubset(summary.columns):
        fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
        pivot = summary.pivot_table(index="Functional", columns="Method", values="Coverage", aggfunc="mean")
        methods = _ordered_methods(pivot.columns)
        x = np.arange(len(pivot.index))
        width = 0.8 / max(len(methods), 1)
        coverage_colors = {
            "GMLM": "#000000",
            "AMLM": "#0072B2",
            "MLM-S": "#E69F00",
            "MLM-F": "#CC79A7",
            "DMLM-S": "#009E73",
        }
        for j, method in enumerate(methods):
            ax.bar(
                x + (j - (len(methods) - 1) / 2) * width,
                pivot[method],
                width=width,
                label=method,
                color=coverage_colors.get(method, METHOD_COLORS.get(method, None)),
                edgecolor="white",
                linewidth=0.6,
            )
        ax.axhline(0.95, color="#D55E00", linestyle="--", linewidth=1.6, label="Nominal 95%")
        ax.set_xticks(x)
        ax.set_xticklabels([_functional_label(name) for name in pivot.index], rotation=0)
        ax.set_ylabel("Coverage probability")
        ax.set_ylim(0, 1.05)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(methods) + 1, 4),
            borderaxespad=0.0,
        )
        outputs.extend(_save(fig, figures_dir, "coverage_probabilities", fmt))
    return outputs


def plot_local_influence_detection(results_dir: Path, figures_dir: Path, fmt: str) -> list[Path]:
    plt = _load_matplotlib()
    df = _read(results_dir, "experiment6_summary.csv")
    if df is None:
        return []
    metrics = [m for m in ["AUC", "Hit@5", "Hit@10"] if m in df.columns]
    outputs: list[Path] = []
    for structure, part in df.groupby("Structure"):
        labels = list(part["Contamination_type"])
        x = np.arange(len(labels))
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
            ax.bar(x, part[metric], color="#54a24b")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(_metric_label(metric))
            clean_metric = metric.lower().replace("@", "at")
            outputs.extend(_save(fig, figures_dir, f"local_influence_detection_{structure}_{clean_metric}", fmt))
            plt.close(fig)

    if "Rank_contaminated_worker" in df.columns:
        wk = df.dropna(subset=["Rank_contaminated_worker"])
        if not wk.empty:
            fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
            wk = wk.sort_values("Structure")
            labels = wk["Structure"].astype(str)
            ax.bar(np.arange(len(wk)), wk["Rank_contaminated_worker"], color="#f58518")
            ax.set_xticks(np.arange(len(wk)))
            ax.set_xticklabels(labels, rotation=0)
            ax.set_ylabel("Rank of contaminated worker")
            max_rank = max(5.0, float(wk["Rank_contaminated_worker"].max()) + 1.0)
            ax.set_ylim(0, max_rank)
            ax.set_yticks(np.arange(0, int(max_rank) + 1, 1))
            outputs.extend(_save(fig, figures_dir, "worker_contamination_rank", fmt))
    return outputs


def main() -> None:
    args = parse_args()
    results_dir = _resolve_results_dir(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading results from: {results_dir}")
    print(f"Writing Chapter 4 figures to: {figures_dir}")

    outputs: list[Path] = []
    outputs.extend(plot_accuracy_communication_tradeoff(results_dir, figures_dir, args.format))
    outputs.extend(plot_pilot_sensitivity(results_dir, figures_dir, args.format))
    outputs.extend(plot_worker_scalability(results_dir, figures_dir, args.format))
    outputs.extend(plot_iteration_effect(results_dir, figures_dir, args.format))
    outputs.extend(plot_asymptotic_normality(results_dir, figures_dir, args.format))
    outputs.extend(plot_local_influence_detection(results_dir, figures_dir, args.format))

    if outputs:
        print("Generated figures:")
        for path in outputs:
            print(f"  {path}")
    else:
        print("No figures were generated. Check that the selected results directory contains experiment CSV files.")


if __name__ == "__main__":
    main()
