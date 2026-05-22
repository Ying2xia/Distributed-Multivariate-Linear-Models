#!/usr/bin/env python3
"""Run DMLM-S real-data applications."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmlm.download_data import download_beijing, download_sarcos
from dmlm.realdata import run_beijing, run_sarcos


def _float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _str_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="If no dataset is specified, both Beijing and SARCOS are run using the default local paths.",
    )
    sub = parser.add_subparsers(dest="dataset", required=False)

    beijing = sub.add_parser("beijing")
    beijing.add_argument("--path", default="data/beijing", help="CSV file or directory of station CSV files.")
    beijing.add_argument("--download", action="store_true", help="Download the dataset automatically if needed.")
    beijing.add_argument("--pilot-proportions", default="0.002,0.005,0.01,0.02")
    beijing.add_argument("--primary-pilot", type=float, default=0.01)
    beijing.add_argument("--rounds", type=int, default=3)
    beijing.add_argument("--seed", type=int, default=20260509)
    beijing.add_argument("--output-dir", default="results")

    sarcos = sub.add_parser("sarcos")
    sarcos.add_argument("--train", default="data/sarcos/sarcos_inv.csv")
    sarcos.add_argument("--test", default="data/sarcos/sarcos_inv_test.csv")
    sarcos.add_argument("--download", action="store_true", help="Download the dataset automatically if needed.")
    sarcos.add_argument("--K-grid", default="20,50")
    sarcos.add_argument("--storage-grid", default="random,motion")
    sarcos.add_argument("--pilot-proportions", default="0.005,0.01,0.02")
    sarcos.add_argument("--primary-pilot", type=float, default=0.01)
    sarcos.add_argument("--rounds", type=int, default=3)
    sarcos.add_argument("--seed", type=int, default=20260509)
    sarcos.add_argument("--output-dir", default="results")
    return parser.parse_args()


def _run_beijing_job(
    *,
    path: str | Path = "data/beijing",
    download: bool = False,
    pilot_proportions: str = "0.002,0.005,0.01,0.02",
    primary_pilot: float = 0.01,
    rounds: int = 3,
    seed: int = 20260509,
    output_dir: str | Path = "results",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = Path(path)
    if download or not path.exists():
        path = download_beijing(path)
    performance, main_results, pilot_sensitivity, obs, wk = run_beijing(
        path,
        pilot_proportions=_float_tuple(pilot_proportions),
        primary_pilot=primary_pilot,
        rounds=rounds,
        seed=seed,
    )
    performance.to_csv(output_dir / "beijing_performance_grid.csv", index=False)
    main_results.to_csv(output_dir / "beijing_main_results.csv", index=False)
    pilot_sensitivity.to_csv(output_dir / "beijing_pilot_sensitivity.csv", index=False)
    obs.to_csv(output_dir / "beijing_observation_influence.csv", index=False)
    wk.to_csv(output_dir / "beijing_worker_influence.csv", index=False)
    print("Beijing main results")
    print(main_results.to_string(index=False))
    print("\nBeijing pilot sensitivity")
    print(pilot_sensitivity.to_string(index=False))
    print("\nBeijing top influential station-hour observations")
    print(obs.head(10).to_string(index=False))
    print("\nBeijing worker influence")
    print(wk.head(12).to_string(index=False))


def _run_sarcos_job(
    *,
    train: str | Path = "data/sarcos/sarcos_inv.csv",
    test: str | Path = "data/sarcos/sarcos_inv_test.csv",
    download: bool = False,
    K_grid: str = "20,50",
    storage_grid: str = "random,motion",
    pilot_proportions: str = "0.005,0.01,0.02",
    primary_pilot: float = 0.01,
    rounds: int = 3,
    seed: int = 20260509,
    output_dir: str | Path = "results",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(train)
    test_path = Path(test)
    if download or not train_path.exists() or not test_path.exists():
        train_path, test_path = download_sarcos(train_path.parent)
    performance, main_results, pilot_sensitivity, wk = run_sarcos(
        train_path,
        test_path,
        K_grid=_int_tuple(K_grid),
        storage_grid=_str_tuple(storage_grid),
        pilot_proportions=_float_tuple(pilot_proportions),
        primary_pilot=primary_pilot,
        rounds=rounds,
        seed=seed,
    )
    performance.to_csv(output_dir / "sarcos_performance_grid.csv", index=False)
    main_results.to_csv(output_dir / "sarcos_main_results.csv", index=False)
    pilot_sensitivity.to_csv(output_dir / "sarcos_pilot_sensitivity.csv", index=False)
    wk.to_csv(output_dir / "sarcos_motion_worker_influence.csv", index=False)
    print("SARCOS main results")
    print(main_results.to_string(index=False))
    print("\nSARCOS pilot sensitivity")
    print(pilot_sensitivity.to_string(index=False))
    print("\nSARCOS motion-regime worker influence")
    print(wk.head(20).to_string(index=False))


def main() -> None:
    args = parse_args()

    if args.dataset is None:
        print("No dataset specified. Running both Beijing and SARCOS with default local paths.")
        _run_beijing_job()
        print("\n" + "=" * 80 + "\n")
        _run_sarcos_job()
    elif args.dataset == "beijing":
        _run_beijing_job(
            path=args.path,
            download=args.download,
            pilot_proportions=args.pilot_proportions,
            primary_pilot=args.primary_pilot,
            rounds=args.rounds,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.dataset == "sarcos":
        _run_sarcos_job(
            train=args.train,
            test=args.test,
            download=args.download,
            K_grid=args.K_grid,
            storage_grid=args.storage_grid,
            pilot_proportions=args.pilot_proportions,
            primary_pilot=args.primary_pilot,
            rounds=args.rounds,
            seed=args.seed,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
