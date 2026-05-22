#!/usr/bin/env python3
"""Run DMLM-S simulation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmlm.experiments import (
    experiment1,
    experiment2,
    experiment3,
    experiment4,
    experiment5,
    experiment6,
    paper_config,
    quick_config,
    write_outputs,
)


EXPERIMENTS = {
    "1": experiment1,
    "2": experiment2,
    "3": experiment3,
    "4": experiment4,
    "5": experiment5,
    "6": experiment6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["1", "2", "3", "4", "5", "6", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="Use a small configuration for code checks.")
    parser.add_argument("--paper", action="store_true", help="Use the paper-scale default configuration. This is the default.")
    parser.add_argument("--replications", type=int, default=None, help="Override Monte Carlo replications.")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reps = args.replications if args.replications is not None else (5 if args.quick else 200)
    if args.quick:
        config = quick_config(replications=reps, seed=args.seed)
    else:
        config = paper_config(replications=reps, seed=args.seed)
    config = config.__class__(**{**config.__dict__, "show_progress": not args.no_progress})

    selected = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    output_dir = Path(args.output_dir)
    for key in selected:
        print(f"Running experiment {key} with N={config.N}, K={config.K}, W={config.replications}")
        raw, summary = EXPERIMENTS[key](config)
        raw_path, summary_path = write_outputs(raw, summary, output_dir=output_dir, stem=f"experiment{key}")
        print(f"  raw:     {raw_path}")
        print(f"  summary: {summary_path}")
        max_rows = 80
        if len(summary) <= max_rows:
            print(summary.to_string(index=False))
        else:
            print(summary.head(max_rows).to_string(index=False))
            print(f"  ... {len(summary) - max_rows} more rows in {summary_path}")


if __name__ == "__main__":
    main()
