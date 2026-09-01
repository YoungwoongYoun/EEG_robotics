"""Run one pooled-global TCFormer input experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_pipeline.experiments.classification.config import load_experiment_config
from eeg_pipeline.experiments.classification.runner import dry_run, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="One input experiment YAML, e.g. configs/classification/baselines/true22.yaml",
    )
    parser.add_argument("--subjects", nargs="+", type=int, help="Dry-run subset only")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", help="Runtime device override, e.g. cuda:1 or cpu")
    parser.add_argument("--dry-run", action="store_true", help="Validate without training")
    parser.add_argument("--overwrite", action="store_true", help="Replace selected completed seeds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, PROJECT_ROOT)
    if args.dry_run:
        dry_run(config, subjects=args.subjects, device_override=args.device)
        return
    output = run_experiment(
        config,
        subjects=args.subjects,
        seeds=args.seeds,
        device_override=args.device,
        overwrite=args.overwrite,
    )
    print(f"Experiment complete: {output}")


if __name__ == "__main__":
    main()
