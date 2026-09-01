"""Run one canonical MI-9 to 22-channel restoration experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_pipeline.experiments.restoration.config import load_restoration_config
from eeg_pipeline.experiments.restoration.runner import (
    dry_run,
    run_all,
    run_inference,
    run_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("train", "infer", "all"),
        default="all",
        help="Learned methods can train and infer separately; interpolation uses infer/all.",
    )
    parser.add_argument("--device", help="Runtime override such as cuda:1 or cpu")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Restart training or replace generated split files instead of resuming/skipping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_restoration_config(args.config, PROJECT_ROOT)
    if args.dry_run:
        dry_run(config, args.device)
        return
    if args.stage == "train":
        output = run_training(config, args.device, overwrite=args.overwrite)
    elif args.stage == "infer":
        output = run_inference(config, args.device, overwrite=args.overwrite)
    else:
        output = run_all(config, args.device, overwrite=args.overwrite)
    print(f"Restoration stage complete: {output}")


if __name__ == "__main__":
    main()
