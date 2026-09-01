"""Evaluate one restored Session-2 input with frozen True-22 TCFormer seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_pipeline.experiments.classification.config import load_experiment_config
from eeg_pipeline.experiments.frozen_oracle import run_frozen_oracle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    oracle = load_experiment_config(
        PROJECT_ROOT / "configs/classification/baselines/true22.yaml", PROJECT_ROOT
    )
    restored = load_experiment_config(args.input_config, PROJECT_ROOT)
    output = run_frozen_oracle(
        oracle,
        restored,
        PROJECT_ROOT,
        device_override=args.device,
        seeds=None if args.seeds is None else tuple(args.seeds),
        dry_run=args.dry_run,
    )
    print(f"Frozen-oracle complete: {output}")


if __name__ == "__main__":
    main()
