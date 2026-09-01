"""Run the formal batch-1 restoration plus TCFormer latency benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_pipeline.experiments.latency import (
    load_latency_benchmark_config,
    run_latency_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/analysis/latency_benchmark.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--warmup-trials", type=int)
    parser.add_argument("--trials-per-subject-class", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_latency_benchmark_config(args.config, PROJECT_ROOT)
    output_dir = args.output_dir
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output = run_latency_benchmark(
        config,
        PROJECT_ROOT,
        device_override=args.device,
        method_ids=None if args.methods is None else tuple(args.methods),
        warmup_trials=args.warmup_trials,
        trials_per_subject_class=args.trials_per_subject_class,
        output_dir_override=output_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(f"Latency benchmark output: {output}")


if __name__ == "__main__":
    main()
