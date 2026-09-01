"""Run leakage-controlled task-relevant Session-2 signal analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_pipeline.experiments.signal_analysis import (
    load_signal_analysis_config,
    run_signal_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/analysis/task_relevant_signal.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    config = load_signal_analysis_config(args.config, PROJECT_ROOT)
    output = run_signal_analysis(
        config, dry_run=args.dry_run, overwrite=args.overwrite,
        skip_figures=args.skip_figures,
    )
    print(f"Signal analysis complete: {output}")


if __name__ == "__main__":
    main()
