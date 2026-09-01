"""Generate manuscript-ready tables, statistics, and figures from saved results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_pipeline.experiments.manuscript_assets import build_manuscript_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manuscript_assets"),
        help="Dedicated output folder inside the project root (default: manuscript_assets)",
    )
    args = parser.parse_args()
    output = build_manuscript_assets(PROJECT_ROOT, args.output_dir)
    print(f"Manuscript assets complete: {output}")


if __name__ == "__main__":
    main()

