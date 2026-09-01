"""Validate and fingerprint all completed manuscript experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_pipeline.experiments.result_freeze import (
    create_results_freeze,
    verify_results_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/manifests/results_freeze_2026-08-11",
        help="New, empty directory for the manifest and validation report.",
    )
    parser.add_argument(
        "--verify-manifest",
        type=Path,
        help="Re-hash an existing freeze instead of creating a new one.",
    )
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="With --verify-manifest, ignore expected source-code changes from later stages.",
    )
    args = parser.parse_args()
    if args.verify_manifest is not None:
        manifest_path = args.verify_manifest
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
        result = verify_results_freeze(
            PROJECT_ROOT, manifest_path, artifacts_only=args.artifacts_only
        )
        print(
            f"Results freeze unchanged: {result['checked_files']} files, "
            f"{result['checked_bytes']} bytes."
        )
        return
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    manifest = create_results_freeze(PROJECT_ROOT, output_dir)
    print(f"Results freeze passed: {output_dir}")
    print(
        f"Validated {manifest['counts']['classification_experiments']} classification and "
        f"{manifest['counts']['restoration_experiments']} restoration experiments."
    )


if __name__ == "__main__":
    main()
