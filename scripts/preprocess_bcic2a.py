"""CLI entry point for leakage-controlled BCIC IV-2a preprocessing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_pipeline.dataset import (
    ChannelMontage,
    PreprocessingConfig,
    build_channel_overlap_dataset,
    build_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocessing" / "canonical_mi9.yaml",
    )
    parser.add_argument("--raw-dir", type=Path, help="Override raw GDF directory")
    parser.add_argument(
        "--labels-dir", type=Path, help="Override evaluation-label directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override artifact output directory"
    )
    parser.add_argument(
        "--no-torch",
        action="store_true",
        help="Write NPZ files only; useful outside the PyTorch GPU environment",
    )
    return parser.parse_args()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise TypeError(f"Expected a YAML mapping in {args.config}")

    raw_dir = args.raw_dir or values.pop("raw_dir")
    labels_dir = args.labels_dir or values.pop("labels_dir")
    output_dir = args.output_dir or values.pop("output_dir")
    montage_values = values.pop("montages", None)
    values["subjects"] = tuple(values.get("subjects", range(1, 10)))
    if args.no_torch:
        values["export_torch"] = False

    config = PreprocessingConfig(
        raw_dir=_project_path(raw_dir),
        labels_dir=_project_path(labels_dir),
        output_dir=_project_path(output_dir),
        **values,
    )
    if montage_values is None:
        summary = build_dataset(config)
    else:
        if not isinstance(montage_values, dict):
            raise TypeError("montages must be a YAML mapping keyed by montage id")
        montages = tuple(
            ChannelMontage(
                id=str(montage_id),
                label=str(montage["label"]),
                channels=tuple(str(channel) for channel in montage["channels"]),
                expected_mi9_overlap=int(montage["expected_mi9_overlap"]),
            )
            for montage_id, montage in montage_values.items()
        )
        summary = build_channel_overlap_dataset(config, montages)
    print(f"Preprocessing complete: {config.output_dir}")
    for split, shapes in summary["split_shapes"].items():
        input_shapes = " ".join(
            f"{key}={shape}" for key, shape in shapes.items() if key.startswith("x_")
        )
        print(f"  {split:10s} {input_shapes}")


if __name__ == "__main__":
    main()
