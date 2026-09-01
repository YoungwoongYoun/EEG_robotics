"""Run one EEG-aware AE loss ablation through restoration and classification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_pipeline.experiments.classification.config import load_experiment_config
from eeg_pipeline.experiments.classification.runner import dry_run as classification_dry_run
from eeg_pipeline.experiments.classification.runner import run_experiment
from eeg_pipeline.experiments.frozen_oracle import run_frozen_oracle
from eeg_pipeline.experiments.restoration.config import load_restoration_config
from eeg_pipeline.experiments.restoration.runner import dry_run as restoration_dry_run
from eeg_pipeline.experiments.restoration.runner import run_all as run_restoration


VARIANTS = {
    "bandpower": "autoencoder_bandpower",
    "spatial": "autoencoder_spatial",
}


def _write_completion(variant: str, stage: str, restoration, classification) -> Path:
    output_dir = (
        PROJECT_ROOT
        / "artifacts/experiments/ablation/autoencoder_eeg_aware/runs"
        / variant
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stage}_complete.json"
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "variant": variant,
        "stage": stage,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "restoration_experiment": str(restoration.output.experiment_dir),
        "restored_arrays": str(restoration.output.arrays_dir),
        "frozen_oracle": str(
            PROJECT_ROOT
            / "artifacts/experiments/classification/frozen_oracle"
            / classification.input.id
        ),
        "matched_classification": str(
            classification.output_dir / classification.name
        ),
    }
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _configs(variant: str):
    name = VARIANTS[variant]
    restoration = load_restoration_config(
        PROJECT_ROOT / f"configs/restoration/{name}.yaml", PROJECT_ROOT
    )
    classification = load_experiment_config(
        PROJECT_ROOT / f"configs/classification/restored/{name}.yaml", PROJECT_ROOT
    )
    oracle = load_experiment_config(
        PROJECT_ROOT / "configs/classification/baselines/true22.yaml", PROJECT_ROOT
    )
    if restoration.output.arrays_dir != classification.input.arrays_dir:
        raise ValueError(f"{name}: restoration output and classifier input differ")
    return restoration, classification, oracle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--device", required=True, help="cuda:0, cuda:1, or cpu")
    parser.add_argument(
        "--stage",
        choices=("restoration", "frozen", "matched", "all"),
        default="all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the selected stage without training or writing results.",
    )
    args = parser.parse_args()
    restoration, classification, oracle = _configs(args.variant)

    if args.dry_run:
        if args.stage in {"restoration", "all"}:
            restoration_dry_run(restoration, args.device)
        if args.stage in {"frozen", "matched"}:
            if not all(
                (classification.input.arrays_dir / f"{split}.npz").is_file()
                for split in ("train", "validation", "test")
            ):
                raise FileNotFoundError(
                    "Restored arrays are required to dry-run frozen or matched stages"
                )
            if args.stage == "frozen":
                run_frozen_oracle(
                    oracle, classification, PROJECT_ROOT,
                    device_override=args.device, dry_run=True,
                )
            else:
                classification_dry_run(classification, device_override=args.device)
        if args.stage == "all":
            print(
                "PREFLIGHT complete for restoration. Frozen and matched dry-runs are "
                "deferred until restored arrays exist."
            )
        return

    if args.stage in {"restoration", "all"}:
        print(f"PIPELINE {args.variant}: restoration", flush=True)
        run_restoration(restoration, args.device)
    if args.stage in {"frozen", "all"}:
        print(f"PIPELINE {args.variant}: frozen oracle", flush=True)
        run_frozen_oracle(
            oracle, classification, PROJECT_ROOT, device_override=args.device
        )
    if args.stage in {"matched", "all"}:
        print(f"PIPELINE {args.variant}: matched TCFormer", flush=True)
        run_experiment(classification, device_override=args.device)
    completion = _write_completion(
        args.variant, args.stage, restoration, classification
    )
    print(f"PIPELINE COMPLETE: {args.variant} ({completion})", flush=True)


if __name__ == "__main__":
    main()
