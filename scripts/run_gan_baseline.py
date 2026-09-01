"""Run the conditional WGAN-GP restoration baseline end to end."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_pipeline.experiments.classification.config import load_experiment_config
from eeg_pipeline.experiments.classification.runner import run_experiment
from eeg_pipeline.experiments.frozen_oracle import run_frozen_oracle
from eeg_pipeline.experiments.restoration.config import load_restoration_config
from eeg_pipeline.experiments.restoration.runner import (
    dry_run,
    run_inference,
    run_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="cuda:0, cuda:1, or cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    restoration = load_restoration_config(
        PROJECT_ROOT / "configs/restoration/wgan_gp.yaml", PROJECT_ROOT
    )
    classification = load_experiment_config(
        PROJECT_ROOT / "configs/classification/restored/wgan_gp.yaml", PROJECT_ROOT
    )
    oracle = load_experiment_config(
        PROJECT_ROOT / "configs/classification/baselines/true22.yaml", PROJECT_ROOT
    )
    if restoration.output.arrays_dir != classification.input.arrays_dir:
        raise ValueError("WGAN-GP restoration output and classifier input differ")
    if args.dry_run:
        dry_run(restoration, args.device)
        print(
            "PREFLIGHT complete for WGAN-GP restoration. Frozen and matched dry-runs "
            "are deferred until restored arrays exist."
        )
        return

    print("PIPELINE WGAN-GP: restoration", flush=True)
    run_training(restoration, args.device)
    training_complete = (
        restoration.output.experiment_dir
        / f"checkpoints/seed_{restoration.seed}/training_complete.json"
    )
    training_status = json.loads(training_complete.read_text(encoding="utf-8"))
    if not training_status.get("validation_plateau_reached", False):
        raise RuntimeError(
            "WGAN-GP did not reach the validation plateau. Inference and "
            "classification were intentionally not started; inspect history.csv."
        )
    run_inference(restoration, args.device)
    print("PIPELINE WGAN-GP: frozen oracle", flush=True)
    run_frozen_oracle(
        oracle, classification, PROJECT_ROOT, device_override=args.device
    )
    print("PIPELINE WGAN-GP: matched TCFormer", flush=True)
    run_experiment(classification, device_override=args.device)
    completion = restoration.output.experiment_dir / "PIPELINE_COMPLETE.json"
    temporary = completion.with_name(f".{completion.name}.tmp")
    temporary.write_text(json.dumps({
        "name": restoration.name,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "restoration_experiment": str(restoration.output.experiment_dir),
        "restored_arrays": str(restoration.output.arrays_dir),
        "frozen_oracle": str(
            PROJECT_ROOT / "artifacts/experiments/classification/frozen_oracle/wgan_gp"
        ),
        "matched_classification": str(classification.output_dir / classification.name),
    }, indent=2) + "\n", encoding="utf-8")
    temporary.replace(completion)
    print(f"PIPELINE COMPLETE: WGAN-GP ({completion})", flush=True)


if __name__ == "__main__":
    main()
