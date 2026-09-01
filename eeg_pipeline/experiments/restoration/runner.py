"""Orchestrate training, classifier-input generation, validation, and metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import RestorationConfig, SPLITS
from .data import (
    RestorationDataRepository,
    RestorationSplit,
    validate_restored_split,
    write_restored_split,
)
from .interpolation import restore_spherical_spline
from .metrics import reconstruction_metrics, write_metric_artifacts
from .training import (
    configure_device,
    dry_run_learned,
    infer_learned,
    train_model,
)


def _write_resolved_config(config: RestorationConfig) -> None:
    config.output.experiment_dir.mkdir(parents=True, exist_ok=True)
    path = config.output.experiment_dir / "resolved_config.json"
    path.write_text(
        json.dumps(config.as_serializable_dict(), indent=2),
        encoding="utf-8",
    )


def dry_run(config: RestorationConfig, device_override: str | None = None) -> None:
    repository = RestorationDataRepository(config)
    if config.method == "spherical_spline":
        restored = restore_spherical_spline(
            repository.splits["train"].subset(2),
            config.source.normalization_dir,
            alpha=float(config.model.get("alpha", 1e-5)),
        )
        validate_restored_split(restored, repository.splits["train"].subset(2), "dry_run")
        print(
            f"DRY RUN {config.name}: method=spherical_spline, "
            f"output={restored.shape}, observed9=exact, writes=none"
        )
        return
    device = configure_device(config, device_override)
    dry_run_learned(config, repository.splits["train"], device)


def run_training(
    config: RestorationConfig,
    device_override: str | None = None,
    *,
    overwrite: bool = False,
) -> Path | None:
    repository = RestorationDataRepository(config)
    _write_resolved_config(config)
    if config.method == "spherical_spline":
        print(f"NO TRAINING required: {config.name}")
        return None
    device = configure_device(config, device_override)
    return train_model(
        config,
        repository.splits["train"],
        repository.splits["validation"],
        device,
        overwrite=overwrite,
    )


def _load_existing_output(
    path: Path,
    config: RestorationConfig,
    source: RestorationSplit,
    split: str,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        required = {config.output.array_key, "y", "subject", "trial_index"}
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
        for key in ("y", "subject", "trial_index"):
            np.testing.assert_array_equal(payload[key], getattr(source, key))
        restored = payload[config.output.array_key]
    validate_restored_split(restored, source, split)
    return restored


def run_inference(
    config: RestorationConfig,
    device_override: str | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    repository = RestorationDataRepository(config)
    _write_resolved_config(config)
    config.output.arrays_dir.mkdir(parents=True, exist_ok=True)
    device = None if config.method == "spherical_spline" else configure_device(
        config, device_override
    )
    split_results = {}
    timing_rows = []
    for split in SPLITS:
        source = repository.splits[split]
        output_path = config.output.arrays_dir / f"{split}.npz"
        if output_path.is_file() and not overwrite:
            restored = _load_existing_output(output_path, config, source, split)
            elapsed = None
            print(f"SKIP validated restoration output: {output_path}")
        else:
            print(f"INFER {config.name} split={split} trials={source.y.size}", flush=True)
            start = time.perf_counter()
            if config.method == "spherical_spline":
                restored = restore_spherical_spline(
                    source,
                    config.source.normalization_dir,
                    alpha=float(config.model.get("alpha", 1e-5)),
                )
                elapsed = time.perf_counter() - start
            else:
                assert device is not None
                restored, elapsed = infer_learned(config, source, device)
            validate_restored_split(restored, source, split)
            write_restored_split(
                output_path, config.output.array_key, restored, source
            )
            print(f"WROTE {output_path}", flush=True)
        aggregate, channels, subjects = reconstruction_metrics(source, restored)
        split_results[split] = (aggregate, channels, subjects)
        timing_rows.append({
            "split": split,
            "trials": int(source.y.size),
            "elapsed_seconds": elapsed,
            "ms_per_trial": None if elapsed is None else 1000.0 * elapsed / source.y.size,
            "reused_existing_output": elapsed is None,
        })
    write_metric_artifacts(config.output.experiment_dir, config.name, split_results)
    metadata = {
        "schema_version": 1,
        "name": config.name,
        "method": config.method,
        "array_key": config.output.array_key,
        "observed_channels": "hard-copied canonical MI-9",
        "primary_metrics": "missing 13 channels",
        "timing": timing_rows,
        "config": config.as_serializable_dict(),
    }
    (config.output.experiment_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return config.output.experiment_dir


def run_all(
    config: RestorationConfig,
    device_override: str | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    run_training(config, device_override, overwrite=overwrite)
    return run_inference(config, device_override, overwrite=overwrite)
