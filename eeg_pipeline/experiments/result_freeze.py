"""Validate and fingerprint the completed experiment results."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from ..channels import MI9_INDICES


EXPECTED_SEEDS = tuple(range(5))
EXPECTED_SUBJECTS = tuple(range(1, 10))
EXPECTED_TEST_TRIALS = 2368
EXPECTED_SPLIT_TRIALS = {"train": 1865, "validation": 463, "test": 2368}


@dataclass(frozen=True)
class ClassificationExperiment:
    group: str
    name: str
    relative_dir: str
    frozen_oracle: bool = False


@dataclass(frozen=True)
class RestorationExperiment:
    name: str
    relative_dir: str
    restored_arrays_dir: str


CLASSIFICATION_EXPERIMENTS = (
    ClassificationExperiment("global_model", "true22", "global_model/true22"),
    ClassificationExperiment("global_model", "direct_mi9", "global_model/direct_mi9"),
    ClassificationExperiment(
        "global_model", "zero_padded_mi9", "global_model/zero_padded_mi9"
    ),
    *(ClassificationExperiment("channel_overlap", f"overlap_{n}", f"channel_overlap/overlap_{n}")
      for n in (0, 2, 4, 7)),
    *(ClassificationExperiment(
        "restoration_benchmarks", name, f"restoration_benchmarks/{name}"
    ) for name in ("spherical_spline", "autoencoder", "ddpm_standard", "autoencoder_eeg_aware")),
    *(ClassificationExperiment(
        "frozen_oracle", name, f"frozen_oracle/{name}", frozen_oracle=True
    ) for name in ("spherical_spline", "autoencoder", "ddpm_standard", "autoencoder_eeg_aware")),
)

RESTORATION_EXPERIMENTS = (
    RestorationExperiment(
        "spherical_spline",
        "spherical_spline/canonical_mi9",
        "spherical_spline/canonical_mi9/arrays",
    ),
    RestorationExperiment(
        "autoencoder",
        "autoencoder/canonical_mi9",
        "autoencoder/canonical_mi9/arrays",
    ),
    RestorationExperiment(
        "ddpm_standard",
        "diffusion/standard_canonical_mi9",
        "diffusion/standard_canonical_mi9/arrays",
    ),
    RestorationExperiment(
        "autoencoder_eeg_aware",
        "autoencoder/eeg_aware_canonical_mi9",
        "autoencoder/eeg_aware_canonical_mi9/arrays",
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required result: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid numeric field {key!r}: {row.get(key)!r}") from error


def _int(row: dict[str, str], key: str) -> int:
    value = _float(row, key)
    if not value.is_integer():
        raise ValueError(f"Expected integer field {key!r}, got {value}")
    return int(value)


def _prediction_columns(frozen_oracle: bool) -> tuple[str, str]:
    if frozen_oracle:
        return "restored_prediction", "restored_accuracy"
    return "predicted_label", "accuracy"


def _validate_prediction_rows(
    path: Path,
    frozen_oracle: bool,
) -> tuple[list[tuple[int, int, int]], float]:
    rows = read_csv(path)
    if len(rows) != EXPECTED_TEST_TRIALS:
        raise ValueError(f"{path}: expected {EXPECTED_TEST_TRIALS} rows, got {len(rows)}")
    prediction_column, _ = _prediction_columns(frozen_oracle)
    keys: list[tuple[int, int, int]] = []
    subjects: set[int] = set()
    correct = 0
    for row in rows:
        subject = _int(row, "subject")
        trial = _int(row, "trial_index")
        label = _int(row, "true_label")
        prediction = _int(row, prediction_column)
        if label not in range(4) or prediction not in range(4):
            raise ValueError(f"{path}: class value outside 0..3")
        subjects.add(subject)
        keys.append((subject, trial, label))
        correct += int(label == prediction)
    if tuple(sorted(subjects)) != EXPECTED_SUBJECTS:
        raise ValueError(f"{path}: unexpected subjects {sorted(subjects)}")
    key_pairs = [(subject, trial) for subject, trial, _ in keys]
    if len(set(key_pairs)) != len(key_pairs):
        raise ValueError(f"{path}: duplicate subject/trial keys")
    return keys, correct / len(rows)


def validate_classification_experiment(
    root: Path,
    experiment: ClassificationExperiment,
    canonical_keys_by_seed: dict[int, list[tuple[int, int, int]]],
) -> dict[str, object]:
    directory = root / "artifacts/experiments/classification" / experiment.relative_dir
    summary = read_csv(directory / "results/experiment_summary.csv")
    seed_rows = read_csv(directory / "results/seed_results.csv")
    subject_rows = read_csv(directory / "results/subject_summary.csv")
    if len(summary) != 1 or len(seed_rows) != 5 or len(subject_rows) != 9:
        raise ValueError(
            f"{directory}: expected 1 summary, 5 seed, and 9 subject rows; got "
            f"{len(summary)}, {len(seed_rows)}, {len(subject_rows)}"
        )
    seed_column = "oracle_seed" if experiment.frozen_oracle else "seed"
    metric_column = "restored_accuracy" if experiment.frozen_oracle else "accuracy"
    seeds = tuple(sorted(_int(row, seed_column) for row in seed_rows))
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"{directory}: expected seeds {EXPECTED_SEEDS}, got {seeds}")
    if _int(summary[0], "n_seeds") != 5 or _int(summary[0], "n_subjects") != 9:
        raise ValueError(f"{directory}: invalid experiment summary counts")

    prediction_accuracies: list[float] = []
    for row in seed_rows:
        seed = _int(row, seed_column)
        if _int(row, "n_test") != EXPECTED_TEST_TRIALS:
            raise ValueError(f"{directory}: seed {seed} has unexpected n_test")
        prediction_path = directory / f"checkpoints/seed_{seed}/predictions.csv"
        keys, recomputed_accuracy = _validate_prediction_rows(
            prediction_path, experiment.frozen_oracle
        )
        if seed in canonical_keys_by_seed and keys != canonical_keys_by_seed[seed]:
            raise ValueError(f"{prediction_path}: labels/order differ from True-22 seed {seed}")
        if experiment.name == "true22" and experiment.group == "global_model":
            canonical_keys_by_seed[seed] = keys
        reported_accuracy = _float(row, metric_column)
        if not np.isclose(recomputed_accuracy, reported_accuracy, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{prediction_path}: recomputed accuracy {recomputed_accuracy} != "
                f"reported {reported_accuracy}"
            )
        prediction_accuracies.append(recomputed_accuracy)

    subject_metric = "restored_accuracy_mean" if experiment.frozen_oracle else "accuracy_mean"
    summary_metric = (
        "restored_accuracy_subject_mean" if experiment.frozen_oracle
        else "accuracy_subject_mean"
    )
    subject_accuracy = np.asarray([_float(row, subject_metric) for row in subject_rows])
    reported_subject_mean = _float(summary[0], summary_metric)
    if not np.isclose(subject_accuracy.mean(), reported_subject_mean, rtol=0.0, atol=1e-12):
        raise ValueError(f"{directory}: subject mean does not reproduce experiment summary")
    return {
        "group": experiment.group,
        "name": experiment.name,
        "frozen_oracle": experiment.frozen_oracle,
        "n_seeds": 5,
        "n_subjects": 9,
        "n_predictions": 5 * EXPECTED_TEST_TRIALS,
        "subject_mean_accuracy": reported_subject_mean,
        "pooled_seed_accuracy_mean": float(np.mean(prediction_accuracies)),
        "relative_dir": directory.relative_to(root).as_posix(),
    }


def validate_restoration_experiment(
    root: Path,
    experiment: RestorationExperiment,
) -> dict[str, object]:
    experiment_dir = root / "artifacts/experiments/restoration" / experiment.relative_dir
    arrays_dir = root / "artifacts/model_inputs/restored" / experiment.restored_arrays_dir
    source_dir = root / "artifacts/preprocessed/bcic2a/canonical_mi9/arrays"
    summaries = read_csv(experiment_dir / "results/reconstruction_summary.csv")
    by_split = {row["split"]: row for row in summaries}
    if set(by_split) != set(EXPECTED_SPLIT_TRIALS):
        raise ValueError(f"{experiment_dir}: unexpected reconstruction summary splits")

    for split, expected_trials in EXPECTED_SPLIT_TRIALS.items():
        row = by_split[split]
        if _int(row, "n_trials") != expected_trials:
            raise ValueError(f"{experiment_dir}: unexpected {split} trial count")
        with np.load(source_dir / f"{split}.npz", allow_pickle=False) as source:
            source_values = {key: source[key] for key in ("x_true22", "x_mi9", "y", "subject", "trial_index")}
        with np.load(arrays_dir / f"{split}.npz", allow_pickle=False) as restored:
            if set(restored.files) != {"x_restored22", "y", "subject", "trial_index"}:
                raise ValueError(f"{arrays_dir}/{split}.npz: unexpected array schema")
            restored_values = {key: restored[key] for key in restored.files}
        signal = restored_values["x_restored22"]
        if signal.shape != source_values["x_true22"].shape or signal.dtype != np.float32:
            raise ValueError(f"{arrays_dir}/{split}.npz: invalid signal shape or dtype")
        if not np.isfinite(signal).all():
            raise ValueError(f"{arrays_dir}/{split}.npz: non-finite restored values")
        for key in ("y", "subject", "trial_index"):
            if not np.array_equal(restored_values[key], source_values[key]):
                raise ValueError(f"{arrays_dir}/{split}.npz: {key} differs from source")
        if not np.array_equal(signal[:, MI9_INDICES, :], source_values["x_mi9"]):
            raise ValueError(f"{arrays_dir}/{split}.npz: observed MI-9 was not copied exactly")

    test = by_split["test"]
    result: dict[str, object] = {
        "name": experiment.name,
        "n_test": EXPECTED_TEST_TRIALS,
        "missing13_mse": _float(test, "missing13_mse"),
        "missing13_mae": _float(test, "missing13_mae"),
        "missing13_correlation": _float(test, "missing13_correlation"),
        "relative_dir": experiment_dir.relative_to(root).as_posix(),
        "restored_arrays_dir": arrays_dir.relative_to(root).as_posix(),
    }
    for key in ("missing13_log_bandpower_mse", "missing13_spatial_correlation_mse"):
        value = test.get(key, "")
        result[key] = None if value == "" else float(value)
    return result


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for path in paths:
        candidates = path.rglob("*") if path.is_dir() else (path,)
        for candidate in candidates:
            if candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                yield candidate


def _artifact_roots(root: Path) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for experiment in CLASSIFICATION_EXPERIMENTS:
        result.append((
            "classification_result",
            root / "artifacts/experiments/classification" / experiment.relative_dir,
        ))
    for experiment in RESTORATION_EXPERIMENTS:
        result.extend((
            ("restoration_result", root / "artifacts/experiments/restoration" / experiment.relative_dir),
            ("restored_input", root / "artifacts/model_inputs/restored" / experiment.restored_arrays_dir),
        ))
    result.extend((
        ("source_input", root / "artifacts/preprocessed/bcic2a/canonical_mi9/arrays"),
        ("source_input", root / "artifacts/preprocessed/bcic2a/canonical_mi9/normalization"),
        ("source_input", root / "artifacts/preprocessed/bcic2a/canonical_mi9/preprocessing_summary.json"),
        ("source_input", root / "artifacts/preprocessed/bcic2a/canonical_mi9/split_manifest.csv"),
        ("source_input", root / "artifacts/preprocessed/bcic2a/channel_overlap/arrays"),
        ("source_input", root / "artifacts/preprocessed/bcic2a/channel_overlap/preprocessing_summary.json"),
        ("source_input", root / "artifacts/preprocessed/bcic2a/channel_overlap/split_manifest.csv"),
    ))
    return result


def _source_snapshot_paths(root: Path) -> list[Path]:
    paths = list((root / "eeg_pipeline").rglob("*.py"))
    paths.extend((root / "scripts").glob("*.py"))
    paths.extend((root / "tests").glob("*.py"))
    paths.extend((root / "configs").rglob("*.yaml"))
    paths.extend(root / name for name in (
        "requirements-gpu-cu124.txt", "requirements-preprocessing.txt"
    ))
    return sorted(path for path in paths if path.is_file())


def _git_metadata(root: Path) -> dict[str, object]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ("git", *args), cwd=root, text=True, capture_output=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "working_tree_clean": status == "",
        "status": status,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_results_freeze(root: Path, output_dir: Path) -> dict[str, object]:
    """Validate completed experiments and write an immutable-reference manifest."""

    root = root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Freeze output already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_keys_by_seed: dict[int, list[tuple[int, int, int]]] = {}
    ordered_experiments = sorted(
        CLASSIFICATION_EXPERIMENTS,
        key=lambda item: not (item.group == "global_model" and item.name == "true22"),
    )
    classification_rows = [
        validate_classification_experiment(root, experiment, canonical_keys_by_seed)
        for experiment in ordered_experiments
    ]
    restoration_rows = [
        validate_restoration_experiment(root, experiment)
        for experiment in RESTORATION_EXPERIMENTS
    ]

    file_rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for role, artifact_root in _artifact_roots(root):
        if not artifact_root.exists():
            raise FileNotFoundError(f"Missing freeze target: {artifact_root}")
        for path in _iter_files((artifact_root,)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            file_rows.append({
                "role": role,
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    for path in _source_snapshot_paths(root):
        file_rows.append({
            "role": "source_snapshot",
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    file_rows.sort(key=lambda row: (str(row["role"]), str(row["path"])))

    manifest = {
        "schema_version": 1,
        "freeze_id": output_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": "passed",
        "protocol": {
            "dataset": "BCI Competition IV Dataset 2a",
            "subjects": list(EXPECTED_SUBJECTS),
            "train": "pooled Session-1 train",
            "validation": "Session-1 validation",
            "test": "held-out Session-2",
            "classifier_seeds": list(EXPECTED_SEEDS),
            "test_trials": EXPECTED_TEST_TRIALS,
        },
        "primary_endpoints": {
            "classification": "subject-mean matched-training Session-2 accuracy",
            "signal": [
                "missing13_log_bandpower_mse",
                "missing13_spatial_correlation_mse",
            ],
        },
        "counts": {
            "classification_experiments": len(classification_rows),
            "restoration_experiments": len(restoration_rows),
            "classification_predictions": sum(
                int(row["n_predictions"]) for row in classification_rows
            ),
            "fingerprinted_files": len(file_rows),
            "fingerprinted_bytes": sum(int(row["size_bytes"]) for row in file_rows),
        },
        "classification": classification_rows,
        "restoration": restoration_rows,
        "git": _git_metadata(root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    _write_csv(output_dir / "classification_metrics.csv", classification_rows)
    _write_csv(output_dir / "restoration_metrics.csv", restoration_rows)
    _write_csv(output_dir / "files.csv", file_rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = f"""# Results freeze: {output_dir.name}

- Status: **PASSED**
- Created (UTC): `{manifest['created_at_utc']}`
- Classification experiments: `{len(classification_rows)}`
- Restoration experiments: `{len(restoration_rows)}`
- Validated prediction rows: `{manifest['counts']['classification_predictions']:,}`
- Fingerprinted files: `{len(file_rows):,}`
- Fingerprinted bytes: `{manifest['counts']['fingerprinted_bytes']:,}`

The freeze verifies seeds `0–4`, nine subjects, 2,368 held-out Session-2 trials per
classifier seed, prediction-derived accuracy, shared label/trial order, restoration
array shape/dtype/finite values, metadata identity, and exact observed MI-9 copying.

`files.csv` records SHA-256 fingerprints without copying the large arrays or checkpoints.
Changing or deleting a listed artifact after this date makes it differ from this freeze.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return manifest


def verify_results_freeze(
    root: Path,
    manifest_path: Path,
    *,
    artifacts_only: bool = False,
) -> dict[str, int]:
    """Re-hash frozen files and fail if any selected file changed or disappeared."""

    root = root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_rows = read_csv(manifest_path.parent / "files.csv")
    expected_count = int(manifest["counts"]["fingerprinted_files"])
    if len(file_rows) != expected_count:
        raise ValueError(
            f"files.csv contains {len(file_rows)} entries; manifest expects {expected_count}"
        )
    if artifacts_only:
        file_rows = [row for row in file_rows if row["role"] != "source_snapshot"]
    checked_bytes = 0
    for row in file_rows:
        path = root / row["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Frozen file is missing: {path}")
        expected_size = int(row["size_bytes"])
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Frozen file size changed: {path} ({actual_size} != {expected_size})"
            )
        actual_digest = sha256_file(path)
        if actual_digest != row["sha256"]:
            raise ValueError(f"Frozen file SHA-256 changed: {path}")
        checked_bytes += actual_size
    return {"checked_files": len(file_rows), "checked_bytes": checked_bytes}
