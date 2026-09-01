"""Independent run artifacts and one study-wide manuscript-oriented record."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from .config import ExperimentConfig
from .training import RunArtifacts

SUMMARY_METRICS = ("accuracy", "macro_f1", "cohen_kappa", "inference_ms_per_trial")


def _temporary_path(path: Path) -> Path:
    """Return a process-unique sibling used for atomic report replacement."""

    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _atomic_text(path: Path, text: str) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = _temporary_path(path)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    temporary = _temporary_path(path)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def config_digest(config: ExperimentConfig) -> str:
    encoded = json.dumps(
        config.as_serializable_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare_experiment_directory(config: ExperimentConfig) -> Path:
    experiment_dir = config.output_dir / config.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = experiment_dir / "experiment_metadata.json"
    digest = config_digest(config)
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != digest:
            raise RuntimeError(
                f"Configuration changed for existing experiment {experiment_dir}. "
                "Use a new experiment name or remove the old experiment directory."
            )
        return experiment_dir
    metadata = {
        "experiment": config.as_serializable_dict(),
        "config_sha256": digest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }
    _atomic_json(metadata_path, metadata)
    _atomic_json(experiment_dir / "resolved_config.json", config.as_serializable_dict())
    return experiment_dir


def run_directory(experiment_dir: Path, seed: int) -> Path:
    return experiment_dir / "checkpoints" / f"seed_{seed}"


def is_complete(path: Path) -> bool:
    return (path / "metrics.json").is_file()


def save_run(path: Path, artifacts: RunArtifacts) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts.checkpoint, path / "best_model.pt")
    _write_csv(path / "history.csv", artifacts.history, list(artifacts.history[0]))
    _write_csv(
        path / "subject_metrics.csv",
        artifacts.subject_metrics,
        list(artifacts.subject_metrics[0]),
    )
    probabilities = artifacts.predictions["probabilities"]
    prediction_rows = []
    for index in range(probabilities.shape[0]):
        row = {
            "subject": int(artifacts.predictions["subject"][index]),
            "subject_id": f"A{int(artifacts.predictions['subject'][index]):02d}",
            "trial_index": int(artifacts.predictions["trial_index"][index]),
            "true_label": int(artifacts.predictions["true_label"][index]),
            "predicted_label": int(artifacts.predictions["predicted_label"][index]),
        }
        row.update({
            f"probability_class_{class_id}": float(value)
            for class_id, value in enumerate(probabilities[index])
        })
        prediction_rows.append(row)
    _write_csv(path / "predictions.csv", prediction_rows, list(prediction_rows[0]))
    confusion_rows = [
        {
            "true_label": class_id,
            **{
                f"predicted_{prediction_id}": int(value)
                for prediction_id, value in enumerate(row)
            },
        }
        for class_id, row in enumerate(artifacts.confusion)
    ]
    _write_csv(path / "confusion_matrix.csv", confusion_rows, list(confusion_rows[0]))
    _atomic_json(path / "metrics.json", artifacts.metrics)


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def _completed_seed_metrics(experiment_dir: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in experiment_dir.glob("checkpoints/seed_*/metrics.json")
    ]
    return sorted(rows, key=lambda row: int(row["seed"]))


def _completed_subject_metrics(experiment_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in experiment_dir.glob("checkpoints/seed_*/subject_metrics.csv"):
        for row in _read_csv(path):
            rows.append({
                **row,
                "subject": int(row["subject"]),
                "seed": int(row["seed"]),
                "n_test": int(row["n_test"]),
                **{metric: float(row[metric]) for metric in SUMMARY_METRICS},
            })
    return sorted(rows, key=lambda row: (int(row["subject"]), int(row["seed"])))


def write_experiment_report(experiment_dir: Path) -> None:
    seed_rows = _completed_seed_metrics(experiment_dir)
    subject_long = _completed_subject_metrics(experiment_dir)
    if not seed_rows or not subject_long:
        return
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(results_dir / "seed_results.csv", seed_rows, list(seed_rows[0]))
    _write_csv(
        results_dir / "subject_results_long.csv",
        subject_long,
        list(subject_long[0]),
    )
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in subject_long:
        grouped[int(row["subject"])].append(row)
    subject_summary = []
    for subject, group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "input_id": group[0]["input_id"],
            "input_label": group[0]["input_label"],
            "subject": subject,
            "subject_id": f"A{subject:02d}",
            "n_seeds": len(group),
        }
        for metric in SUMMARY_METRICS:
            mean, std = _mean_std([float(value[metric]) for value in group])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        subject_summary.append(row)
    _write_csv(
        results_dir / "subject_summary.csv",
        subject_summary,
        list(subject_summary[0]),
    )
    experiment_summary: dict[str, Any] = {
        "input_id": seed_rows[0]["input_id"],
        "input_label": seed_rows[0]["input_label"],
        "n_channels": seed_rows[0]["n_channels"],
        "n_seeds": len(seed_rows),
        "n_subjects": len(subject_summary),
        "trainable_parameters": seed_rows[0]["trainable_parameters"],
        "epochs": len(_read_csv(
            experiment_dir / "checkpoints" / f"seed_{seed_rows[0]['seed']}" / "history.csv"
        )),
        "optimizer": seed_rows[0]["optimizer"],
        "learning_rate": seed_rows[0]["learning_rate"],
        "scheduler": seed_rows[0]["scheduler"],
        "sr_augmentation": seed_rows[0]["sr_augmentation"],
        "sr_segments": seed_rows[0]["sr_segments"],
    }
    for metric in SUMMARY_METRICS:
        mean, std = _mean_std([float(row[f"{metric}_mean"]) for row in subject_summary])
        experiment_summary[f"{metric}_subject_mean"] = mean
        experiment_summary[f"{metric}_subject_std"] = std
    _write_csv(
        results_dir / "experiment_summary.csv",
        [experiment_summary],
        list(experiment_summary),
    )
    _write_experiment_markdown(experiment_dir / "report.md", experiment_summary, subject_summary)


def _format_percent(mean: float, std: float) -> str:
    return f"{100 * mean:.2f} ± {100 * std:.2f}"


def _write_experiment_markdown(
    path: Path,
    summary: dict[str, Any],
    subject_rows: list[dict[str, Any]],
) -> None:
    lines = [
        f"# {summary['input_label']}",
        "",
        "One TCFormer was trained per seed on pooled Session-1 train data from all selected subjects. "
        "Checkpoint selection used pooled Session-1 validation loss; Session 2 was used once for final evaluation.",
        "The classifier uses the paper-aligned global configuration with S&R augmentation, "
        f"{summary['optimizer']} optimization, and {summary['scheduler']} scheduling.",
        "",
        "| Seeds | Subjects | Channels | Parameters | Accuracy (%) | Macro-F1 (%) | Kappa |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['n_seeds']} | {summary['n_subjects']} | {summary['n_channels']} | "
        f"{int(summary['trainable_parameters']):,} | "
        f"{_format_percent(summary['accuracy_subject_mean'], summary['accuracy_subject_std'])} | "
        f"{_format_percent(summary['macro_f1_subject_mean'], summary['macro_f1_subject_std'])} | "
        f"{summary['cohen_kappa_subject_mean']:.3f} ± {summary['cohen_kappa_subject_std']:.3f} |",
        "",
        "## Session-2 accuracy by subject",
        "",
        "| Subject | Accuracy (%) | Macro-F1 (%) | Kappa |",
        "|---|---:|---:|---:|",
    ]
    for row in subject_rows:
        lines.append(
            f"| {row['subject_id']} | "
            f"{_format_percent(row['accuracy_mean'], row['accuracy_std'])} | "
            f"{_format_percent(row['macro_f1_mean'], row['macro_f1_std'])} | "
            f"{row['cohen_kappa_mean']:.3f} ± {row['cohen_kappa_std']:.3f} |"
        )
    _atomic_text(path, "\n".join(lines) + "\n")


def _paired_statistics(subject_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["input_id"]), int(row["subject"])): float(row["accuracy_mean"])
        for row in subject_rows
    }
    input_ids = sorted({str(row["input_id"]) for row in subject_rows})
    if "true22" not in input_ids:
        return []
    rows = []
    for comparison in (value for value in input_ids if value != "true22"):
        subjects = sorted(
            subject for input_id, subject in lookup
            if input_id == "true22" and (comparison, subject) in lookup
        )
        if len(subjects) < 2:
            continue
        reference = np.asarray([lookup[("true22", subject)] for subject in subjects])
        compared = np.asarray([lookup[(comparison, subject)] for subject in subjects])
        differences = reference - compared
        difference_std = float(differences.std(ddof=1))
        if np.isclose(difference_std, 0.0):
            ci_low = ci_high = float(differences.mean())
            paired_t_statistic = 0.0
            paired_t_p = 1.0
        else:
            ci_low, ci_high = stats.t.interval(
                0.95, len(subjects) - 1,
                loc=float(differences.mean()), scale=float(stats.sem(differences)),
            )
            paired_t = stats.ttest_rel(reference, compared)
            paired_t_statistic = float(paired_t.statistic)
            paired_t_p = float(paired_t.pvalue)
        try:
            wilcoxon = stats.wilcoxon(differences, method="auto")
            wilcoxon_statistic = float(wilcoxon.statistic)
            wilcoxon_p = float(wilcoxon.pvalue)
        except ValueError:
            wilcoxon_statistic = 0.0
            wilcoxon_p = 1.0
        rows.append({
            "reference_input": "true22",
            "comparison_input": comparison,
            "n_subjects": len(subjects),
            "difference_mean": float(differences.mean()),
            "difference_std": difference_std,
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "cohen_dz": float(differences.mean()) / difference_std if difference_std else 0.0,
            "paired_t_statistic": paired_t_statistic,
            "paired_t_p": paired_t_p,
            "wilcoxon_statistic": wilcoxon_statistic,
            "wilcoxon_p": wilcoxon_p,
        })
    return rows


def write_study_record(study_dir: Path) -> None:
    summaries: list[dict[str, Any]] = []
    subjects: list[dict[str, Any]] = []
    for experiment_dir in sorted(path for path in study_dir.iterdir() if path.is_dir()):
        summary_path = experiment_dir / "results" / "experiment_summary.csv"
        subject_path = experiment_dir / "results" / "subject_summary.csv"
        if not summary_path.is_file() or not subject_path.is_file():
            continue
        summary = _read_csv(summary_path)[0]
        summaries.append({
            **summary,
            **{key: float(value) for key, value in summary.items() if key.endswith(("_mean", "_std"))},
        })
        for row in _read_csv(subject_path):
            subjects.append({
                **row,
                "subject": int(row["subject"]),
                **{key: float(value) for key, value in row.items() if key.endswith(("_mean", "_std"))},
            })
    if not summaries:
        return
    comparison_dir = study_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    paired = _paired_statistics(subjects)
    if paired:
        _write_csv(comparison_dir / "paired_statistics.csv", paired, list(paired[0]))
    labels = {str(row["input_id"]): str(row["input_label"]) for row in summaries}
    lines = [
        "# Global-model classification experiment record",
        "",
        "Protocol: all selected subjects' Session-1 trials are pooled to train one global model per seed; "
        "Session-1 validation selects checkpoints; held-out Session 2 is the final test set. "
        "The same subject cohort appears across sessions, so this is not unseen-subject evaluation.",
        "",
        "## Completed input experiments",
        "",
        "| Input | Seeds | Subjects | Channels | Accuracy (%) | Macro-F1 (%) | Kappa |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['input_label']} | {row['n_seeds']} | {row['n_subjects']} | "
            f"{row['n_channels']} | "
            f"{_format_percent(row['accuracy_subject_mean'], row['accuracy_subject_std'])} | "
            f"{_format_percent(row['macro_f1_subject_mean'], row['macro_f1_subject_std'])} | "
            f"{row['cohen_kappa_subject_mean']:.3f} ± {row['cohen_kappa_subject_std']:.3f} |"
        )
    input_ids = [str(row["input_id"]) for row in summaries]
    subject_ids = sorted({int(row["subject"]) for row in subjects})
    lookup = {(str(row["input_id"]), int(row["subject"])): row for row in subjects}
    lines.extend([
        "", "## Session-2 accuracy by subject", "",
        "| Subject | " + " | ".join(labels[value] + " (%)" for value in input_ids) + " |",
        "|---|" + "---:|" * len(input_ids),
    ])
    for subject in subject_ids:
        values = []
        for input_id in input_ids:
            row = lookup.get((input_id, subject))
            values.append("—" if row is None else _format_percent(row["accuracy_mean"], row["accuracy_std"]))
        lines.append(f"| A{subject:02d} | " + " | ".join(values) + " |")
    if paired:
        lines.extend([
            "", "## Paired accuracy statistics", "",
            "| Comparison | True-22 difference (pp) | 95% CI (pp) | Cohen's dz | Paired t p | Wilcoxon p |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in paired:
            lines.append(
                f"| True-22 vs {labels[row['comparison_input']]} | "
                f"{100 * row['difference_mean']:.2f} | "
                f"[{100 * row['ci95_low']:.2f}, {100 * row['ci95_high']:.2f}] | "
                f"{row['cohen_dz']:.3f} | {row['paired_t_p']:.4g} | {row['wilcoxon_p']:.4g} |"
            )
    _atomic_text(study_dir / "EXPERIMENT_RECORD.md", "\n".join(lines) + "\n")
