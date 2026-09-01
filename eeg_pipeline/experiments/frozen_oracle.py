"""Evaluate restored Session-2 EEG with frozen True-22 TCFormer checkpoints."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .classification.config import ExperimentConfig
from .classification.data import InputDataRepository
from .classification.metrics import classification_metrics
from .classification.training import build_model, configure_device


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _aligned_test_dataset(
    oracle_config: ExperimentConfig,
    input_config: ExperimentConfig,
) -> TensorDataset:
    oracle = InputDataRepository(oracle_config.input).pooled(oracle_config.subjects).test
    restored = InputDataRepository(input_config.input).pooled(input_config.subjects).test
    oracle_x, oracle_y, oracle_subject, oracle_trial = oracle.tensors
    restored_x, restored_y, restored_subject, restored_trial = restored.tensors
    for name, left, right in (
        ("label", oracle_y, restored_y),
        ("subject", oracle_subject, restored_subject),
        ("trial_index", oracle_trial, restored_trial),
    ):
        if not torch.equal(left, right):
            raise ValueError(f"True-22 and restored test {name} arrays are not aligned")
    return TensorDataset(
        oracle_x,
        restored_x,
        oracle_y,
        oracle_subject,
        oracle_trial,
    )


def _load_oracle(
    oracle_config: ExperimentConfig,
    checkpoint_path: Path,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing True-22 checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("seed") != seed or payload.get("input", {}).get("id") != "true22":
        raise ValueError(f"Checkpoint identity mismatch: {checkpoint_path}")
    if payload.get("model_args") != oracle_config.model.args:
        raise ValueError(f"Checkpoint model configuration mismatch: {checkpoint_path}")
    model = build_model(oracle_config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def _evaluate_seed(
    oracle_config: ExperimentConfig,
    input_config: ExperimentConfig,
    dataset: TensorDataset,
    checkpoint_path: Path,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    model = _load_oracle(oracle_config, checkpoint_path, seed, device)
    loader = DataLoader(
        dataset,
        batch_size=input_config.training.batch_size,
        shuffle=False,
        num_workers=input_config.training.num_workers,
        pin_memory=True,
    )
    amp_enabled = input_config.training.amp and device.type == "cuda"
    collected: dict[str, list[np.ndarray]] = {
        "label": [], "oracle_prediction": [], "restored_prediction": [],
        "oracle_probability": [], "restored_probability": [],
        "subject": [], "trial_index": [],
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for oracle_x, restored_x, y, subject, trial_index in loader:
        oracle_x = oracle_x.to(device, non_blocking=True)
        restored_x = restored_x.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            oracle_probability = model(oracle_x).softmax(dim=1)
            restored_probability = model(restored_x).softmax(dim=1)
        collected["label"].append(y.numpy())
        collected["oracle_prediction"].append(oracle_probability.argmax(1).cpu().numpy())
        collected["restored_prediction"].append(restored_probability.argmax(1).cpu().numpy())
        collected["oracle_probability"].append(oracle_probability.float().cpu().numpy())
        collected["restored_probability"].append(restored_probability.float().cpu().numpy())
        collected["subject"].append(subject.numpy())
        collected["trial_index"].append(trial_index.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    values = {key: np.concatenate(parts) for key, parts in collected.items()}
    restored_metrics, confusion = classification_metrics(
        values["label"], values["restored_prediction"], oracle_config.model.n_classes
    )
    oracle_metrics, _ = classification_metrics(
        values["label"], values["oracle_prediction"], oracle_config.model.n_classes
    )
    agreement = values["restored_prediction"] == values["oracle_prediction"]
    probability_l1 = np.abs(
        values["restored_probability"] - values["oracle_probability"]
    ).mean(axis=1)
    metrics = {
        "input_id": input_config.input.id,
        "input_label": input_config.input.label,
        "scope": "frozen_true22_oracle_session2",
        "oracle_seed": seed,
        "n_test": len(dataset),
        "restored_accuracy": restored_metrics["accuracy"],
        "restored_macro_f1": restored_metrics["macro_f1"],
        "restored_cohen_kappa": restored_metrics["cohen_kappa"],
        "oracle_true22_accuracy": oracle_metrics["accuracy"],
        "prediction_agreement": float(agreement.mean()),
        "probability_l1": float(probability_l1.mean()),
        "seconds": elapsed,
        "ms_per_trial_for_two_forwards": 1000.0 * elapsed / len(dataset),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
    }
    subject_rows = []
    for subject_id in sorted(int(value) for value in np.unique(values["subject"])):
        mask = values["subject"] == subject_id
        subject_metric, _ = classification_metrics(
            values["label"][mask], values["restored_prediction"][mask],
            oracle_config.model.n_classes,
        )
        oracle_subject_metric, _ = classification_metrics(
            values["label"][mask], values["oracle_prediction"][mask],
            oracle_config.model.n_classes,
        )
        subject_rows.append({
            "input_id": input_config.input.id,
            "subject": subject_id,
            "subject_id": f"A{subject_id:02d}",
            "oracle_seed": seed,
            "restored_accuracy": subject_metric["accuracy"],
            "restored_macro_f1": subject_metric["macro_f1"],
            "restored_cohen_kappa": subject_metric["cohen_kappa"],
            "oracle_true22_accuracy": oracle_subject_metric["accuracy"],
            "prediction_agreement": float(agreement[mask].mean()),
            "probability_l1": float(probability_l1[mask].mean()),
        })
    prediction_rows = []
    for index in range(len(dataset)):
        row = {
            "subject": int(values["subject"][index]),
            "trial_index": int(values["trial_index"][index]),
            "true_label": int(values["label"][index]),
            "oracle_prediction": int(values["oracle_prediction"][index]),
            "restored_prediction": int(values["restored_prediction"][index]),
        }
        for class_id in range(oracle_config.model.n_classes):
            row[f"oracle_probability_{class_id}"] = float(values["oracle_probability"][index, class_id])
            row[f"restored_probability_{class_id}"] = float(values["restored_probability"][index, class_id])
        prediction_rows.append(row)
    return metrics, subject_rows, prediction_rows, confusion


def _aggregate(output_dir: Path, input_config: ExperimentConfig) -> None:
    seed_rows = []
    subject_rows = []
    for seed_dir in sorted((output_dir / "checkpoints").glob("seed_*")):
        metrics_path = seed_dir / "metrics.json"
        subjects_path = seed_dir / "subject_metrics.csv"
        if not metrics_path.is_file() or not subjects_path.is_file():
            continue
        seed_rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        with subjects_path.open(newline="", encoding="utf-8") as handle:
            subject_rows.extend(list(csv.DictReader(handle)))
    if not seed_rows:
        raise RuntimeError("No completed frozen-oracle seeds")
    results = output_dir / "results"
    results.mkdir(exist_ok=True)
    _atomic_csv(results / "seed_results.csv", seed_rows)
    grouped = {}
    for row in subject_rows:
        grouped.setdefault(int(row["subject"]), []).append(row)
    summaries = []
    for subject, rows in sorted(grouped.items()):
        summary = {"input_id": input_config.input.id, "subject": subject, "subject_id": f"A{subject:02d}", "n_seeds": len(rows)}
        for metric in ("restored_accuracy", "restored_macro_f1", "restored_cohen_kappa", "oracle_true22_accuracy", "prediction_agreement", "probability_l1"):
            values = np.asarray([float(row[metric]) for row in rows])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summaries.append(summary)
    _atomic_csv(results / "subject_summary.csv", summaries)
    experiment = {
        "input_id": input_config.input.id,
        "input_label": input_config.input.label,
        "n_seeds": len(seed_rows),
        "n_subjects": len(summaries),
    }
    for metric in ("restored_accuracy", "restored_macro_f1", "restored_cohen_kappa", "oracle_true22_accuracy", "prediction_agreement", "probability_l1"):
        values = np.asarray([float(row[f"{metric}_mean"]) for row in summaries])
        experiment[f"{metric}_subject_mean"] = float(values.mean())
        experiment[f"{metric}_subject_std"] = float(values.std(ddof=1))
    _atomic_csv(results / "experiment_summary.csv", [experiment])
    lines = [
        f"# Frozen True-22 oracle: {input_config.input.label}", "",
        "The five validation-selected True-22 TCFormer checkpoints were frozen and evaluated on aligned restored Session-2 trials. No classifier was retrained.", "",
        "| Subjects | Seeds | Restored accuracy (%) | True-22 oracle accuracy (%) | Macro-F1 (%) | Kappa | Oracle agreement (%) | Probability L1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {len(summaries)} | {len(seed_rows)} | "
        f"{100*experiment['restored_accuracy_subject_mean']:.2f} ± {100*experiment['restored_accuracy_subject_std']:.2f} | "
        f"{100*experiment['oracle_true22_accuracy_subject_mean']:.2f} ± {100*experiment['oracle_true22_accuracy_subject_std']:.2f} | "
        f"{100*experiment['restored_macro_f1_subject_mean']:.2f} ± {100*experiment['restored_macro_f1_subject_std']:.2f} | "
        f"{experiment['restored_cohen_kappa_subject_mean']:.3f} ± {experiment['restored_cohen_kappa_subject_std']:.3f} | "
        f"{100*experiment['prediction_agreement_subject_mean']:.2f} ± {100*experiment['prediction_agreement_subject_std']:.2f} | "
        f"{experiment['probability_l1_subject_mean']:.4f} ± {experiment['probability_l1_subject_std']:.4f} |",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_frozen_oracle(
    oracle_config: ExperimentConfig,
    input_config: ExperimentConfig,
    project_root: Path,
    device_override: str | None = None,
    seeds: tuple[int, ...] | None = None,
    dry_run: bool = False,
) -> Path:
    if oracle_config.input.id != "true22" or oracle_config.input.n_channels != 22:
        raise ValueError("Frozen oracle must be the True-22 experiment")
    if (
        input_config.input.category != "restored"
        or input_config.input.n_channels != 22
        or input_config.input.transform != "none"
    ):
        raise ValueError("Frozen-oracle restored input must be native 22-channel")
    if oracle_config.subjects != input_config.subjects:
        raise ValueError("Oracle and restored input subjects must match")
    selected = oracle_config.seeds if seeds is None else seeds
    if not selected or not set(selected).issubset(oracle_config.seeds):
        raise ValueError("Seeds must be a non-empty subset of True-22 seeds")
    dataset = _aligned_test_dataset(oracle_config, input_config)
    device = configure_device(oracle_config, device_override)
    oracle_dir = oracle_config.output_dir / oracle_config.name
    output_dir = project_root / "artifacts/experiments/classification/frozen_oracle" / input_config.input.id
    if dry_run:
        checkpoint = oracle_dir / "checkpoints" / f"seed_{selected[0]}" / "best_model.pt"
        model = _load_oracle(oracle_config, checkpoint, selected[0], device)
        oracle_x, restored_x, *_ = dataset[0]
        with torch.inference_mode():
            oracle_logits = model(oracle_x.unsqueeze(0).to(device))
            restored_logits = model(restored_x.unsqueeze(0).to(device))
        print(
            f"DRY RUN frozen-oracle {input_config.input.id}: test={len(dataset)}, "
            f"oracle={tuple(oracle_x.shape)}, restored={tuple(restored_x.shape)}, "
            f"logits={tuple(restored_logits.shape)}, finite="
            f"{bool(torch.isfinite(oracle_logits).all() and torch.isfinite(restored_logits).all())}, "
            f"device={device}, writes=none"
        )
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "oracle_config": oracle_config.as_serializable_dict(),
        "input_config": input_config.as_serializable_dict(),
        "oracle_checkpoint_dir": str(oracle_dir / "checkpoints"),
        "scope": "frozen_true22_oracle_session2",
    }
    resolved_path = output_dir / "resolved_config.json"
    if resolved_path.is_file() and json.loads(resolved_path.read_text(encoding="utf-8")) != resolved:
        raise RuntimeError(f"Frozen-oracle configuration changed: {output_dir}")
    _atomic_json(resolved_path, resolved)
    for seed in selected:
        seed_dir = output_dir / "checkpoints" / f"seed_{seed}"
        if (seed_dir / "metrics.json").is_file():
            print(f"SKIP frozen-oracle {input_config.input.id} seed={seed}")
            continue
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = oracle_dir / "checkpoints" / f"seed_{seed}" / "best_model.pt"
        metrics, subject_rows, prediction_rows, confusion = _evaluate_seed(
            oracle_config, input_config, dataset, checkpoint, seed, device
        )
        _atomic_json(seed_dir / "metrics.json", metrics)
        _atomic_csv(seed_dir / "subject_metrics.csv", subject_rows)
        _atomic_csv(seed_dir / "predictions.csv", prediction_rows)
        confusion_rows = [
            {"true_label": index, **{f"predicted_{j}": int(value) for j, value in enumerate(row)}}
            for index, row in enumerate(confusion)
        ]
        _atomic_csv(seed_dir / "confusion_matrix.csv", confusion_rows)
        print(
            f"DONE frozen-oracle {input_config.input.id} seed={seed} "
            f"accuracy={metrics['restored_accuracy']:.4f} "
            f"agreement={metrics['prediction_agreement']:.4f}",
            flush=True,
        )
    _aggregate(output_dir, input_config)
    return output_dir
