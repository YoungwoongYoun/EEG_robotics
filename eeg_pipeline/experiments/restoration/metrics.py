"""Missing-channel-first reconstruction metrics and reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...channels import EEG_CHANNELS_22, MI9_INDICES
from .data import MISSING_INDICES, RestorationSplit


SAMPLING_RATE = 250.0
MI_BANDS = ((8.0, 13.0), (13.0, 30.0))
METRIC_EPSILON = 1e-6


def _channel_metrics(true: np.ndarray, restored: np.ndarray) -> dict[str, np.ndarray]:
    error = restored.astype(np.float64) - true.astype(np.float64)
    mse = np.mean(error * error, axis=(0, 2))
    mae = np.mean(np.abs(error), axis=(0, 2))
    true_centered = true.astype(np.float64) - np.mean(true, axis=(0, 2), keepdims=True)
    restored_centered = (
        restored.astype(np.float64) - np.mean(restored, axis=(0, 2), keepdims=True)
    )
    numerator = np.sum(true_centered * restored_centered, axis=(0, 2))
    denominator = np.sqrt(
        np.sum(true_centered * true_centered, axis=(0, 2))
        * np.sum(restored_centered * restored_centered, axis=(0, 2))
    )
    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    feature_metrics = _eeg_feature_metrics(true, restored)
    return {
        "mse": mse,
        "mae": mae,
        "correlation": correlation,
        **feature_metrics,
    }


def _eeg_feature_metrics(
    true: np.ndarray,
    restored: np.ndarray,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Return per-channel mu/beta log-power and spatial-correlation MSE."""

    if true.shape != restored.shape or true.ndim != 3 or true.shape[1] != 22:
        raise ValueError("EEG feature inputs must have matching [N, 22, T] shapes")
    frequencies = np.fft.rfftfreq(true.shape[-1], d=1.0 / SAMPLING_RATE)
    masks = [
        (frequencies >= low) & (frequencies < high)
        for low, high in MI_BANDS
    ]
    masks = [mask for mask in masks if np.any(mask)]
    if not masks:
        raise ValueError("EEG trial is too short for the configured mu/beta bands")
    band_sum = np.zeros(22, dtype=np.float64)
    spatial_sum = np.zeros(22, dtype=np.float64)
    trials = 0
    for start in range(0, true.shape[0], batch_size):
        stop = min(start + batch_size, true.shape[0])
        true_batch = true[start:stop].astype(np.float64, copy=False)
        restored_batch = restored[start:stop].astype(np.float64, copy=False)
        true_power = np.abs(np.fft.rfft(true_batch, axis=-1)) ** 2
        restored_power = np.abs(np.fft.rfft(restored_batch, axis=-1)) ** 2
        band_error = []
        for mask in masks:
            true_band = np.log(true_power[..., mask].mean(axis=-1) + METRIC_EPSILON)
            restored_band = np.log(
                restored_power[..., mask].mean(axis=-1) + METRIC_EPSILON
            )
            band_error.append((restored_band - true_band) ** 2)
        band_sum += np.stack(band_error).mean(axis=0).sum(axis=0)

        def correlation(values: np.ndarray) -> np.ndarray:
            centered = values - values.mean(axis=-1, keepdims=True)
            scale = np.sqrt(
                np.mean(centered * centered, axis=-1, keepdims=True) + METRIC_EPSILON
            )
            normalized = centered / scale
            return normalized @ np.swapaxes(normalized, 1, 2) / values.shape[-1]

        spatial_error = (correlation(restored_batch) - correlation(true_batch)) ** 2
        spatial_sum += spatial_error.mean(axis=2).sum(axis=0)
        trials += stop - start
    return {
        "log_bandpower_mse": band_sum / trials,
        "spatial_correlation_mse": spatial_sum / trials,
    }


def reconstruction_metrics(
    source: RestorationSplit,
    restored22: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return aggregate, per-channel, and per-subject reconstruction metrics."""

    metrics = _channel_metrics(source.x_true22, restored22)
    per_channel = []
    observed = set(MI9_INDICES)
    for index, name in enumerate(EEG_CHANNELS_22):
        per_channel.append({
            "channel_index": index,
            "channel": name,
            "group": "observed9" if index in observed else "missing13",
            **{key: float(values[index]) for key, values in metrics.items()},
        })
    aggregate: dict[str, Any] = {"n_trials": int(source.y.size)}
    groups = {"all22": tuple(range(22)), "observed9": MI9_INDICES, "missing13": MISSING_INDICES}
    for group, indices in groups.items():
        for metric, values in metrics.items():
            aggregate[f"{group}_{metric}"] = float(np.mean(values[list(indices)]))

    subject_rows = []
    for subject in sorted(int(value) for value in np.unique(source.subject)):
        mask = source.subject == subject
        values = _channel_metrics(source.x_true22[mask], restored22[mask])
        row: dict[str, Any] = {
            "subject": subject,
            "subject_id": f"A{subject:02d}",
            "n_trials": int(mask.sum()),
        }
        for metric, channel_values in values.items():
            row[f"missing13_{metric}"] = float(
                np.mean(channel_values[list(MISSING_INDICES)])
            )
            row[f"all22_{metric}"] = float(np.mean(channel_values))
        subject_rows.append(row)
    return aggregate, per_channel, subject_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_metric_artifacts(
    experiment_dir: Path,
    method_name: str,
    split_results: dict[str, tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]],
) -> None:
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows = []
    for split, (aggregate, channels, subjects) in split_results.items():
        aggregate_rows.append({"method": method_name, "split": split, **aggregate})
        _write_csv(
            results_dir / f"{split}_per_channel_metrics.csv",
            [{"method": method_name, "split": split, **row} for row in channels],
        )
        _write_csv(
            results_dir / f"{split}_subject_metrics.csv",
            [{"method": method_name, "split": split, **row} for row in subjects],
        )
    _write_csv(results_dir / "reconstruction_summary.csv", aggregate_rows)
    test = next(row for row in aggregate_rows if row["split"] == "test")
    report = (
        f"# {method_name} restoration report\n\n"
        "Primary reconstruction results are computed only on the 13 missing channels. "
        "Observed MI-9 channels are hard-copied and full-22 metrics are secondary.\n\n"
        "| Split | Trials | Missing-13 MSE | MAE | Correlation | "
        "mu/beta log-power MSE | Spatial-correlation MSE |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    for row in aggregate_rows:
        report += (
            f"| {row['split']} | {row['n_trials']} | {row['missing13_mse']:.6f} | "
            f"{row['missing13_mae']:.6f} | {row['missing13_correlation']:.4f} | "
            f"{row['missing13_log_bandpower_mse']:.6f} | "
            f"{row['missing13_spatial_correlation_mse']:.6f} |\n"
        )
    report += (
        "\n## Held-out Session-2 primary result\n\n"
        f"Missing-13 MSE `{test['missing13_mse']:.6f}`, MAE "
        f"`{test['missing13_mae']:.6f}`, correlation "
        f"`{test['missing13_correlation']:.4f}`, mu/beta log-power MSE "
        f"`{test['missing13_log_bandpower_mse']:.6f}`, spatial-correlation MSE "
        f"`{test['missing13_spatial_correlation_mse']:.6f}`.\n"
    )
    (experiment_dir / "report.md").write_text(report, encoding="utf-8")
    with (results_dir / "reconstruction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate_rows, handle, indent=2)
