"""End-to-end, leakage-controlled Stage-D signal analysis runner."""

from __future__ import annotations

import csv
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ...channels import CLASS_NAMES, EEG_CHANNELS_22
from .config import SignalAnalysisConfig
from .data import EEGSplit, load_method_test, load_source
from .figures import (
    plot_error_topographies,
    plot_metric_summary,
    plot_tfr_errors,
    plot_true_topographies,
)
from .metrics import (
    MISSING_INDICES,
    affine_invariant_distance,
    class_conditional_tfr,
    csp_log_variance,
    feature_correlation,
    fit_ovr_csp,
    log_relative_bandpower,
    normalized_covariance,
)
from .statistics import paired_endpoint_rows


CLASSES = tuple(range(len(CLASS_NAMES)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fit_session1_csp(config: SignalAnalysisConfig) -> dict[int, np.ndarray]:
    train = load_source(config.source_arrays_dir / "train.npz")
    validation = load_source(config.source_arrays_dir / "validation.npz")
    filters = {}
    subjects = sorted(int(value) for value in np.unique(train.subject))
    for subject in subjects:
        train_mask = train.subject == subject
        validation_mask = validation.subject == subject
        x = np.concatenate((train.x_true22[train_mask], validation.x_true22[validation_mask]))
        y = np.concatenate((train.y[train_mask], validation.y[validation_mask]))
        filters[subject] = fit_ovr_csp(
            x, y, CLASSES, config.covariance_ridge, config.csp_filters_per_class
        )
    del train, validation
    gc.collect()
    return filters


def _method_metrics(
    method_id: str,
    x: np.ndarray,
    source: EEGSplit,
    true_bandpower: np.ndarray,
    true_csp: dict[int, np.ndarray],
    csp_filters: dict[int, np.ndarray],
    config: SignalAnalysisConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    restored_bandpower = log_relative_bandpower(
        x, config.sampling_rate, config.bands, config.total_band
    )
    band_rows = []
    covariance_rows = []
    csp_rows = []
    for subject in sorted(int(value) for value in np.unique(source.subject)):
        subject_mask = source.subject == subject
        method_csp = csp_log_variance(x[subject_mask], csp_filters[subject])
        reference_csp = true_csp[subject]
        csp_rows.append({
            "method": method_id,
            "subject": subject,
            "subject_id": f"A{subject:02d}",
            "n_trials": int(subject_mask.sum()),
            "csp_feature_mse": float(np.mean((method_csp - reference_csp) ** 2)),
            "csp_feature_correlation": feature_correlation(method_csp, reference_csp),
        })
        for class_id, class_name in enumerate(CLASS_NAMES):
            mask = subject_mask & (source.y == class_id)
            if not np.any(mask):
                raise ValueError(f"A{subject:02d} has no Session-2 {class_name} trials")
            true_mean = true_bandpower[mask].mean(axis=0)
            method_mean = restored_bandpower[mask].mean(axis=0)
            squared = (method_mean - true_mean) ** 2
            for channel_index, channel in enumerate(EEG_CHANNELS_22):
                for band_index, (band_name, _, _) in enumerate(config.bands):
                    band_rows.append({
                        "method": method_id,
                        "subject": subject,
                        "subject_id": f"A{subject:02d}",
                        "class": class_id,
                        "class_name": class_name,
                        "channel_index": channel_index,
                        "channel": channel,
                        "channel_group": "missing13" if channel_index in MISSING_INDICES else "observed9",
                        "band": band_name,
                        "true_log_relative_power": float(true_mean[channel_index, band_index]),
                        "method_log_relative_power": float(method_mean[channel_index, band_index]),
                        "squared_error": float(squared[channel_index, band_index]),
                    })
            covariance_rows.append({
                "method": method_id,
                "subject": subject,
                "subject_id": f"A{subject:02d}",
                "class": class_id,
                "class_name": class_name,
                "n_trials": int(mask.sum()),
                "covariance_distance": affine_invariant_distance(
                    normalized_covariance(source.x_true22[mask], config.covariance_ridge),
                    normalized_covariance(x[mask], config.covariance_ridge),
                ),
            })
    return band_rows, covariance_rows, csp_rows, restored_bandpower


def _subject_summary(
    methods: tuple[str, ...],
    band_rows: list[dict[str, Any]],
    covariance_rows: list[dict[str, Any]],
    csp_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    subjects = sorted({int(row["subject"]) for row in csp_rows})
    for method in methods:
        for subject in subjects:
            band = [float(row["squared_error"]) for row in band_rows
                    if row["method"] == method and row["subject"] == subject
                    and row["channel_group"] == "missing13"]
            covariance = [float(row["covariance_distance"]) for row in covariance_rows
                          if row["method"] == method and row["subject"] == subject]
            csp = next(row for row in csp_rows
                       if row["method"] == method and row["subject"] == subject)
            rows.append({
                "method": method,
                "subject": subject,
                "subject_id": f"A{subject:02d}",
                "bandpower_mse": float(np.mean(band)),
                "covariance_distance": float(np.mean(covariance)),
                "csp_feature_mse": float(csp["csp_feature_mse"]),
                "csp_feature_correlation": float(csp["csp_feature_correlation"]),
            })
    return rows


def _aggregate_summary(subject_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in dict.fromkeys(str(row["method"]) for row in subject_rows):
        selected = [row for row in subject_rows if row["method"] == method]
        summary: dict[str, Any] = {"method": method, "n_subjects": len(selected)}
        for metric in (
            "bandpower_mse", "covariance_distance", "csp_feature_mse",
            "csp_feature_correlation",
        ):
            values = np.asarray([float(row[metric]) for row in selected])
            summary[metric] = float(values.mean())
            summary[f"{metric}_subject_std"] = float(values.std(ddof=1))
        rows.append(summary)
    return rows


def _statistical_rows(
    subject_rows: list[dict[str, Any]], config: SignalAnalysisConfig
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(config.random_seed)
    rows = []
    for endpoint in ("bandpower_mse", "covariance_distance", "csp_feature_mse"):
        values: dict[str, dict[int, float]] = {}
        for row in subject_rows:
            values.setdefault(str(row["method"]), {})[int(row["subject"])] = float(row[endpoint])
        rows.extend(paired_endpoint_rows(
            values, "autoencoder_eeg_aware", endpoint,
            config.bootstrap_samples, rng,
        ))
    return rows


def _write_report(
    output_dir: Path,
    config: SignalAnalysisConfig,
    aggregate: list[dict[str, Any]],
    statistics: list[dict[str, Any]],
) -> None:
    labels = {method.id: method.label for method in config.methods}
    lines = [
        "# Task-relevant Session-2 signal preservation", "",
        "CSP filters were fitted separately for each subject using only Session-1 True-22 "
        "training and validation trials. The fixed filters were then applied to aligned "
        "held-out Session-2 inputs. All other endpoints are class-conditional Session-2 "
        "comparisons against True-22; no Session-2 result was used for model selection.", "",
        "| Method | μ/β log-relative-power MSE | Covariance AIRM | CSP feature MSE | CSP correlation |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {labels[str(row['method'])]} | {row['bandpower_mse']:.6f} | "
            f"{row['covariance_distance']:.6f} | {row['csp_feature_mse']:.6f} | "
            f"{row['csp_feature_correlation']:.4f} |"
        )
    lines.extend(["", "## Predeclared subject-level comparisons", ""])
    for row in statistics:
        lines.append(
            f"- `{row['endpoint']}`: EEG-aware AE vs `{row['comparison']}`, "
            f"comparison − EEG-aware difference `{row['mean_difference_comparison_minus_reference']:.6f}`, "
            f"EEG-aware wins `{row['reference_wins']}/{row['n_subjects']}`, "
            f"raw/Holm p `{row['wilcoxon_raw_p']:.4f}/{row['holm_adjusted_p']:.4f}`."
        )
    lines.extend([
        "", "Error endpoints are lower-is-better. Each endpoint has its own five-comparison "
        "Holm family. Direct MI-9 has no 22-channel spatial representation, so zero-padded "
        "MI-9 is its signal-space reference. Figures use all nine subjects and four classes; "
        "no representative subject was selected.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_signal_analysis(
    config: SignalAnalysisConfig,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    skip_figures: bool = False,
) -> Path:
    """Run Stage-D analysis atomically; refuse to overwrite unless explicitly requested."""

    required = [config.source_arrays_dir / f"{split}.npz" for split in ("train", "validation", "test")]
    required.extend(method.test_path for method in config.methods if method.test_path is not None)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Stage-D inputs:\n" + "\n".join(map(str, missing)))
    if dry_run:
        print(f"Validated {len(config.methods)} methods; output={config.output_dir}")
        return config.output_dir
    if config.output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {config.output_dir}. Use --overwrite intentionally."
        )

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{config.id}.", dir=config.output_dir.parent))
    try:
        results_dir = temporary / "results"
        figures_dir = temporary / "figures"
        results_dir.mkdir()
        if not skip_figures:
            figures_dir.mkdir()
        print("Fitting subject-specific CSP filters on Session-1 True-22 only...")
        csp_filters = _fit_session1_csp(config)
        source = load_source(config.source_arrays_dir / "test.npz")
        true_bandpower = log_relative_bandpower(
            source.x_true22, config.sampling_rate, config.bands, config.total_band
        )
        true_csp = {
            subject: csp_log_variance(
                source.x_true22[source.subject == subject], filters
            )
            for subject, filters in csp_filters.items()
        }
        all_band_rows: list[dict[str, Any]] = []
        all_covariance_rows: list[dict[str, Any]] = []
        all_csp_rows: list[dict[str, Any]] = []
        tfr_values: dict[str, np.ndarray] = {}
        frequencies = times = None
        labels = {method.id: method.label for method in config.methods}
        for method in config.methods:
            print(f"Analyzing {method.label}...")
            x = load_method_test(method, source)
            band, covariance, csp, _ = _method_metrics(
                method.id, x, source, true_bandpower, true_csp, csp_filters, config
            )
            all_band_rows.extend(band)
            all_covariance_rows.extend(covariance)
            all_csp_rows.extend(csp)
            if not skip_figures:
                frequencies, times, tfr_values[method.id] = class_conditional_tfr(
                    x, source.y, CLASSES, config.sampling_rate, config.total_band,
                    config.tfr_window_samples, config.tfr_overlap_samples,
                )
            if method.kind == "restored" or method.kind == "zero_padded":
                del x
            gc.collect()
        method_ids = tuple(method.id for method in config.methods)
        subject_rows = _subject_summary(
            method_ids, all_band_rows, all_covariance_rows, all_csp_rows
        )
        aggregate = _aggregate_summary(subject_rows)
        statistical_rows = _statistical_rows(subject_rows, config)
        _write_csv(results_dir / "bandpower_detail.csv", all_band_rows)
        _write_csv(results_dir / "covariance_by_subject_class.csv", all_covariance_rows)
        _write_csv(results_dir / "csp_by_subject.csv", all_csp_rows)
        _write_csv(results_dir / "subject_metrics.csv", subject_rows)
        _write_csv(results_dir / "method_summary.csv", aggregate)
        _write_csv(results_dir / "paired_statistics.csv", statistical_rows)
        if not skip_figures:
            assert frequencies is not None and times is not None
            np.savez_compressed(
                results_dir / "grand_average_tfr.npz",
                frequencies=frequencies, times=times,
                **{f"method_{method}": value for method, value in tfr_values.items()},
            )
            true_values = np.empty((len(CLASSES), 22, len(config.bands)))
            for class_id in CLASSES:
                true_values[class_id] = true_bandpower[source.y == class_id].mean(axis=0)
            plot_true_topographies(
                true_values, tuple(item[0] for item in config.bands), CLASS_NAMES,
                figures_dir / "true22_class_bandpower_topography.png",
            )
            error_topographies = {}
            for method in method_ids:
                if method in {"true22", "zero_padded_mi9"}:
                    continue
                rows = [row for row in all_band_rows if row["method"] == method]
                error = np.zeros((22, len(config.bands)))
                for channel in range(22):
                    for band_index, (band_name, _, _) in enumerate(config.bands):
                        values = [float(row["squared_error"]) for row in rows
                                  if row["channel_index"] == channel and row["band"] == band_name]
                        error[channel, band_index] = np.sqrt(np.mean(values))
                error_topographies[method] = error
            plot_error_topographies(
                error_topographies, labels, tuple(item[0] for item in config.bands),
                figures_dir / "restoration_bandpower_error_topography.png",
            )
            plot_tfr_errors(
                tfr_values, labels, frequencies, times,
                figures_dir / "restoration_time_frequency_error.png",
            )
            plot_metric_summary(aggregate, labels, figures_dir / "signal_metric_summary.png")
        _write_report(temporary, config, aggregate, statistical_rows)
        completion = {
            "experiment_id": config.id,
            "status": "complete",
            "n_methods": len(config.methods),
            "n_subjects": len(csp_filters),
            "n_test_trials": int(source.y.size),
            "csp_fit": "subject-specific Session-1 True-22 train+validation only",
            "session2_model_selection": False,
            "figures_generated": not skip_figures,
        }
        (temporary / "PIPELINE_COMPLETE.json").write_text(
            json.dumps(completion, indent=2) + "\n", encoding="utf-8"
        )
        if config.output_dir.exists():
            shutil.rmtree(config.output_dir)
        os.replace(temporary, config.output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return config.output_dir
