"""Build reproducible, manuscript-ready tables, statistics, and figures."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

from eeg_pipeline.experiments.signal_analysis.statistics import holm_adjust


METHOD_ORDER = (
    "direct_mi9",
    "zero_padded_mi9",
    "spherical_spline",
    "autoencoder",
    "autoencoder_eeg_aware",
    "ddpm_standard",
    "wgan_gp",
    "true22",
)
METHOD_LABELS = {
    "direct_mi9": "Direct MI-9",
    "zero_padded_mi9": "Zero-padded MI-9",
    "spherical_spline": "Spherical spline",
    "autoencoder": "Autoencoder",
    "autoencoder_bandpower": "AE + bandpower",
    "autoencoder_spatial": "AE + spatial",
    "autoencoder_eeg_aware": "EEG-aware AE",
    "ddpm_standard": "Standard DDPM",
    "wgan_gp": "Conditional WGAN-GP",
    "true22": "True 22-channel",
}
COLORS = {
    "direct_mi9": "#4C78A8",
    "zero_padded_mi9": "#9ECAE9",
    "spherical_spline": "#59A14F",
    "autoencoder": "#F28E2B",
    "autoencoder_bandpower": "#EDC948",
    "autoencoder_spatial": "#B07AA1",
    "autoencoder_eeg_aware": "#E15759",
    "ddpm_standard": "#76B7B2",
    "wgan_gp": "#AF7AA1",
    "true22": "#303030",
}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_812


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required result is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Required result is empty: {path}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required result is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"Inconsistent columns in table: {path}")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(value) for value in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(clean(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 300}), (".pdf", {})):
        target = stem.with_suffix(suffix)
        with tempfile.NamedTemporaryFile(
            suffix=suffix, dir=target.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            fig.savefig(temporary, bbox_inches="tight", facecolor="white", **kwargs)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(fig)


def _classification_paths(root: Path, method: str) -> tuple[Path, Path]:
    if method in {"direct_mi9", "zero_padded_mi9", "true22"}:
        base = root / "artifacts/experiments/classification/global_model" / method / "results"
    else:
        base = root / "artifacts/experiments/classification/restoration_benchmarks" / method / "results"
    return base / "experiment_summary.csv", base / "subject_summary.csv"


def _load_classification(root: Path, methods: Iterable[str]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, float]]]:
    summaries: dict[str, dict[str, str]] = {}
    subjects: dict[str, dict[str, float]] = {}
    for method in methods:
        summary_path, subject_path = _classification_paths(root, method)
        summary = _read_csv(summary_path)[0]
        if summary["input_id"] != method:
            raise ValueError(f"Method mismatch in {summary_path}: {summary['input_id']}")
        subject_rows = _read_csv(subject_path)
        if len(subject_rows) != 9:
            raise ValueError(f"Expected 9 subjects for {method}, got {len(subject_rows)}")
        summaries[method] = summary
        subjects[method] = {
            row["subject_id"]: float(row["accuracy_mean"]) for row in subject_rows
        }
    expected = set(subjects[next(iter(subjects))])
    for method, values in subjects.items():
        if set(values) != expected:
            raise ValueError(f"Subject alignment failed for {method}")
    return summaries, subjects


def paired_statistics(
    subjects: dict[str, dict[str, float]],
    comparisons: Sequence[tuple[str, str]],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Return subject-level paired inference, with Holm correction as one family."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    raw_p = []
    for reference, comparison in comparisons:
        if reference not in subjects or comparison not in subjects:
            raise KeyError(f"Unknown comparison: {reference} vs {comparison}")
        subject_ids = sorted(set(subjects[reference]) & set(subjects[comparison]))
        if len(subject_ids) < 2:
            raise ValueError(f"Too few aligned subjects: {reference} vs {comparison}")
        difference = np.asarray(
            [subjects[reference][key] - subjects[comparison][key] for key in subject_ids],
            dtype=np.float64,
        )
        nonzero = difference[~np.isclose(difference, 0.0)]
        if nonzero.size == 0:
            statistic, p_value, rank_biserial = 0.0, 1.0, 0.0
        else:
            result = stats.wilcoxon(difference, alternative="two-sided", method="auto")
            statistic, p_value = float(result.statistic), float(result.pvalue)
            ranks = stats.rankdata(np.abs(nonzero))
            rank_biserial = float(
                (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
            )
        sampled = difference[
            rng.integers(0, difference.size, size=(bootstrap_samples, difference.size))
        ].mean(axis=1)
        difference_std = float(difference.std(ddof=1))
        rows.append({
            "comparison": f"{reference}_vs_{comparison}",
            "reference": reference,
            "compared": comparison,
            "n_subjects": len(subject_ids),
            "mean_difference_accuracy_pp": 100.0 * float(difference.mean()),
            "bootstrap_ci95_low_pp": 100.0 * float(np.quantile(sampled, 0.025)),
            "bootstrap_ci95_high_pp": 100.0 * float(np.quantile(sampled, 0.975)),
            "reference_wins": int(np.sum(difference > 0)),
            "ties": int(np.sum(np.isclose(difference, 0.0))),
            "wilcoxon_statistic": statistic,
            "wilcoxon_raw_p": p_value,
            "rank_biserial_reference_better": rank_biserial,
            "cohen_dz": float(difference.mean()) / difference_std if difference_std else 0.0,
        })
        raw_p.append(p_value)
    adjusted = holm_adjust(np.asarray(raw_p, dtype=np.float64))
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = float(value)
    return rows


def _bootstrap_mean_ci(values: Sequence[float], seed: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, data.size, size=(BOOTSTRAP_SAMPLES, data.size))]
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _main_classification_tables(
    root: Path, output: Path
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, float]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    summaries, subjects = _load_classification(root, METHOD_ORDER)
    direct = float(summaries["direct_mi9"]["accuracy_subject_mean"])
    ceiling = float(summaries["true22"]["accuracy_subject_mean"])
    denominator = ceiling - direct
    if denominator <= 0:
        raise ValueError("True-22 must exceed Direct MI-9 for recovery-ratio reporting")

    table_rows = []
    for index, method in enumerate(METHOD_ORDER):
        summary = summaries[method]
        accuracy = float(summary["accuracy_subject_mean"])
        ci_low, ci_high = _bootstrap_mean_ci(list(subjects[method].values()), BOOTSTRAP_SEED + index)
        table_rows.append({
            "method": method,
            "label": METHOD_LABELS[method],
            "n_subjects": 9,
            "n_classifier_seeds": int(summary["n_seeds"]),
            "accuracy_mean_percent": 100.0 * accuracy,
            "accuracy_subject_sd_percent": 100.0 * float(summary["accuracy_subject_std"]),
            "accuracy_bootstrap_ci95_low_percent": 100.0 * ci_low,
            "accuracy_bootstrap_ci95_high_percent": 100.0 * ci_high,
            "macro_f1_mean_percent": 100.0 * float(summary["macro_f1_subject_mean"]),
            "macro_f1_subject_sd_percent": 100.0 * float(summary["macro_f1_subject_std"]),
            "cohen_kappa_mean": float(summary["cohen_kappa_subject_mean"]),
            "cohen_kappa_subject_sd": float(summary["cohen_kappa_subject_std"]),
            "recovery_ratio": (accuracy - direct) / denominator,
        })
    _write_csv(output / "tables/table_03_main_classification.csv", table_rows)

    primary_pairs = (
        ("autoencoder_eeg_aware", "direct_mi9"),
        ("autoencoder_eeg_aware", "autoencoder"),
        ("autoencoder_eeg_aware", "spherical_spline"),
        ("autoencoder_eeg_aware", "wgan_gp"),
        ("autoencoder_eeg_aware", "ddpm_standard"),
        ("true22", "autoencoder_eeg_aware"),
    )
    paired = paired_statistics(subjects, primary_pairs)
    _write_csv(output / "statistics/classifier_primary_paired.csv", paired)
    _write_csv(output / "tables/table_05_primary_classifier_statistics.csv", paired)
    friedman = stats.friedmanchisquare(
        *(np.asarray([subjects[method][key] for key in sorted(subjects[method])]) for method in METHOD_ORDER)
    )
    omnibus = [{
        "test": "Friedman",
        "endpoint": "matched_accuracy",
        "n_methods": len(METHOD_ORDER),
        "n_subjects": 9,
        "statistic": float(friedman.statistic),
        "p_value": float(friedman.pvalue),
    }]
    _write_csv(output / "statistics/classifier_omnibus.csv", omnibus)
    return summaries, subjects, table_rows, paired


def _protocol_table(root: Path, output: Path) -> list[dict[str, Any]]:
    summary = _read_json(
        root / "artifacts/preprocessed/bcic2a/canonical_mi9/preprocessing_summary.json"
    )
    shapes = summary["split_shapes"]
    config = summary["config"]
    rows = [
        {"item": "Dataset", "value": "BCI Competition IV Dataset 2a; A01-A09"},
        {"item": "Generalization", "value": "Same-cohort inter-session; not unseen-subject"},
        {"item": "Development split", "value": f"Session 1: train n={shapes['train']['y'][0]}, validation n={shapes['validation']['y'][0]}"},
        {"item": "Final test", "value": f"Held-out Session 2; accepted trials n={shapes['test']['y'][0]}"},
        {"item": "MI-9 channels", "value": ", ".join(summary["mi9_channels"])},
        {"item": "Target", "value": "22 EEG channels referenced to the available MI-9 average"},
        {"item": "Epoch", "value": f"{config['epoch_tmin']:.0f}-{config['epoch_tmax']:.0f} s; 1,001 samples at 250 Hz"},
        {"item": "Filter", "value": f"{config['low_frequency']:.0f}-{config['high_frequency']:.0f} Hz band-pass"},
        {"item": "Normalization", "value": summary["normalization"]},
        {"item": "Classifier", "value": "Pooled-global TCFormer; five seeds (0-4); matched input training"},
        {"item": "Inference unit", "value": "Subject; classifier seeds averaged within subject"},
    ]
    _write_csv(output / "tables/table_01_protocol.csv", rows)
    return rows


def _method_table(root: Path, output: Path) -> list[dict[str, Any]]:
    config_paths = {
        "spherical_spline": root / "configs/restoration/spherical_spline.yaml",
        "autoencoder": root / "configs/restoration/autoencoder.yaml",
        "autoencoder_eeg_aware": root / "configs/restoration/autoencoder_eeg_aware.yaml",
        "ddpm_standard": root / "configs/restoration/ddpm_standard.yaml",
        "wgan_gp": root / "configs/restoration/wgan_gp.yaml",
    }
    complexity = {
        row["method"]: row for row in _read_csv(
            root / "artifacts/experiments/system/latency_benchmark/results/complexity_summary.csv"
        )
    }
    completion_paths = {
        "autoencoder": root / "artifacts/experiments/restoration/autoencoder/canonical_mi9/checkpoints/seed_0/training_complete.json",
        "autoencoder_eeg_aware": root / "artifacts/experiments/restoration/autoencoder/eeg_aware_canonical_mi9/checkpoints/seed_0/training_complete.json",
        "ddpm_standard": root / "artifacts/experiments/restoration/diffusion/standard_canonical_mi9/checkpoints/seed_0/training_complete.json",
        "wgan_gp": root / "artifacts/experiments/restoration/gan/wgan_gp_canonical_mi9/checkpoints/seed_0/training_complete.json",
    }
    family = {
        "spherical_spline": "Classical interpolation",
        "autoencoder": "Deterministic neural",
        "autoencoder_eeg_aware": "Deterministic EEG-aware neural",
        "ddpm_standard": "Diffusion generative",
        "wgan_gp": "Adversarial generative",
    }
    rows = []
    for method, path in config_paths.items():
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        training = config.get("training", {})
        inference = config.get("inference", {})
        loss = config.get("loss", {})
        completion = _read_json(completion_paths[method]) if method in completion_paths else {}
        if method == "autoencoder":
            objective = "Missing-channel time-domain MSE"
        elif method == "autoencoder_eeg_aware":
            objective = "Time MSE + 0.1 bandpower + 1.0 spatial"
        elif method == "ddpm_standard":
            objective = "Noise-prediction MSE; linear beta schedule"
        elif method == "wgan_gp":
            objective = "1.0 reconstruction + 0.1 adversarial; GP=10"
        else:
            objective = "Spherical spline; alpha=1e-5"
        rows.append({
            "method": method,
            "label": METHOD_LABELS[method],
            "family": family[method],
            "training_seed": config.get("seed", "N/A"),
            "objective": objective,
            "optimizer": training.get("optimizer", "N/A"),
            "learning_rate": training.get("learning_rate", "N/A"),
            "batch_size": training.get("batch_size", "N/A"),
            "max_epochs": training.get("epochs", "N/A"),
            "selected_epoch": completion.get("best_epoch", "N/A"),
            "stop_reason": completion.get("stop_reason", "deterministic/no training"),
            "sampler": inference.get("sampler", "deterministic"),
            "sampling_steps": inference.get("sampling_steps", 1),
            "restoration_inference_parameters": int(complexity[method]["restoration_inference_parameters"]),
            "restoration_training_parameters": int(complexity[method]["restoration_training_parameters"]),
            "loss_weights": (
                f"time={loss.get('time_weight')}, band={loss.get('bandpower_weight')}, spatial={loss.get('spatial_weight')}"
                if loss else "N/A"
            ),
        })
    _write_csv(output / "tables/table_02_restoration_methods.csv", rows)
    return rows


def _signal_table(root: Path, output: Path) -> list[dict[str, Any]]:
    signal = {
        row["method"]: row for row in _read_csv(
            root / "artifacts/experiments/analysis/task_relevant_signal/results/method_summary.csv"
        )
    }
    reconstruction_paths = {
        "spherical_spline": root / "artifacts/experiments/restoration/spherical_spline/canonical_mi9/results/reconstruction_summary.csv",
        "autoencoder": root / "artifacts/experiments/restoration/autoencoder/canonical_mi9/results/reconstruction_summary.csv",
        "autoencoder_eeg_aware": root / "artifacts/experiments/restoration/autoencoder/eeg_aware_canonical_mi9/results/reconstruction_summary.csv",
        "ddpm_standard": root / "artifacts/experiments/restoration/diffusion/standard_canonical_mi9/results/reconstruction_summary.csv",
        "wgan_gp": root / "artifacts/experiments/restoration/gan/wgan_gp_canonical_mi9/results/reconstruction_summary.csv",
    }
    reconstruction = {}
    for method, path in reconstruction_paths.items():
        test_rows = [row for row in _read_csv(path) if row["split"] == "test"]
        if len(test_rows) != 1:
            raise ValueError(f"Expected one test row in {path}")
        reconstruction[method] = test_rows[0]
    rows = []
    for method in ("zero_padded_mi9", "spherical_spline", "autoencoder", "autoencoder_eeg_aware", "ddpm_standard", "wgan_gp"):
        waveform = reconstruction.get(method)
        row = signal[method]
        rows.append({
            "method": method,
            "label": METHOD_LABELS[method],
            "missing13_mse": "N/A" if waveform is None else float(waveform["missing13_mse"]),
            "missing13_mae": "N/A" if waveform is None else float(waveform["missing13_mae"]),
            "missing13_correlation": "N/A" if waveform is None else float(waveform["missing13_correlation"]),
            "class_mu_beta_relative_power_mse": float(row["bandpower_mse"]),
            "covariance_airm": float(row["covariance_distance"]),
            "csp_feature_mse": float(row["csp_feature_mse"]),
            "csp_feature_correlation": float(row["csp_feature_correlation"]),
        })
    _write_csv(output / "tables/table_04_signal_metrics.csv", rows)
    signal_stats = _read_csv(
        root / "artifacts/experiments/analysis/task_relevant_signal/results/paired_statistics.csv"
    )
    _write_csv(output / "statistics/signal_endpoint_paired.csv", signal_stats)
    _write_csv(output / "tables/table_05b_signal_endpoint_statistics.csv", signal_stats)
    return rows


def _ablation_tables(root: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    methods = (
        "autoencoder",
        "autoencoder_bandpower",
        "autoencoder_spatial",
        "autoencoder_eeg_aware",
    )
    summaries, subjects = _load_classification(root, methods)
    weights = {
        "autoencoder": "time=1, band=0, spatial=0",
        "autoencoder_bandpower": "time=1, band=0.1, spatial=0",
        "autoencoder_spatial": "time=1, band=0, spatial=1",
        "autoencoder_eeg_aware": "time=1, band=0.1, spatial=1",
    }
    rows = []
    for method in methods:
        summary = summaries[method]
        rows.append({
            "variant": method,
            "label": METHOD_LABELS[method],
            "loss_weights": weights[method],
            "accuracy_mean_percent": 100.0 * float(summary["accuracy_subject_mean"]),
            "accuracy_subject_sd_percent": 100.0 * float(summary["accuracy_subject_std"]),
            "macro_f1_mean_percent": 100.0 * float(summary["macro_f1_subject_mean"]),
            "macro_f1_subject_sd_percent": 100.0 * float(summary["macro_f1_subject_std"]),
            "cohen_kappa_mean": float(summary["cohen_kappa_subject_mean"]),
            "cohen_kappa_subject_sd": float(summary["cohen_kappa_subject_std"]),
        })
    _write_csv(output / "tables/table_06_ae_ablation.csv", rows)
    paired = paired_statistics(
        subjects,
        (
            ("autoencoder_eeg_aware", "autoencoder"),
            ("autoencoder_eeg_aware", "autoencoder_bandpower"),
            ("autoencoder_eeg_aware", "autoencoder_spatial"),
        ),
        seed=BOOTSTRAP_SEED + 100,
    )
    _write_csv(output / "statistics/ae_ablation_paired.csv", paired)
    return rows, paired


def _latency_table(root: Path, output: Path) -> list[dict[str, Any]]:
    source = {
        row["method"]: row for row in _read_csv(
            root / "artifacts/experiments/system/latency_benchmark/results/method_summary.csv"
        )
    }
    rows = []
    for method in METHOD_ORDER:
        row = source[method]
        rows.append({
            "method": method,
            "label": METHOD_LABELS[method],
            "restoration_median_ms": float(row["restoration_ms_median"]),
            "restoration_p95_ms": float(row["restoration_ms_p95"]),
            "processing_median_ms": float(row["end_to_end_ms_median"]),
            "processing_p95_ms": float(row["end_to_end_ms_p95"]),
            "total_inference_parameters": int(row["total_inference_parameters"]),
            "peak_allocated_gpu_mb": float(row["peak_gpu_memory_mb"]),
            "sampling_steps": int(row["sampling_steps"]),
        })
    _write_csv(output / "tables/table_07_processing_cost.csv", rows)
    return rows


def _frozen_table(root: Path, output: Path) -> list[dict[str, Any]]:
    rows = []
    for method in ("spherical_spline", "autoencoder", "autoencoder_eeg_aware", "ddpm_standard", "wgan_gp"):
        source = _read_csv(
            root / f"artifacts/experiments/classification/frozen_oracle/{method}/results/experiment_summary.csv"
        )[0]
        rows.append({
            "method": method,
            "label": METHOD_LABELS[method],
            "frozen_accuracy_mean_percent": 100.0 * float(source["restored_accuracy_subject_mean"]),
            "frozen_accuracy_subject_sd_percent": 100.0 * float(source["restored_accuracy_subject_std"]),
            "oracle_agreement_mean_percent": 100.0 * float(source["prediction_agreement_subject_mean"]),
            "oracle_agreement_subject_sd_percent": 100.0 * float(source["prediction_agreement_subject_std"]),
            "probability_l1_mean": float(source["probability_l1_subject_mean"]),
            "probability_l1_subject_sd": float(source["probability_l1_subject_std"]),
        })
    _write_csv(output / "supplementary/table_s01_frozen_oracle.csv", rows)
    return rows


def _subject_matrix(output: Path, subjects: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for subject_id in sorted(subjects["direct_mi9"]):
        row: dict[str, Any] = {"subject_id": subject_id}
        for method in METHOD_ORDER:
            row[f"{method}_accuracy_percent"] = 100.0 * subjects[method][subject_id]
        rows.append(row)
    _write_csv(output / "supplementary/table_s02_subject_accuracy.csv", rows)
    return rows


def _figure_pipeline(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x: float, y: float, width: float, height: float, text: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.5, edgecolor=color, facecolor=color + "18",
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color="#555555", linewidth=1.4))

    box(0.2, 1.8, 1.8, 1.1, "Canonical MI-9\nSession 2", "#4C78A8")
    box(2.7, 0.25, 2.6, 0.75, "Direct 9-channel", "#4C78A8")
    box(2.7, 1.15, 2.6, 0.75, "Zero padding", "#9ECAE9")
    box(2.7, 2.05, 2.6, 0.75, "Spherical spline", "#59A14F")
    box(2.7, 2.95, 2.6, 0.75, "AE / EEG-aware AE", "#E15759")
    box(2.7, 3.85, 2.6, 0.75, "DDPM / WGAN-GP", "#76B7B2")
    box(6.0, 1.45, 2.2, 1.6, "Matched TCFormer\n(same capacity)\n5 seeds", "#F28E2B")
    box(9.0, 0.55, 2.5, 1.0, "MI decoding\nAccuracy / F1 / kappa", "#B07AA1")
    box(9.0, 1.95, 2.5, 1.0, "Signal fidelity\nSpectral / AIRM / CSP", "#59A14F")
    box(9.0, 3.35, 2.5, 1.0, "Processing cost\nLatency / memory", "#777777")
    for y in (0.625, 1.525, 2.425, 3.325, 4.225):
        arrow(2.0, 2.35, 2.7, y)
        arrow(5.3, y, 6.0, 2.25)
    for y in (1.05, 2.45, 3.85):
        arrow(8.2, 2.25, 9.0, y)
    ax.text(6.0, 0.05, "True-22 is evaluated separately as the measured-channel ceiling.", fontsize=8.5, color="#444444")
    ax.set_title("Leakage-controlled comparison of low-channel EEG restoration methods", fontsize=14, weight="bold", pad=8)
    _save_figure(fig, output / "figures/figure_01_study_pipeline")


def _figure_accuracy(output: Path, subjects: dict[str, dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.7))
    subject_ids = sorted(subjects["direct_mi9"])
    offsets = np.linspace(-0.16, 0.16, len(subject_ids))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for index, method in enumerate(METHOD_ORDER):
        values = 100.0 * np.asarray([subjects[method][key] for key in subject_ids])
        ax.scatter(
            index + offsets, values, s=27, color=COLORS[method], alpha=0.48,
            edgecolors="white", linewidths=0.4, zorder=2,
        )
        sampled = values[rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))].mean(axis=1)
        low, high = np.quantile(sampled, (0.025, 0.975))
        ax.errorbar(
            index, values.mean(), yerr=[[values.mean() - low], [high - values.mean()]],
            fmt="o", markersize=7, color=COLORS[method], markeredgecolor="black",
            markeredgewidth=0.7, capsize=4, linewidth=2.0, zorder=3,
        )
    ax.axhline(25.0, color="#888888", linestyle="--", linewidth=1.0, label="Chance level")
    ax.set_xticks(range(len(METHOD_ORDER)), [METHOD_LABELS[m].replace(" ", "\n", 1) for m in METHOD_ORDER], fontsize=8.5)
    ax.set_ylabel("Session-2 accuracy (%)")
    ax.set_ylim(20, 92)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Matched TCFormer decoding across subjects", weight="bold")
    ax.text(0.01, 0.98, "Points: five-seed subject means; bars: 95% subject bootstrap CI", transform=ax.transAxes, va="top", fontsize=8.5, color="#444444")
    _save_figure(fig, output / "figures/figure_02_matched_accuracy")


def _figure_signal_task(
    output: Path, signal_rows: Sequence[dict[str, Any]], classification_rows: Sequence[dict[str, Any]]
) -> None:
    methods = ("spherical_spline", "autoencoder", "autoencoder_eeg_aware", "ddpm_standard", "wgan_gp")
    signal = {row["method"]: row for row in signal_rows}
    classification = {row["method"]: row for row in classification_rows}
    endpoints = (
        ("class_mu_beta_relative_power_mse", "Class μ/β power MSE", True),
        ("covariance_airm", "Covariance AIRM", False),
        ("csp_feature_mse", "CSP feature MSE", False),
        ("accuracy_mean_percent", "Matched accuracy (%)", False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for ax, (field, title, log_scale) in zip(axes.flat, endpoints, strict=True):
        values = [
            float(classification[m][field]) if field == "accuracy_mean_percent" else float(signal[m][field])
            for m in methods
        ]
        ax.bar(range(len(methods)), values, color=[COLORS[m] for m in methods], edgecolor="white")
        ax.set_xticks(range(len(methods)), ["Spline", "AE", "EEG-aware\nAE", "DDPM", "WGAN-GP"], fontsize=8.5)
        ax.set_title(title, weight="bold")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(axis="y", color="#E2E2E2", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Signal fidelity does not monotonically predict MI decoding", fontsize=14, weight="bold")
    fig.tight_layout()
    _save_figure(fig, output / "figures/figure_03_signal_task_tradeoff")


def _figure_latency(output: Path, latency_rows: Sequence[dict[str, Any]], classification_rows: Sequence[dict[str, Any]]) -> None:
    accuracy = {row["method"]: float(row["accuracy_mean_percent"]) for row in classification_rows}
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    for row in latency_rows:
        method = row["method"]
        x = float(row["processing_median_ms"])
        y = accuracy[method]
        ax.scatter(x, y, s=80, color=COLORS[method], edgecolor="black", linewidth=0.5, zorder=3)
        offsets = {
            "true22": (5, 7),
            "direct_mi9": (5, 5),
            "zero_padded_mi9": (5, -14),
            "spherical_spline": (5, 7),
            "autoencoder": (5, -13),
            "autoencoder_eeg_aware": (5, 5),
            "ddpm_standard": (5, 5),
            "wgan_gp": (5, 5),
        }
        offset = offsets[method]
        ax.annotate(METHOD_LABELS[method], (x, y), xytext=offset, textcoords="offset points", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("Median processing latency per trial (ms, log scale)")
    ax.set_ylabel("Matched Session-2 accuracy (%)")
    ax.grid(color="#DDDDDD", linewidth=0.7, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Accuracy-processing cost trade-off on NVIDIA RTX A6000", weight="bold")
    fig.subplots_adjust(bottom=0.18)
    ax.text(0.01, -0.17, "Processing: CPU-resident input to CPU-returned prediction; excludes ROS2 and actuation.", transform=ax.transAxes, fontsize=8, color="#444444")
    _save_figure(fig, output / "figures/figure_04_accuracy_latency")


def _figure_ablation(root: Path, output: Path) -> None:
    methods = (
        "autoencoder",
        "autoencoder_bandpower",
        "autoencoder_spatial",
        "autoencoder_eeg_aware",
    )
    _, subjects = _load_classification(root, methods)
    subject_ids = sorted(subjects["autoencoder"])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    offsets = np.linspace(-0.13, 0.13, len(subject_ids))
    rng = np.random.default_rng(BOOTSTRAP_SEED + 200)
    for index, method in enumerate(methods):
        values = 100.0 * np.asarray([subjects[method][key] for key in subject_ids])
        ax.scatter(
            index + offsets, values, s=29, color=COLORS[method], alpha=0.5,
            edgecolors="white", linewidths=0.4, zorder=2,
        )
        sampled = values[rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))].mean(axis=1)
        low, high = np.quantile(sampled, (0.025, 0.975))
        ax.errorbar(
            index, values.mean(), yerr=[[values.mean() - low], [high - values.mean()]],
            fmt="o", markersize=8, color=COLORS[method], markeredgecolor="black",
            markeredgewidth=0.7, capsize=4, linewidth=2.0, zorder=3,
        )
    ax.set_xticks(range(len(methods)), [METHOD_LABELS[method] for method in methods], fontsize=9)
    ax.set_ylabel("Session-2 accuracy (%)")
    ax.set_ylim(35, 88)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("EEG-aware autoencoder objective ablation", weight="bold")
    ax.text(0.01, 0.98, "Points: five-seed subject means; bars: 95% subject bootstrap CI", transform=ax.transAxes, va="top", fontsize=8.5, color="#444444")
    _save_figure(fig, output / "figures/figure_05_ae_ablation")


def _write_table_markdown(
    output: Path,
    protocol: Sequence[dict[str, Any]],
    methods: Sequence[dict[str, Any]],
    classification: Sequence[dict[str, Any]],
    signal: Sequence[dict[str, Any]],
    paired: Sequence[dict[str, Any]],
    ablation: Sequence[dict[str, Any]],
    ablation_paired: Sequence[dict[str, Any]],
    latency: Sequence[dict[str, Any]],
    frozen: Sequence[dict[str, Any]],
) -> None:
    sections = ["# Manuscript-ready tables", ""]
    sections.extend(["## Table 1. Evaluation protocol", "", _markdown_table(
        ["Item", "Fixed value"], [[row["item"], row["value"]] for row in protocol]
    ), ""])
    sections.extend(["## Table 2. Restoration methods and reproducibility settings", "", _markdown_table(
        ["Method", "Family", "Objective", "Optimizer", "LR", "Batch", "Max/selected epoch", "Sampler/steps", "Inference params"],
        [[row["label"], row["family"], row["objective"], row["optimizer"], row["learning_rate"], row["batch_size"], f"{row['max_epochs']}/{row['selected_epoch']}", f"{row['sampler']}/{row['sampling_steps']}", f"{row['restoration_inference_parameters']:,}"] for row in methods]
    ), ""])
    sections.extend(["## Table 3. Main matched-classifier results", "", _markdown_table(
        ["Input", "Accuracy, %", "95% CI, %", "Macro-F1, %", "Kappa", "Recovery"],
        [[row["label"], f"{row['accuracy_mean_percent']:.2f} ± {row['accuracy_subject_sd_percent']:.2f}", f"[{row['accuracy_bootstrap_ci95_low_percent']:.2f}, {row['accuracy_bootstrap_ci95_high_percent']:.2f}]", f"{row['macro_f1_mean_percent']:.2f} ± {row['macro_f1_subject_sd_percent']:.2f}", f"{row['cohen_kappa_mean']:.3f} ± {row['cohen_kappa_subject_sd']:.3f}", f"{row['recovery_ratio']:.3f}"] for row in classification]
    ), "", "Values are five-seed means within subject followed by mean ± SD across nine subjects. Confidence intervals are 10,000-sample subject bootstrap intervals.", ""])
    sections.extend(["## Table 4. Held-out signal and task-information endpoints", "", _markdown_table(
        ["Method", "Missing-13 MSE", "Correlation", "μ/β MSE", "AIRM", "CSP MSE", "CSP r"],
        [[row["label"], row["missing13_mse"] if row["missing13_mse"] == "N/A" else f"{row['missing13_mse']:.4f}", row["missing13_correlation"] if row["missing13_correlation"] == "N/A" else f"{row['missing13_correlation']:.4f}", f"{row['class_mu_beta_relative_power_mse']:.5f}", f"{row['covariance_airm']:.4f}", f"{row['csp_feature_mse']:.4f}", f"{row['csp_feature_correlation']:.4f}"] for row in signal]
    ), "", "Zero-padding's missing-channel log-power error is a structural zero-power reference, not finite reconstruction fidelity.", ""])
    sections.extend(["## Table 5. Primary paired classifier statistics", "", _markdown_table(
        ["Comparison", "Difference, pp", "95% CI, pp", "Wins", "Rank-biserial", "Cohen dz", "Raw p", "Holm p"],
        [[f"{METHOD_LABELS[row['reference']]} vs {METHOD_LABELS[row['compared']]}", f"{row['mean_difference_accuracy_pp']:.2f}", f"[{row['bootstrap_ci95_low_pp']:.2f}, {row['bootstrap_ci95_high_pp']:.2f}]", f"{row['reference_wins']}/9", f"{row['rank_biserial_reference_better']:.3f}", f"{row['cohen_dz']:.3f}", f"{row['wilcoxon_raw_p']:.4f}", f"{row['holm_adjusted_p']:.4f}"] for row in paired]
    ), ""])
    sections.extend(["## Table 6. EEG-aware objective ablation", "", _markdown_table(
        ["Variant", "Loss weights", "Accuracy, %", "Macro-F1, %", "Kappa"],
        [[row["label"], row["loss_weights"], f"{row['accuracy_mean_percent']:.2f} ± {row['accuracy_subject_sd_percent']:.2f}", f"{row['macro_f1_mean_percent']:.2f} ± {row['macro_f1_subject_sd_percent']:.2f}", f"{row['cohen_kappa_mean']:.3f} ± {row['cohen_kappa_subject_sd']:.3f}"] for row in ablation]
    ), "", "### Table 6b. Paired ablation statistics", "", _markdown_table(
        ["Comparison", "Difference, pp", "95% CI, pp", "Rank-biserial", "Holm p"],
        [[f"{METHOD_LABELS[row['reference']]} vs {METHOD_LABELS[row['compared']]}", f"{row['mean_difference_accuracy_pp']:.2f}", f"[{row['bootstrap_ci95_low_pp']:.2f}, {row['bootstrap_ci95_high_pp']:.2f}]", f"{row['rank_biserial_reference_better']:.3f}", f"{row['holm_adjusted_p']:.4f}"] for row in ablation_paired]
    ), ""])
    sections.extend(["## Table 7. Formal batch-1 processing cost", "", _markdown_table(
        ["Method", "Restoration median/p95, ms", "Processing median/p95, ms", "Inference params", "Peak GPU MB", "Steps"],
        [[row["label"], f"{row['restoration_median_ms']:.3f}/{row['restoration_p95_ms']:.3f}", f"{row['processing_median_ms']:.3f}/{row['processing_p95_ms']:.3f}", f"{row['total_inference_parameters']:,}", f"{row['peak_allocated_gpu_mb']:.1f}", row["sampling_steps"]] for row in latency]
    ), "", "Processing begins with a CPU-resident trial and ends with a CPU-returned prediction. Model loading, file I/O, ROS2 transport, safety filtering, and actuation are excluded.", ""])
    sections.extend(["## Supplementary Table S1. Frozen-oracle diagnostic", "", _markdown_table(
        ["Method", "Frozen accuracy, %", "Oracle agreement, %", "Probability L1"],
        [[row["label"], f"{row['frozen_accuracy_mean_percent']:.2f} ± {row['frozen_accuracy_subject_sd_percent']:.2f}", f"{row['oracle_agreement_mean_percent']:.2f} ± {row['oracle_agreement_subject_sd_percent']:.2f}", f"{row['probability_l1_mean']:.4f} ± {row['probability_l1_subject_sd']:.4f}"] for row in frozen]
    ), ""])
    _atomic_text(output / "TABLES.md", "\n".join(sections))


def _write_captions(output: Path) -> None:
    captions = """# Figure captions

## Figure 1

Leakage-controlled evaluation framework. All learned restoration models and matched TCFormer classifiers were developed using pooled Session-1 data, while Session 2 remained held out for final evaluation. True 22-channel EEG served only as the measured-channel ceiling.

## Figure 2

Matched TCFormer Session-2 accuracy for the eight retained input conditions. Each point is one subject's mean across five classifier seeds; error bars show 95% bootstrap confidence intervals across nine subjects. Separate, same-capacity classifiers were trained for each input representation.

## Figure 3

Signal fidelity and downstream task utility across restoration methods. Lower values indicate better fidelity for the spectral, covariance, and CSP-error panels, whereas higher matched accuracy is better. The discordant ordering illustrates that global signal similarity does not necessarily preserve trial-level motor-imagery information.

## Figure 4

Accuracy-processing cost trade-off measured with batch size 1 on an NVIDIA RTX A6000. Processing latency spans a CPU-resident trial through restoration and matched TCFormer prediction returned to CPU; acquisition, ROS2 transport, safety filtering, and robot actuation are excluded.

## Figure 5

EEG-aware autoencoder loss ablation. Each point is one subject's mean across five TCFormer seeds; error bars show 95% bootstrap confidence intervals across nine subjects. The combined loss achieved the highest mean, but its advantage over individual AE variants was not statistically significant after Holm correction.

## Supplementary topography and time-frequency figures

The packaged supplementary figures reproduce the all-subject, all-class Stage-D analyses without favorable-subject selection. Their original detailed captions and numerical sources remain in `artifacts/experiments/analysis/task_relevant_signal/`.
"""
    _atomic_text(output / "FIGURE_CAPTIONS.md", captions)


def _write_results_summary(
    output: Path,
    classification: Sequence[dict[str, Any]],
    paired: Sequence[dict[str, Any]],
    ablation_paired: Sequence[dict[str, Any]],
) -> None:
    results = {row["method"]: row for row in classification}
    primary = {row["comparison"]: row for row in paired}
    direct = results["direct_mi9"]["accuracy_mean_percent"]
    eeg_aware = results["autoencoder_eeg_aware"]["accuracy_mean_percent"]
    true22 = results["true22"]["accuracy_mean_percent"]
    summary = f"""# Results summary and writing guidance

## Primary result

True 22-channel input achieved `{true22:.2f}%`, compared with `{direct:.2f}%` for Direct MI-9. EEG-aware AE achieved the highest restored-input mean accuracy (`{eeg_aware:.2f}%`) and recovered `{results['autoencoder_eeg_aware']['recovery_ratio']:.3f}` of the Direct-to-True gap.

EEG-aware AE exceeded Direct MI-9 by `{primary['autoencoder_eeg_aware_vs_direct_mi9']['mean_difference_accuracy_pp']:.2f}` percentage points (95% bootstrap CI `{primary['autoencoder_eeg_aware_vs_direct_mi9']['bootstrap_ci95_low_pp']:.2f}` to `{primary['autoencoder_eeg_aware_vs_direct_mi9']['bootstrap_ci95_high_pp']:.2f}`; Holm-adjusted `p={primary['autoencoder_eeg_aware_vs_direct_mi9']['holm_adjusted_p']:.4f}`). It also exceeded DDPM and WGAN-GP after Holm correction. Differences from the baseline AE and spherical spline were not significant.

## Ablation result

The combined EEG-aware loss had the highest average among the four equal-capacity AE objectives. None of its three component comparisons remained significant after Holm correction; the smallest adjusted p-value was `{min(float(row['holm_adjusted_p']) for row in ablation_paired):.4f}`. Present this as an objective-level trend supported more strongly by spectral fidelity than by classification superiority.

## Interpretation order

1. Establish the significant measured-channel ceiling using Direct MI-9 versus True-22.
2. Compare representative classical, deterministic, diffusion, and adversarial restoration families under the same protocol.
3. Report EEG-aware AE as the best observed multi-criteria trade-off, not a universally superior method.
4. Use spectral, covariance, and CSP results to show that signal proximity does not monotonically determine MI decoding.
5. Close with processing cost. State explicitly that it is not physical ROS2 end-to-end latency.

## Claims to avoid

- Diffusion is necessary, novel, or superior.
- EEG-aware AE is statistically better than baseline AE or spherical spline.
- Restored EEG is equivalent to measured True-22.
- Results establish unseen-subject generalization.
- The complete ROS2/robot system has an 11 ms end-to-end latency.
"""
    _atomic_text(output / "RESULTS_SUMMARY.md", summary)


def _copy_supplementary_figures(root: Path, output: Path) -> None:
    source = root / "artifacts/experiments/analysis/task_relevant_signal/figures"
    mapping = {
        "true22_class_bandpower_topography.png": "figure_s01_true22_class_bandpower_topography.png",
        "restoration_bandpower_error_topography.png": "figure_s02_restoration_bandpower_error_topography.png",
        "restoration_time_frequency_error.png": "figure_s03_restoration_time_frequency_error.png",
        "signal_metric_summary.png": "figure_s04_signal_metric_summary.png",
    }
    target_dir = output / "supplementary/figures"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in mapping.items():
        source_path = source / source_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Required figure is missing: {source_path}")
        shutil.copy2(source_path, target_dir / target_name)


def _source_manifest(root: Path, output: Path) -> list[dict[str, Any]]:
    sources = [
        root / "artifacts/preprocessed/bcic2a/canonical_mi9/preprocessing_summary.json",
        root / "artifacts/experiments/analysis/task_relevant_signal/results/method_summary.csv",
        root / "artifacts/experiments/analysis/task_relevant_signal/results/paired_statistics.csv",
        root / "artifacts/experiments/system/latency_benchmark/results/method_summary.csv",
        root / "artifacts/experiments/system/latency_benchmark/results/complexity_summary.csv",
    ]
    for method in METHOD_ORDER:
        sources.extend(_classification_paths(root, method))
    for method in ("autoencoder_bandpower", "autoencoder_spatial"):
        sources.extend(_classification_paths(root, method))
    sources.extend(root / "configs/restoration" / name for name in (
        "spherical_spline.yaml",
        "autoencoder.yaml",
        "autoencoder_eeg_aware.yaml",
        "ddpm_standard.yaml",
        "wgan_gp.yaml",
    ))
    sources.extend(root / path for path in (
        "artifacts/experiments/restoration/autoencoder/canonical_mi9/checkpoints/seed_0/training_complete.json",
        "artifacts/experiments/restoration/autoencoder/eeg_aware_canonical_mi9/checkpoints/seed_0/training_complete.json",
        "artifacts/experiments/restoration/diffusion/standard_canonical_mi9/checkpoints/seed_0/training_complete.json",
        "artifacts/experiments/restoration/gan/wgan_gp_canonical_mi9/checkpoints/seed_0/training_complete.json",
        "artifacts/experiments/restoration/spherical_spline/canonical_mi9/results/reconstruction_summary.csv",
        "artifacts/experiments/restoration/autoencoder/canonical_mi9/results/reconstruction_summary.csv",
        "artifacts/experiments/restoration/autoencoder/eeg_aware_canonical_mi9/results/reconstruction_summary.csv",
        "artifacts/experiments/restoration/diffusion/standard_canonical_mi9/results/reconstruction_summary.csv",
        "artifacts/experiments/restoration/gan/wgan_gp_canonical_mi9/results/reconstruction_summary.csv",
    ))
    for method in ("spherical_spline", "autoencoder", "autoencoder_eeg_aware", "ddpm_standard", "wgan_gp"):
        sources.append(
            root / f"artifacts/experiments/classification/frozen_oracle/{method}/results/experiment_summary.csv"
        )
    source_figure_dir = root / "artifacts/experiments/analysis/task_relevant_signal/figures"
    sources.extend(source_figure_dir / name for name in (
        "true22_class_bandpower_topography.png",
        "restoration_bandpower_error_topography.png",
        "restoration_time_frequency_error.png",
        "signal_metric_summary.png",
    ))
    rows = []
    for path in sorted(set(sources)):
        data = path.read_bytes()
        rows.append({
            "source_path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        })
    _write_csv(output / "SOURCE_MANIFEST.csv", rows)
    return rows


def _write_readme(output: Path, omnibus: dict[str, str], source_count: int) -> None:
    text = f"""# Manuscript assets

Generated from frozen or completed experiment artifacts. This folder is the convenient writing layer; the authoritative raw outputs remain under `artifacts/`.

## Contents

- `TABLES.md`: all manuscript tables in copy-ready Markdown.
- `RESULTS_SUMMARY.md`: result interpretation order and defensible claim boundaries.
- `MAIN_VS_SUPPLEMENTARY.md`: fixed allocation between the concise article and detailed supplement.
- `tables/`: machine-readable main tables as CSV.
- `statistics/`: subject-level omnibus, paired bootstrap CI, Wilcoxon, Holm, rank-biserial, and Cohen's dz outputs.
- `figures/`: main figures in 300-dpi PNG and vector PDF.
- `supplementary/`: frozen-oracle and subject-level tables plus all-subject signal figures.
- `FIGURE_CAPTIONS.md`: manuscript-ready English captions.
- `SOURCE_MANIFEST.csv`: {source_count} input files and their SHA-256 hashes.

## Statistical contract

- Unit of inference: subject (`n=9`).
- Five classifier seeds are averaged within subject before inference.
- Confidence intervals: 10,000 paired or subject bootstrap resamples, seed `{BOOTSTRAP_SEED}`.
- Pairwise test: two-sided Wilcoxon signed-rank.
- Multiplicity: Holm correction within each prespecified family.
- Effect sizes: matched-pairs rank-biserial correlation and Cohen's dz.
- Eight-method omnibus: Friedman statistic `{float(omnibus['statistic']):.4f}`, p `{float(omnibus['p_value']):.6g}`.

## Rebuild

```bash
.venv/bin/python scripts/build_manuscript_assets.py
.venv/bin/python scripts/build_supplementary_docx.py
```

Both commands are CPU-only and do not train or run any neural network. The first fails
if required inputs are missing, empty, or not aligned across nine subjects; the second
uses LibreOffice to write `supplementary information.docx` in the project root.

## Claim boundary

The latency table reports model processing from a CPU-resident trial to a CPU-returned prediction. It does not include EEG acquisition, ROS2 transport, safety filtering, robot actuation, or full physical-system latency. ROS2 is therefore treated as an application integration layer, not as a newly validated end-to-end timing result.
"""
    _atomic_text(output / "README.md", text)


def build_manuscript_assets(project_root: Path, output_dir: Path) -> Path:
    """Build the complete manuscript package from existing immutable results."""

    root = project_root.resolve()
    output = output_dir if output_dir.is_absolute() else root / output_dir
    output = output.resolve()
    if output == root or root not in output.parents:
        raise ValueError("Output directory must be a dedicated folder inside the project root")
    for directory in ("tables", "statistics", "figures", "supplementary"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    protocol = _protocol_table(root, output)
    methods = _method_table(root, output)
    summaries, subjects, classification, paired = _main_classification_tables(root, output)
    signal = _signal_table(root, output)
    ablation, ablation_paired = _ablation_tables(root, output)
    latency = _latency_table(root, output)
    frozen = _frozen_table(root, output)
    _subject_matrix(output, subjects)

    _write_table_markdown(
        output, protocol, methods, classification, signal, paired,
        ablation, ablation_paired, latency, frozen,
    )
    _figure_pipeline(output)
    _figure_accuracy(output, subjects)
    _figure_signal_task(output, signal, classification)
    _figure_latency(output, latency, classification)
    _figure_ablation(root, output)
    _copy_supplementary_figures(root, output)
    _write_captions(output)
    _write_results_summary(output, classification, paired, ablation_paired)
    sources = _source_manifest(root, output)
    omnibus = _read_csv(output / "statistics/classifier_omnibus.csv")[0]
    _write_readme(output, omnibus, len(sources))

    completion = {
        "status": "complete",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_methods": len(summaries),
        "n_subjects": len(subjects["direct_mi9"]),
        "n_source_files": len(sources),
        "output_dir": str(output.relative_to(root)),
    }
    _atomic_text(output / "BUILD_COMPLETE.json", json.dumps(completion, indent=2) + "\n")
    return output
