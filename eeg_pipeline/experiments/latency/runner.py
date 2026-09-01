"""Formal, restartable batch-1 benchmark for restoration plus matched TCFormer."""

from __future__ import annotations

import csv
import gc
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ...channels import MI9_INDICES
from ..classification.config import ExperimentConfig, load_experiment_config
from ..classification.training import build_model as build_classifier
from ..restoration.config import RestorationConfig, load_restoration_config
from ..restoration.data import MISSING_INDICES, load_subject_normalization
from ..restoration.interpolation import spherical_spline_matrix
from ..restoration.training import (
    _ddpm_sample,
    _finalize_restoration_batch,
    build_diffusion_schedule,
    load_best_model,
)
from .config import LatencyBenchmarkConfig, LatencyMethodConfig


@dataclass(frozen=True)
class SourceTest:
    x_mi9: np.ndarray
    x_true22: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    trial_index: np.ndarray


@dataclass
class PreparedMethod:
    method: LatencyMethodConfig
    classifier_config: ExperimentConfig
    classifier: torch.nn.Module
    restoration_config: RestorationConfig | None
    restorer: torch.nn.Module | None
    restore: Callable[[np.ndarray, int, int], tuple[torch.Tensor, float, float]]
    classifier_checkpoint: Path
    restoration_checkpoint: Path | None
    classifier_parameters: int
    restoration_inference_parameters: int
    restoration_training_parameters: int
    classifier_training_seconds: float | None
    restoration_training_seconds: float | None
    matched_accuracy_subject_mean: float


def _load_source(path: Path) -> SourceTest:
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical Session-2 source: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {"x_mi9", "x_true22", "y", "subject", "trial_index"}
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        source = SourceTest(**{key: payload[key] for key in required})
    if source.x_mi9.ndim != 3 or source.x_mi9.shape[1] != 9:
        raise ValueError(f"Unexpected MI-9 shape: {source.x_mi9.shape}")
    if source.x_true22.shape != (source.x_mi9.shape[0], 22, source.x_mi9.shape[2]):
        raise ValueError(f"Unexpected True-22 shape: {source.x_true22.shape}")
    n_trials = source.y.size
    if any(value.shape != (n_trials,) for value in (source.subject, source.trial_index)):
        raise ValueError("Session-2 metadata shape mismatch")
    if not np.isfinite(source.x_mi9).all() or not np.isfinite(source.x_true22).all():
        raise ValueError("Session-2 source contains non-finite EEG")
    np.testing.assert_array_equal(source.x_true22[:, MI9_INDICES], source.x_mi9)
    return source


def stratified_latency_indices(
    source: SourceTest,
    trials_per_subject_class: int,
    seed: int = 0,
) -> np.ndarray:
    """Select each cell's earliest trials and apply a fixed, result-independent order."""

    selected = []
    for subject in range(1, 10):
        for class_id in range(4):
            indices = np.flatnonzero((source.subject == subject) & (source.y == class_id))
            if indices.size < trials_per_subject_class:
                raise ValueError(
                    f"A{subject:02d}/class {class_id} has only {indices.size} trials; "
                    f"need {trials_per_subject_class}"
                )
            selected.extend(indices[:trials_per_subject_class].tolist())
    selected_array = np.asarray(selected, dtype=np.int64)
    return selected_array[np.random.default_rng(seed).permutation(selected_array.size)]


def summarize_timings(rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ("transfer_ms", "restoration_ms", "classification_ms", "end_to_end_ms"):
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        result.update({
            f"{metric}_mean": float(values.mean()),
            f"{metric}_std": float(values.std(ddof=1)),
            f"{metric}_median": float(np.median(values)),
            f"{metric}_p95": float(np.quantile(values, 0.95)),
            f"{metric}_min": float(values.min()),
            f"{metric}_max": float(values.max()),
        })
    return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, function: Callable[[], Any]) -> tuple[Any, float]:
    _sync(device)
    start = time.perf_counter_ns()
    value = function()
    _sync(device)
    return value, (time.perf_counter_ns() - start) / 1_000_000.0


def _classifier_checkpoint(config: ExperimentConfig, seed: int) -> Path:
    return config.output_dir / config.name / "checkpoints" / f"seed_{seed}" / "best_model.pt"


def _load_classifier(
    config: ExperimentConfig, seed: int, device: torch.device
) -> tuple[torch.nn.Module, Path, float | None]:
    checkpoint = _classifier_checkpoint(config, seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing matched TCFormer checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("seed") != seed or payload.get("input", {}).get("id") != config.input.id:
        raise ValueError(f"Classifier checkpoint identity mismatch: {checkpoint}")
    if payload.get("model_args") != config.model.args:
        raise ValueError(f"Classifier model configuration mismatch: {checkpoint}")
    model = build_classifier(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metrics_path = checkpoint.parent / "metrics.json"
    training_seconds = None
    if metrics_path.is_file():
        training_seconds = float(json.loads(metrics_path.read_text())["training_seconds"])
    return model, checkpoint, training_seconds


def _matched_accuracy(config: ExperimentConfig) -> float:
    path = config.output_dir / config.name / "results/experiment_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one classifier experiment summary row: {path}")
    return float(rows[0]["accuracy_subject_mean"])


def _restoration_checkpoint(config: RestorationConfig) -> Path:
    return config.output.experiment_dir / "checkpoints" / f"seed_{config.seed}" / "best_model.pt"


def _training_metadata(config: RestorationConfig) -> tuple[int, float | None]:
    completion = (
        config.output.experiment_dir / "checkpoints" / f"seed_{config.seed}"
        / "training_complete.json"
    )
    if not completion.is_file():
        return 0, None
    payload = json.loads(completion.read_text(encoding="utf-8"))
    parameters = int(payload.get("parameters", payload.get("generator_parameters", 0)))
    return parameters, float(payload["training_seconds_this_run"])


def _prepare_method(
    method: LatencyMethodConfig,
    project_root: Path,
    source: SourceTest,
    classifier_seed: int,
    device: torch.device,
) -> PreparedMethod:
    classifier_config = load_experiment_config(method.classification_config, project_root)
    if classifier_config.name != method.id:
        raise ValueError(
            f"Latency id/classifier id mismatch: {method.id}/{classifier_config.name}"
        )
    classifier, classifier_checkpoint, classifier_training_seconds = _load_classifier(
        classifier_config, classifier_seed, device
    )
    classifier_parameters = sum(parameter.numel() for parameter in classifier.parameters())
    restoration_config = None
    restorer = None
    restoration_checkpoint = None
    restoration_inference_parameters = 0
    restoration_training_parameters = 0
    restoration_training_seconds = None

    if method.kind == "true22":
        def restore(values: np.ndarray, subject: int, trial_index: int):
            del subject, trial_index
            tensor, transfer_ms = _timed(
                device, lambda: torch.from_numpy(values).float().unsqueeze(0).to(device)
            )
            return tensor, 0.0, transfer_ms
    elif method.kind == "direct_mi9":
        def restore(values: np.ndarray, subject: int, trial_index: int):
            del subject, trial_index
            tensor, transfer_ms = _timed(
                device, lambda: torch.from_numpy(values).float().unsqueeze(0).to(device)
            )
            return tensor, 0.0, transfer_ms
    elif method.kind == "zero_padded":
        def restore(values: np.ndarray, subject: int, trial_index: int):
            del subject, trial_index
            def zero_pad() -> np.ndarray:
                output = np.zeros((22, values.shape[-1]), dtype=np.float32)
                output[np.asarray(MI9_INDICES)] = values
                return output
            output, restoration_ms = _timed(torch.device("cpu"), zero_pad)
            tensor, transfer_ms = _timed(
                device, lambda: torch.from_numpy(output).unsqueeze(0).to(device)
            )
            return tensor, restoration_ms, transfer_ms
    elif method.kind == "spherical_spline":
        assert method.restoration_config is not None
        restoration_config = load_restoration_config(method.restoration_config, project_root)
        if restoration_config.method != "spherical_spline" or restoration_config.name != method.id:
            raise ValueError(f"Invalid spherical-spline config for {method.id}")
        matrix = spherical_spline_matrix(float(restoration_config.model.get("alpha", 1e-5)))
        normalization = {
            subject: load_subject_normalization(restoration_config.source.normalization_dir, subject)
            for subject in range(1, 10)
        }
        def restore(values: np.ndarray, subject: int, trial_index: int):
            del trial_index
            def interpolate() -> np.ndarray:
                mean, std = normalization[subject]
                observed = np.asarray(MI9_INDICES)
                physical = values.astype(np.float64) * std[observed, None] + mean[observed, None]
                output = np.empty((22, values.shape[-1]), dtype=np.float64)
                output[observed] = physical
                output[np.asarray(MISSING_INDICES)] = matrix @ physical
                output = (output - mean[:, None]) / std[:, None]
                result = output.astype(np.float32)
                result[observed] = values
                return result
            output, restoration_ms = _timed(torch.device("cpu"), interpolate)
            tensor, transfer_ms = _timed(
                device, lambda: torch.from_numpy(output).unsqueeze(0).to(device)
            )
            return tensor, restoration_ms, transfer_ms
    else:
        assert method.restoration_config is not None
        restoration_config = load_restoration_config(method.restoration_config, project_root)
        if restoration_config.name != method.id:
            raise ValueError(
                f"Latency id/restoration id mismatch: {method.id}/{restoration_config.name}"
            )
        full_model = load_best_model(restoration_config, device)
        restoration_checkpoint = _restoration_checkpoint(restoration_config)
        restoration_training_parameters, restoration_training_seconds = _training_metadata(
            restoration_config
        )
        if restoration_config.method == "wgan_gp":
            restorer = full_model.generator
            restoration_inference_parameters = sum(
                parameter.numel() for parameter in restorer.parameters()
            )
            del full_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            restorer = full_model
            restoration_inference_parameters = sum(
                parameter.numel() for parameter in restorer.parameters()
            )
        schedule = (
            build_diffusion_schedule(restoration_config, device)
            if restoration_config.method == "ddpm" else None
        )
        amp_enabled = restoration_config.training.amp and device.type == "cuda"
        def restore(values: np.ndarray, subject: int, trial_index: int):
            input_tensor, transfer_ms = _timed(
                device, lambda: torch.from_numpy(values).float().unsqueeze(0).to(device)
            )
            generator = torch.Generator(device=device).manual_seed(
                restoration_config.inference.seed + subject * 10_000 + trial_index
            )
            def infer() -> torch.Tensor:
                with torch.inference_mode(), torch.amp.autocast(
                    device_type=device.type, enabled=amp_enabled
                ):
                    if restoration_config.method in {"autoencoder", "wgan_gp"}:
                        output = restorer(input_tensor)
                    elif restoration_config.method == "ddpm" and schedule is not None:
                        output = _ddpm_sample(
                            restorer, input_tensor, restoration_config, schedule, generator
                        )
                    else:
                        raise ValueError(f"Unsupported latency restorer: {restoration_config.method}")
                    return _finalize_restoration_batch(output, input_tensor)
            output, restoration_ms = _timed(device, infer)
            return output, restoration_ms, transfer_ms

    return PreparedMethod(
        method=method,
        classifier_config=classifier_config,
        classifier=classifier,
        restoration_config=restoration_config,
        restorer=restorer,
        restore=restore,
        classifier_checkpoint=classifier_checkpoint,
        restoration_checkpoint=restoration_checkpoint,
        classifier_parameters=classifier_parameters,
        restoration_inference_parameters=restoration_inference_parameters,
        restoration_training_parameters=restoration_training_parameters,
        classifier_training_seconds=classifier_training_seconds,
        restoration_training_seconds=restoration_training_seconds,
        matched_accuracy_subject_mean=_matched_accuracy(classifier_config),
    )


def _run_trial(
    prepared: PreparedMethod,
    source: SourceTest,
    source_index: int,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    method = prepared.method
    subject = int(source.subject[source_index])
    trial_index = int(source.trial_index[source_index])
    values = (
        source.x_true22[source_index]
        if method.kind == "true22" else source.x_mi9[source_index]
    )
    _sync(device)
    total_start = time.perf_counter_ns()
    restored, restoration_ms, transfer_ms = prepared.restore(values, subject, trial_index)
    amp_enabled = prepared.classifier_config.training.amp and device.type == "cuda"
    def classify() -> torch.Tensor:
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type, enabled=amp_enabled
        ):
            return prepared.classifier(restored.unsqueeze(1)).argmax(dim=1)
    prediction, classification_ms = _timed(device, classify)
    predicted_class = int(prediction.cpu().item())
    _sync(device)
    end_to_end_ms = (time.perf_counter_ns() - total_start) / 1_000_000.0
    row = {
        "method": method.id,
        "source_index": source_index,
        "subject": subject,
        "subject_id": f"A{subject:02d}",
        "class": int(source.y[source_index]),
        "trial_index": trial_index,
        "transfer_ms": transfer_ms,
        "restoration_ms": restoration_ms,
        "classification_ms": classification_ms,
        "end_to_end_ms": end_to_end_ms,
        "prediction": predicted_class,
        "correct": int(predicted_class == int(source.y[source_index])),
    }
    return row, restored


def _validate_online_output(
    prepared: PreparedMethod, output: torch.Tensor, source: SourceTest, source_index: int
) -> None:
    expected_channels = prepared.classifier_config.input.n_channels
    if tuple(output.shape) != (1, expected_channels, source.x_mi9.shape[-1]):
        raise ValueError(f"Unexpected online output for {prepared.method.id}: {tuple(output.shape)}")
    if not bool(torch.isfinite(output).all().item()):
        raise ValueError(f"Non-finite online output for {prepared.method.id}")
    if expected_channels == 22 and prepared.method.kind != "true22":
        observed = output[0, list(MI9_INDICES)].float().cpu().numpy()
        np.testing.assert_array_equal(observed, source.x_mi9[source_index])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _method_signature(
    config: LatencyBenchmarkConfig,
    method: LatencyMethodConfig,
    device: torch.device,
    warmup_trials: int,
    trials_per_subject_class: int,
) -> dict[str, Any]:
    method_values = asdict(method)
    method_values["classification_config"] = str(method.classification_config)
    method_values["restoration_config"] = (
        None if method.restoration_config is None else str(method.restoration_config)
    )
    return {
        "benchmark_id": config.id,
        "method": method_values,
        "classifier_seed": config.classifier_seed,
        "benchmark_seed": config.benchmark_seed,
        "cpu_threads": config.cpu_threads,
        "allow_tf32": config.allow_tf32,
        "device": str(device),
        "warmup_trials": warmup_trials,
        "trials_per_subject_class": trials_per_subject_class,
        "timing": "perf_counter_ns with CUDA synchronize before/after each stage",
        "batch_size": 1,
    }


def _benchmark_method(
    prepared: PreparedMethod,
    source: SourceTest,
    indices: np.ndarray,
    config: LatencyBenchmarkConfig,
    device: torch.device,
    warmup_trials: int,
    trials_per_subject_class: int,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    method_dir = output_dir / "methods" / prepared.method.id
    summary_path = method_dir / "summary.json"
    signature = _method_signature(
        config, prepared.method, device, warmup_trials, trials_per_subject_class
    )
    if summary_path.is_file() and not overwrite:
        summary = json.loads(summary_path.read_text())
        if summary.get("signature") != signature:
            raise ValueError(
                f"Completed latency method has a different signature: {method_dir}. "
                "Use a new output directory or --overwrite."
            )
        print(f"SKIP completed latency method: {prepared.method.id}")
        return summary["summary"]
    if method_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Incomplete latency method exists: {method_dir}. Use --overwrite intentionally."
            )
        shutil.rmtree(method_dir)
    method_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{prepared.method.id}.", dir=method_dir.parent))
    try:
        for position in range(warmup_trials):
            source_index = int(indices[position % len(indices)])
            _, output = _run_trial(prepared, source, source_index, device)
        _validate_online_output(prepared, output, source, source_index)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rows = []
        for order, source_index_value in enumerate(indices):
            row, _ = _run_trial(prepared, source, int(source_index_value), device)
            row["measurement_order"] = order
            rows.append(row)
        timing = summarize_timings(rows)
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda" else None
        )
        restoration_checkpoint_bytes = (
            prepared.restoration_checkpoint.stat().st_size
            if prepared.restoration_checkpoint is not None else 0
        )
        summary = {
            "method": prepared.method.id,
            "label": prepared.method.label,
            "kind": prepared.method.kind,
            "device": str(device),
            "n_trials": len(rows),
            "warmup_trials": warmup_trials,
            "classifier_seed": config.classifier_seed,
            "matched_accuracy_subject_mean": prepared.matched_accuracy_subject_mean,
            "sanity_subset_accuracy": float(np.mean([row["correct"] for row in rows])),
            "classifier_parameters": prepared.classifier_parameters,
            "restoration_inference_parameters": prepared.restoration_inference_parameters,
            "restoration_training_parameters": prepared.restoration_training_parameters,
            "total_inference_parameters": (
                prepared.classifier_parameters + prepared.restoration_inference_parameters
            ),
            "classifier_checkpoint_bytes": prepared.classifier_checkpoint.stat().st_size,
            "restoration_checkpoint_bytes": restoration_checkpoint_bytes,
            "total_checkpoint_bytes": (
                prepared.classifier_checkpoint.stat().st_size + restoration_checkpoint_bytes
            ),
            "classifier_training_seconds_seed0": prepared.classifier_training_seconds,
            "restoration_training_seconds_seed0": prepared.restoration_training_seconds,
            "sampling_steps": (
                prepared.restoration_config.inference.sampling_steps
                if prepared.restoration_config is not None
                and prepared.restoration_config.method == "ddpm" else 1
            ),
            "peak_gpu_memory_mb": peak_memory_mb,
            **timing,
        }
        _write_csv(temporary / "latency_by_trial.csv", rows)
        (temporary / "summary.json").write_text(
            json.dumps({"signature": signature, "summary": summary}, indent=2) + "\n"
        )
        os.replace(temporary, method_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def _system_info(device: torch.device, config: LatencyBenchmarkConfig) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "configured_cpu_threads": config.cpu_threads,
        "allow_tf32": config.allow_tf32,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info.update({
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        })
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True, capture_output=True, text=True, timeout=10,
            )
            info["nvidia_driver"] = result.stdout.splitlines()[device.index or 0].strip()
        except (OSError, subprocess.SubprocessError, IndexError):
            info["nvidia_driver"] = "unavailable"
    return info


def _write_aggregate(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    results = output_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    _write_csv(results / "method_summary.csv", summaries)
    complexity_keys = (
        "method", "label", "device", "sampling_steps", "classifier_parameters",
        "restoration_inference_parameters", "restoration_training_parameters",
        "total_inference_parameters", "classifier_checkpoint_bytes",
        "restoration_checkpoint_bytes", "total_checkpoint_bytes",
        "classifier_training_seconds_seed0", "restoration_training_seconds_seed0",
        "peak_gpu_memory_mb",
    )
    _write_csv(
        results / "complexity_summary.csv",
        [{key: row.get(key) for key in complexity_keys} for row in summaries],
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for row in summaries:
        axis.scatter(
            float(row["end_to_end_ms_median"]),
            100 * float(row["matched_accuracy_subject_mean"]),
            s=45,
        )
        axis.annotate(
            str(row["label"]),
            (float(row["end_to_end_ms_median"]), 100 * float(row["matched_accuracy_subject_mean"])),
            xytext=(4, 4), textcoords="offset points", fontsize=8,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Batch-1 end-to-end processing latency, median (ms; log scale)")
    axis.set_ylabel("Matched Session-2 accuracy (%)")
    axis.set_title("Accuracy–latency comparison")
    figure.savefig(figure_dir / "accuracy_latency_tradeoff.png", dpi=200)
    plt.close(figure)
    lines = [
        "# Formal batch-1 latency benchmark", "",
        "Warm-up, model loading, spline-matrix construction, and file I/O are excluded. "
        "End-to-end processing starts with one CPU-resident trial and ends when its "
        "matched TCFormer prediction is available on CPU. CUDA synchronization brackets "
        "every measured stage.", "",
        "| Method | Restore median/p95 (ms) | Classify median/p95 (ms) | End-to-end median/p95 (ms) | Peak GPU MB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        peak = "N/A" if row["peak_gpu_memory_mb"] is None else f"{row['peak_gpu_memory_mb']:.1f}"
        lines.append(
            f"| {row['label']} | {row['restoration_ms_median']:.3f}/{row['restoration_ms_p95']:.3f} | "
            f"{row['classification_ms_median']:.3f}/{row['classification_ms_p95']:.3f} | "
            f"{row['end_to_end_ms_median']:.3f}/{row['end_to_end_ms_p95']:.3f} | {peak} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_latency_benchmark(
    config: LatencyBenchmarkConfig,
    project_root: Path,
    *,
    device_override: str | None = None,
    method_ids: tuple[str, ...] | None = None,
    warmup_trials: int | None = None,
    trials_per_subject_class: int | None = None,
    output_dir_override: Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Path:
    """Run selected methods and aggregate only when every configured method is complete."""

    device = torch.device(device_override or "cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    output_dir = output_dir_override or config.output_dir
    warmup = warmup_trials if warmup_trials is not None else config.warmup_trials
    cell_trials = (
        trials_per_subject_class
        if trials_per_subject_class is not None else config.trials_per_subject_class
    )
    if warmup < 1 or cell_trials < 1:
        raise ValueError("Warm-up and trials per subject/class must be positive")
    selected = config.methods
    if method_ids is not None:
        unknown = set(method_ids) - {method.id for method in config.methods}
        if unknown:
            raise ValueError(f"Unknown latency methods: {sorted(unknown)}")
        selected = tuple(method for method in config.methods if method.id in set(method_ids))
    source_path = config.source_arrays_dir / "test.npz"
    required = [source_path]
    for method in selected:
        required.append(method.classification_config)
        if method.restoration_config is not None:
            required.append(method.restoration_config)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing latency inputs:\n" + "\n".join(map(str, missing)))
    source = _load_source(source_path)
    indices = stratified_latency_indices(source, cell_trials, config.benchmark_seed)
    if dry_run:
        for method in selected:
            classifier_config = load_experiment_config(method.classification_config, project_root)
            checkpoint = _classifier_checkpoint(classifier_config, config.classifier_seed)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if method.restoration_config is not None:
                restoration_config = load_restoration_config(method.restoration_config, project_root)
                if restoration_config.method != "spherical_spline" and not _restoration_checkpoint(restoration_config).is_file():
                    raise FileNotFoundError(_restoration_checkpoint(restoration_config))
        print(
            f"Validated {len(selected)} methods, {len(indices)} measured trials, "
            f"warmup={warmup}, device={device}, output={output_dir}"
        )
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "system_info.json").write_text(
        json.dumps(_system_info(device, config), indent=2) + "\n", encoding="utf-8"
    )
    summaries = []
    for method in selected:
        print(f"PREPARE latency method: {method.label}", flush=True)
        prepared = _prepare_method(
            method, project_root, source, config.classifier_seed, device
        )
        print(f"BENCHMARK latency method: {method.label}", flush=True)
        summaries.append(_benchmark_method(
            prepared, source, indices, config, device, warmup, cell_trials,
            output_dir, overwrite,
        ))
        del prepared
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    completed_summaries = []
    for method in config.methods:
        summary_path = output_dir / "methods" / method.id / "summary.json"
        if summary_path.is_file():
            completed_summaries.append(json.loads(summary_path.read_text())["summary"])
    if len(completed_summaries) == len(config.methods):
        by_method = {row["method"]: row for row in completed_summaries}
        ordered = [by_method[method.id] for method in config.methods]
        _write_aggregate(output_dir, ordered)
        completion = {
            "experiment_id": config.id,
            "status": "complete",
            "device": str(device),
            "n_methods": len(ordered),
            "n_trials_per_method": len(indices),
            "warmup_trials": warmup,
            "batch_size": 1,
            "classifier_seed": config.classifier_seed,
        }
        (output_dir / "PIPELINE_COMPLETE.json").write_text(
            json.dumps(completion, indent=2) + "\n", encoding="utf-8"
        )
        print(f"LATENCY PIPELINE COMPLETE: {output_dir}", flush=True)
    else:
        print(
            f"PARTIAL latency benchmark: {len(completed_summaries)}/{len(config.methods)} methods",
            flush=True,
        )
    return output_dir
