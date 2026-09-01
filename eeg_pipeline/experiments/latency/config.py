"""Strict configuration for the formal batch-1 latency benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_KINDS = ("true22", "direct_mi9", "zero_padded", "spherical_spline", "learned")


@dataclass(frozen=True)
class LatencyMethodConfig:
    id: str
    label: str
    kind: str
    classification_config: Path
    restoration_config: Path | None = None


@dataclass(frozen=True)
class LatencyBenchmarkConfig:
    id: str
    source_arrays_dir: Path
    output_dir: Path
    classifier_seed: int
    benchmark_seed: int
    cpu_threads: int
    allow_tf32: bool
    warmup_trials: int
    trials_per_subject_class: int
    methods: tuple[LatencyMethodConfig, ...]


def _resolve(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_latency_benchmark_config(path: Path, project_root: Path) -> LatencyBenchmarkConfig:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Latency benchmark config must be a mapping")
    methods = []
    for raw in payload.get("methods", []):
        if not isinstance(raw, dict):
            raise ValueError("Each latency method must be a mapping")
        kind = str(raw.get("kind", ""))
        if kind not in SUPPORTED_KINDS:
            raise ValueError(f"Unsupported latency method kind: {kind}")
        restoration_config = None
        if kind in {"spherical_spline", "learned"}:
            restoration_config = _resolve(
                project_root, raw.get("restoration_config"), "restoration_config"
            )
        methods.append(LatencyMethodConfig(
            id=str(raw.get("id", "")),
            label=str(raw.get("label", "")),
            kind=kind,
            classification_config=_resolve(
                project_root, raw.get("classification_config"), "classification_config"
            ),
            restoration_config=restoration_config,
        ))
    if not methods or any(not method.id or not method.label for method in methods):
        raise ValueError("At least one fully named latency method is required")
    if len({method.id for method in methods}) != len(methods):
        raise ValueError("Latency method ids must be unique")
    required_kinds = {"true22", "direct_mi9", "zero_padded", "spherical_spline", "learned"}
    if not required_kinds.issubset({method.kind for method in methods}):
        raise ValueError(f"Latency benchmark must cover kinds: {sorted(required_kinds)}")
    warmup = int(payload.get("warmup_trials", 0))
    trials = int(payload.get("trials_per_subject_class", 0))
    seed = int(payload.get("classifier_seed", -1))
    benchmark_seed = int(payload.get("benchmark_seed", -1))
    cpu_threads = int(payload.get("cpu_threads", 0))
    if warmup < 1 or trials < 1 or seed < 0 or benchmark_seed < 0 or cpu_threads < 1:
        raise ValueError(
            "warmup/trial/thread counts must be positive and seed must be non-negative"
        )
    return LatencyBenchmarkConfig(
        id=str(payload.get("id", "latency_benchmark")),
        source_arrays_dir=_resolve(project_root, payload.get("source_arrays_dir"), "source_arrays_dir"),
        output_dir=_resolve(project_root, payload.get("output_dir"), "output_dir"),
        classifier_seed=seed,
        benchmark_seed=benchmark_seed,
        cpu_threads=cpu_threads,
        allow_tf32=bool(payload.get("allow_tf32", True)),
        warmup_trials=warmup,
        trials_per_subject_class=trials,
        methods=tuple(methods),
    )
