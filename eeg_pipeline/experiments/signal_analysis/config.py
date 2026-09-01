"""Configuration schema for the Stage-D signal analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MethodConfig:
    id: str
    label: str
    kind: str
    test_path: Path | None = None
    array_key: str | None = None


@dataclass(frozen=True)
class SignalAnalysisConfig:
    id: str
    source_arrays_dir: Path
    output_dir: Path
    methods: tuple[MethodConfig, ...]
    sampling_rate: float
    bands: tuple[tuple[str, float, float], ...]
    total_band: tuple[float, float]
    covariance_ridge: float
    csp_filters_per_class: int
    bootstrap_samples: int
    random_seed: int
    tfr_window_samples: int
    tfr_overlap_samples: int


def _path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_signal_analysis_config(path: Path, project_root: Path) -> SignalAnalysisConfig:
    """Load and validate one Stage-D YAML configuration."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Signal-analysis configuration must be a mapping")
    methods = []
    valid_kinds = {"true22", "zero_padded", "restored"}
    for raw in payload.get("methods", []):
        if not isinstance(raw, dict):
            raise ValueError("Each method must be a mapping")
        kind = str(raw.get("kind", ""))
        if kind not in valid_kinds:
            raise ValueError(f"Unsupported method kind: {kind}")
        test_path = None
        array_key = None
        if kind == "restored":
            test_path = _path(project_root, raw.get("test_path"), "method.test_path")
            array_key = str(raw.get("array_key", ""))
            if not array_key:
                raise ValueError("Restored methods require array_key")
        methods.append(MethodConfig(
            id=str(raw.get("id", "")),
            label=str(raw.get("label", "")),
            kind=kind,
            test_path=test_path,
            array_key=array_key,
        ))
    if not methods or any(not item.id or not item.label for item in methods):
        raise ValueError("At least one fully named method is required")
    if len({item.id for item in methods}) != len(methods):
        raise ValueError("Method ids must be unique")
    kinds = {item.kind for item in methods}
    if "true22" not in kinds or "zero_padded" not in kinds:
        raise ValueError("true22 and zero_padded references are required")
    if sum(item.kind == "true22" for item in methods) != 1:
        raise ValueError("Exactly one true22 reference is required")
    if sum(item.kind == "zero_padded" for item in methods) != 1:
        raise ValueError("Exactly one zero_padded reference is required")
    if not any(item.kind == "restored" for item in methods):
        raise ValueError("At least one restored method is required")
    if "autoencoder_eeg_aware" not in {item.id for item in methods}:
        raise ValueError("The predeclared autoencoder_eeg_aware reference is required")

    analysis = payload.get("analysis", {})
    raw_bands = analysis.get("bands", {})
    if not isinstance(raw_bands, dict) or not raw_bands:
        raise ValueError("analysis.bands must be a non-empty mapping")
    bands = []
    for name, bounds in raw_bands.items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"Band {name} must contain [low, high]")
        low, high = map(float, bounds)
        if not 0 <= low < high:
            raise ValueError(f"Invalid band bounds for {name}")
        bands.append((str(name), low, high))
    total_band = tuple(map(float, analysis.get("total_band", [])))
    if len(total_band) != 2 or total_band[0] >= total_band[1]:
        raise ValueError("analysis.total_band must be [low, high]")
    if any(low < total_band[0] or high > total_band[1] for _, low, high in bands):
        raise ValueError("Every analysis band must be inside total_band")

    ridge = float(analysis.get("covariance_ridge", 0.0))
    filters = int(analysis.get("csp_filters_per_class", 0))
    bootstrap = int(analysis.get("bootstrap_samples", 0))
    window = int(analysis.get("tfr_window_samples", 0))
    overlap = int(analysis.get("tfr_overlap_samples", -1))
    if ridge <= 0 or filters < 2 or filters % 2:
        raise ValueError("covariance_ridge must be positive and CSP filters/class positive-even")
    if bootstrap < 100 or window < 16 or not 0 <= overlap < window:
        raise ValueError("Invalid bootstrap or TFR window settings")
    return SignalAnalysisConfig(
        id=str(payload.get("id", "task_relevant_signal")),
        source_arrays_dir=_path(project_root, payload.get("source_arrays_dir"), "source_arrays_dir"),
        output_dir=_path(project_root, payload.get("output_dir"), "output_dir"),
        methods=tuple(methods),
        sampling_rate=float(analysis.get("sampling_rate", 250.0)),
        bands=tuple(bands),
        total_band=(total_band[0], total_band[1]),
        covariance_ridge=ridge,
        csp_filters_per_class=filters,
        bootstrap_samples=bootstrap,
        random_seed=int(analysis.get("random_seed", 0)),
        tfr_window_samples=window,
        tfr_overlap_samples=overlap,
    )
