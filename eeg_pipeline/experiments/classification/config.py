"""Configuration for one pooled multi-subject classification experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_TRANSFORMS = ("none", "zero_pad_mi9_to_22")


@dataclass(frozen=True)
class InputConfig:
    id: str
    label: str
    category: str
    arrays_dir: Path
    array_key: str
    n_channels: int
    transform: str = "none"

    def validate(self) -> None:
        if not self.id or "/" in self.id:
            raise ValueError("Input id must be a non-empty path-safe string")
        if not self.label:
            raise ValueError("Input label must not be empty")
        if self.category not in {"baseline", "restored"}:
            raise ValueError("Input category must be 'baseline' or 'restored'")
        if not self.array_key:
            raise ValueError("array_key must not be empty")
        if self.n_channels < 1:
            raise ValueError("n_channels must be positive")
        if self.transform not in SUPPORTED_TRANSFORMS:
            raise ValueError(f"Unsupported input transform: {self.transform}")
        if self.transform == "zero_pad_mi9_to_22" and self.n_channels != 22:
            raise ValueError("zero_pad_mi9_to_22 requires n_channels: 22")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 125
    batch_size: int = 48
    learning_rate: float = 9e-4
    weight_decay: float = 1e-3
    optimizer: str = "adam"
    beta_1: float = 0.5
    beta_2: float = 0.999
    scheduler: str = "warmup_cosine"
    warmup_epochs: int = 3
    sr_augmentation: bool = True
    sr_segments: int = 7
    early_stopping_patience: int = 125
    early_stopping_min_delta: float = 0.0
    num_workers: int = 0
    amp: bool = True
    deterministic: bool = True
    allow_tf32: bool = True


@dataclass(frozen=True)
class ModelConfig:
    n_classes: int = 4
    reference_url: str = ""
    reference_commit: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: Path
    subjects: tuple[int, ...]
    seeds: tuple[int, ...]
    device: str
    input: InputConfig
    training: TrainingConfig
    model: ModelConfig

    def validate(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError("Experiment name must be a non-empty path-safe string")
        if self.name != self.input.id:
            raise ValueError("Experiment name and input id must match")
        if not self.subjects or any(subject < 1 or subject > 9 for subject in self.subjects):
            raise ValueError("Subjects must be selected from 1 through 9")
        if len(set(self.subjects)) != len(self.subjects):
            raise ValueError("Subjects must not contain duplicates")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("Seeds must be non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must not contain duplicates")
        if self.training.epochs < 1 or self.training.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.training.learning_rate <= 0 or self.training.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.training.optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be 'adam' or 'adamw'")
        if not (0 <= self.training.beta_1 < 1 and 0 <= self.training.beta_2 < 1):
            raise ValueError("Optimizer beta values must be in [0, 1)")
        if self.training.scheduler not in {"none", "warmup_cosine"}:
            raise ValueError("scheduler must be 'none' or 'warmup_cosine'")
        if not (0 <= self.training.warmup_epochs < self.training.epochs):
            raise ValueError("warmup_epochs must be in [0, epochs)")
        if self.training.sr_segments < 2:
            raise ValueError("sr_segments must be at least 2")
        if self.training.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if self.model.n_classes != 4:
            raise ValueError("BCIC IV-2a requires four output classes")
        self.input.validate()

    def as_serializable_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        values["input"]["arrays_dir"] = str(self.input.arrays_dir)
        values["subjects"] = list(self.subjects)
        values["seeds"] = list(self.seeds)
        return values


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_experiment_config(path: Path, project_root: Path) -> ExperimentConfig:
    """Load and validate a single-input experiment YAML file."""

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    values = dict(loaded)
    input_values = dict(values.pop("input"))
    input_values["arrays_dir"] = _resolve_path(
        project_root, input_values["arrays_dir"]
    )
    config = ExperimentConfig(
        name=str(values.pop("name")),
        output_dir=_resolve_path(project_root, values.pop("output_dir")),
        subjects=tuple(int(value) for value in values.pop("subjects")),
        seeds=tuple(int(value) for value in values.pop("seeds")),
        device=str(values.pop("device", "cuda:0")),
        input=InputConfig(**input_values),
        training=TrainingConfig(**dict(values.pop("training", {}))),
        model=ModelConfig(**dict(values.pop("model", {}))),
    )
    if values:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(values)}")
    config.validate()
    return config
