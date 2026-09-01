"""Configuration for one canonical MI-9 to 22-channel restoration experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_METHODS = ("spherical_spline", "autoencoder", "ddpm", "wgan_gp")
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SourceConfig:
    arrays_dir: Path
    normalization_dir: Path
    input_key: str = "x_mi9"
    target_key: str = "x_true22"


@dataclass(frozen=True)
class OutputConfig:
    arrays_dir: Path
    experiment_dir: Path
    array_key: str = "x_restored22"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    optimizer: str = "adam"
    beta_1: float = 0.9
    beta_2: float = 0.999
    patience: int = 20
    min_delta: float = 0.0
    num_workers: int = 0
    amp: bool = True
    deterministic: bool = True
    allow_tf32: bool = True


@dataclass(frozen=True)
class InferenceConfig:
    batch_size: int = 32
    seed: int = 1000
    sampler: str = "deterministic"
    sampling_steps: int = 50
    eta: float = 0.0


@dataclass(frozen=True)
class RestorationConfig:
    name: str
    method: str
    seed: int
    device: str
    source: SourceConfig
    output: OutputConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    model: dict[str, Any] = field(default_factory=dict)
    diffusion: dict[str, Any] = field(default_factory=dict)
    gan: dict[str, Any] = field(default_factory=dict)
    loss: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError("name must be a non-empty path-safe string")
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported restoration method: {self.method}")
        if self.seed < 0 or self.inference.seed < 0:
            raise ValueError("Training and inference seeds must be non-negative")
        if not self.source.input_key or not self.source.target_key:
            raise ValueError("Source array keys must not be empty")
        if not self.output.array_key:
            raise ValueError("Output array_key must not be empty")
        if self.training.epochs < 1 or self.training.batch_size < 1:
            raise ValueError("Training epochs and batch_size must be positive")
        if self.training.learning_rate <= 0 or self.training.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.training.optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be adam or adamw")
        if not (0 <= self.training.beta_1 < 1 and 0 <= self.training.beta_2 < 1):
            raise ValueError("Optimizer beta values must be in [0, 1)")
        if self.training.patience < 1 or self.inference.batch_size < 1:
            raise ValueError("patience and inference batch_size must be positive")
        if self.method == "ddpm":
            required = {"timesteps", "beta_start", "beta_end"}
            missing = required - set(self.diffusion)
            if missing:
                raise ValueError(f"DDPM configuration is missing: {sorted(missing)}")
            timesteps = int(self.diffusion["timesteps"])
            if timesteps < 2:
                raise ValueError("DDPM timesteps must be at least 2")
            beta_start = float(self.diffusion["beta_start"])
            beta_end = float(self.diffusion["beta_end"])
            if not 0.0 < beta_start <= beta_end < 1.0:
                raise ValueError("DDPM betas must satisfy 0 < beta_start <= beta_end < 1")
            alpha_bar_terminal = 1.0
            for index in range(timesteps):
                fraction = index / (timesteps - 1)
                beta = beta_start + fraction * (beta_end - beta_start)
                alpha_bar_terminal *= 1.0 - beta
            if not math.isfinite(alpha_bar_terminal) or alpha_bar_terminal > 1e-3:
                raise ValueError(
                    "The terminal diffusion state retains too much signal: "
                    f"alpha_bar_T={alpha_bar_terminal:.6g}. Increase timesteps or beta_end "
                    "so inference can validly start from Gaussian noise."
                )
            if not 1 <= self.inference.sampling_steps <= timesteps:
                raise ValueError("sampling_steps must be between 1 and diffusion timesteps")
            if self.inference.sampler not in {"ddpm", "ddim"}:
                raise ValueError("DDPM inference sampler must be ddpm or ddim")
            if (
                self.inference.sampler == "ddpm"
                and self.inference.sampling_steps != timesteps
            ):
                raise ValueError("Ancestral DDPM sampling must use every diffusion timestep")
            if not 0.0 <= self.inference.eta <= 1.0:
                raise ValueError("DDIM eta must be in [0, 1]")
        elif self.method == "wgan_gp":
            required = {
                "gradient_penalty_weight",
                "adversarial_weight",
                "reconstruction_weight",
                "critic_steps",
            }
            missing = required - set(self.gan)
            if missing:
                raise ValueError(f"WGAN-GP configuration is missing: {sorted(missing)}")
            if float(self.gan["gradient_penalty_weight"]) <= 0:
                raise ValueError("gradient_penalty_weight must be positive")
            if float(self.gan["adversarial_weight"]) <= 0:
                raise ValueError("adversarial_weight must be positive")
            if float(self.gan["reconstruction_weight"]) <= 0:
                raise ValueError("reconstruction_weight must be positive")
            if int(self.gan["critic_steps"]) < 1:
                raise ValueError("critic_steps must be positive")
        elif self.inference.sampler != "deterministic":
            raise ValueError("Non-diffusion inference sampler must be deterministic")
        if self.method != "wgan_gp" and self.gan:
            raise ValueError("GAN settings are supported only for wgan_gp")
        if self.loss:
            if self.method != "autoencoder":
                raise ValueError("Custom restoration losses are supported only for autoencoder")
            required = {
                "objective", "time_weight", "bandpower_weight", "spatial_weight",
                "sampling_rate", "bands", "epsilon",
            }
            missing = required - set(self.loss)
            if missing:
                raise ValueError(f"EEG-aware loss configuration is missing: {sorted(missing)}")
            if self.loss["objective"] != "eeg_spectral_spatial":
                raise ValueError("Unsupported autoencoder loss objective")
            weights = (
                float(self.loss["time_weight"]),
                float(self.loss["bandpower_weight"]),
                float(self.loss["spatial_weight"]),
            )
            if weights[0] <= 0 or any(weight < 0 for weight in weights[1:]):
                raise ValueError("time_weight must be positive and auxiliary weights non-negative")
            sampling_rate = float(self.loss["sampling_rate"])
            if sampling_rate <= 0 or float(self.loss["epsilon"]) <= 0:
                raise ValueError("sampling_rate and epsilon must be positive")
            bands = self.loss["bands"]
            if not isinstance(bands, list) or not bands:
                raise ValueError("bands must be a non-empty list")
            nyquist = sampling_rate / 2.0
            for band in bands:
                if len(band) != 2 or not 0 <= float(band[0]) < float(band[1]) <= nyquist:
                    raise ValueError("Each frequency band must lie within [0, Nyquist]")

    def as_serializable_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["source"]["arrays_dir"] = str(self.source.arrays_dir)
        values["source"]["normalization_dir"] = str(self.source.normalization_dir)
        values["output"]["arrays_dir"] = str(self.output.arrays_dir)
        values["output"]["experiment_dir"] = str(self.output.experiment_dir)
        return values

    def training_signature(self) -> dict[str, Any]:
        """Fields that must match before a learned checkpoint can be reused."""

        values = self.as_serializable_dict()
        return {
            key: values[key]
            for key in (
                "method", "seed", "source", "training", "model", "diffusion", "gan", "loss"
            )
        }


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_restoration_config(path: Path, project_root: Path) -> RestorationConfig:
    """Load and strictly validate a restoration YAML file."""

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    values = dict(loaded)
    source_values = dict(values.pop("source"))
    output_values = dict(values.pop("output"))
    source_values["arrays_dir"] = _resolve(project_root, source_values["arrays_dir"])
    source_values["normalization_dir"] = _resolve(
        project_root, source_values["normalization_dir"]
    )
    output_values["arrays_dir"] = _resolve(project_root, output_values["arrays_dir"])
    output_values["experiment_dir"] = _resolve(
        project_root, output_values["experiment_dir"]
    )
    config = RestorationConfig(
        name=str(values.pop("name")),
        method=str(values.pop("method")),
        seed=int(values.pop("seed", 0)),
        device=str(values.pop("device", "cuda:0")),
        source=SourceConfig(**source_values),
        output=OutputConfig(**output_values),
        training=TrainingConfig(**dict(values.pop("training", {}))),
        inference=InferenceConfig(**dict(values.pop("inference", {}))),
        model=dict(values.pop("model", {})),
        diffusion=dict(values.pop("diffusion", {})),
        gan=dict(values.pop("gan", {})),
        loss=dict(values.pop("loss", {})),
    )
    if values:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(values)}")
    config.validate()
    return config
