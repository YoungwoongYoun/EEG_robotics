"""Dataset assembly and serialization for the revised BCIC IV-2a protocol."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .channels import EEG_CHANNELS_22, MI9_CHANNELS, MI9_INDICES
from .preprocessing import (
    apply_standardizer,
    fit_channelwise_standardizer,
    load_bcic2a_session,
    select_average_referenced_channels,
    stratified_train_val_indices,
)


@dataclass(frozen=True)
class PreprocessingConfig:
    raw_dir: Path
    labels_dir: Path
    output_dir: Path
    subjects: tuple[int, ...] = tuple(range(1, 10))
    low_frequency: float = 8.0
    high_frequency: float = 30.0
    epoch_tmin: float = 0.0
    epoch_tmax: float = 4.0
    validation_fraction: float = 0.2
    random_seed: int = 42
    reject_flagged_trials: bool = True
    export_torch: bool = True

    def validate(self) -> None:
        if not self.subjects:
            raise ValueError("At least one subject is required")
        if any(subject < 1 or subject > 9 for subject in self.subjects):
            raise ValueError("BCIC IV-2a subject IDs must be between 1 and 9")
        if not 0.0 < self.low_frequency < self.high_frequency:
            raise ValueError("Expected 0 < low_frequency < high_frequency")
        if not self.epoch_tmin < self.epoch_tmax:
            raise ValueError("epoch_tmin must be smaller than epoch_tmax")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")


@dataclass(frozen=True)
class ChannelMontage:
    """One fixed nine-channel input used in the overlap sensitivity study."""

    id: str
    label: str
    channels: tuple[str, ...]
    expected_mi9_overlap: int

    @property
    def array_key(self) -> str:
        return f"x_{self.id}"

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(EEG_CHANNELS_22.index(channel) for channel in self.channels)

    def validate(self) -> None:
        if not self.id or "/" in self.id:
            raise ValueError("Montage id must be a non-empty path-safe string")
        if not self.label:
            raise ValueError(f"Montage {self.id} needs a label")
        if len(self.channels) != 9 or len(set(self.channels)) != 9:
            raise ValueError(f"Montage {self.id} must contain nine unique channels")
        unknown = set(self.channels) - set(EEG_CHANNELS_22)
        if unknown:
            raise ValueError(f"Montage {self.id} has unknown channels: {sorted(unknown)}")
        canonical_order = tuple(sorted(self.channels, key=EEG_CHANNELS_22.index))
        if self.channels != canonical_order:
            raise ValueError(
                f"Montage {self.id} channels must follow canonical EEG_CHANNELS_22 order"
            )
        actual_overlap = len(set(self.channels) & set(MI9_CHANNELS))
        if actual_overlap != self.expected_mi9_overlap:
            raise ValueError(
                f"Montage {self.id} expected MI-9 overlap {self.expected_mi9_overlap}, "
                f"got {actual_overlap}"
            )


def _append_split(
    aggregate: dict[str, list[np.ndarray]],
    *,
    x: np.ndarray,
    y: np.ndarray,
    subject: int,
    trial_index: np.ndarray,
) -> None:
    aggregate["x_true22"].append(x)
    aggregate["x_mi9"].append(x[:, MI9_INDICES, :])
    aggregate["y"].append(y.astype(np.int64, copy=False))
    aggregate["subject"].append(np.full(y.shape, subject, dtype=np.int64))
    aggregate["trial_index"].append(trial_index.astype(np.int64, copy=False))


def _empty_aggregate() -> dict[str, list[np.ndarray]]:
    return {key: [] for key in ("x_true22", "x_mi9", "y", "subject", "trial_index")}


def _concatenate(aggregate: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate(parts, axis=0) for key, parts in aggregate.items()}


def _save_npz_splits(output_dir: Path, splits: dict[str, dict[str, np.ndarray]]) -> None:
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    for split_name, payload in splits.items():
        np.savez_compressed(arrays_dir / f"{split_name}.npz", **payload)


def _save_torch_splits(output_dir: Path, splits: dict[str, dict[str, np.ndarray]]) -> None:
    try:
        import torch
        from torch.utils.data import TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "export_torch=true but PyTorch is unavailable. Install the GPU-server "
            "requirements or pass --no-torch."
        ) from exc

    torch_dir = output_dir / "torch_datasets"
    torch_dir.mkdir(parents=True, exist_ok=True)
    for split_name, payload in splits.items():
        labels = torch.from_numpy(payload["y"]).long()
        true22 = torch.from_numpy(payload["x_true22"]).float().unsqueeze(1)
        mi9 = torch.from_numpy(payload["x_mi9"]).float().unsqueeze(1)
        torch.save(TensorDataset(true22, labels), torch_dir / f"{split_name}_dataset.pt")
        torch.save(TensorDataset(mi9, labels), torch_dir / f"{split_name}_mi9_dataset.pt")


def _write_manifest(output_dir: Path, splits: dict[str, dict[str, np.ndarray]]) -> None:
    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "subject", "session", "trial_index", "label"),
        )
        writer.writeheader()
        for split_name, payload in splits.items():
            session = "session_2" if split_name == "test" else "session_1"
            for subject, trial_index, label in zip(
                payload["subject"], payload["trial_index"], payload["y"]
            ):
                writer.writerow(
                    {
                        "split": split_name,
                        "subject": f"A{int(subject):02d}",
                        "session": session,
                        "trial_index": int(trial_index),
                        "label": int(label),
                    }
                )


def build_dataset(config: PreprocessingConfig) -> dict[str, Any]:
    """Build official inter-session splits and write reusable artifacts."""

    config.validate()
    input_directories = (
        (config.raw_dir, "raw_dir"),
        (config.labels_dir, "labels_dir"),
    )
    for directory, name in input_directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"{name} does not exist: {directory}")

    aggregates = {name: _empty_aggregate() for name in ("train", "validation", "test")}
    normalization_dir = config.output_dir / "normalization"
    normalization_dir.mkdir(parents=True, exist_ok=True)
    subject_summaries: list[dict[str, Any]] = []

    for subject in config.subjects:
        subject_name = f"A{subject:02d}"
        training_path = config.raw_dir / f"{subject_name}T.gdf"
        test_path = config.raw_dir / f"{subject_name}E.gdf"
        label_path = config.labels_dir / f"{subject_name}E.mat"
        for required in (training_path, test_path, label_path):
            if not required.is_file():
                raise FileNotFoundError(f"Required dataset file is missing: {required}")

        common = {
            "low_frequency": config.low_frequency,
            "high_frequency": config.high_frequency,
            "epoch_tmin": config.epoch_tmin,
            "epoch_tmax": config.epoch_tmax,
            "reject_flagged_trials": config.reject_flagged_trials,
        }
        session_1 = load_bcic2a_session(
            training_path,
            session="train",
            evaluation_label_path=None,
            **common,
        )
        session_2 = load_bcic2a_session(
            test_path,
            session="test",
            evaluation_label_path=label_path,
            **common,
        )

        train_idx, validation_idx = stratified_train_val_indices(
            session_1.y,
            validation_fraction=config.validation_fraction,
            seed=config.random_seed + subject,
        )
        stats = fit_channelwise_standardizer(session_1.x[train_idx])
        np.savez_compressed(
            normalization_dir / f"{subject_name}.npz",
            mean=stats.mean,
            std=stats.std,
        )

        normalized = {
            "train": apply_standardizer(session_1.x[train_idx], stats),
            "validation": apply_standardizer(session_1.x[validation_idx], stats),
            "test": apply_standardizer(session_2.x, stats),
        }
        _append_split(
            aggregates["train"],
            x=normalized["train"],
            y=session_1.y[train_idx],
            subject=subject,
            trial_index=session_1.trial_index[train_idx],
        )
        _append_split(
            aggregates["validation"],
            x=normalized["validation"],
            y=session_1.y[validation_idx],
            subject=subject,
            trial_index=session_1.trial_index[validation_idx],
        )
        _append_split(
            aggregates["test"],
            x=normalized["test"],
            y=session_2.y,
            subject=subject,
            trial_index=session_2.trial_index,
        )
        subject_summaries.append(
            {
                "subject": subject_name,
                "train_trials": int(train_idx.size),
                "validation_trials": int(validation_idx.size),
                "test_trials": int(session_2.y.size),
                "session_1_rejected_trials": session_1.rejected_trial_index.tolist(),
                "session_2_rejected_trials": session_2.rejected_trial_index.tolist(),
            }
        )

    splits = {name: _concatenate(value) for name, value in aggregates.items()}
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _save_npz_splits(config.output_dir, splits)
    if config.export_torch:
        _save_torch_splits(config.output_dir, splits)
    _write_manifest(config.output_dir, splits)

    summary = {
        "protocol": "same-cohort inter-session",
        "session_1_usage": "training and validation",
        "session_2_usage": "final test only",
        "reference": "MI-9 average reference applied to all 22 target channels",
        "normalization": "per-subject, per-channel, fit on Session-1 training only",
        "eeg_channels": list(EEG_CHANNELS_22),
        "mi9_channels": list(MI9_CHANNELS),
        "mi9_indices": list(MI9_INDICES),
        "subjects": subject_summaries,
        "split_shapes": {
            name: {key: list(value.shape) for key, value in payload.items()}
            for name, payload in splits.items()
        },
        "config": {
            **asdict(config),
            "raw_dir": str(config.raw_dir),
            "labels_dir": str(config.labels_dir),
            "output_dir": str(config.output_dir),
            "subjects": list(config.subjects),
        },
    }
    with (config.output_dir / "preprocessing_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_channel_overlap_dataset(
    config: PreprocessingConfig,
    montages: tuple[ChannelMontage, ...],
) -> dict[str, Any]:
    """Build all fixed nine-channel overlap inputs while loading raw data once."""

    config.validate()
    if not montages:
        raise ValueError("At least one channel-overlap montage is required")
    if len({montage.id for montage in montages}) != len(montages):
        raise ValueError("Channel-overlap montage ids must be unique")
    for montage in montages:
        montage.validate()
    if config.export_torch:
        raise ValueError(
            "Channel-overlap preprocessing writes reusable NPZ arrays only; "
            "set export_torch: false"
        )

    for directory, name in ((config.raw_dir, "raw_dir"), (config.labels_dir, "labels_dir")):
        if not directory.is_dir():
            raise FileNotFoundError(f"{name} does not exist: {directory}")

    array_keys = tuple(montage.array_key for montage in montages)
    aggregates = {
        split: {
            **{key: [] for key in array_keys},
            "y": [],
            "subject": [],
            "trial_index": [],
        }
        for split in ("train", "validation", "test")
    }
    normalization_dir = config.output_dir / "normalization"
    normalization_dir.mkdir(parents=True, exist_ok=True)
    subject_summaries: list[dict[str, Any]] = []

    for subject in config.subjects:
        subject_name = f"A{subject:02d}"
        training_path = config.raw_dir / f"{subject_name}T.gdf"
        test_path = config.raw_dir / f"{subject_name}E.gdf"
        label_path = config.labels_dir / f"{subject_name}E.mat"
        for required in (training_path, test_path, label_path):
            if not required.is_file():
                raise FileNotFoundError(f"Required dataset file is missing: {required}")

        common = {
            "low_frequency": config.low_frequency,
            "high_frequency": config.high_frequency,
            "epoch_tmin": config.epoch_tmin,
            "epoch_tmax": config.epoch_tmax,
            "reject_flagged_trials": config.reject_flagged_trials,
            "reference_indices": None,
        }
        session_1 = load_bcic2a_session(
            training_path,
            session="train",
            evaluation_label_path=None,
            **common,
        )
        session_2 = load_bcic2a_session(
            test_path,
            session="test",
            evaluation_label_path=label_path,
            **common,
        )
        train_idx, validation_idx = stratified_train_val_indices(
            session_1.y,
            validation_fraction=config.validation_fraction,
            seed=config.random_seed + subject,
        )

        for montage in montages:
            session_1_selected = select_average_referenced_channels(
                session_1.x, montage.indices
            )
            session_2_selected = select_average_referenced_channels(
                session_2.x, montage.indices
            )
            stats = fit_channelwise_standardizer(session_1_selected[train_idx])
            montage_normalization_dir = normalization_dir / montage.id
            montage_normalization_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                montage_normalization_dir / f"{subject_name}.npz",
                mean=stats.mean,
                std=stats.std,
            )
            normalized = {
                "train": apply_standardizer(session_1_selected[train_idx], stats),
                "validation": apply_standardizer(
                    session_1_selected[validation_idx], stats
                ),
                "test": apply_standardizer(session_2_selected, stats),
            }
            for split, values in normalized.items():
                aggregates[split][montage.array_key].append(values)

        split_metadata = {
            "train": (session_1.y[train_idx], session_1.trial_index[train_idx]),
            "validation": (
                session_1.y[validation_idx],
                session_1.trial_index[validation_idx],
            ),
            "test": (session_2.y, session_2.trial_index),
        }
        for split, (labels, trial_indices) in split_metadata.items():
            aggregates[split]["y"].append(labels.astype(np.int64, copy=False))
            aggregates[split]["subject"].append(
                np.full(labels.shape, subject, dtype=np.int64)
            )
            aggregates[split]["trial_index"].append(
                trial_indices.astype(np.int64, copy=False)
            )

        subject_summaries.append(
            {
                "subject": subject_name,
                "train_trials": int(train_idx.size),
                "validation_trials": int(validation_idx.size),
                "test_trials": int(session_2.y.size),
                "session_1_rejected_trials": session_1.rejected_trial_index.tolist(),
                "session_2_rejected_trials": session_2.rejected_trial_index.tolist(),
            }
        )

    splits = {name: _concatenate(value) for name, value in aggregates.items()}
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _save_npz_splits(config.output_dir, splits)
    _write_manifest(config.output_dir, splits)
    summary = {
        "protocol": "same-cohort inter-session channel-overlap sensitivity",
        "session_1_usage": "training and validation",
        "session_2_usage": "final test only",
        "reference": "each nine-channel montage average-referenced to itself",
        "normalization": "per-subject and per-montage, fit on Session-1 training only",
        "eeg_channels": list(EEG_CHANNELS_22),
        "canonical_mi9_channels": list(MI9_CHANNELS),
        "montages": [
            {
                "id": montage.id,
                "label": montage.label,
                "array_key": montage.array_key,
                "channels": list(montage.channels),
                "indices": list(montage.indices),
                "mi9_overlap_count": montage.expected_mi9_overlap,
                "mi9_overlap_fraction": montage.expected_mi9_overlap / 9,
            }
            for montage in montages
        ],
        "subjects": subject_summaries,
        "split_shapes": {
            name: {key: list(value.shape) for key, value in payload.items()}
            for name, payload in splits.items()
        },
        "config": {
            **asdict(config),
            "raw_dir": str(config.raw_dir),
            "labels_dir": str(config.labels_dir),
            "output_dir": str(config.output_dir),
            "subjects": list(config.subjects),
        },
    }
    with (config.output_dir / "preprocessing_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    return summary
