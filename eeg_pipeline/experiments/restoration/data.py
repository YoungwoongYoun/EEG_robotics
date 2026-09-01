"""Source-data loading and generated-restoration artifact validation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...channels import EEG_CHANNELS_22, MI9_INDICES
from .config import RestorationConfig, SPLITS


METADATA_KEYS = ("y", "subject", "trial_index")
MISSING_INDICES = tuple(index for index in range(22) if index not in MI9_INDICES)


@dataclass(frozen=True)
class RestorationSplit:
    x_mi9: np.ndarray
    x_true22: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    trial_index: np.ndarray

    def subset(self, size: int) -> "RestorationSplit":
        if size < 1:
            raise ValueError("Subset size must be positive")
        stop = min(size, self.y.size)
        return RestorationSplit(
            x_mi9=self.x_mi9[:stop],
            x_true22=self.x_true22[:stop],
            y=self.y[:stop],
            subject=self.subject[:stop],
            trial_index=self.trial_index[:stop],
        )


class RestorationDataRepository:
    """Load canonical source splits without merging train, validation, and test."""

    def __init__(self, config: RestorationConfig):
        self.config = config
        self.splits = {split: self._load_split(split) for split in SPLITS}
        self._validate_disjoint_splits()

    def _load_split(self, split: str) -> RestorationSplit:
        path = self.config.source.arrays_dir / f"{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source split: {path}")
        with np.load(path, allow_pickle=False) as payload:
            required = {
                self.config.source.input_key,
                self.config.source.target_key,
                *METADATA_KEYS,
            }
            missing = required - set(payload.files)
            if missing:
                raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
            values = {key: payload[key] for key in required}
        result = RestorationSplit(
            x_mi9=values[self.config.source.input_key],
            x_true22=values[self.config.source.target_key],
            y=values["y"],
            subject=values["subject"],
            trial_index=values["trial_index"],
        )
        validate_source_split(result, split)
        return result

    def _validate_disjoint_splits(self) -> None:
        keys = {
            split: set(zip(values.subject.tolist(), values.trial_index.tolist()))
            for split, values in self.splits.items()
        }
        if keys["train"] & keys["validation"]:
            raise ValueError("Training and validation source splits overlap")


def validate_source_split(payload: RestorationSplit, split: str) -> None:
    """Validate shape, metadata, and canonical observed-channel identity."""

    if payload.x_mi9.ndim != 3 or payload.x_mi9.shape[1] != len(MI9_INDICES):
        raise ValueError(f"Unexpected {split} MI-9 shape: {payload.x_mi9.shape}")
    if payload.x_true22.ndim != 3 or payload.x_true22.shape[1] != len(EEG_CHANNELS_22):
        raise ValueError(f"Unexpected {split} True-22 shape: {payload.x_true22.shape}")
    if payload.x_mi9.shape[0] != payload.x_true22.shape[0]:
        raise ValueError(f"Mismatched source trial count in {split}")
    if payload.x_mi9.shape[2] != payload.x_true22.shape[2]:
        raise ValueError(f"Mismatched source time length in {split}")
    n_trials = payload.x_mi9.shape[0]
    if any(value.ndim != 1 or value.size != n_trials for value in (
        payload.y, payload.subject, payload.trial_index
    )):
        raise ValueError(f"Mismatched source metadata in {split}")
    if not np.isfinite(payload.x_mi9).all() or not np.isfinite(payload.x_true22).all():
        raise ValueError(f"Non-finite source EEG in {split}")
    np.testing.assert_allclose(
        payload.x_true22[:, MI9_INDICES, :],
        payload.x_mi9,
        rtol=0.0,
        atol=1e-6,
        err_msg=f"Canonical MI-9 does not match True-22 in {split}",
    )


def load_subject_normalization(
    normalization_dir: Path,
    subject: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = normalization_dir / f"A{subject:02d}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing normalization statistics: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"mean", "std"}:
            raise KeyError(f"Unexpected normalization schema in {path}: {payload.files}")
        mean = payload["mean"]
        std = payload["std"]
    if mean.shape != (22,) or std.shape != (22,) or np.any(std <= 0):
        raise ValueError(f"Invalid normalization statistics in {path}")
    return mean, std


def enforce_observed_channels(
    restored22: np.ndarray,
    x_mi9: np.ndarray,
) -> np.ndarray:
    """Hard-copy observed normalized MI-9 into a generated 22-channel array."""

    if restored22.ndim != 3 or restored22.shape[1] != 22:
        raise ValueError("restored22 must have shape [N, 22, T]")
    if x_mi9.shape != (restored22.shape[0], 9, restored22.shape[2]):
        raise ValueError("x_mi9 shape does not match restored22")
    restored22[:, MI9_INDICES, :] = x_mi9
    return restored22


def validate_restored_split(
    restored22: np.ndarray,
    source: RestorationSplit,
    split: str,
) -> None:
    expected = source.x_true22.shape
    if restored22.shape != expected:
        raise ValueError(f"Unexpected restored {split} shape: {restored22.shape}, expected {expected}")
    if restored22.dtype != np.float32:
        raise TypeError(f"Restored {split} must be float32, got {restored22.dtype}")
    if not np.isfinite(restored22).all():
        raise ValueError(f"Non-finite restored EEG in {split}")
    np.testing.assert_array_equal(
        restored22[:, MI9_INDICES, :],
        source.x_mi9,
        err_msg=f"Observed MI-9 was not preserved exactly in {split}",
    )


def write_restored_split(
    path: Path,
    array_key: str,
    restored22: np.ndarray,
    source: RestorationSplit,
) -> None:
    """Atomically write one classifier-ready restoration split."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        array_key: restored22,
        "y": source.y,
        "subject": source.subject,
        "trial_index": source.trial_index,
    }
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".npz", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(temporary_path, **payload)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
