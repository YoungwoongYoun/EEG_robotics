"""Aligned source and restoration inputs for Stage-D analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...channels import MI9_INDICES
from .config import MethodConfig


@dataclass(frozen=True)
class EEGSplit:
    x_true22: np.ndarray
    x_mi9: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    trial_index: np.ndarray


def load_source(path: Path) -> EEGSplit:
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical source split: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {"x_true22", "x_mi9", "y", "subject", "trial_index"}
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        split = EEGSplit(**{key: payload[key] for key in required})
    if split.x_true22.ndim != 3 or split.x_true22.shape[1] != 22:
        raise ValueError(f"Unexpected True-22 shape: {split.x_true22.shape}")
    if split.x_mi9.shape != (split.x_true22.shape[0], 9, split.x_true22.shape[2]):
        raise ValueError(f"Unexpected MI-9 shape: {split.x_mi9.shape}")
    n_trials = split.x_true22.shape[0]
    if any(value.shape != (n_trials,) for value in (split.y, split.subject, split.trial_index)):
        raise ValueError("Source metadata length mismatch")
    if not np.isfinite(split.x_true22).all() or not np.isfinite(split.x_mi9).all():
        raise ValueError("Source contains non-finite EEG")
    np.testing.assert_allclose(
        split.x_true22[:, MI9_INDICES], split.x_mi9, rtol=0.0, atol=1e-6
    )
    return split


def load_method_test(method: MethodConfig, source: EEGSplit) -> np.ndarray:
    """Load one aligned Session-2 22-channel method without retaining NPZ handles."""

    if method.kind == "true22":
        return source.x_true22
    if method.kind == "zero_padded":
        result = np.zeros_like(source.x_true22)
        result[:, MI9_INDICES] = source.x_mi9
        return result
    assert method.test_path is not None and method.array_key is not None
    if not method.test_path.is_file():
        raise FileNotFoundError(f"Missing restored test input: {method.test_path}")
    with np.load(method.test_path, allow_pickle=False) as payload:
        required = {method.array_key, "y", "subject", "trial_index"}
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"{method.test_path} is missing {sorted(missing)}")
        restored = payload[method.array_key]
        metadata = {key: payload[key] for key in ("y", "subject", "trial_index")}
    if restored.shape != source.x_true22.shape or restored.dtype != np.float32:
        raise ValueError(
            f"Unexpected {method.id} shape/dtype: {restored.shape}/{restored.dtype}"
        )
    if not np.isfinite(restored).all():
        raise ValueError(f"{method.id} contains non-finite EEG")
    for key, expected in (
        ("y", source.y), ("subject", source.subject), ("trial_index", source.trial_index)
    ):
        np.testing.assert_array_equal(metadata[key], expected, err_msg=f"{method.id} {key} mismatch")
    np.testing.assert_array_equal(
        restored[:, MI9_INDICES], source.x_mi9,
        err_msg=f"{method.id} did not exactly preserve canonical MI-9",
    )
    return restored
