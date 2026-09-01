"""Spherical-spline restoration in physical units with training-only scaling."""

from __future__ import annotations

import numpy as np

from ...channels import EEG_CHANNELS_22, MI9_CHANNELS, MI9_INDICES
from .data import (
    MISSING_INDICES,
    RestorationSplit,
    enforce_observed_channels,
    load_subject_normalization,
)


def spherical_spline_matrix(alpha: float = 1e-5) -> np.ndarray:
    """Build the Perrin spherical-spline mapping from MI-9 to missing-13."""

    import mne
    from mne.channels.interpolation import _make_interpolation_matrix

    montage = mne.channels.make_standard_montage("standard_1020")
    positions = montage.get_positions()["ch_pos"]
    missing_channels = tuple(EEG_CHANNELS_22[index] for index in MISSING_INDICES)
    absent = set((*MI9_CHANNELS, *missing_channels)) - set(positions)
    if absent:
        raise ValueError(f"standard_1020 lacks required channels: {sorted(absent)}")
    source_positions = np.asarray([positions[channel] for channel in MI9_CHANNELS])
    target_positions = np.asarray([positions[channel] for channel in missing_channels])
    matrix = _make_interpolation_matrix(source_positions, target_positions, alpha=alpha)
    if matrix.shape != (len(MISSING_INDICES), len(MI9_INDICES)):
        raise RuntimeError(f"Unexpected spherical-spline matrix shape: {matrix.shape}")
    return matrix


def restore_spherical_spline(
    source: RestorationSplit,
    normalization_dir,
    *,
    alpha: float = 1e-5,
) -> np.ndarray:
    """Restore each subject before reapplying that subject's target scaling."""

    matrix = spherical_spline_matrix(alpha)
    restored = np.empty_like(source.x_true22, dtype=np.float32)
    for subject in sorted(int(value) for value in np.unique(source.subject)):
        trial_mask = source.subject == subject
        mean, std = load_subject_normalization(normalization_dir, subject)
        physical_mi9 = (
            source.x_mi9[trial_mask].astype(np.float64)
            * std[np.asarray(MI9_INDICES)][None, :, None]
            + mean[np.asarray(MI9_INDICES)][None, :, None]
        )
        physical_missing = np.einsum("cm,nmt->nct", matrix, physical_mi9, optimize=True)
        restored_subject = np.empty(
            (int(trial_mask.sum()), 22, source.x_mi9.shape[2]), dtype=np.float64
        )
        restored_subject[:, MI9_INDICES, :] = physical_mi9
        restored_subject[:, MISSING_INDICES, :] = physical_missing
        restored_subject = (
            restored_subject - mean[None, :, None]
        ) / std[None, :, None]
        restored[trial_mask] = restored_subject.astype(np.float32)
    return enforce_observed_channels(restored, source.x_mi9)
