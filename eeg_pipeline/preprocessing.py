"""Leakage-controlled preprocessing for BCI Competition IV Dataset 2a.

The public functions in the first half of this module operate only on NumPy
arrays and are unit-testable without EEG files. MNE and SciPy are imported only
inside the file-loading functions so importing the package stays lightweight.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .channels import (
    EEG_CHANNELS_22,
    EVALUATION_CUE_EVENT_ID,
    MI9_INDICES,
    REJECTED_TRIAL_EVENT_ID,
    TRAIN_CUE_EVENT_IDS,
)


@dataclass(frozen=True)
class EpochData:
    """One subject/session after filtering, epoching, and referencing."""

    x: np.ndarray
    y: np.ndarray
    trial_index: np.ndarray
    rejected_trial_index: np.ndarray


@dataclass(frozen=True)
class StandardizationStats:
    """Per-channel statistics fitted on Session-1 training trials only."""

    mean: np.ndarray
    std: np.ndarray


def apply_available_average_reference(
    epochs: np.ndarray,
    channel_indices: Iterable[int],
) -> np.ndarray:
    """Reference every channel to the mean of the deployment-available montage.

    Args:
        epochs: EEG array with shape ``[trials, 22, time]``.
        channel_indices: Available-channel indices in the 22-channel order.
    """

    epochs = np.asarray(epochs)
    indices = np.asarray(tuple(channel_indices), dtype=int)
    if epochs.ndim != 3:
        raise ValueError(f"Expected [trials, channels, time], got {epochs.shape}")
    if epochs.shape[1] != len(EEG_CHANNELS_22):
        raise ValueError(
            f"Expected {len(EEG_CHANNELS_22)} channels, got {epochs.shape[1]}"
        )
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= epochs.shape[1]):
        raise ValueError("channel_indices must contain valid channel indices")
    if np.unique(indices).size != indices.size:
        raise ValueError("channel_indices must not contain duplicates")

    reference = epochs[:, indices, :].mean(axis=1, keepdims=True)
    return epochs - reference


def apply_mi9_average_reference(
    epochs: np.ndarray,
    mi9_indices: Iterable[int] = MI9_INDICES,
) -> np.ndarray:
    """Reference all 22 channels to the canonical deployment MI-9 average."""

    return apply_available_average_reference(epochs, mi9_indices)


def select_average_referenced_channels(
    epochs: np.ndarray,
    channel_indices: Iterable[int],
) -> np.ndarray:
    """Select one montage after referencing it to its own channel average."""

    indices = tuple(channel_indices)
    referenced = apply_available_average_reference(epochs, indices)
    return referenced[:, indices, :]


def stratified_train_val_indices(
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split Session 1 while preserving every class in train and validation."""

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        if class_indices.size < 2:
            raise ValueError(f"Class {label!r} needs at least two trials")
        shuffled = rng.permutation(class_indices)
        validation_size = round(class_indices.size * validation_fraction)
        validation_size = min(max(validation_size, 1), class_indices.size - 1)
        validation_parts.append(shuffled[:validation_size])
        train_parts.append(shuffled[validation_size:])

    train_indices = np.sort(np.concatenate(train_parts))
    validation_indices = np.sort(np.concatenate(validation_parts))
    return train_indices, validation_indices


def fit_channelwise_standardizer(
    training_epochs: np.ndarray,
    epsilon: float = 1e-8,
) -> StandardizationStats:
    """Fit per-channel z-score parameters across training trials and time."""

    training_epochs = np.asarray(training_epochs)
    if training_epochs.ndim != 3 or training_epochs.shape[0] == 0:
        raise ValueError("training_epochs must be a non-empty [N, C, T] array")
    mean = training_epochs.mean(axis=(0, 2))
    std = training_epochs.std(axis=(0, 2))
    std = np.where(std < epsilon, 1.0, std)
    return StandardizationStats(mean=mean, std=std)


def apply_standardizer(
    epochs: np.ndarray,
    stats: StandardizationStats,
) -> np.ndarray:
    """Apply training-only channel statistics to an epoch array."""

    epochs = np.asarray(epochs)
    if epochs.ndim != 3:
        raise ValueError("epochs must have shape [N, C, T]")
    if epochs.shape[1] != stats.mean.shape[0] or stats.mean.shape != stats.std.shape:
        raise ValueError("Channel count does not match standardization statistics")
    normalized = (epochs - stats.mean[None, :, None]) / stats.std[None, :, None]
    return normalized.astype(np.float32, copy=False)


def artifact_trial_mask(
    cue_samples: np.ndarray,
    artifact_samples: np.ndarray,
    sampling_frequency: float,
    seconds_before_cue: float = 2.0,
    seconds_after_cue: float = 4.0,
) -> np.ndarray:
    """Return a mask for cues belonging to trials flagged with event 1023.

    Dataset 2a places the MI cue two seconds after trial start. Associating the
    rejection event over this full trial interval is more robust than relying on
    MNE's BAD-prefix convention, because GDF event ``1023`` is not named BAD.
    """

    cue_samples = np.asarray(cue_samples, dtype=np.int64)
    artifact_samples = np.asarray(artifact_samples, dtype=np.int64)
    before = round(seconds_before_cue * sampling_frequency)
    after = round(seconds_after_cue * sampling_frequency)
    return np.asarray(
        [np.any((artifact_samples >= cue - before) & (artifact_samples <= cue + after))
         for cue in cue_samples],
        dtype=bool,
    )


def _events_for_descriptions(raw: Any, event_id: dict[str, int]) -> np.ndarray:
    import mne

    available = set(raw.annotations.description)
    selected = {name: value for name, value in event_id.items() if name in available}
    if not selected:
        return np.empty((0, 3), dtype=np.int64)
    events, _ = mne.events_from_annotations(raw, event_id=selected, verbose=False)
    return events


def _load_evaluation_labels(path: Path) -> np.ndarray:
    from scipy.io import loadmat

    payload = loadmat(path)
    if "classlabel" not in payload:
        raise KeyError(f"{path} does not contain 'classlabel'")
    labels = np.asarray(payload["classlabel"]).reshape(-1).astype(np.int64) - 1
    if labels.size == 0 or np.any((labels < 0) | (labels > 3)):
        raise ValueError(f"Invalid evaluation labels in {path}")
    return labels


def load_bcic2a_session(
    gdf_path: Path,
    *,
    session: str,
    evaluation_label_path: Path | None,
    low_frequency: float,
    high_frequency: float,
    epoch_tmin: float,
    epoch_tmax: float,
    reject_flagged_trials: bool,
    reference_indices: Iterable[int] | None = MI9_INDICES,
) -> EpochData:
    """Load one session, optionally referencing before normalization is fitted."""

    import mne

    if session not in {"train", "test"}:
        raise ValueError("session must be 'train' or 'test'")

    raw = mne.io.read_raw_gdf(
        str(gdf_path),
        preload=True,
        eog=(22, 23, 24),
        verbose=False,
    )
    if len(raw.ch_names) < 25:
        raise ValueError(f"Expected at least 25 channels in {gdf_path}")

    rename_map = {
        original: canonical
        for original, canonical in zip(raw.ch_names[:22], EEG_CHANNELS_22)
    }
    raw.rename_channels(rename_map)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
    raw.filter(
        l_freq=low_frequency,
        h_freq=high_frequency,
        picks=list(EEG_CHANNELS_22),
        fir_design="firwin",
        verbose=False,
    )

    cue_event_id = TRAIN_CUE_EVENT_IDS if session == "train" else EVALUATION_CUE_EVENT_ID
    cue_events = _events_for_descriptions(raw, cue_event_id)
    if cue_events.size == 0:
        raise ValueError(f"No MI cue events found in {gdf_path}")

    if session == "train":
        labels = cue_events[:, 2].astype(np.int64) - 1
    else:
        if evaluation_label_path is None:
            raise ValueError("evaluation_label_path is required for the test session")
        labels = _load_evaluation_labels(evaluation_label_path)
        if labels.size != cue_events.shape[0]:
            raise ValueError(
                f"Evaluation label/event mismatch for {gdf_path}: "
                f"{labels.size} labels vs {cue_events.shape[0]} cues"
            )

    epochs = mne.Epochs(
        raw,
        cue_events,
        event_id=None,
        tmin=epoch_tmin,
        tmax=epoch_tmax,
        baseline=None,
        picks=list(EEG_CHANNELS_22),
        preload=True,
        proj=False,
        reject_by_annotation=False,
        verbose=False,
    )
    data = epochs.get_data().astype(np.float32, copy=False)
    if data.shape[0] != labels.size:
        raise RuntimeError("MNE dropped epochs before explicit artifact handling")

    artifact_events = _events_for_descriptions(raw, REJECTED_TRIAL_EVENT_ID)
    artifact_samples = (
        artifact_events[:, 0] if artifact_events.size else np.empty(0, dtype=np.int64)
    )
    rejected = artifact_trial_mask(cue_events[:, 0], artifact_samples, raw.info["sfreq"])
    trial_index = np.arange(labels.size, dtype=np.int64)
    rejected_trial_index = trial_index[rejected]

    if reject_flagged_trials:
        keep = ~rejected
        data = data[keep]
        labels = labels[keep]
        trial_index = trial_index[keep]

    if reference_indices is not None:
        data = apply_available_average_reference(data, reference_indices)
    return EpochData(
        x=data,
        y=labels,
        trial_index=trial_index,
        rejected_trial_index=rejected_trial_index,
    )
