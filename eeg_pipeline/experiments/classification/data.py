"""Load one input representation as pooled Session-1/Session-2 datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ...channels import MI9_INDICES
from .config import InputConfig


def zero_pad_mi9(x_mi9: np.ndarray) -> np.ndarray:
    """Embed normalized MI-9 at canonical positions in a 22-channel tensor."""

    if x_mi9.ndim != 3 or x_mi9.shape[1] != len(MI9_INDICES):
        raise ValueError("x_mi9 must have shape [N, 9, T]")
    padded = np.zeros((x_mi9.shape[0], 22, x_mi9.shape[2]), dtype=x_mi9.dtype)
    padded[:, MI9_INDICES, :] = x_mi9
    return padded


@dataclass(frozen=True)
class GlobalSplits:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset


class InputDataRepository:
    """Validate one representation and pool selected subjects without mixing sessions."""

    def __init__(self, input_config: InputConfig):
        self.input = input_config
        self._splits = {
            split: self._load_split(input_config.arrays_dir / f"{split}.npz")
            for split in ("train", "validation", "test")
        }
        self._validate_protocol()

    def _load_split(self, path) -> dict[str, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input split: {path}")
        with np.load(path) as payload:
            required = {self.input.array_key, "y", "subject", "trial_index"}
            missing = required - set(payload.files)
            if missing:
                raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
            return {key: payload[key] for key in required}

    def _transform(self, x: np.ndarray) -> np.ndarray:
        if self.input.transform == "zero_pad_mi9_to_22":
            return zero_pad_mi9(x)
        return x

    def _validate_protocol(self) -> None:
        train_keys = set(zip(
            self._splits["train"]["subject"].tolist(),
            self._splits["train"]["trial_index"].tolist(),
        ))
        validation_keys = set(zip(
            self._splits["validation"]["subject"].tolist(),
            self._splits["validation"]["trial_index"].tolist(),
        ))
        if train_keys & validation_keys:
            raise ValueError("Training and validation splits overlap")

        for split, payload in self._splits.items():
            x = self._transform(payload[self.input.array_key])
            y = payload["y"]
            subject = payload["subject"]
            trial_index = payload["trial_index"]
            if x.ndim != 3 or x.shape[1] != self.input.n_channels:
                raise ValueError(
                    f"Unexpected {split} shape for {self.input.id}: {x.shape}"
                )
            if not (x.shape[0] == y.shape[0] == subject.shape[0] == trial_index.shape[0]):
                raise ValueError(f"Mismatched sample counts in {split}")
            if not np.isfinite(x).all():
                raise ValueError(f"Non-finite samples in {split}")
            if np.any((y < 0) | (y > 3)):
                raise ValueError(f"Invalid labels in {split}")

    def subjects(self) -> tuple[int, ...]:
        observed = set(self._splits["train"]["subject"].tolist())
        return tuple(sorted(int(value) for value in observed))

    def pooled(self, subjects: tuple[int, ...]) -> GlobalSplits:
        requested = set(subjects)
        datasets = {}
        for split, payload in self._splits.items():
            mask = np.isin(payload["subject"], subjects)
            observed = set(int(value) for value in payload["subject"][mask])
            missing = requested - observed
            if missing:
                raise ValueError(f"Subjects absent from {split}: {sorted(missing)}")
            x = np.ascontiguousarray(self._transform(payload[self.input.array_key][mask]))
            y = np.ascontiguousarray(payload["y"][mask])
            subject = np.ascontiguousarray(payload["subject"][mask])
            trial_index = np.ascontiguousarray(payload["trial_index"][mask])
            datasets[split] = TensorDataset(
                torch.from_numpy(x).float().unsqueeze(1),
                torch.from_numpy(y).long(),
                torch.from_numpy(subject).long(),
                torch.from_numpy(trial_index).long(),
            )
        return GlobalSplits(
            train=datasets["train"],
            validation=datasets["validation"],
            test=datasets["test"],
        )
