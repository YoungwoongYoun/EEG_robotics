"""Predeclared grand-average figures for the Stage-D analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from ...channels import EEG_CHANNELS_22


def _sensor_positions() -> np.ndarray:
    montage = mne.channels.make_standard_montage("standard_1020")
    positions = montage.get_positions()["ch_pos"]
    missing = [channel for channel in EEG_CHANNELS_22 if channel not in positions]
    if missing:
        raise ValueError(f"Montage is missing EEG channels: {missing}")
    return np.asarray([positions[channel][:2] for channel in EEG_CHANNELS_22])


def plot_true_topographies(
    values: np.ndarray,
    band_names: tuple[str, ...],
    class_names: tuple[str, ...],
    output: Path,
) -> None:
    """Plot True-22 grand-average class-conditional log-relative power."""

    positions = _sensor_positions()
    figure, axes = plt.subplots(
        len(class_names), len(band_names), figsize=(6, 10), constrained_layout=True
    )
    lower, upper = np.quantile(values, [0.02, 0.98])
    for class_index, class_name in enumerate(class_names):
        for band_index, band_name in enumerate(band_names):
            axis = axes[class_index, band_index]
            mne.viz.plot_topomap(
                values[class_index, :, band_index], positions, axes=axis, show=False,
                cmap="RdBu_r", vlim=(float(lower), float(upper)), contours=4,
            )
            axis.set_title(f"{class_name} — {band_name}")
    figure.suptitle("True-22 class-conditional log relative power")
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_error_topographies(
    error_rmse: dict[str, np.ndarray],
    labels: dict[str, str],
    band_names: tuple[str, ...],
    output: Path,
) -> None:
    """Plot grand-average restoration error without subject selection."""

    positions = _sensor_positions()
    methods = tuple(error_rmse)
    figure, axes = plt.subplots(
        len(methods), len(band_names),
        figsize=(6, max(3, 2.2 * len(methods))), constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    upper = max(float(np.quantile(value, 0.98)) for value in error_rmse.values())
    for method_index, method in enumerate(methods):
        for band_index, band_name in enumerate(band_names):
            axis = axes[method_index, band_index]
            mne.viz.plot_topomap(
                error_rmse[method][:, band_index], positions, axes=axis, show=False,
                cmap="magma", vlim=(0.0, upper), contours=4,
            )
            axis.set_title(f"{labels[method]} — {band_name}")
    figure.suptitle("Class-conditional log-relative-power RMSE")
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_tfr_errors(
    tfr: dict[str, np.ndarray],
    labels: dict[str, str],
    frequencies: np.ndarray,
    times: np.ndarray,
    output: Path,
) -> None:
    """Plot class-averaged missing-channel TFR absolute error for every restoration."""

    methods = tuple(
        method for method in tfr
        if method not in {"true22", "zero_padded_mi9"}
    )
    columns = 2
    rows = int(np.ceil(len(methods) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10, 3.2 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    errors = {method: np.abs(tfr[method] - tfr["true22"]).mean(axis=0) for method in methods}
    upper = max(float(np.quantile(value, 0.98)) for value in errors.values())
    for axis, method in zip(axes, methods, strict=False):
        image = axis.imshow(
            errors[method], origin="lower", aspect="auto", cmap="magma",
            extent=(times[0], times[-1], frequencies[0], frequencies[-1]),
            vmin=0.0, vmax=upper,
        )
        axis.set_title(labels[method])
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Frequency (Hz)")
    for axis in axes[len(methods):]:
        axis.set_visible(False)
    figure.colorbar(image, ax=axes[:len(methods)].tolist(), label="Absolute log-power error")
    figure.suptitle("Grand-average missing-channel time-frequency error")
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_metric_summary(
    rows: list[dict[str, float | str]],
    labels: dict[str, str],
    output: Path,
) -> None:
    endpoints = (
        ("bandpower_mse", "Log-relative-power MSE"),
        ("covariance_distance", "Covariance AIRM distance"),
        ("csp_feature_mse", "CSP feature MSE"),
    )
    methods = tuple(row["method"] for row in rows if row["method"] != "true22")
    figure, axes = plt.subplots(1, len(endpoints), figsize=(15, 4), constrained_layout=True)
    for axis, (key, title) in zip(axes, endpoints, strict=True):
        means = [float(next(row[key] for row in rows if row["method"] == method)) for method in methods]
        stds = [float(next(row[f"{key}_subject_std"] for row in rows if row["method"] == method)) for method in methods]
        positions = np.arange(len(methods))
        axis.bar(positions, means, yerr=stds, capsize=3)
        axis.set_xticks(positions, [labels[method] for method in methods], rotation=45, ha="right")
        axis.set_title(title)
        axis.set_ylabel("Lower is better")
        axis.set_yscale("log")
    figure.savefig(output, dpi=200)
    plt.close(figure)
