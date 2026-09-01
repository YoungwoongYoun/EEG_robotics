"""Band-power, covariance, CSP, and time-frequency signal-preservation metrics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import linalg, signal

from ...channels import MI9_INDICES


EPSILON = 1e-12
MISSING_INDICES = np.asarray([index for index in range(22) if index not in MI9_INDICES])


def log_relative_bandpower(
    x: np.ndarray,
    sampling_rate: float,
    bands: tuple[tuple[str, float, float], ...],
    total_band: tuple[float, float],
) -> np.ndarray:
    """Return log relative FFT power with shape [trial, channel, band]."""

    frequencies = np.fft.rfftfreq(x.shape[-1], d=1.0 / sampling_rate)
    power = np.abs(np.fft.rfft(x.astype(np.float64, copy=False), axis=-1)) ** 2
    total_mask = (frequencies >= total_band[0]) & (frequencies < total_band[1])
    if not np.any(total_mask):
        raise ValueError("No FFT bins fall inside total_band")
    total = power[..., total_mask].sum(axis=-1)
    values = []
    for _, low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            raise ValueError(f"No FFT bins fall inside band {low}-{high} Hz")
        relative = power[..., mask].sum(axis=-1) / np.maximum(total, EPSILON)
        values.append(np.log(np.maximum(relative, EPSILON)))
    return np.stack(values, axis=-1)


def normalized_covariance(x: np.ndarray, ridge: float) -> np.ndarray:
    """Average trace-normalized trial covariance and make it strictly SPD."""

    if x.ndim != 3 or x.shape[1] != 22:
        raise ValueError("Covariance input must have shape [N, 22, T]")
    centered = x.astype(np.float64, copy=False) - x.mean(axis=-1, keepdims=True)
    covariances = centered @ np.swapaxes(centered, 1, 2)
    covariances /= max(x.shape[-1] - 1, 1)
    traces = np.trace(covariances, axis1=1, axis2=2)
    covariances /= np.maximum(traces[:, None, None], EPSILON)
    covariance = covariances.mean(axis=0)
    covariance = (covariance + covariance.T) / 2.0
    scale = np.trace(covariance) / covariance.shape[0]
    return covariance + ridge * max(scale, EPSILON) * np.eye(covariance.shape[0])


def affine_invariant_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Dimension-normalized affine-invariant Riemannian SPD distance."""

    eigenvalues = linalg.eigvalsh(right, left, check_finite=False)
    eigenvalues = np.maximum(eigenvalues, EPSILON)
    return float(np.linalg.norm(np.log(eigenvalues)) / np.sqrt(left.shape[0]))


def fit_ovr_csp(
    x: np.ndarray,
    y: np.ndarray,
    classes: Iterable[int],
    ridge: float,
    filters_per_class: int,
) -> np.ndarray:
    """Fit deterministic one-vs-rest CSP filters from Session-1 True-22 only."""

    if filters_per_class < 2 or filters_per_class % 2:
        raise ValueError("filters_per_class must be a positive even number")
    filters = []
    half = filters_per_class // 2
    for class_id in classes:
        positive = x[y == class_id]
        negative = x[y != class_id]
        if positive.size == 0 or negative.size == 0:
            raise ValueError(f"CSP class {class_id} is absent")
        class_cov = normalized_covariance(positive, ridge)
        rest_cov = normalized_covariance(negative, ridge)
        _, eigenvectors = linalg.eigh(
            class_cov, class_cov + rest_cov, check_finite=False
        )
        indices = np.concatenate((np.arange(half), np.arange(-half, 0)))
        filters.append(eigenvectors[:, indices].T)
    return np.concatenate(filters, axis=0)


def csp_log_variance(x: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Apply fixed CSP filters and return normalized log-variance features."""

    projected = np.einsum("fc,nct->nft", filters, x.astype(np.float64, copy=False))
    variances = np.var(projected, axis=-1, ddof=1)
    relative = variances / np.maximum(variances.sum(axis=1, keepdims=True), EPSILON)
    return np.log(np.maximum(relative, EPSILON))


def feature_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.ravel().astype(np.float64)
    right_flat = right.ravel().astype(np.float64)
    left_flat -= left_flat.mean()
    right_flat -= right_flat.mean()
    denominator = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    return float(left_flat @ right_flat / denominator) if denominator > EPSILON else 0.0


def class_conditional_tfr(
    x: np.ndarray,
    y: np.ndarray,
    classes: tuple[int, ...],
    sampling_rate: float,
    total_band: tuple[float, float],
    window_samples: int,
    overlap_samples: int,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average missing-channel log spectrogram for each class, in bounded batches."""

    class_sum: dict[int, np.ndarray | None] = {class_id: None for class_id in classes}
    class_trials = {class_id: 0 for class_id in classes}
    frequencies: np.ndarray | None = None
    times: np.ndarray | None = None
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        frequencies, times, spectra = signal.spectrogram(
            x[start:stop, MISSING_INDICES].astype(np.float64, copy=False),
            fs=sampling_rate,
            window="hann",
            nperseg=window_samples,
            noverlap=overlap_samples,
            detrend=False,
            scaling="density",
            mode="psd",
            axis=-1,
        )
        mask = (frequencies >= total_band[0]) & (frequencies <= total_band[1])
        spectra = spectra[:, :, mask].mean(axis=1)
        batch_labels = y[start:stop]
        for class_id in classes:
            selected = spectra[batch_labels == class_id]
            if selected.size == 0:
                continue
            contribution = selected.sum(axis=0)
            class_sum[class_id] = (
                contribution if class_sum[class_id] is None
                else class_sum[class_id] + contribution
            )
            class_trials[class_id] += selected.shape[0]
    assert frequencies is not None and times is not None
    mask = (frequencies >= total_band[0]) & (frequencies <= total_band[1])
    averages = []
    for class_id in classes:
        if class_trials[class_id] == 0 or class_sum[class_id] is None:
            raise ValueError(f"TFR class {class_id} is absent")
        averages.append(np.log(np.maximum(class_sum[class_id] / class_trials[class_id], EPSILON)))
    return frequencies[mask], times, np.stack(averages)
