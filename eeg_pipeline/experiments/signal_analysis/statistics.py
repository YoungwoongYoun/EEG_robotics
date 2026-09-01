"""Subject-level paired inference for Stage-D error endpoints."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, wilcoxon


def holm_adjust(raw_p: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment in original row order."""

    raw_p = np.asarray(raw_p, dtype=np.float64)
    order = np.argsort(raw_p)
    adjusted_sorted = np.maximum.accumulate(
        (raw_p.size - np.arange(raw_p.size)) * raw_p[order]
    )
    result = np.empty_like(raw_p)
    result[order] = np.minimum(adjusted_sorted, 1.0)
    return result


def paired_endpoint_rows(
    subject_values: dict[str, dict[int, float]],
    reference: str,
    endpoint: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, float | int | str]]:
    """Compare lower-is-better endpoints to a reference and Holm-correct the family."""

    if reference not in subject_values:
        raise KeyError(f"Missing statistical reference: {reference}")
    rows = []
    raw = []
    for method, values in subject_values.items():
        if method == reference or method == "true22":
            continue
        subjects = sorted(set(values) & set(subject_values[reference]))
        if not subjects:
            raise ValueError(f"No aligned subjects for {method} vs {reference}")
        difference = np.asarray([
            values[subject] - subject_values[reference][subject] for subject in subjects
        ], dtype=np.float64)
        if np.allclose(difference, 0.0):
            p_value = 1.0
            rank_biserial = 0.0
        else:
            p_value = float(wilcoxon(difference, alternative="two-sided").pvalue)
            nonzero = difference[~np.isclose(difference, 0.0)]
            ranks = rankdata(np.abs(nonzero))
            rank_biserial = float(
                (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
            )
        samples = difference[rng.integers(0, difference.size, (bootstrap_samples, difference.size))]
        bootstrap_means = samples.mean(axis=1)
        row: dict[str, float | int | str] = {
            "endpoint": endpoint,
            "reference": reference,
            "comparison": method,
            "n_subjects": difference.size,
            "mean_difference_comparison_minus_reference": float(difference.mean()),
            "bootstrap_ci_low": float(np.quantile(bootstrap_means, 0.025)),
            "bootstrap_ci_high": float(np.quantile(bootstrap_means, 0.975)),
            "reference_wins": int(np.sum(difference > 0)),
            "ties": int(np.sum(np.isclose(difference, 0.0))),
            "wilcoxon_raw_p": p_value,
            "rank_biserial_reference_better": rank_biserial,
        }
        rows.append(row)
        raw.append(p_value)
    adjusted = holm_adjust(np.asarray(raw))
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = float(value)
    return rows
