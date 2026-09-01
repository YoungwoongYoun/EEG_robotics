import tempfile
import unittest
from pathlib import Path

import numpy as np

from eeg_pipeline.channels import MI9_INDICES
from eeg_pipeline.experiments.restoration.data import (
    MISSING_INDICES,
    RestorationSplit,
    enforce_observed_channels,
    validate_restored_split,
    write_restored_split,
)
from eeg_pipeline.experiments.restoration.metrics import reconstruction_metrics


def source_split() -> RestorationSplit:
    rng = np.random.default_rng(4)
    true22 = rng.normal(size=(4, 22, 16)).astype(np.float32)
    return RestorationSplit(
        x_mi9=true22[:, MI9_INDICES, :].copy(),
        x_true22=true22,
        y=np.asarray([0, 1, 2, 3]),
        subject=np.asarray([1, 1, 2, 2]),
        trial_index=np.asarray([0, 1, 0, 1]),
    )


class RestorationDataTests(unittest.TestCase):
    def test_observed_channels_are_hard_copied_and_validated(self):
        source = source_split()
        restored = np.zeros_like(source.x_true22)
        enforce_observed_channels(restored, source.x_mi9)
        validate_restored_split(restored, source, "test")
        np.testing.assert_array_equal(restored[:, MI9_INDICES], source.x_mi9)

    def test_atomic_output_contains_classifier_contract(self):
        source = source_split()
        restored = enforce_observed_channels(np.zeros_like(source.x_true22), source.x_mi9)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.npz"
            write_restored_split(path, "x_restored22", restored, source)
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(
                    set(payload.files),
                    {"x_restored22", "y", "subject", "trial_index"},
                )
                np.testing.assert_array_equal(payload["subject"], source.subject)

    def test_primary_metrics_exclude_perfect_observed_channels(self):
        source = source_split()
        restored = source.x_true22.copy()
        restored[:, MISSING_INDICES, :] += 1.0
        aggregate, channels, subjects = reconstruction_metrics(source, restored)
        self.assertAlmostEqual(aggregate["observed9_mse"], 0.0)
        self.assertAlmostEqual(aggregate["missing13_mse"], 1.0)
        self.assertGreater(aggregate["missing13_mse"], aggregate["all22_mse"])
        self.assertIn("missing13_log_bandpower_mse", aggregate)
        self.assertIn("missing13_spatial_correlation_mse", aggregate)
        self.assertGreaterEqual(aggregate["missing13_log_bandpower_mse"], 0.0)
        self.assertGreaterEqual(aggregate["missing13_spatial_correlation_mse"], 0.0)
        self.assertEqual(len(channels), 22)
        self.assertEqual(len(subjects), 2)


if __name__ == "__main__":
    unittest.main()
