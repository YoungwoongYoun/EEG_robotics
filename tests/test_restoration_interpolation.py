import tempfile
import unittest
from pathlib import Path

import numpy as np

from eeg_pipeline.channels import MI9_INDICES
from eeg_pipeline.experiments.restoration.data import RestorationSplit
from eeg_pipeline.experiments.restoration.interpolation import (
    restore_spherical_spline,
    spherical_spline_matrix,
)


class RestorationInterpolationTests(unittest.TestCase):
    def test_matrix_and_restoration_contract(self):
        matrix = spherical_spline_matrix()
        self.assertEqual(matrix.shape, (13, 9))
        rng = np.random.default_rng(3)
        true22 = rng.normal(size=(2, 22, 20)).astype(np.float32)
        source = RestorationSplit(
            x_mi9=true22[:, MI9_INDICES].copy(),
            x_true22=true22,
            y=np.asarray([0, 1]),
            subject=np.asarray([1, 1]),
            trial_index=np.asarray([0, 1]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            normalization = Path(temporary)
            np.savez_compressed(
                normalization / "A01.npz",
                mean=np.zeros(22, dtype=np.float32),
                std=np.ones(22, dtype=np.float32),
            )
            restored = restore_spherical_spline(source, normalization)
        self.assertEqual(restored.shape, true22.shape)
        self.assertTrue(np.isfinite(restored).all())
        np.testing.assert_array_equal(restored[:, MI9_INDICES], source.x_mi9)


if __name__ == "__main__":
    unittest.main()
