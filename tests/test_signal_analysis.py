from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eeg_pipeline.experiments.signal_analysis.config import load_signal_analysis_config
from eeg_pipeline.experiments.signal_analysis.metrics import (
    affine_invariant_distance,
    csp_log_variance,
    fit_ovr_csp,
    log_relative_bandpower,
    normalized_covariance,
)
from eeg_pipeline.experiments.signal_analysis.statistics import holm_adjust


class SignalAnalysisTests(unittest.TestCase):
    def test_signal_analysis_config_loads_project_paths(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        config_path = root / "analysis.yaml"
        config_path.write_text(
        """
id: test
source_arrays_dir: source
output_dir: output
methods:
  - {id: true22, label: True, kind: true22}
  - {id: zero, label: Zero, kind: zero_padded}
  - id: autoencoder_eeg_aware
    label: Restored
    kind: restored
    test_path: restored.npz
    array_key: x_restored22
analysis:
  bands: {mu: [8, 13], beta: [13, 30]}
  total_band: [8, 30]
  covariance_ridge: 0.001
  csp_filters_per_class: 2
  bootstrap_samples: 100
  tfr_window_samples: 32
  tfr_overlap_samples: 16
""",
            encoding="utf-8",
        )
        config = load_signal_analysis_config(config_path, root)
        self.assertEqual(config.source_arrays_dir, root / "source")
        self.assertEqual(config.methods[-1].test_path, root / "restored.npz")


    def test_bandpower_and_covariance_identity(self) -> None:
        time = np.arange(0, 4, 1 / 250.0)
        signal_value = np.sin(2 * np.pi * 10 * time)
        x = np.tile(signal_value, (8, 22, 1)).astype(np.float32)
        bandpower = log_relative_bandpower(
            x, 250.0, (("mu", 8.0, 13.0), ("beta", 13.0, 30.0)), (8.0, 30.0)
        )
        self.assertEqual(bandpower.shape, (8, 22, 2))
        self.assertTrue(np.all(bandpower[..., 0] > bandpower[..., 1]))
        covariance = normalized_covariance(x, 0.001)
        self.assertAlmostEqual(affine_invariant_distance(covariance, covariance), 0.0, places=10)


    def test_csp_fit_and_transform_are_finite(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(size=(32, 22, 128)).astype(np.float32)
        y = np.repeat(np.arange(4), 8)
        x[y == 0, 0] *= 2.0
        filters = fit_ovr_csp(x, y, range(4), ridge=0.001, filters_per_class=2)
        features = csp_log_variance(x, filters)
        self.assertEqual(filters.shape, (8, 22))
        self.assertEqual(features.shape, (32, 8))
        self.assertTrue(np.isfinite(features).all())


    def test_holm_adjustment_preserves_order_and_monotonicity(self) -> None:
        raw = np.asarray([0.04, 0.001, 0.02])
        adjusted = holm_adjust(raw)
        np.testing.assert_allclose(adjusted, [0.04, 0.003, 0.04])
        self.assertTrue(np.all(adjusted >= raw))


if __name__ == "__main__":
    unittest.main()
