from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eeg_pipeline.experiments.latency.config import load_latency_benchmark_config
from eeg_pipeline.experiments.latency.runner import (
    SourceTest,
    _method_signature,
    stratified_latency_indices,
    summarize_timings,
)


class LatencyBenchmarkTests(unittest.TestCase):
    def test_config_resolves_all_required_method_kinds(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        path = root / "latency.yaml"
        path.write_text(
            """
id: latency
source_arrays_dir: source
output_dir: output
classifier_seed: 0
benchmark_seed: 17
cpu_threads: 1
allow_tf32: true
warmup_trials: 2
trials_per_subject_class: 1
methods:
  - {id: true22, label: True, kind: true22, classification_config: true.yaml}
  - {id: direct_mi9, label: Direct, kind: direct_mi9, classification_config: direct.yaml}
  - {id: zero_padded_mi9, label: Zero, kind: zero_padded, classification_config: zero.yaml}
  - id: spherical_spline
    label: Spline
    kind: spherical_spline
    classification_config: spline-clf.yaml
    restoration_config: spline-rest.yaml
  - id: autoencoder
    label: AE
    kind: learned
    classification_config: ae-clf.yaml
    restoration_config: ae-rest.yaml
""",
            encoding="utf-8",
        )
        config = load_latency_benchmark_config(path, root)
        self.assertEqual(config.classifier_seed, 0)
        self.assertEqual(config.methods[-1].restoration_config, root / "ae-rest.yaml")
        signature = _method_signature(
            config, config.methods[-1], __import__("torch").device("cpu"), 2, 1
        )
        json.dumps(signature)

    def test_stratified_selection_has_every_subject_class_once(self) -> None:
        subject = np.repeat(np.arange(1, 10), 8)
        labels = np.tile(np.repeat(np.arange(4), 2), 9)
        trials = len(labels)
        source = SourceTest(
            x_mi9=np.zeros((trials, 9, 16), dtype=np.float32),
            x_true22=np.zeros((trials, 22, 16), dtype=np.float32),
            y=labels,
            subject=subject,
            trial_index=np.arange(trials),
        )
        indices = stratified_latency_indices(source, 1, seed=17)
        self.assertEqual(indices.size, 36)
        pairs = {(int(subject[index]), int(labels[index])) for index in indices}
        self.assertEqual(len(pairs), 36)
        np.testing.assert_array_equal(
            indices, stratified_latency_indices(source, 1, seed=17)
        )

    def test_timing_summary_reports_expected_quantiles(self) -> None:
        rows = [
            {
                "transfer_ms": value,
                "restoration_ms": 2 * value,
                "classification_ms": 3 * value,
                "end_to_end_ms": 6 * value,
            }
            for value in (1.0, 2.0, 3.0, 4.0)
        ]
        summary = summarize_timings(rows)
        self.assertEqual(summary["transfer_ms_median"], 2.5)
        self.assertAlmostEqual(summary["end_to_end_ms_p95"], 23.1)


if __name__ == "__main__":
    unittest.main()
