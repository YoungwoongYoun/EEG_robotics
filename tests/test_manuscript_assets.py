from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from eeg_pipeline.experiments.manuscript_assets import (
    METHOD_ORDER,
    build_manuscript_assets,
    paired_statistics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ManuscriptStatisticsTests(unittest.TestCase):
    def test_paired_statistics_direction_and_holm_family(self) -> None:
        values = {
            "better": {f"A{i:02d}": float(i + 1) for i in range(1, 10)},
            "worse": {f"A{i:02d}": float(i) for i in range(1, 10)},
            "same": {f"A{i:02d}": float(i + 1) for i in range(1, 10)},
        }
        rows = paired_statistics(
            values,
            (("better", "worse"), ("better", "same")),
            bootstrap_samples=100,
            seed=1,
        )
        self.assertEqual(rows[0]["reference_wins"], 9)
        self.assertAlmostEqual(rows[0]["mean_difference_accuracy_pp"], 100.0)
        self.assertEqual(rows[1]["ties"], 9)
        self.assertEqual(rows[1]["wilcoxon_raw_p"], 1.0)
        self.assertGreaterEqual(rows[0]["holm_adjusted_p"], rows[0]["wilcoxon_raw_p"])

    def test_build_rejects_project_root_as_output(self) -> None:
        with self.assertRaises(ValueError):
            build_manuscript_assets(PROJECT_ROOT, PROJECT_ROOT)


class ManuscriptAssetIntegrationTests(unittest.TestCase):
    def test_builds_complete_package_from_saved_results(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            output = Path(temporary) / "package"
            build_manuscript_assets(PROJECT_ROOT, output)
            completion = json.loads((output / "BUILD_COMPLETE.json").read_text())
            self.assertEqual(completion["status"], "complete")
            self.assertEqual(completion["n_methods"], len(METHOD_ORDER))
            self.assertTrue((output / "TABLES.md").is_file())
            self.assertTrue((output / "figures/figure_02_matched_accuracy.pdf").is_file())
            with (output / "statistics/classifier_primary_paired.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            existing = next(row for row in rows if row["comparison"] == "autoencoder_eeg_aware_vs_direct_mi9")
            self.assertAlmostEqual(float(existing["mean_difference_accuracy_pp"]), 1.9335, places=3)
            self.assertAlmostEqual(float(existing["holm_adjusted_p"]), 0.03125, places=8)


if __name__ == "__main__":
    unittest.main()
