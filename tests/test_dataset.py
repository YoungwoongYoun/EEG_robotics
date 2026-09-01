import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eeg_pipeline.dataset import (
    ChannelMontage,
    PreprocessingConfig,
    build_channel_overlap_dataset,
    build_dataset,
)
from eeg_pipeline.preprocessing import EpochData


class DatasetBuildTests(unittest.TestCase):
    def test_build_uses_session_one_for_train_val_and_session_two_for_test(self):
        rng = np.random.default_rng(7)
        session_1 = EpochData(
            x=rng.normal(size=(20, 22, 16)).astype(np.float32),
            y=np.repeat(np.arange(4), 5),
            trial_index=np.arange(20),
            rejected_trial_index=np.asarray([3]),
        )
        session_2 = EpochData(
            x=rng.normal(loc=10.0, size=(8, 22, 16)).astype(np.float32),
            y=np.repeat(np.arange(4), 2),
            trial_index=np.arange(8),
            rejected_trial_index=np.asarray([2]),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            labels_dir = root / "labels"
            output_dir = root / "output"
            raw_dir.mkdir()
            labels_dir.mkdir()
            for filename in ("A01T.gdf", "A01E.gdf"):
                (raw_dir / filename).touch()
            (labels_dir / "A01E.mat").touch()

            config = PreprocessingConfig(
                raw_dir=raw_dir,
                labels_dir=labels_dir,
                output_dir=output_dir,
                subjects=(1,),
                validation_fraction=0.2,
                random_seed=42,
                export_torch=False,
            )
            with patch(
                "eeg_pipeline.dataset.load_bcic2a_session",
                side_effect=(session_1, session_2),
            ):
                summary = build_dataset(config)

            train = np.load(output_dir / "arrays" / "train.npz")
            validation = np.load(output_dir / "arrays" / "validation.npz")
            test = np.load(output_dir / "arrays" / "test.npz")
            self.assertEqual(train["x_true22"].shape[0], 16)
            self.assertEqual(validation["x_true22"].shape[0], 4)
            self.assertEqual(test["x_true22"].shape[0], 8)
            self.assertEqual(test["x_mi9"].shape[1], 9)
            self.assertEqual(summary["protocol"], "same-cohort inter-session")

            stats = np.load(output_dir / "normalization" / "A01.npz")
            train_indices = train["trial_index"]
            expected_mean = session_1.x[train_indices].mean(axis=(0, 2))
            np.testing.assert_allclose(stats["mean"], expected_mean)

            with (output_dir / "split_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["session"] for row in rows if row["split"] == "test"}, {"session_2"})
            self.assertEqual(
                {row["session"] for row in rows if row["split"] != "test"},
                {"session_1"},
            )

            saved_summary = json.loads(
                (output_dir / "preprocessing_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_summary["subjects"][0]["subject"], "A01")

    def test_channel_overlap_build_writes_all_montages_from_one_split(self):
        rng = np.random.default_rng(11)
        session_1 = EpochData(
            x=rng.normal(size=(20, 22, 16)).astype(np.float32),
            y=np.repeat(np.arange(4), 5),
            trial_index=np.arange(20),
            rejected_trial_index=np.asarray([], dtype=np.int64),
        )
        session_2 = EpochData(
            x=rng.normal(size=(8, 22, 16)).astype(np.float32),
            y=np.repeat(np.arange(4), 2),
            trial_index=np.arange(8),
            rejected_trial_index=np.asarray([], dtype=np.int64),
        )
        montages = (
            ChannelMontage(
                id="overlap_0",
                label="0/9 overlap",
                channels=("Fz", "FC3", "FC4", "C5", "C6", "CP3", "CP4", "P1", "P2"),
                expected_mi9_overlap=0,
            ),
            ChannelMontage(
                id="overlap_2",
                label="2/9 overlap",
                channels=("Fz", "FC3", "FC4", "C3", "C4", "CP3", "CP4", "P1", "P2"),
                expected_mi9_overlap=2,
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            labels_dir = root / "labels"
            output_dir = root / "output"
            raw_dir.mkdir()
            labels_dir.mkdir()
            for filename in ("A01T.gdf", "A01E.gdf"):
                (raw_dir / filename).touch()
            (labels_dir / "A01E.mat").touch()
            config = PreprocessingConfig(
                raw_dir=raw_dir,
                labels_dir=labels_dir,
                output_dir=output_dir,
                subjects=(1,),
                validation_fraction=0.2,
                random_seed=42,
                export_torch=False,
            )
            with patch(
                "eeg_pipeline.dataset.load_bcic2a_session",
                side_effect=(session_1, session_2),
            ) as loader:
                summary = build_channel_overlap_dataset(config, montages)

            self.assertIsNone(loader.call_args_list[0].kwargs["reference_indices"])
            train = np.load(output_dir / "arrays" / "train.npz")
            self.assertEqual(train["x_overlap_0"].shape, (16, 9, 16))
            self.assertEqual(train["x_overlap_2"].shape, (16, 9, 16))
            np.testing.assert_allclose(
                train["x_overlap_0"].mean(axis=(0, 2)), 0.0, atol=1e-5
            )
            self.assertTrue(
                (output_dir / "normalization" / "overlap_0" / "A01.npz").is_file()
            )
            self.assertEqual(summary["montages"][1]["mi9_overlap_count"], 2)


if __name__ == "__main__":
    unittest.main()
