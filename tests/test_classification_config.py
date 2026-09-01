import unittest
from pathlib import Path

from eeg_pipeline.experiments.classification.config import load_experiment_config
from eeg_pipeline.experiments.classification.training import build_model
from eeg_pipeline.experiments.restoration.config import load_restoration_config


class ClassificationConfigTests(unittest.TestCase):
    def test_baseline_configs_are_independent_and_capacity_is_preserved(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "true22": (22, 127_212),
            "zero_padded_mi9": (22, 127_212),
            "direct_mi9": (9, 124_716),
        }
        for input_id, (channels, parameters) in expected.items():
            config = load_experiment_config(
                root / "configs" / "classification" / "baselines" / f"{input_id}.yaml",
                root,
            )
            self.assertEqual(config.name, input_id)
            self.assertEqual(config.input.id, input_id)
            self.assertEqual(config.input.n_channels, channels)
            self.assertEqual(config.seeds, (0, 1, 2, 3, 4))
            self.assertEqual(config.training.epochs, 125)
            self.assertEqual(config.training.batch_size, 48)
            self.assertTrue(config.training.sr_augmentation)
            self.assertEqual(config.training.optimizer, "adam")
            self.assertEqual(config.training.scheduler, "warmup_cosine")
            self.assertEqual(config.model.args["F1"], 32)
            self.assertTrue(config.model.args["use_group_attn"])
            self.assertEqual(
                config.model.reference_commit,
                "74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5",
            )
            self.assertEqual(
                sum(p.numel() for p in build_model(config).parameters() if p.requires_grad),
                parameters,
            )

    def test_channel_overlap_configs_reuse_identical_direct9_classifier(self):
        root = Path(__file__).resolve().parents[1]
        expected_keys = {
            "overlap_0": "x_overlap_0",
            "overlap_2": "x_overlap_2",
            "overlap_4": "x_overlap_4",
            "overlap_7": "x_overlap_7",
        }
        configs = []
        for input_id, array_key in expected_keys.items():
            config = load_experiment_config(
                root / "configs" / "classification" / "channel_overlap" / f"{input_id}.yaml",
                root,
            )
            self.assertEqual(config.input.array_key, array_key)
            self.assertEqual(config.input.n_channels, 9)
            self.assertEqual(config.seeds, (0, 1, 2, 3, 4))
            self.assertEqual(
                sum(p.numel() for p in build_model(config).parameters() if p.requires_grad),
                124_716,
            )
            configs.append(config)

        first = configs[0]
        for config in configs[1:]:
            self.assertEqual(config.training, first.training)
            self.assertEqual(config.model, first.model)

    def test_restored_configs_reuse_true22_classifier(self):
        root = Path(__file__).resolve().parents[1]
        true22 = load_experiment_config(
            root / "configs/classification/baselines/true22.yaml", root
        )
        for input_id in (
            "spherical_spline", "autoencoder", "autoencoder_eeg_aware", "ddpm_standard"
        ):
            config = load_experiment_config(
                root / "configs/classification/restored" / f"{input_id}.yaml", root
            )
            self.assertEqual(config.input.category, "restored")
            self.assertEqual(config.input.n_channels, 22)
            self.assertEqual(config.input.array_key, "x_restored22")
            self.assertEqual(config.training, true22.training)
            self.assertEqual(config.model, true22.model)

    def test_restoration_outputs_are_exact_classifier_inputs(self):
        root = Path(__file__).resolve().parents[1]
        for input_id in (
            "spherical_spline", "autoencoder", "autoencoder_eeg_aware", "ddpm_standard"
        ):
            restoration = load_restoration_config(
                root / "configs/restoration" / f"{input_id}.yaml", root
            )
            classification = load_experiment_config(
                root / "configs/classification/restored" / f"{input_id}.yaml", root
            )
            self.assertEqual(
                classification.input.arrays_dir,
                restoration.output.arrays_dir,
            )
            self.assertEqual(
                classification.input.array_key,
                restoration.output.array_key,
            )
            self.assertEqual(
                classification.output_dir,
                root / "artifacts/experiments/classification/restoration_benchmarks",
            )


if __name__ == "__main__":
    unittest.main()
