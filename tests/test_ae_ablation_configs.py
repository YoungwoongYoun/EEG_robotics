import unittest
from pathlib import Path

from eeg_pipeline.experiments.classification.config import load_experiment_config
from eeg_pipeline.experiments.classification.training import build_model
from eeg_pipeline.experiments.restoration.config import load_restoration_config
from eeg_pipeline.experiments.restoration.models import build_restoration_model


class AutoencoderAblationConfigTests(unittest.TestCase):
    def test_ablation_configs_isolate_one_auxiliary_loss(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "autoencoder_bandpower": (0.1, 0.0),
            "autoencoder_spatial": (0.0, 1.0),
        }
        restored_outputs = set()
        experiment_outputs = set()
        for name, (bandpower, spatial) in expected.items():
            restoration = load_restoration_config(
                root / f"configs/restoration/{name}.yaml", root
            )
            classification = load_experiment_config(
                root / f"configs/classification/restored/{name}.yaml", root
            )
            self.assertEqual(restoration.loss["time_weight"], 1.0)
            self.assertEqual(restoration.loss["bandpower_weight"], bandpower)
            self.assertEqual(restoration.loss["spatial_weight"], spatial)
            self.assertEqual(classification.input.arrays_dir, restoration.output.arrays_dir)
            self.assertEqual(classification.input.array_key, restoration.output.array_key)
            self.assertNotIn(restoration.output.arrays_dir, restored_outputs)
            self.assertNotIn(restoration.output.experiment_dir, experiment_outputs)
            restored_outputs.add(restoration.output.arrays_dir)
            experiment_outputs.add(restoration.output.experiment_dir)

    def test_ablation_capacity_matches_combined_models(self):
        root = Path(__file__).resolve().parents[1]
        restoration_parameters = []
        classifier_parameters = []
        for name in (
            "autoencoder_bandpower",
            "autoencoder_spatial",
            "autoencoder_eeg_aware",
        ):
            restoration = load_restoration_config(
                root / f"configs/restoration/{name}.yaml", root
            )
            classification = load_experiment_config(
                root / f"configs/classification/restored/{name}.yaml", root
            )
            restoration_parameters.append(sum(
                parameter.numel()
                for parameter in build_restoration_model(
                    restoration.method, restoration.model
                ).parameters()
            ))
            classifier_parameters.append(sum(
                parameter.numel()
                for parameter in build_model(classification).parameters()
            ))
        self.assertEqual(restoration_parameters, [209_398] * 3)
        self.assertEqual(classifier_parameters, [127_212] * 3)


if __name__ == "__main__":
    unittest.main()
