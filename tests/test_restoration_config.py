import tempfile
import unittest
from pathlib import Path

import yaml

from eeg_pipeline.experiments.restoration.config import load_restoration_config


class RestorationConfigTests(unittest.TestCase):
    def test_project_configs_have_separate_outputs_and_expected_methods(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "spherical_spline.yaml": ("spherical_spline", "spherical_spline"),
            "autoencoder.yaml": ("autoencoder", "autoencoder"),
            "autoencoder_eeg_aware.yaml": ("autoencoder_eeg_aware", "autoencoder"),
            "ddpm_standard.yaml": ("ddpm_standard", "ddpm"),
        }
        outputs = set()
        for filename, (name, method) in expected.items():
            config = load_restoration_config(root / "configs/restoration" / filename, root)
            self.assertEqual(config.name, name)
            self.assertEqual(config.method, method)
            self.assertEqual(config.output.array_key, "x_restored22")
            self.assertNotIn(config.output.arrays_dir, outputs)
            outputs.add(config.output.arrays_dir)

    def test_ancestral_ddpm_rejects_skipped_steps(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "configs/restoration/ddpm_standard.yaml").open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        values["inference"]["sampler"] = "ddpm"
        values["inference"]["sampling_steps"] = 50
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.yaml"
            path.write_text(yaml.safe_dump(values), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every diffusion timestep"):
                load_restoration_config(path, root)

    def test_ddpm_rejects_terminal_state_that_is_not_nearly_gaussian(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "configs/restoration/ddpm_standard.yaml").open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        values["diffusion"]["timesteps"] = 200
        values["inference"]["sampling_steps"] = 50
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad_terminal.yaml"
            path.write_text(yaml.safe_dump(values), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "retains too much signal"):
                load_restoration_config(path, root)

    def test_eeg_aware_autoencoder_has_valid_physiological_loss(self):
        root = Path(__file__).resolve().parents[1]
        config = load_restoration_config(
            root / "configs/restoration/autoencoder_eeg_aware.yaml", root
        )
        self.assertEqual(config.loss["objective"], "eeg_spectral_spatial")
        self.assertEqual(config.loss["bands"], [[8.0, 13.0], [13.0, 30.0]])
        self.assertIn("loss", config.training_signature())


if __name__ == "__main__":
    unittest.main()
