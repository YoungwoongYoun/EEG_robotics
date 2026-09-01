import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from eeg_pipeline.channels import MI9_INDICES
from eeg_pipeline.experiments.restoration.config import (
    InferenceConfig,
    OutputConfig,
    RestorationConfig,
    SourceConfig,
    TrainingConfig,
)
from eeg_pipeline.experiments.restoration.data import RestorationSplit
from eeg_pipeline.experiments.restoration.training import (
    _finalize_restoration_batch,
    _loss,
    configure_device,
    infer_learned,
    log_bandpower_loss,
    masked_noise_loss,
    spatial_correlation_loss,
    train_model,
)


def tiny_split(trials: int = 4) -> RestorationSplit:
    rng = np.random.default_rng(11)
    true22 = rng.normal(size=(trials, 22, 16)).astype(np.float32)
    return RestorationSplit(
        x_mi9=true22[:, MI9_INDICES].copy(),
        x_true22=true22,
        y=np.arange(trials) % 4,
        subject=np.ones(trials, dtype=np.int64),
        trial_index=np.arange(trials),
    )


class RestorationTrainingTests(unittest.TestCase):
    def test_eeg_aware_loss_accepts_autocast_like_half_output(self):
        class HalfOutput(torch.nn.Module):
            def forward(self, observed):
                return torch.zeros(
                    observed.shape[0], 22, observed.shape[-1], dtype=torch.float16
                )

        true22 = torch.randn(2, 22, 251)
        observed = true22[:, MI9_INDICES].clone()
        losses = _loss(
            HalfOutput(),
            "autoencoder",
            observed,
            true22,
            None,
            0,
            {
                "time_weight": 1.0,
                "bandpower_weight": 0.1,
                "spatial_weight": 1.0,
                "sampling_rate": 250.0,
                "bands": [[8.0, 13.0], [13.0, 30.0]],
                "epsilon": 1e-6,
            },
        )
        self.assertTrue(torch.isfinite(losses.total))

    def test_eeg_losses_are_zero_for_identity_and_backpropagate(self):
        true22 = torch.randn(2, 22, 251)
        restored = true22.clone().requires_grad_(True)
        band = log_bandpower_loss(
            restored, true22, 250.0, [[8.0, 13.0], [13.0, 30.0]], 1e-6
        )
        spatial = spatial_correlation_loss(restored, true22, 1e-6)
        self.assertAlmostEqual(float(band), 0.0, places=7)
        self.assertAlmostEqual(float(spatial), 0.0, places=7)
        perturbed = (true22 + 0.2 * torch.randn_like(true22)).requires_grad_(True)
        loss = log_bandpower_loss(
            perturbed, true22, 250.0, [[8.0, 13.0], [13.0, 30.0]], 1e-6
        ) + spatial_correlation_loss(perturbed, true22, 1e-6)
        loss.backward()
        self.assertGreater(float(loss), 0.0)
        self.assertGreater(float(perturbed.grad.abs().sum()), 0.0)

    def test_half_precision_output_accepts_float_observed_channels(self):
        output = torch.zeros(2, 22, 8, dtype=torch.float16)
        observed = torch.randn(2, 9, 8, dtype=torch.float32)
        finalized = _finalize_restoration_batch(output, observed)
        self.assertEqual(finalized.dtype, torch.float32)
        torch.testing.assert_close(
            finalized[:, MI9_INDICES, :],
            observed,
            rtol=0.0,
            atol=0.0,
        )

    def test_ddpm_loss_ignores_observed_channel_predictions(self):
        noise = torch.randn(2, 22, 8)
        first = torch.zeros_like(noise)
        second = first.clone()
        second[:, MI9_INDICES, :] = 1000.0
        self.assertEqual(
            float(masked_noise_loss(first, noise)),
            float(masked_noise_loss(second, noise)),
        )

    def test_autoencoder_training_checkpoint_and_inference_are_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = RestorationConfig(
                name="tiny_autoencoder",
                method="autoencoder",
                seed=3,
                device="cpu",
                source=SourceConfig(root, root),
                output=OutputConfig(root / "arrays", root / "experiment"),
                training=TrainingConfig(
                    epochs=1,
                    batch_size=2,
                    learning_rate=1e-3,
                    patience=1,
                    amp=False,
                    deterministic=True,
                ),
                inference=InferenceConfig(batch_size=2),
                model={
                    "hidden_channels": 16,
                    "bottleneck_channels": 8,
                    "dilations": [1],
                    "dropout": 0.0,
                },
            )
            config.validate()
            device = configure_device(config)
            source = tiny_split()
            checkpoint = train_model(config, source, source, device)
            self.assertTrue(checkpoint.is_file())
            completion = json.loads(
                (checkpoint.parent / "training_complete.json").read_text(encoding="utf-8")
            )
            self.assertEqual(completion["stop_reason"], "max_epochs_reached")
            self.assertFalse(completion["validation_plateau_reached"])
            self.assertTrue(completion["best_epoch_near_end"])
            restored, elapsed = infer_learned(config, source, device)
            self.assertEqual(restored.shape, source.x_true22.shape)
            np.testing.assert_array_equal(restored[:, MI9_INDICES], source.x_mi9)
            self.assertTrue(np.isfinite(restored).all())
            self.assertGreaterEqual(elapsed, 0.0)
            skipped = train_model(config, source, source, device)
            self.assertEqual(skipped, checkpoint)
            changed = replace(
                config,
                training=replace(config.training, learning_rate=5e-4),
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                train_model(changed, source, source, device)


if __name__ == "__main__":
    unittest.main()
