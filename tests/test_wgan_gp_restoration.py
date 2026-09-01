import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from eeg_pipeline.channels import MI9_INDICES
from eeg_pipeline.experiments.classification.config import load_experiment_config
from eeg_pipeline.experiments.restoration.config import (
    OutputConfig,
    TrainingConfig,
    load_restoration_config,
)
from eeg_pipeline.experiments.restoration.data import RestorationSplit
from eeg_pipeline.experiments.restoration.models import build_restoration_model
from eeg_pipeline.experiments.restoration.training import (
    configure_device,
    infer_learned,
    train_model,
    wgan_gradient_penalty,
)


def tiny_split(trials: int = 4) -> RestorationSplit:
    rng = np.random.default_rng(29)
    true22 = rng.normal(size=(trials, 22, 33)).astype(np.float32)
    return RestorationSplit(
        x_mi9=true22[:, MI9_INDICES].copy(),
        x_true22=true22,
        y=np.arange(trials) % 4,
        subject=np.ones(trials, dtype=np.int64),
        trial_index=np.arange(trials),
    )


class WganGpRestorationTests(unittest.TestCase):
    def test_project_config_matches_classifier_and_generator_capacity(self):
        root = Path(__file__).resolve().parents[1]
        restoration = load_restoration_config(
            root / "configs/restoration/wgan_gp.yaml", root
        )
        classification = load_experiment_config(
            root / "configs/classification/restored/wgan_gp.yaml", root
        )
        model = build_restoration_model(restoration.method, restoration.model)
        self.assertEqual(restoration.method, "wgan_gp")
        self.assertEqual(restoration.output.arrays_dir, classification.input.arrays_dir)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.generator.parameters()),
            209_398,
        )
        self.assertGreater(
            sum(parameter.numel() for parameter in model.critic.parameters()), 0
        )

    def test_gradient_penalty_is_finite_and_backpropagates(self):
        model = build_restoration_model(
            "wgan_gp",
            {
                "generator": {
                    "hidden_channels": 16,
                    "bottleneck_channels": 8,
                    "dilations": [1],
                    "dropout": 0.0,
                },
                "critic": {"base_channels": 8, "channel_multipliers": [1, 2]},
            },
        )
        condition = torch.randn(2, 22, 33)
        real = torch.randn(2, 22, 33)
        fake = torch.randn(2, 22, 33)
        penalty = wgan_gradient_penalty(model.critic, condition, real, fake)
        penalty.backward()
        self.assertTrue(torch.isfinite(penalty))
        self.assertGreater(sum(
            float(parameter.grad.abs().sum())
            for parameter in model.critic.parameters()
            if parameter.grad is not None
        ), 0.0)

    def test_tiny_training_checkpoint_resume_and_inference(self):
        root = Path(__file__).resolve().parents[1]
        base = load_restoration_config(root / "configs/restoration/wgan_gp.yaml", root)
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            config = replace(
                base,
                device="cpu",
                output=OutputConfig(
                    temporary_root / "arrays", temporary_root / "experiment"
                ),
                training=TrainingConfig(
                    epochs=1,
                    batch_size=2,
                    learning_rate=1e-4,
                    optimizer="adam",
                    beta_1=0.5,
                    beta_2=0.9,
                    patience=1,
                    amp=False,
                    deterministic=True,
                ),
                model={
                    "generator": {
                        "hidden_channels": 16,
                        "bottleneck_channels": 8,
                        "dilations": [1],
                        "dropout": 0.0,
                    },
                    "critic": {"base_channels": 8, "channel_multipliers": [1, 2]},
                },
                gan={
                    "gradient_penalty_weight": 10.0,
                    "adversarial_weight": 0.1,
                    "reconstruction_weight": 1.0,
                    "critic_steps": 1,
                },
            )
            config.validate()
            source = tiny_split()
            checkpoint = train_model(
                config, source, source, configure_device(config)
            )
            self.assertTrue(checkpoint.is_file())
            completion = json.loads(
                (checkpoint.parent / "training_complete.json").read_text()
            )
            self.assertEqual(completion["selection_metric"], "validation_missing13_mse")
            self.assertGreater(completion["critic_parameters"], 0)
            restored, _ = infer_learned(
                config, source, configure_device(config)
            )
            self.assertEqual(restored.shape, source.x_true22.shape)
            np.testing.assert_array_equal(restored[:, MI9_INDICES], source.x_mi9)
            self.assertTrue(np.isfinite(restored).all())
            self.assertEqual(
                train_model(config, source, source, configure_device(config)), checkpoint
            )


if __name__ == "__main__":
    unittest.main()
