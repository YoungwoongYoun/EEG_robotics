import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from eeg_pipeline.experiments.classification.config import (
    ExperimentConfig,
    InputConfig,
    ModelConfig,
    TrainingConfig,
)
from eeg_pipeline.experiments.classification.data import GlobalSplits
from eeg_pipeline.experiments.classification.training import (
    _build_optimizer,
    _build_scheduler,
    segmentation_reconstruction,
    train_global,
)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(16, 4)

    def forward(self, x):
        return self.classifier(x.flatten(start_dim=1))


def make_dataset(size: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(size)
    return TensorDataset(
        torch.randn(size, 1, 2, 8, generator=generator),
        torch.arange(size) % 4,
        1 + torch.arange(size) % 2,
        torch.arange(size),
    )


class ClassificationTrainingTests(unittest.TestCase):
    def test_paper_aligned_optimizer_and_schedule(self):
        model = nn.Linear(2, 2)
        config = ExperimentConfig(
            name="true22",
            output_dir=Path("output"),
            subjects=(1,),
            seeds=(0,),
            device="cpu",
            input=InputConfig("true22", "True", "baseline", Path("input"), "x", 2),
            training=TrainingConfig(),
            model=ModelConfig(),
        )
        optimizer = _build_optimizer(model, config)
        scheduler = _build_scheduler(optimizer, config)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertIsNotNone(scheduler)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        for _ in range(3):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0009)

    def test_segmentation_reconstruction_doubles_batch_and_preserves_classes(self):
        x = torch.arange(4 * 1 * 2 * 14, dtype=torch.float32).reshape(4, 1, 2, 14)
        y = torch.tensor([0, 0, 1, 1])
        augmented_x, augmented_y = segmentation_reconstruction(x, y, segments=7)
        self.assertEqual(tuple(augmented_x.shape), (8, 1, 2, 14))
        self.assertEqual(torch.bincount(augmented_y).tolist(), [4, 4])

    def test_segmentation_reconstruction_rejects_incompatible_length(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            segmentation_reconstruction(
                torch.zeros(2, 1, 2, 8), torch.tensor([0, 1]), segments=7
            )

    def test_global_training_reports_each_subject_and_uses_validation_for_selection(self):
        config = ExperimentConfig(
            name="true22",
            output_dir=Path("output"),
            subjects=(1, 2),
            seeds=(13,),
            device="cpu",
            input=InputConfig("true22", "True", "baseline", Path("input"), "x", 2),
            training=TrainingConfig(
                epochs=2,
                batch_size=4,
                scheduler="none",
                warmup_epochs=0,
                sr_augmentation=False,
                early_stopping_patience=2,
                amp=False,
            ),
            model=ModelConfig(),
        )
        splits = GlobalSplits(make_dataset(8), make_dataset(4), make_dataset(4))
        with patch(
            "eeg_pipeline.experiments.classification.training.build_model",
            return_value=TinyClassifier(),
        ):
            artifacts = train_global(config, seed=13, splits=splits, device=torch.device("cpu"))
        self.assertEqual(artifacts.metrics["scope"], "pooled_multi_subject_inter_session")
        self.assertEqual(artifacts.metrics["n_test"], 4)
        self.assertEqual({row["subject"] for row in artifacts.subject_metrics}, {1, 2})
        self.assertEqual(len(artifacts.predictions["subject"]), 4)
        self.assertIn(artifacts.metrics["best_epoch"], (1, 2))


if __name__ == "__main__":
    unittest.main()
