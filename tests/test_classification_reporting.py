import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from eeg_pipeline.experiments.classification.config import (
    ExperimentConfig,
    InputConfig,
    ModelConfig,
    TrainingConfig,
)
from eeg_pipeline.experiments.classification.reporting import (
    prepare_experiment_directory,
    run_directory,
    save_run,
    write_experiment_report,
    write_study_record,
)
from eeg_pipeline.experiments.classification.training import RunArtifacts


def config(root: Path, input_id: str, label: str) -> ExperimentConfig:
    channels = 9 if input_id == "direct_mi9" else 22
    return ExperimentConfig(
        name=input_id,
        output_dir=root / "global_model",
        subjects=(1, 2),
        seeds=(13,),
        device="cpu",
        input=InputConfig(input_id, label, "baseline", root / "input", "x", channels),
        training=TrainingConfig(epochs=1),
        model=ModelConfig(),
    )


def artifacts(input_id: str, label: str, accuracy: float) -> RunArtifacts:
    subject_rows = [
        {
            "input_id": input_id,
            "input_label": label,
            "subject": subject,
            "subject_id": f"A{subject:02d}",
            "seed": 13,
            "n_test": 2,
            "inference_ms_per_trial": 1.0,
            "accuracy": accuracy,
            "macro_f1": accuracy,
            "cohen_kappa": accuracy - 0.25,
            "class_0_recall": 1.0,
            "class_1_recall": 1.0,
            "class_2_recall": 1.0,
            "class_3_recall": 1.0,
        }
        for subject in (1, 2)
    ]
    return RunArtifacts(
        metrics={
            "input_id": input_id,
            "input_label": label,
            "seed": 13,
            "n_channels": 9 if input_id == "direct_mi9" else 22,
            "trainable_parameters": 100,
            "optimizer": "adam",
            "learning_rate": 0.0009,
            "scheduler": "warmup_cosine",
            "sr_augmentation": True,
            "sr_segments": 7,
            "accuracy": accuracy,
            "macro_f1": accuracy,
            "cohen_kappa": accuracy - 0.25,
            "inference_ms_per_trial": 1.0,
        },
        subject_metrics=subject_rows,
        history=[{"epoch": 1, "train_loss": 1.0}],
        predictions={
            "subject": np.asarray([1, 2]),
            "trial_index": np.asarray([0, 0]),
            "true_label": np.asarray([0, 1]),
            "predicted_label": np.asarray([0, 1]),
            "probabilities": np.asarray([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]]),
        },
        confusion=np.eye(4, dtype=int),
        checkpoint={"model_state_dict": {"weight": torch.ones(1)}},
    )


class ClassificationReportingTests(unittest.TestCase):
    def test_independent_checkpoints_and_one_study_record_are_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for input_id, label, accuracy in (
                ("true22", "True 22-channel", 0.75),
                ("direct_mi9", "Direct MI-9", 0.60),
            ):
                cfg = config(root, input_id, label)
                experiment_dir = prepare_experiment_directory(cfg)
                save_run(run_directory(experiment_dir, 13), artifacts(input_id, label, accuracy))
                write_experiment_report(experiment_dir)
            study_dir = root / "global_model"
            write_study_record(study_dir)
            self.assertTrue((study_dir / "true22" / "checkpoints" / "seed_13" / "best_model.pt").is_file())
            self.assertTrue((study_dir / "true22" / "report.md").is_file())
            self.assertTrue((study_dir / "true22" / "results" / "subject_summary.csv").is_file())
            record = (study_dir / "EXPERIMENT_RECORD.md").read_text()
            self.assertIn("True 22-channel", record)
            self.assertIn("Direct MI-9", record)
            self.assertIn("15.00", record)
            self.assertTrue((study_dir / "comparison" / "paired_statistics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
