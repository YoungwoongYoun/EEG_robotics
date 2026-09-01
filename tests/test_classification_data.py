import tempfile
import unittest
from pathlib import Path

import numpy as np

from eeg_pipeline.channels import MI9_INDICES
from eeg_pipeline.experiments.classification.config import InputConfig
from eeg_pipeline.experiments.classification.data import InputDataRepository


def write_arrays(arrays_dir: Path) -> None:
    for split, trials in (("train", [0, 1]), ("validation", [2]), ("test", [0])):
        subjects = np.asarray([1, 2] if len(trials) == 2 else [1])
        np.savez_compressed(
            arrays_dir / f"{split}.npz",
            x_true22=np.zeros((len(trials), 22, 16), dtype=np.float32),
            x_mi9=np.ones((len(trials), 9, 16), dtype=np.float32),
            y=np.asarray([index % 4 for index in trials]),
            subject=subjects,
            trial_index=np.asarray(trials),
        )


class ClassificationDataTests(unittest.TestCase):
    def test_selected_subjects_are_pooled_and_subject_ids_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            arrays_dir = Path(temporary)
            write_arrays(arrays_dir)
            config = InputConfig("true22", "True", "baseline", arrays_dir, "x_true22", 22)
            splits = InputDataRepository(config).pooled((1,))
            self.assertEqual(tuple(splits.train.tensors[0].shape), (1, 1, 22, 16))
            self.assertEqual(splits.train.tensors[2].tolist(), [1])
            self.assertEqual(tuple(splits.test.tensors[0].shape), (1, 1, 22, 16))

    def test_zero_padding_uses_canonical_positions(self):
        with tempfile.TemporaryDirectory() as temporary:
            arrays_dir = Path(temporary)
            write_arrays(arrays_dir)
            config = InputConfig(
                "zero_padded_mi9", "Zero", "baseline", arrays_dir,
                "x_mi9", 22, "zero_pad_mi9_to_22",
            )
            padded = InputDataRepository(config).pooled((1,)).test.tensors[0][0, 0].numpy()
            np.testing.assert_array_equal(padded[list(MI9_INDICES)], 1.0)
            missing = sorted(set(range(22)) - set(MI9_INDICES))
            np.testing.assert_array_equal(padded[missing], 0.0)


if __name__ == "__main__":
    unittest.main()
