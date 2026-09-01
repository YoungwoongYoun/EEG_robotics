import unittest

import numpy as np

from eeg_pipeline.experiments.classification.metrics import (
    classification_metrics,
    confusion_matrix,
)


class ClassificationMetricTests(unittest.TestCase):
    def test_perfect_predictions(self):
        targets = np.asarray([0, 1, 2, 3])
        metrics, matrix = classification_metrics(targets, targets, n_classes=4)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["cohen_kappa"], 1.0)
        np.testing.assert_array_equal(matrix, np.eye(4, dtype=int))

    def test_known_confusion_matrix(self):
        targets = np.asarray([0, 0, 1, 1])
        predicted = np.asarray([0, 1, 1, 1])
        matrix = confusion_matrix(targets, predicted, n_classes=2)
        np.testing.assert_array_equal(matrix, [[1, 1], [0, 2]])
        metrics, _ = classification_metrics(targets, predicted, n_classes=2)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["cohen_kappa"], 0.5)


if __name__ == "__main__":
    unittest.main()
