import unittest

import numpy as np

from eeg_pipeline.channels import MI9_INDICES
from eeg_pipeline.preprocessing import (
    StandardizationStats,
    apply_available_average_reference,
    apply_mi9_average_reference,
    apply_standardizer,
    artifact_trial_mask,
    fit_channelwise_standardizer,
    select_average_referenced_channels,
    stratified_train_val_indices,
)


class PreprocessingTests(unittest.TestCase):
    def test_mi9_reference_has_zero_available_channel_mean(self):
        rng = np.random.default_rng(1)
        epochs = rng.normal(size=(3, 22, 8))
        referenced = apply_mi9_average_reference(epochs)
        np.testing.assert_allclose(
            referenced[:, MI9_INDICES, :].mean(axis=1),
            0.0,
            atol=1e-12,
        )

    def test_arbitrary_montage_is_referenced_to_its_own_average(self):
        rng = np.random.default_rng(9)
        epochs = rng.normal(size=(3, 22, 8))
        indices = (0, 1, 5, 6, 12, 13, 17, 18, 20)
        selected = select_average_referenced_channels(epochs, indices)
        self.assertEqual(selected.shape, (3, 9, 8))
        np.testing.assert_allclose(selected.mean(axis=1), 0.0, atol=1e-12)

        with self.assertRaises(ValueError):
            apply_available_average_reference(epochs, (0, 0, 1))

    def test_stratified_split_is_disjoint_and_reproducible(self):
        labels = np.repeat(np.arange(4), 10)
        first = stratified_train_val_indices(labels, 0.2, seed=42)
        second = stratified_train_val_indices(labels, 0.2, seed=42)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertEqual(set(np.unique(labels[first[0]])), {0, 1, 2, 3})
        self.assertEqual(set(np.unique(labels[first[1]])), {0, 1, 2, 3})

    def test_standardizer_uses_supplied_training_statistics(self):
        train = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
        stats = fit_channelwise_standardizer(train)
        transformed = apply_standardizer(train, stats)
        np.testing.assert_allclose(transformed.mean(axis=(0, 2)), 0.0, atol=1e-6)
        np.testing.assert_allclose(transformed.std(axis=(0, 2)), 1.0, atol=1e-6)

        held_out = np.ones((1, 3, 4), dtype=float) * 100
        expected = (held_out - stats.mean[None, :, None]) / stats.std[None, :, None]
        np.testing.assert_allclose(apply_standardizer(held_out, stats), expected)

    def test_constant_channel_uses_safe_scale(self):
        stats = fit_channelwise_standardizer(np.ones((2, 3, 4)))
        np.testing.assert_array_equal(stats.std, np.ones(3))

    def test_artifact_event_is_associated_with_full_trial_window(self):
        cues = np.asarray([1000, 3000, 5000])
        artifacts = np.asarray([500, 5900])
        mask = artifact_trial_mask(cues, artifacts, sampling_frequency=250.0)
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_standardizer_rejects_channel_mismatch(self):
        stats = StandardizationStats(mean=np.zeros(2), std=np.ones(2))
        with self.assertRaises(ValueError):
            apply_standardizer(np.zeros((1, 3, 4)), stats)


if __name__ == "__main__":
    unittest.main()
