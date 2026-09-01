import unittest

import torch

from eeg_pipeline.experiments.restoration.models import (
    ConditionalDDPMDenoiser,
    TemporalAutoencoder,
)


class RestorationModelTests(unittest.TestCase):
    def test_autoencoder_preserves_length_and_outputs_22_channels(self):
        model = TemporalAutoencoder(hidden_channels=16, bottleneck_channels=8, dilations=(1, 2))
        output = model(torch.randn(2, 9, 33))
        self.assertEqual(tuple(output.shape), (2, 22, 33))

    def test_ddpm_denoiser_preserves_length_and_outputs_22_channels(self):
        model = ConditionalDDPMDenoiser(base_channels=8, time_dimension=16)
        output = model(torch.randn(2, 66, 33), torch.tensor([0, 3]))
        self.assertEqual(tuple(output.shape), (2, 22, 33))


if __name__ == "__main__":
    unittest.main()
