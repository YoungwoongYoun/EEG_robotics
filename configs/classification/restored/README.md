# Restored-input configurations

Place one YAML per restoration method and variant here. The initial P1 study is:

```text
restored/
├── spherical_spline.yaml
├── autoencoder.yaml
├── autoencoder_bandpower.yaml
├── autoencoder_spatial.yaml
├── autoencoder_eeg_aware.yaml
├── wgan_gp.yaml
└── ddpm_standard.yaml
```

Each file runs independently and writes to its own experiment directory.
All variants reuse the exact True-22 TCFormer training configuration. Do not tune
the classifier separately for one restoration method.

`autoencoder_eeg_aware.yaml` is the downstream matched-training configuration
for the spectral-spatial AE objective. It remains separate from `autoencoder.yaml`,
so the baseline-vs-EEG-aware ablation cannot overwrite arrays or classifier results.
The bandpower-only and spatial-only YAML files point to their own restored arrays
but reuse the exact True-22 TCFormer training and model configuration.
`wgan_gp.yaml` applies that same classifier protocol to the conditional GAN output.
