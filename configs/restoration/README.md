# Restoration experiment configurations

Every YAML defines one canonical MI-9 to restored-22 experiment. Generated
classifier inputs are written under
`artifacts/model_inputs/restored/<method>/canonical_mi9/arrays/`; checkpoints,
metrics, and reports are written separately under
`artifacts/experiments/restoration/<method>/canonical_mi9/`.

Run `--dry-run` first. Learned methods support separate resumable training and
inference stages. Existing completed training is skipped, partial training
resumes from `last_model.pt`, and validated split outputs are skipped unless
`--overwrite` is explicitly supplied.

The EEG-aware objective ablation reuses the same autoencoder and training
settings. `autoencoder_bandpower.yaml` enables only the mu/beta log-power
auxiliary term, while `autoencoder_spatial.yaml` enables only the
spatial-correlation term. Their arrays and checkpoints use independent
`bandpower_canonical_mi9` and `spatial_canonical_mi9` directories.

`wgan_gp.yaml` is the sole additional external learned baseline. It reuses the
time-only AE generator and adds a conditional temporal critic with WGAN gradient
penalty. It intentionally excludes the proposed bandpower and spatial losses.

```bash
python scripts/run_restoration.py \
  --config configs/restoration/autoencoder.yaml \
  --device cpu --dry-run
```

The common output key is `x_restored22`. Observed canonical MI-9 channels are
hard-copied exactly. Primary reconstruction metrics are computed on the missing
13 channels.
