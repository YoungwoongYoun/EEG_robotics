# EEG Robotics

Code and publication assets for a leakage-controlled comparison of EEG channel
restoration methods for low-channel motor-imagery decoding, with a ROS2/Gazebo
assistive-wheelchair integration example.

## Study scope

The benchmark uses BCI Competition IV Dataset 2a (subjects A01-A09). Session 1
is used for training and validation, and Session 2 is held out for final testing.
The observed montage contains nine motor-area channels:

`FC1, FCz, FC2, C3, Cz, C4, CP1, CPz, CP2`

The repository compares:

- native MI-9 and zero-padded MI-9 baselines;
- measured 22-channel EEG as a protocol-specific ceiling;
- spherical-spline interpolation;
- time-domain and spectral-spatial autoencoders;
- conditional WGAN-GP;
- standard conditional DDPM with DDIM-100 inference.

Each representation is evaluated with a separately trained pooled-global
TCFormer using five classifier seeds. Signal analyses include waveform,
class-conditional mu/beta power, covariance AIRM, and CSP preservation.

## Repository layout

```text
eeg_pipeline/       preprocessing, restoration, classification, statistics
configs/            fixed YAML experiment configurations
scripts/            command-line experiment entry points
tests/              unit and artifact-contract tests
notebooks/          preprocessing execution notebook
data/README.md      expected Dataset 2a file layout
manuscript_assets/  publication tables, statistics, and figures
ros2_src/           TCFormer intent node, safety filter, robot and Gazebo packages
```

Raw EEG data, trained experiment checkpoints, generated arrays, and experiment
logs are intentionally not included in the repository.

## Environment

Create a Python environment and install the project requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-gpu-cu124.txt
pip install -r requirements-preprocessing.txt
```

The recorded GPU experiments used PyTorch 2.6.0 with CUDA 12.4. Select a
different official PyTorch build when required by the host GPU and driver.

## Dataset preparation

Download BCI Competition IV Dataset 2a and place the GDF and label files as
described in `data/README.md`. Then run:

```bash
python scripts/preprocess_bcic2a.py \
  --config configs/preprocessing/canonical_mi9.yaml
```

The command creates train, validation, and test arrays under `artifacts/`.
Normalization is fitted independently for each subject using Session-1 training
data only.

## Experiment commands

Validate a configuration without training:

```bash
python scripts/run_classification.py \
  --config configs/classification/baselines/direct_mi9.yaml \
  --device cpu --dry-run

python scripts/run_restoration.py \
  --config configs/restoration/autoencoder_eeg_aware.yaml \
  --device cpu --dry-run
```

Run the primary stages:

```bash
python scripts/run_classification.py \
  --config configs/classification/baselines/true22.yaml

python scripts/run_restoration.py \
  --config configs/restoration/autoencoder_eeg_aware.yaml --stage train

python scripts/run_restoration.py \
  --config configs/restoration/autoencoder_eeg_aware.yaml --stage infer

python scripts/run_classification.py \
  --config configs/classification/restored/autoencoder_eeg_aware.yaml

python scripts/run_signal_analysis.py
python scripts/run_latency_benchmark.py --device cuda:0
```

Additional entry points cover frozen-oracle evaluation, the AE objective
ablation, WGAN-GP, result freezing, and manuscript-asset generation.

## Tests

```bash
pytest -q
```

The tests cover preprocessing leakage controls, input contracts, restoration
losses, hard-copy preservation of observed channels, classifier configuration,
statistics, latency sampling, and result-manifest validation.

## Publication assets

`manuscript_assets/` contains the final CSV tables, statistical outputs, PNG/PDF
figures, supplementary figures, and a SHA-256 source manifest. These are compact
publication outputs; the underlying large checkpoints and prediction artifacts
are not included.

## ROS2 integration

`ros2_src/` contains the ROS2 packages for:

- loading prepared 22-channel EEG trials;
- TCFormer intent inference;
- mapping four MI classes to velocity commands;
- LiDAR-based pass, deceleration, and stop rules;
- ROS-Gazebo bridge, wheelchair description, and simulation bring-up.

The included ROS2 example begins with prepared EEG tensors. It does not include
online EEG acquisition, online restoration inside a ROS2 node, or measured
amplifier-to-actuator latency.
