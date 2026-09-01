# Classification configurations

Each YAML file describes exactly one classifier input experiment. Run conditions
separately so each input has independent checkpoints and results.

The baseline YAML files share the paper-aligned global TCFormer configuration.
Only the input representation and required channel count differ. Do not tune a
single restoration method's classifier independently.

```text
classification/
├── baselines/
│   ├── true22.yaml
│   ├── zero_padded_mi9.yaml
│   └── direct_mi9.yaml
├── channel_overlap/
│   ├── overlap_0.yaml
│   ├── overlap_2.yaml
│   ├── overlap_4.yaml
│   └── overlap_7.yaml
└── restored/
    └── README.md
```

A restored-input experiment should copy a baseline YAML, use a unique `name` and
`input.id`, set `category: restored`, point `input.arrays_dir` to its method and
variant directory under `artifacts/model_inputs/restored/`, and set the stored
array key and channel count. The classifier code does not need a method-specific
branch.

The four `channel_overlap` YAMLs reuse the exact direct-nine TCFormer and
training configuration. They differ only in experiment identity and the array
key created by `configs/preprocessing/channel_overlap.yaml`. Their results share
one study directory and one `EXPERIMENT_RECORD.md`.
