"""Run one resumable pooled-global input experiment at a time."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import InputDataRepository
from .reporting import (
    is_complete,
    prepare_experiment_directory,
    run_directory,
    save_run,
    write_experiment_report,
    write_study_record,
)
from .training import (
    build_model,
    configure_device,
    segmentation_reconstruction,
    train_global,
)


def _selected(values: tuple[int, ...], override: Iterable[int] | None) -> tuple[int, ...]:
    return tuple(values if override is None else override)


def _validate_selection(
    config: ExperimentConfig,
    subjects: Iterable[int] | None,
    seeds: Iterable[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    selected_subjects = tuple(int(value) for value in _selected(config.subjects, subjects))
    selected_seeds = tuple(int(value) for value in _selected(config.seeds, seeds))
    if not set(selected_subjects).issubset(config.subjects):
        raise ValueError("Subject override must be a subset of the config")
    if not set(selected_seeds).issubset(config.seeds):
        raise ValueError("Seed override must be a subset of the config")
    if not selected_subjects or not selected_seeds:
        raise ValueError("At least one subject and seed are required")
    return selected_subjects, selected_seeds


def dry_run(
    config: ExperimentConfig,
    *,
    subjects: Iterable[int] | None = None,
    device_override: str | None = None,
) -> None:
    """Validate all split arrays and one forward pass without training."""

    selected_subjects, _ = _validate_selection(config, subjects, None)
    repository = InputDataRepository(config.input)
    splits = repository.pooled(selected_subjects)
    device = configure_device(config, device_override)
    model = build_model(config).to(device).eval()
    x, _, _, _ = splits.train[0]
    augmentation_text = "disabled"
    if config.training.sr_augmentation:
        sample_count = min(config.training.batch_size, len(splits.train))
        batch_x = torch.stack([splits.train[index][0] for index in range(sample_count)])
        batch_y = torch.stack([splits.train[index][1] for index in range(sample_count)])
        augmented_x, augmented_y = segmentation_reconstruction(
            batch_x,
            batch_y,
            config.training.sr_segments,
        )
        augmentation_text = (
            f"S&R {sample_count}->{len(augmented_y)} "
            f"(segments={config.training.sr_segments}, shape={tuple(augmented_x.shape)})"
        )
    with torch.inference_mode():
        output = model(x.unsqueeze(0).to(device))
    parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
    print(
        f"DRY RUN {config.input.id}: pooled_subjects={len(selected_subjects)}, "
        f"train={len(splits.train)}, validation={len(splits.validation)}, "
        f"test={len(splits.test)}, input={tuple(x.shape)}, output={tuple(output.shape)}, "
        f"parameters={parameters:,}, augmentation={augmentation_text}, device={device}"
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    subjects: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    device_override: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Train one independent input condition, one global model per seed."""

    selected_subjects, selected_seeds = _validate_selection(config, subjects, seeds)
    if selected_subjects != config.subjects:
        raise ValueError(
            "A subject override changes the pooled experiment identity; create a separate config instead"
        )
    experiment_dir = prepare_experiment_directory(config)
    repository = InputDataRepository(config.input)
    splits = repository.pooled(selected_subjects)
    device = configure_device(config, device_override)
    for seed in selected_seeds:
        output = run_directory(experiment_dir, seed)
        if is_complete(output) and not overwrite:
            print(f"SKIP completed: {config.input.id} global seed={seed}")
            continue
        if output.exists() and overwrite:
            shutil.rmtree(output)
        print(
            f"START: {config.input.id} global seed={seed} on {device} "
            f"({len(splits.train)} train / {len(splits.validation)} val / {len(splits.test)} test)"
        )
        artifacts = train_global(config, seed, splits, device)
        save_run(output, artifacts)
        write_experiment_report(experiment_dir)
        write_study_record(config.output_dir)
        print(
            f"DONE: {config.input.id} global seed={seed} "
            f"accuracy={artifacts.metrics['accuracy']:.4f}"
        )
    write_experiment_report(experiment_dir)
    write_study_record(config.output_dir)
    return experiment_dir
