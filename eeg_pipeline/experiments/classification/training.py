"""Train one pooled global TCFormer and evaluate held-out Session 2."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ...models import TCFormer
from .config import ExperimentConfig
from .data import GlobalSplits
from .metrics import classification_metrics


@dataclass(frozen=True)
class RunArtifacts:
    metrics: dict[str, Any]
    subject_metrics: list[dict[str, Any]]
    history: list[dict[str, float | int]]
    predictions: dict[str, np.ndarray]
    confusion: np.ndarray
    checkpoint: dict[str, Any]


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def configure_device(config: ExperimentConfig, override: str | None = None) -> torch.device:
    requested = override or config.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    device = torch.device(requested)
    torch.backends.cuda.matmul.allow_tf32 = config.training.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.training.allow_tf32
    return device


def build_model(config: ExperimentConfig) -> TCFormer:
    return TCFormer(
        n_channels=config.input.n_channels,
        n_classes=config.model.n_classes,
        **config.model.args,
    )


def _loader(
    dataset: TensorDataset,
    config: ExperimentConfig,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=config.training.num_workers > 0,
        generator=generator,
    )


def segmentation_reconstruction(
    x: torch.Tensor,
    y: torch.Tensor,
    segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Double a batch with official-style same-class S&R synthetic trials."""

    if x.ndim != 4 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("Expected x [B, 1, C, T] and matching y [B]")
    if x.shape[-1] % segments != 0:
        raise ValueError(
            f"Input length {x.shape[-1]} must be divisible by sr_segments={segments}"
        )
    synthetic_x = []
    synthetic_y = []
    for class_id in torch.unique(y, sorted=True):
        class_trials = x[y == class_id]
        n_class = class_trials.shape[0]
        chunks = torch.cat(torch.chunk(class_trials, chunks=segments, dim=-1), dim=0)
        choices = torch.randint(
            0,
            n_class,
            size=(n_class, segments),
            device=x.device,
        )
        offsets = torch.arange(segments, device=x.device) * n_class
        chosen = chunks[choices + offsets]
        reconstructed = chosen.permute(0, 2, 3, 1, 4).reshape_as(class_trials)
        synthetic_x.append(reconstructed)
        synthetic_y.append(torch.full_like(y[y == class_id], class_id))
    combined_x = torch.cat((x, *synthetic_x), dim=0)
    combined_y = torch.cat((y, *synthetic_y), dim=0)
    permutation = torch.randperm(combined_x.shape[0], device=x.device)
    return combined_x[permutation], combined_y[permutation]


def _build_optimizer(
    model: nn.Module,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    kwargs = {
        "lr": config.training.learning_rate,
        "betas": (config.training.beta_1, config.training.beta_2),
        "weight_decay": config.training.weight_decay,
    }
    if config.training.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _warmup_cosine_factor(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return float(epoch) / float(max(1, warmup_epochs))
    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return 0.5 * (1.0 + np.cos(np.pi * progress))


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.training.scheduler == "none":
        return None
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _warmup_cosine_factor(
            epoch,
            config.training.warmup_epochs,
            config.training.epochs,
        ),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    sr_augmentation: bool = False,
    sr_segments: int = 7,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for x, y, _, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if training and sr_augmentation:
                x, y = segmentation_reconstruction(x, y, sr_segments)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch_size = y.size(0)
            loss_sum += float(loss.detach()) * batch_size
            correct += int((logits.argmax(dim=1) == y).sum())
            count += batch_size
    if count == 0:
        raise ValueError("Cannot run an epoch on an empty dataset")
    return loss_sum / count, correct / count


@torch.inference_mode()
def _predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, dict[str, np.ndarray], float]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "true_label": [],
        "predicted_label": [],
        "probabilities": [],
        "subject": [],
        "trial_index": [],
    }
    loss_sum = 0.0
    count = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for x, y, subject, trial_index in loader:
        x = x.to(device, non_blocking=True)
        y_device = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y_device)
        probability = logits.softmax(dim=1)
        collected["true_label"].append(y.numpy())
        collected["predicted_label"].append(probability.argmax(dim=1).cpu().numpy())
        collected["probabilities"].append(probability.cpu().numpy())
        collected["subject"].append(subject.numpy())
        collected["trial_index"].append(trial_index.numpy())
        loss_sum += float(loss) * y.size(0)
        count += y.size(0)
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (
        loss_sum / count,
        {key: np.concatenate(values) for key, values in collected.items()},
        time.perf_counter() - start,
    )


def _subject_metrics(
    config: ExperimentConfig,
    seed: int,
    predictions: dict[str, np.ndarray],
    inference_ms_per_trial: float,
) -> list[dict[str, Any]]:
    rows = []
    for subject in sorted(int(value) for value in np.unique(predictions["subject"])):
        mask = predictions["subject"] == subject
        metrics, _ = classification_metrics(
            predictions["true_label"][mask],
            predictions["predicted_label"][mask],
            config.model.n_classes,
        )
        rows.append({
            "input_id": config.input.id,
            "input_label": config.input.label,
            "subject": subject,
            "subject_id": f"A{subject:02d}",
            "seed": seed,
            "n_test": int(mask.sum()),
            "inference_ms_per_trial": inference_ms_per_trial,
            **metrics,
        })
    return rows


def train_global(
    config: ExperimentConfig,
    seed: int,
    splits: GlobalSplits,
    device: torch.device,
) -> RunArtifacts:
    """Pool all selected Session-1 subjects into one model; test on Session 2."""

    seed_everything(seed, config.training.deterministic)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    amp_enabled = config.training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    train_loader = _loader(splits.train, config, shuffle=True, seed=seed)
    validation_loader = _loader(splits.validation, config, shuffle=False, seed=seed)
    test_loader = _loader(splits.test, config, shuffle=False, seed=seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    training_start = time.perf_counter()

    for epoch in range(1, config.training.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler, amp_enabled=amp_enabled,
            sr_augmentation=config.training.sr_augmentation,
            sr_segments=config.training.sr_segments,
        )
        validation_loss, validation_accuracy = _run_epoch(
            model, validation_loader, criterion, device,
            optimizer=None, scaler=scaler, amp_enabled=amp_enabled,
        )
        history.append({
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })
        print(
            f"[{config.input.id} global seed={seed}] {epoch:03d}/{config.training.epochs} "
            f"train={train_accuracy:.4f} val={validation_accuracy:.4f} "
            f"val_loss={validation_loss:.6f} lr={learning_rate:.7f}",
            flush=True,
        )
        improved = validation_loss < (
            best_validation_loss - config.training.early_stopping_min_delta
        )
        if improved:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.training.early_stopping_patience:
                break
        if scheduler is not None:
            scheduler.step()

    training_seconds = time.perf_counter() - training_start
    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint")
    model.load_state_dict(best_state)
    test_loss, predictions, test_seconds = _predict(
        model, test_loader, criterion, device, amp_enabled
    )
    metrics, confusion = classification_metrics(
        predictions["true_label"], predictions["predicted_label"], config.model.n_classes
    )
    inference_ms = 1000.0 * test_seconds / len(splits.test)
    metrics.update({
        "input_id": config.input.id,
        "input_label": config.input.label,
        "category": config.input.category,
        "scope": "pooled_multi_subject_inter_session",
        "seed": seed,
        "subjects": list(config.subjects),
        "n_channels": config.input.n_channels,
        "n_train": len(splits.train),
        "n_train_effective_per_epoch": len(splits.train) * (
            2 if config.training.sr_augmentation else 1
        ),
        "optimizer": config.training.optimizer,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "scheduler": config.training.scheduler,
        "warmup_epochs": config.training.warmup_epochs,
        "sr_augmentation": config.training.sr_augmentation,
        "sr_segments": config.training.sr_segments,
        "n_validation": len(splits.validation),
        "n_test": len(splits.test),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_loss": best_validation_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "test_loss": test_loss,
        "training_seconds": training_seconds,
        "test_seconds": test_seconds,
        "inference_ms_per_trial": inference_ms,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda" else 0.0
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
    })
    checkpoint = {
        "model_state_dict": best_state,
        "input": config.as_serializable_dict()["input"],
        "training": config.as_serializable_dict()["training"],
        "scope": "pooled_multi_subject_inter_session",
        "subjects": list(config.subjects),
        "seed": seed,
        "n_classes": config.model.n_classes,
        "model_args": config.model.args,
        "best_epoch": best_epoch,
        "validation_loss": best_validation_loss,
    }
    return RunArtifacts(
        metrics=metrics,
        subject_metrics=_subject_metrics(config, seed, predictions, inference_ms),
        history=history,
        predictions=predictions,
        confusion=confusion,
        checkpoint=checkpoint,
    )
