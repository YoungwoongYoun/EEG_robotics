"""Resumable learned-restoration training and deterministic inference."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ...channels import MI9_INDICES
from .config import RestorationConfig
from .data import MISSING_INDICES, RestorationSplit, enforce_observed_channels
from .models import build_restoration_model


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


@dataclass(frozen=True)
class LossComponents:
    total: torch.Tensor
    reconstruction: torch.Tensor
    bandpower: torch.Tensor
    spatial: torch.Tensor


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def configure_device(config: RestorationConfig, override: str | None = None) -> torch.device:
    requested = override or config.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    device = torch.device(requested)
    torch.backends.cuda.matmul.allow_tf32 = config.training.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.training.allow_tf32
    return device


def build_diffusion_schedule(
    config: RestorationConfig,
    device: torch.device,
) -> DiffusionSchedule:
    timesteps = int(config.diffusion["timesteps"])
    betas = torch.linspace(
        float(config.diffusion["beta_start"]),
        float(config.diffusion["beta_end"]),
        timesteps,
        device=device,
        dtype=torch.float32,
    )
    alphas = 1.0 - betas
    return DiffusionSchedule(betas, alphas, torch.cumprod(alphas, dim=0))


def _condition(x_mi9: torch.Tensor, time_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch = x_mi9.shape[0]
    condition = torch.zeros((batch, 22, time_length), device=x_mi9.device, dtype=x_mi9.dtype)
    mask = torch.zeros_like(condition)
    condition[:, MI9_INDICES, :] = x_mi9
    mask[:, MI9_INDICES, :] = 1.0
    return condition, mask


def _make_loader(
    source: RestorationSplit,
    config: RestorationConfig,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(source.x_mi9).float(),
        torch.from_numpy(source.x_true22).float(),
    )
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=config.training.num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def _build_optimizer(model: nn.Module, config: RestorationConfig) -> torch.optim.Optimizer:
    kwargs = {
        "lr": config.training.learning_rate,
        "betas": (config.training.beta_1, config.training.beta_2),
        "weight_decay": config.training.weight_decay,
    }
    if config.training.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _conditional_full22(
    predicted22: torch.Tensor,
    x_mi9: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a differentiable full-22 candidate and its embedded condition."""

    condition, mask = _condition(x_mi9, predicted22.shape[-1])
    return condition + predicted22 * (1.0 - mask), condition


def wgan_gradient_penalty(
    critic: nn.Module,
    condition22: torch.Tensor,
    real22: torch.Tensor,
    fake22: torch.Tensor,
) -> torch.Tensor:
    """WGAN-GP penalty over candidate EEG while keeping the condition fixed."""

    if real22.shape != fake22.shape or real22.shape != condition22.shape:
        raise ValueError("Gradient-penalty inputs must have matching shapes")
    alpha = torch.rand(
        (real22.shape[0], 1, 1), device=real22.device, dtype=real22.dtype
    )
    interpolated = (alpha * real22 + (1.0 - alpha) * fake22).requires_grad_(True)
    scores = critic(condition22, interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_norm = gradients.flatten(1).norm(2, dim=1)
    return (gradient_norm - 1.0).square().mean()


def _run_wgan_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    config: RestorationConfig,
    device: torch.device,
    generator_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    sums = {
        "generator_total": 0.0,
        "reconstruction": 0.0,
        "adversarial": 0.0,
        "critic": 0.0,
        "gradient_penalty": 0.0,
        "wasserstein_gap": 0.0,
    }
    total_trials = 0
    critic_steps = int(config.gan["critic_steps"])
    gp_weight = float(config.gan["gradient_penalty_weight"])
    reconstruction_weight = float(config.gan["reconstruction_weight"])
    adversarial_weight = float(config.gan["adversarial_weight"])
    for x_mi9, x_true22 in loader:
        x_mi9 = x_mi9.to(device, non_blocking=True)
        x_true22 = x_true22.to(device, non_blocking=True)
        with torch.no_grad():
            fake22, condition22 = _conditional_full22(model(x_mi9), x_mi9)
            real22, _ = _conditional_full22(x_true22, x_mi9)
        critic_loss_sum = 0.0
        penalty_sum = 0.0
        gap_sum = 0.0
        for _ in range(critic_steps):
            critic_optimizer.zero_grad(set_to_none=True)
            real_score = model.critic(condition22, real22)
            fake_score = model.critic(condition22, fake22)
            penalty = wgan_gradient_penalty(
                model.critic, condition22, real22, fake22
            )
            critic_loss = fake_score.mean() - real_score.mean() + gp_weight * penalty
            critic_loss.backward()
            critic_optimizer.step()
            critic_loss_sum += float(critic_loss.detach())
            penalty_sum += float(penalty.detach())
            gap_sum += float((real_score.mean() - fake_score.mean()).detach())

        for parameter in model.critic.parameters():
            parameter.requires_grad_(False)
        generator_optimizer.zero_grad(set_to_none=True)
        predicted22 = model(x_mi9)
        generated22, condition22 = _conditional_full22(predicted22, x_mi9)
        reconstruction = F.mse_loss(
            generated22[:, MISSING_INDICES, :],
            x_true22[:, MISSING_INDICES, :],
        )
        adversarial = -model.critic(condition22, generated22).mean()
        generator_total = (
            reconstruction_weight * reconstruction
            + adversarial_weight * adversarial
        )
        generator_total.backward()
        generator_optimizer.step()
        for parameter in model.critic.parameters():
            parameter.requires_grad_(True)

        batch_size = x_mi9.shape[0]
        values = {
            "generator_total": float(generator_total.detach()),
            "reconstruction": float(reconstruction.detach()),
            "adversarial": float(adversarial.detach()),
            "critic": critic_loss_sum / critic_steps,
            "gradient_penalty": penalty_sum / critic_steps,
            "wasserstein_gap": gap_sum / critic_steps,
        }
        for name, value in values.items():
            sums[name] += value * batch_size
        total_trials += batch_size
    if not total_trials:
        raise ValueError("Cannot train WGAN-GP on an empty split")
    return {name: value / total_trials for name, value in sums.items()}


@torch.inference_mode()
def _run_wgan_validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    config: RestorationConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums = {"reconstruction": 0.0, "adversarial": 0.0, "wasserstein_gap": 0.0}
    total_trials = 0
    for x_mi9, x_true22 in loader:
        x_mi9 = x_mi9.to(device, non_blocking=True)
        x_true22 = x_true22.to(device, non_blocking=True)
        generated22, condition22 = _conditional_full22(model(x_mi9), x_mi9)
        real22, _ = _conditional_full22(x_true22, x_mi9)
        reconstruction = F.mse_loss(
            generated22[:, MISSING_INDICES, :],
            x_true22[:, MISSING_INDICES, :],
        )
        fake_score = model.critic(condition22, generated22)
        real_score = model.critic(condition22, real22)
        values = {
            "reconstruction": float(reconstruction),
            "adversarial": float(-fake_score.mean()),
            "wasserstein_gap": float(real_score.mean() - fake_score.mean()),
        }
        batch_size = x_mi9.shape[0]
        for name, value in values.items():
            sums[name] += value * batch_size
        total_trials += batch_size
    if not total_trials:
        raise ValueError("Cannot validate WGAN-GP on an empty split")
    return {name: value / total_trials for name, value in sums.items()}


def _loss(
    model: nn.Module,
    method: str,
    x_mi9: torch.Tensor,
    x_true22: torch.Tensor,
    schedule: DiffusionSchedule | None,
    diffusion_timesteps: int,
    loss_config: dict[str, Any] | None = None,
    *,
    validation_generator: torch.Generator | None = None,
) -> LossComponents:
    if method == "autoencoder":
        predicted = model(x_mi9)
        reconstruction = F.mse_loss(
            predicted[:, MISSING_INDICES, :], x_true22[:, MISSING_INDICES, :]
        )
        bandpower = reconstruction.new_zeros(())
        spatial = reconstruction.new_zeros(())
        if loss_config:
            # CUDA autocast may return float16 while observed MI-9 stays float32.
            # Feature losses use float32 FFT/correlation and preserve gradients.
            finalized = predicted.float()
            finalized[:, MI9_INDICES, :] = x_mi9.float()
            bandpower = log_bandpower_loss(
                finalized,
                x_true22,
                float(loss_config["sampling_rate"]),
                loss_config["bands"],
                float(loss_config["epsilon"]),
            )
            spatial = spatial_correlation_loss(finalized, x_true22, float(loss_config["epsilon"]))
        total = (
            float(loss_config.get("time_weight", 1.0) if loss_config else 1.0)
            * reconstruction
            + float(loss_config.get("bandpower_weight", 0.0) if loss_config else 0.0)
            * bandpower
            + float(loss_config.get("spatial_weight", 0.0) if loss_config else 0.0)
            * spatial
        )
        return LossComponents(total, reconstruction, bandpower, spatial)
    if method != "ddpm" or schedule is None:
        raise ValueError(f"Unsupported learned method: {method}")
    random_kwargs = (
        {"generator": validation_generator} if validation_generator is not None else {}
    )
    timesteps = torch.randint(
        0,
        diffusion_timesteps,
        (x_true22.shape[0],),
        device=x_true22.device,
        **random_kwargs,
    )
    noise = torch.randn(
        x_true22.shape,
        device=x_true22.device,
        dtype=x_true22.dtype,
        **random_kwargs,
    )
    alpha_bar = schedule.alpha_bars[timesteps].view(-1, 1, 1)
    noisy = alpha_bar.sqrt() * x_true22 + (1.0 - alpha_bar).sqrt() * noise
    condition, mask = _condition(x_mi9, x_true22.shape[-1])
    # Match reverse diffusion: observed channels are clean and fixed at every step.
    noisy[:, MI9_INDICES, :] = x_mi9
    predicted_noise = model(torch.cat((noisy, condition, mask), dim=1), timesteps)
    # Only missing channels are generated; observed-channel noise is not an endpoint.
    reconstruction = masked_noise_loss(predicted_noise, noise)
    zero = reconstruction.new_zeros(())
    return LossComponents(reconstruction, reconstruction, zero, zero)


def log_bandpower_loss(
    restored22: torch.Tensor,
    true22: torch.Tensor,
    sampling_rate: float,
    bands: list[list[float]] | list[tuple[float, float]],
    epsilon: float,
) -> torch.Tensor:
    """Match log band power on the 13 reconstructed channels."""

    if restored22.shape != true22.shape or restored22.ndim != 3:
        raise ValueError("Bandpower inputs must have matching [B, 22, T] shapes")
    frequencies = torch.fft.rfftfreq(
        restored22.shape[-1], d=1.0 / sampling_rate, device=restored22.device
    )
    restored_power = torch.fft.rfft(restored22.float(), dim=-1).abs().square()
    true_power = torch.fft.rfft(true22.float(), dim=-1).abs().square()
    losses = []
    for low, high in bands:
        selected = (frequencies >= float(low)) & (frequencies < float(high))
        if not selected.any():
            raise ValueError(f"No FFT bins fall inside band [{low}, {high})")
        restored_band = restored_power[:, MISSING_INDICES, :][:, :, selected].mean(-1)
        true_band = true_power[:, MISSING_INDICES, :][:, :, selected].mean(-1)
        losses.append(
            F.mse_loss(torch.log(restored_band + epsilon), torch.log(true_band + epsilon))
        )
    return torch.stack(losses).mean()


def spatial_correlation_loss(
    restored22: torch.Tensor,
    true22: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Match missing-to-all temporal correlation structure within each trial."""

    if restored22.shape != true22.shape or restored22.ndim != 3:
        raise ValueError("Spatial inputs must have matching [B, 22, T] shapes")

    def correlation(values: torch.Tensor) -> torch.Tensor:
        centered = values.float() - values.float().mean(dim=-1, keepdim=True)
        normalized = centered / torch.sqrt(centered.square().mean(dim=-1, keepdim=True) + epsilon)
        return normalized @ normalized.transpose(1, 2) / values.shape[-1]

    restored_correlation = correlation(restored22)
    true_correlation = correlation(true22)
    return F.mse_loss(
        restored_correlation[:, MISSING_INDICES, :],
        true_correlation[:, MISSING_INDICES, :],
    )


def masked_noise_loss(predicted_noise: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Noise-prediction loss over the 13 generated channels only."""

    if predicted_noise.shape != noise.shape or predicted_noise.ndim != 3:
        raise ValueError("Noise tensors must have matching [B, 22, T] shapes")
    if predicted_noise.shape[1] != 22:
        raise ValueError("Noise tensors must contain 22 channels")
    return F.mse_loss(
        predicted_noise[:, MISSING_INDICES, :],
        noise[:, MISSING_INDICES, :],
    )


def _finalize_restoration_batch(
    output: torch.Tensor,
    x_mi9: torch.Tensor,
) -> torch.Tensor:
    """Return float32 output with the observed channels copied exactly."""

    if output.ndim != 3 or output.shape[1] != 22:
        raise ValueError("Restoration output must have shape [B, 22, T]")
    if x_mi9.shape != (output.shape[0], len(MI9_INDICES), output.shape[2]):
        raise ValueError("Observed MI-9 shape does not match restoration output")
    # CUDA autocast can return float16 from the autoencoder while the source
    # remains float32. Normalize before indexed assignment to avoid a dtype
    # mismatch and to preserve the classifier-input float32 contract.
    finalized = output.float()
    finalized[:, MI9_INDICES, :] = x_mi9.float()
    return finalized


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    config: RestorationConfig,
    device: torch.device,
    schedule: DiffusionSchedule | None,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    amp_enabled = config.training.amp and device.type == "cuda"
    loss_sums = {"total": 0.0, "reconstruction": 0.0, "bandpower": 0.0, "spatial": 0.0}
    total_trials = 0
    validation_generator = None
    if not training:
        validation_generator = torch.Generator(device=device).manual_seed(config.seed + 10_000)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for x_mi9, x_true22 in loader:
            x_mi9 = x_mi9.to(device, non_blocking=True)
            x_true22 = x_true22.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                losses = _loss(
                    model,
                    config.method,
                    x_mi9,
                    x_true22,
                    schedule,
                    int(config.diffusion.get("timesteps", 0)),
                    config.loss,
                    validation_generator=validation_generator,
                )
            if training:
                scaler.scale(losses.total).backward()
                scaler.step(optimizer)
                scaler.update()
            batch_size = x_mi9.shape[0]
            for name in loss_sums:
                loss_sums[name] += float(getattr(losses, name).detach()) * batch_size
            total_trials += batch_size
    if not total_trials:
        raise ValueError("Cannot train or validate on an empty split")
    return {name: value / total_trials for name, value in loss_sums.items()}


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_validation_loss: float,
    best_epoch: int,
    epochs_without_improvement: int,
    config: RestorationConfig,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "config": config.as_serializable_dict(),
    }


def _train_wgan_model(
    config: RestorationConfig,
    train: RestorationSplit,
    validation: RestorationSplit,
    device: torch.device,
    *,
    overwrite: bool,
) -> Path:
    """Train and resume a conditional WGAN-GP using validation reconstruction MSE."""

    experiment_dir = config.output.experiment_dir
    checkpoint_dir = experiment_dir / "checkpoints" / f"seed_{config.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pt"
    last_path = checkpoint_dir / "last_model.pt"
    complete_path = checkpoint_dir / "training_complete.json"
    history_path = checkpoint_dir / "history.csv"
    if complete_path.is_file() and not overwrite:
        _validate_checkpoint_signature(best_path, config)
        print(f"SKIP completed restoration training: {config.name} seed={config.seed}")
        return best_path

    seed_everything(config.seed, config.training.deterministic)
    model = build_restoration_model(config.method, config.model).to(device)
    generator_optimizer = _build_optimizer(model.generator, config)
    critic_optimizer = _build_optimizer(model.critic, config)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    if last_path.is_file() and not overwrite:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        _assert_training_signature(checkpoint, config, last_path)
        model.load_state_dict(checkpoint["model_state"])
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if history_path.is_file():
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
        print(f"RESUME {config.name} from epoch {start_epoch}")

    train_loader = _make_loader(train, config, shuffle=True, seed=config.seed)
    validation_loader = _make_loader(validation, config, shuffle=False, seed=config.seed)
    training_start = time.perf_counter()
    final_epoch = start_epoch - 1
    stop_reason = "max_epochs_reached"
    for epoch in range(start_epoch, config.training.epochs + 1):
        train_metrics = _run_wgan_training_epoch(
            model,
            train_loader,
            config,
            device,
            generator_optimizer,
            critic_optimizer,
        )
        validation_metrics = _run_wgan_validation_epoch(
            model, validation_loader, config, device
        )
        validation_loss = validation_metrics["reconstruction"]
        improved = validation_loss < best_validation_loss - config.training.min_delta
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append({
            "epoch": epoch,
            "train_generator_total": train_metrics["generator_total"],
            "train_reconstruction_loss": train_metrics["reconstruction"],
            "validation_reconstruction_loss": validation_loss,
            "train_adversarial_loss": train_metrics["adversarial"],
            "validation_adversarial_loss": validation_metrics["adversarial"],
            "train_critic_loss": train_metrics["critic"],
            "train_gradient_penalty": train_metrics["gradient_penalty"],
            "train_wasserstein_gap": train_metrics["wasserstein_gap"],
            "validation_wasserstein_gap": validation_metrics["wasserstein_gap"],
            "best_validation_loss": best_validation_loss,
            "improved": int(improved),
        })
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "generator_optimizer_state": generator_optimizer.state_dict(),
            "critic_optimizer_state": critic_optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "config": config.as_serializable_dict(),
        }
        torch.save(payload, last_path)
        if improved:
            torch.save(payload, best_path)
        _write_history(history_path, history)
        print(
            f"[{config.name} seed={config.seed}] {epoch:03d}/{config.training.epochs} "
            f"train_recon={train_metrics['reconstruction']:.6f} "
            f"val_recon={validation_loss:.6f} "
            f"critic={train_metrics['critic']:.6f} "
            f"gp={train_metrics['gradient_penalty']:.6f} "
            f"gap={validation_metrics['wasserstein_gap']:.6f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
        final_epoch = epoch
        if epochs_without_improvement >= config.training.patience:
            print(f"EARLY STOP {config.name}: patience={config.training.patience}")
            stop_reason = "validation_plateau"
            break
    if not best_path.is_file():
        raise RuntimeError("WGAN-GP training finished without a best checkpoint")
    completion = {
        "name": config.name,
        "method": config.method,
        "seed": config.seed,
        "selection_metric": "validation_missing13_mse",
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "last_epoch": final_epoch,
        "max_epochs": config.training.epochs,
        "patience": config.training.patience,
        "epochs_without_improvement": epochs_without_improvement,
        "stop_reason": stop_reason,
        "validation_plateau_reached": stop_reason == "validation_plateau",
        "best_epoch_near_end": best_epoch > max(0, final_epoch - config.training.patience),
        "training_seconds_this_run": time.perf_counter() - training_start,
        "generator_parameters": sum(p.numel() for p in model.generator.parameters()),
        "critic_parameters": sum(p.numel() for p in model.critic.parameters()),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    complete_path.write_text(json.dumps(completion, indent=2), encoding="utf-8")
    if stop_reason == "max_epochs_reached":
        print(
            f"WARNING {config.name}: reached max_epochs={config.training.epochs} "
            "before the validation plateau criterion; inspect history.csv before inference.",
            flush=True,
        )
    return best_path


def train_model(
    config: RestorationConfig,
    train: RestorationSplit,
    validation: RestorationSplit,
    device: torch.device,
    *,
    overwrite: bool = False,
) -> Path:
    """Train one learned restoration model, resuming the last epoch by default."""

    if config.method == "spherical_spline":
        raise ValueError("Spherical spline has no training stage")
    if config.method == "wgan_gp":
        return _train_wgan_model(
            config, train, validation, device, overwrite=overwrite
        )
    experiment_dir = config.output.experiment_dir
    checkpoint_dir = experiment_dir / "checkpoints" / f"seed_{config.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pt"
    last_path = checkpoint_dir / "last_model.pt"
    complete_path = checkpoint_dir / "training_complete.json"
    history_path = checkpoint_dir / "history.csv"
    if complete_path.is_file() and not overwrite:
        _validate_checkpoint_signature(best_path, config)
        print(f"SKIP completed restoration training: {config.name} seed={config.seed}")
        return best_path

    seed_everything(config.seed, config.training.deterministic)
    model = build_restoration_model(config.method, config.model).to(device)
    optimizer = _build_optimizer(model, config)
    amp_enabled = config.training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    schedule = build_diffusion_schedule(config, device) if config.method == "ddpm" else None
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    if last_path.is_file() and not overwrite:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        _assert_training_signature(checkpoint, config, last_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if history_path.is_file():
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
        print(f"RESUME {config.name} from epoch {start_epoch}")

    train_loader = _make_loader(train, config, shuffle=True, seed=config.seed)
    validation_loader = _make_loader(validation, config, shuffle=False, seed=config.seed)
    training_start = time.perf_counter()
    final_epoch = start_epoch - 1
    stop_reason = "max_epochs_reached"
    for epoch in range(start_epoch, config.training.epochs + 1):
        train_losses = _run_epoch(
            model, train_loader, config, device, schedule,
            optimizer=optimizer, scaler=scaler,
        )
        validation_losses = _run_epoch(
            model, validation_loader, config, device, schedule,
            optimizer=None, scaler=scaler,
        )
        train_loss = train_losses["total"]
        validation_loss = validation_losses["total"]
        improved = validation_loss < best_validation_loss - config.training.min_delta
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_reconstruction_loss": train_losses["reconstruction"],
            "validation_reconstruction_loss": validation_losses["reconstruction"],
            "train_bandpower_loss": train_losses["bandpower"],
            "validation_bandpower_loss": validation_losses["bandpower"],
            "train_spatial_loss": train_losses["spatial"],
            "validation_spatial_loss": validation_losses["spatial"],
            "best_validation_loss": best_validation_loss,
            "improved": int(improved),
        })
        payload = _checkpoint_payload(
            model, optimizer, scaler, epoch, best_validation_loss, best_epoch,
            epochs_without_improvement, config,
        )
        torch.save(payload, last_path)
        if improved:
            torch.save(payload, best_path)
        _write_history(history_path, history)
        print(
            f"[{config.name} seed={config.seed}] {epoch:03d}/{config.training.epochs} "
            f"train_loss={train_loss:.6f} val_loss={validation_loss:.6f} "
            f"train_band={train_losses['bandpower']:.6f} "
            f"val_band={validation_losses['bandpower']:.6f} "
            f"train_spatial={train_losses['spatial']:.6f} "
            f"val_spatial={validation_losses['spatial']:.6f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
        final_epoch = epoch
        if epochs_without_improvement >= config.training.patience:
            print(f"EARLY STOP {config.name}: patience={config.training.patience}")
            stop_reason = "validation_plateau"
            break
    if not best_path.is_file():
        raise RuntimeError("Training finished without a best checkpoint")
    completion = {
        "name": config.name,
        "method": config.method,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "last_epoch": final_epoch,
        "max_epochs": config.training.epochs,
        "patience": config.training.patience,
        "epochs_without_improvement": epochs_without_improvement,
        "stop_reason": stop_reason,
        "validation_plateau_reached": stop_reason == "validation_plateau",
        "best_epoch_near_end": best_epoch > max(0, final_epoch - config.training.patience),
        "training_seconds_this_run": time.perf_counter() - training_start,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    complete_path.write_text(json.dumps(completion, indent=2), encoding="utf-8")
    if stop_reason == "max_epochs_reached":
        print(
            f"WARNING {config.name}: reached max_epochs={config.training.epochs} "
            "before the validation plateau criterion; inspect history.csv before inference.",
            flush=True,
        )
    return best_path


def load_best_model(
    config: RestorationConfig,
    device: torch.device,
) -> nn.Module:
    checkpoint = (
        config.output.experiment_dir
        / "checkpoints"
        / f"seed_{config.seed}"
        / "best_model.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Train the restoration model first: {checkpoint}")
    model = build_restoration_model(config.method, config.model).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    _assert_training_signature(payload, config, checkpoint)
    model.load_state_dict(payload["model_state"])
    return model.eval()


def _stored_training_signature(payload: dict[str, Any]) -> dict[str, Any]:
    stored = payload.get("config")
    if not isinstance(stored, dict):
        raise ValueError("Checkpoint lacks its resolved training configuration")
    signature = {
        key: stored[key]
        for key in ("method", "seed", "source", "training", "model", "diffusion")
    }
    signature["gan"] = stored.get("gan", {})
    # Earlier checkpoints used the same default time-domain objective.
    signature["loss"] = stored.get("loss", {})
    return signature


def _assert_training_signature(
    payload: dict[str, Any],
    config: RestorationConfig,
    checkpoint: Path,
) -> None:
    if _stored_training_signature(payload) != config.training_signature():
        raise ValueError(
            f"Checkpoint configuration does not match the current experiment: {checkpoint}. "
            "Use a new experiment name/path or explicitly restart with --overwrite."
        )


def _validate_checkpoint_signature(checkpoint: Path, config: RestorationConfig) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Completion marker exists but best checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _assert_training_signature(payload, config, checkpoint)


@torch.inference_mode()
def _ddpm_sample(
    model: nn.Module,
    x_mi9: torch.Tensor,
    config: RestorationConfig,
    schedule: DiffusionSchedule,
    generator: torch.Generator,
) -> torch.Tensor:
    condition, mask = _condition(x_mi9, x_mi9.shape[-1])
    current = torch.randn(
        condition.shape, device=x_mi9.device, dtype=x_mi9.dtype, generator=generator
    )
    current[:, MI9_INDICES, :] = x_mi9
    total_steps = int(config.diffusion["timesteps"])
    if config.inference.sampler == "ddpm":
        selected = list(reversed(range(total_steps)))
    else:
        selected = np.linspace(
            0, total_steps - 1, config.inference.sampling_steps, dtype=np.int64
        ).tolist()[::-1]
    for position, timestep in enumerate(selected):
        t = torch.full((x_mi9.shape[0],), timestep, device=x_mi9.device, dtype=torch.long)
        predicted_noise = model(torch.cat((current, condition, mask), dim=1), t)
        if config.inference.sampler == "ddpm":
            beta = schedule.betas[timestep]
            alpha = schedule.alphas[timestep]
            alpha_bar = schedule.alpha_bars[timestep]
            mean = (current - beta * predicted_noise / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
            if timestep > 0:
                current = mean + torch.sqrt(beta) * torch.randn(
                    current.shape,
                    device=current.device,
                    dtype=current.dtype,
                    generator=generator,
                )
            else:
                current = mean
        else:
            previous = selected[position + 1] if position + 1 < len(selected) else -1
            alpha_bar = schedule.alpha_bars[timestep]
            previous_alpha_bar = (
                schedule.alpha_bars[previous]
                if previous >= 0
                else torch.ones((), device=current.device)
            )
            predicted_x0 = (
                current - torch.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha_bar)
            eta = config.inference.eta
            sigma = eta * torch.sqrt(
                torch.clamp(
                    (1.0 - previous_alpha_bar)
                    / (1.0 - alpha_bar)
                    * (1.0 - alpha_bar / previous_alpha_bar),
                    min=0.0,
                )
            )
            direction = torch.sqrt(
                torch.clamp(1.0 - previous_alpha_bar - sigma.square(), min=0.0)
            ) * predicted_noise
            current = torch.sqrt(previous_alpha_bar) * predicted_x0 + direction
            if eta > 0 and previous >= 0:
                current = current + sigma * torch.randn(
                    current.shape,
                    device=current.device,
                    dtype=current.dtype,
                    generator=generator,
                )
        current[:, MI9_INDICES, :] = x_mi9
    return current


@torch.inference_mode()
def infer_learned(
    config: RestorationConfig,
    source: RestorationSplit,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    model = load_best_model(config, device)
    dataset = TensorDataset(torch.from_numpy(source.x_mi9).float())
    loader = DataLoader(
        dataset,
        batch_size=config.inference.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    schedule = build_diffusion_schedule(config, device) if config.method == "ddpm" else None
    generator = torch.Generator(device=device).manual_seed(config.inference.seed)
    restored = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for (x_mi9,) in loader:
        x_mi9 = x_mi9.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=config.training.amp and device.type == "cuda",
        ):
            if config.method in {"autoencoder", "wgan_gp"}:
                output = model(x_mi9)
            elif config.method == "ddpm" and schedule is not None:
                output = _ddpm_sample(model, x_mi9, config, schedule, generator)
            else:
                raise ValueError(f"Unsupported learned method: {config.method}")
        output = _finalize_restoration_batch(output, x_mi9)
        restored.append(output.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    restored_array = np.concatenate(restored).astype(np.float32, copy=False)
    restored_array = enforce_observed_channels(restored_array, source.x_mi9)
    return restored_array, elapsed


def dry_run_learned(
    config: RestorationConfig,
    source: RestorationSplit,
    device: torch.device,
) -> None:
    """Run one forward/loss pass without creating artifacts."""

    seed_everything(config.seed, config.training.deterministic)
    model = build_restoration_model(config.method, config.model).to(device)
    x_mi9 = torch.from_numpy(source.x_mi9[:2]).float().to(device)
    x_true22 = torch.from_numpy(source.x_true22[:2]).float().to(device)
    schedule = build_diffusion_schedule(config, device) if config.method == "ddpm" else None
    if config.method == "wgan_gp":
        generated22, condition22 = _conditional_full22(model(x_mi9), x_mi9)
        real22, _ = _conditional_full22(x_true22, x_mi9)
        reconstruction = F.mse_loss(
            generated22[:, MISSING_INDICES, :],
            x_true22[:, MISSING_INDICES, :],
        )
        real_score = model.critic(condition22, real22).mean()
        fake_score = model.critic(condition22, generated22).mean()
        penalty = wgan_gradient_penalty(
            model.critic, condition22, real22, generated22.detach()
        )
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"DRY RUN {config.name}: input={tuple(x_mi9.shape)}, "
            f"target={tuple(x_true22.shape)}, reconstruction={float(reconstruction):.6f}, "
            f"critic_gap={float(real_score - fake_score):.6f}, "
            f"gradient_penalty={float(penalty):.6f}, parameters={parameters:,}, "
            f"device={device}"
        )
        return
    with torch.inference_mode():
        losses = _loss(
            model, config.method, x_mi9, x_true22, schedule,
            int(config.diffusion.get("timesteps", 0)),
            config.loss,
            validation_generator=torch.Generator(device=device).manual_seed(0),
        )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"DRY RUN {config.name}: input={tuple(x_mi9.shape)}, "
        f"target={tuple(x_true22.shape)}, loss={float(losses.total):.6f}, "
        f"reconstruction={float(losses.reconstruction):.6f}, "
        f"bandpower={float(losses.bandpower):.6f}, "
        f"spatial={float(losses.spatial):.6f}, "
        f"parameters={parameters:,}, device={device}"
    )
