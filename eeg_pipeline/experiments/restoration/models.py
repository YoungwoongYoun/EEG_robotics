"""Deterministic autoencoder and the manuscript's standard conditional DDPM."""

from __future__ import annotations

import math

import torch
from torch import nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TemporalAutoencoder(nn.Module):
    """Length-preserving temporal bottleneck mapping nine channels to 22."""

    def __init__(
        self,
        hidden_channels: int = 64,
        bottleneck_channels: int = 32,
        dilations: tuple[int, ...] | list[int] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_channels % 8:
            raise ValueError("hidden_channels must be divisible by 8")
        self.encoder_in = nn.Conv1d(9, hidden_channels, 7, padding=3)
        self.encoder = nn.Sequential(*[
            ResidualTemporalBlock(hidden_channels, int(dilation), dropout)
            for dilation in dilations
        ])
        self.to_bottleneck = nn.Sequential(
            nn.Conv1d(hidden_channels, bottleneck_channels, 1),
            nn.SiLU(),
        )
        self.from_bottleneck = nn.Sequential(
            nn.Conv1d(bottleneck_channels, hidden_channels, 1),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(*[
            ResidualTemporalBlock(hidden_channels, int(dilation), dropout)
            for dilation in reversed(tuple(dilations))
        ])
        self.output = nn.Conv1d(hidden_channels, 22, 1)

    def forward(self, x_mi9: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.encoder_in(x_mi9))
        decoded = self.from_bottleneck(self.to_bottleneck(encoded))
        return self.output(self.decoder(decoded + encoded))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedded = torch.cat((angles.sin(), angles.cos()), dim=1)
        if self.dimension % 2:
            embedded = torch.nn.functional.pad(embedded, (0, 1))
        return embedded


class TimeResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dimension: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dimension, out_channels)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(x) + self.time_projection(time_embedding).unsqueeze(-1)
        hidden = self.conv2(self.activation(hidden))
        return self.activation(hidden + self.skip(x))


class ConditionalDDPMDenoiser(nn.Module):
    """Length-preserving denoiser carried over from the submitted DDPM notebook."""

    def __init__(self, base_channels: int = 64, time_dimension: int = 128):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dimension)
        self.blocks = nn.ModuleList((
            TimeResidualBlock(66, base_channels, time_dimension),
            TimeResidualBlock(base_channels, base_channels * 2, time_dimension),
            TimeResidualBlock(base_channels * 2, base_channels * 2, time_dimension),
            TimeResidualBlock(base_channels * 2, base_channels, time_dimension),
            TimeResidualBlock(base_channels, base_channels, time_dimension),
        ))
        self.output = nn.Conv1d(base_channels, 22, 1)

    def forward(self, model_input: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        embedded = self.time_embedding(timesteps)
        hidden = model_input
        for block in self.blocks:
            hidden = block(hidden, embedded)
        return self.output(hidden)


class ConditionalEEGCritic(nn.Module):
    """Sample-independent temporal critic conditioned on the observed MI-9."""

    def __init__(
        self,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] | list[int] = (1, 2, 4, 4),
        kernel_size: int = 7,
        stride: int = 2,
    ):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("critic kernel_size must be odd and at least 3")
        if stride < 1:
            raise ValueError("critic stride must be positive")
        layers: list[nn.Module] = []
        input_channels = 44  # candidate full-22 plus observed MI-9 embedded in 22 channels
        for multiplier in channel_multipliers:
            output_channels = base_channels * int(multiplier)
            layers.extend((
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                ),
                nn.LeakyReLU(0.2),
            ))
            input_channels = output_channels
        self.features = nn.Sequential(*layers)
        self.output = nn.Conv1d(input_channels, 1, 1)

    def forward(self, condition22: torch.Tensor, candidate22: torch.Tensor) -> torch.Tensor:
        if condition22.shape != candidate22.shape or candidate22.shape[1] != 22:
            raise ValueError("Critic inputs must have matching [B, 22, T] shapes")
        patch_scores = self.output(self.features(torch.cat((condition22, candidate22), dim=1)))
        return patch_scores.mean(dim=(1, 2))


class ConditionalWGANRestorer(nn.Module):
    """Existing temporal AE generator plus a conditional WGAN critic."""

    def __init__(self, generator: dict | None = None, critic: dict | None = None):
        super().__init__()
        self.generator = TemporalAutoencoder(**(generator or {}))
        self.critic = ConditionalEEGCritic(**(critic or {}))

    def forward(self, x_mi9: torch.Tensor) -> torch.Tensor:
        return self.generator(x_mi9)


def build_restoration_model(method: str, model_args: dict) -> nn.Module:
    if method == "autoencoder":
        return TemporalAutoencoder(**model_args)
    if method == "ddpm":
        return ConditionalDDPMDenoiser(**model_args)
    if method == "wgan_gp":
        return ConditionalWGANRestorer(**model_args)
    raise ValueError(f"Method does not use a learned model: {method}")
