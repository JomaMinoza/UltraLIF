# -*- coding: utf-8 -*-
"""
Spike encoding utilities for converting data to spike trains.

Functions:
    rate_encode:     Poisson rate coding (spike probability ∝ intensity).
    latency_encode:  Latency coding (higher intensity → earlier spike).
    temporal_encode: Phase-based temporal coding.
    direct_encode:   Repeat input at each timestep (no spike conversion).

Classes:
    SpikeEncoder: Configurable module wrapping the above functions.
"""

import torch
import torch.nn as nn


def rate_encode(x: torch.Tensor, timesteps: int = 30, gain: float = 1.0) -> torch.Tensor:
    """
    Rate coding: convert pixel values to Poisson spike trains.

    Higher pixel intensity → higher spike probability per timestep.

    Args:
        x: Input tensor [batch, features] with values in [0, 1].
        timesteps: Number of time steps T.
        gain: Scaling factor for spike rate.

    Returns:
        Spike train [batch, T, features] with values in {0, 1}.

    Example:
        >>> x = torch.rand(32, 784)
        >>> spikes = rate_encode(x, timesteps=30)
        >>> assert spikes.shape == (32, 30, 784)
    """
    x = x.clamp(0, 1)
    x_expanded = x.unsqueeze(1).expand(-1, timesteps, -1)
    spikes = (torch.rand_like(x_expanded) < x_expanded * gain).float()
    return spikes


def latency_encode(x: torch.Tensor, timesteps: int = 30, tau: float = 5.0) -> torch.Tensor:
    """
    Latency coding: higher values spike earlier.

    Args:
        x: Input tensor [batch, features] with values in (0, 1].
        timesteps: Number of time steps.
        tau: Time constant (higher tau → slower responses).

    Returns:
        Spike train [batch, T, features] with at most one spike per neuron.
    """
    x = x.clamp(0.01, 1)
    spike_times = (tau * (-torch.log(x))).long().clamp(0, timesteps - 1)
    batch, features = x.shape
    spikes = torch.zeros(batch, timesteps, features, device=x.device)
    for b in range(batch):
        for f in range(features):
            spikes[b, spike_times[b, f].item(), f] = 1.0
    return spikes


def temporal_encode(x: torch.Tensor, timesteps: int = 30) -> torch.Tensor:
    """
    Phase coding: encode input as phase-shifted sine wave patterns.

    Args:
        x: Input tensor [batch, features] with values in [0, 1].
        timesteps: Number of time steps.

    Returns:
        Spike train [batch, T, features].
    """
    t = torch.linspace(0, 1, timesteps, device=x.device)
    phase = x.unsqueeze(1) * 2 * 3.14159
    waves = torch.sin(t.view(1, -1, 1) * 2 * 3.14159 - phase)
    return (waves > 0.9).float()


def direct_encode(x: torch.Tensor, timesteps: int = 30) -> torch.Tensor:
    """
    Direct coding: repeat the input value at each timestep (no discretization).

    Useful as a non-spiking baseline.

    Args:
        x: Input tensor [batch, features].
        timesteps: Number of time steps.

    Returns:
        Tensor [batch, T, features].
    """
    return x.unsqueeze(1).expand(-1, timesteps, -1)


class SpikeEncoder(nn.Module):
    """
    Configurable spike encoder module.

    Args:
        method: Encoding method ('rate', 'latency', 'temporal', 'direct').
        timesteps: Number of time steps.
        **kwargs: Passed to the underlying encoding function.

    Example:
        >>> encoder = SpikeEncoder(method='rate', timesteps=30, gain=0.5)
        >>> spikes = encoder(torch.rand(32, 784))
    """

    def __init__(self, method: str = "rate", timesteps: int = 30, **kwargs):
        super().__init__()
        self.method = method
        self.timesteps = timesteps
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "rate":
            return rate_encode(x, self.timesteps, **self.kwargs)
        elif self.method == "latency":
            return latency_encode(x, self.timesteps, **self.kwargs)
        elif self.method == "temporal":
            return temporal_encode(x, self.timesteps, **self.kwargs)
        elif self.method == "direct":
            return direct_encode(x, self.timesteps)
        else:
            raise ValueError(f"Unknown encoding method: {self.method!r}")
