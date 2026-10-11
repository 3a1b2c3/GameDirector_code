"""Rectified-flow scheduler identical to StatePlay Causal Forcing++."""

import torch


class FlowMatchScheduler:
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        sigmas = torch.linspace(1.0, 0.0, self.num_train_timesteps + 1)[:-1]
        self.sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        timestep = self.timesteps
        weights = torch.exp(
            -2 * ((timestep - self.num_train_timesteps / 2) / self.num_train_timesteps) ** 2
        )
        weights = weights - weights.min()
        self.weights = weights * (self.num_train_timesteps / weights.sum())

    def to(self, device):
        self.sigmas = self.sigmas.to(device)
        self.timesteps = self.timesteps.to(device)
        self.weights = self.weights.to(device)
        return self

    def sample_pf_timestep_ids(self, batch, frames, device):
        return torch.randint(0, self.num_train_timesteps, (batch, frames), device=device)

    def add_noise_pf(self, clean, noise, ids):
        sigma = self.sigmas.to(clean.device, clean.dtype)[ids]
        if clean.ndim != 5:
            raise ValueError(f"per-frame video noising expects rank 5, got {clean.ndim}")
        sigma = sigma[:, None, :, None, None]
        return (1 - sigma) * clean + sigma * noise

    def training_weight(self, ids):
        return self.weights.to(ids.device)[ids]

    def add_noise(self, clean, noise, ids):
        sigma = self.sigmas.to(clean.device, clean.dtype)[ids]
        sigma = sigma.view(-1, *([1] * (clean.ndim - 1)))
        return (1 - sigma) * clean + sigma * noise

    def set_timesteps(self, num_inference_steps, device=None):
        sigmas = torch.linspace(1.0, 0.0, int(num_inference_steps) + 1)[:-1]
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        if device is not None:
            sigmas = sigmas.to(device)
        self.infer_sigmas = sigmas
        self.infer_timesteps = sigmas * self.num_train_timesteps
        return self.infer_timesteps
