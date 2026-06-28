"""Optimal-Gap Timestep Sampler (OGS).

Theory: The copilot's Bayes-optimal loss floor is L_Z^*(t) = E[tr Cov(R_t | Z_t)].
Training should focus on timesteps where the *learnable gap* is large and the
copilot hasn't closed it yet, rather than timesteps where raw residual is large
but mostly irreducible noise.

Sampling distribution:
  p(t) ∝ p_0(t) · [learnable_frac(t)]^α · [G_copilot(t) + eps]^β

where:
  learnable_frac(t) = max(L_base(t) - L_floor(t), 0) / (L_base(t) + eps)
  G_copilot(t) = L_copilot(t) - L_floor(t)
"""

from __future__ import annotations

import numpy as np


class OptimalGapTimestepSampler:

    def __init__(
        self,
        num_total_timesteps: int = 1000,
        num_bins: int = 50,
        ema_decay: float = 0.99,
        warmup_steps: int = 200,
        alpha: float = 1.0,
        beta: float = 1.0,
        floor_ema_decay: float = 0.999,
        eps: float = 1e-8,
        min_timestep_id: int = 0,
        max_timestep_id: int = 1000,
    ):
        self.num_total_timesteps = num_total_timesteps
        self.num_bins = num_bins
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.beta = beta
        self.floor_ema_decay = floor_ema_decay
        self.eps = eps
        self.min_timestep_id = min_timestep_id
        self.max_timestep_id = max_timestep_id

        self.bin_size = max(1, num_total_timesteps // num_bins)

        self.l_base = np.zeros(num_bins, dtype=np.float64)
        self.l_copilot = np.zeros(num_bins, dtype=np.float64)
        self.l_floor = np.full(num_bins, np.inf, dtype=np.float64)
        self.bin_counts = np.zeros(num_bins, dtype=np.int64)
        self.total_updates = 0

        self._cached_weights: np.ndarray | None = None
        self._cache_dirty = True

    def _timestep_id_to_bin(self, timestep_id: int) -> int:
        return min(int(timestep_id) // self.bin_size, self.num_bins - 1)

    @property
    def is_warmed_up(self) -> bool:
        if self.total_updates < self.warmup_steps:
            return False
        active = np.sum(self.bin_counts > 0)
        return active >= 0.8 * self.num_bins

    def update(self, timestep_id: int, residual_norm_sq: float, copilot_loss: float):
        b = self._timestep_id_to_bin(timestep_id)
        decay = self.ema_decay

        if self.bin_counts[b] == 0:
            self.l_base[b] = residual_norm_sq
            self.l_copilot[b] = copilot_loss
            self.l_floor[b] = copilot_loss
        else:
            self.l_base[b] = decay * self.l_base[b] + (1 - decay) * residual_norm_sq
            self.l_copilot[b] = decay * self.l_copilot[b] + (1 - decay) * copilot_loss
            floor_decay = self.floor_ema_decay
            candidate = floor_decay * self.l_floor[b] + (1 - floor_decay) * copilot_loss
            self.l_floor[b] = min(self.l_floor[b], candidate)

        self.bin_counts[b] += 1
        self.total_updates += 1
        self._cache_dirty = True

    def _compute_weights(self) -> np.ndarray:
        weights = np.ones(self.num_bins, dtype=np.float64)

        active_mask = self.bin_counts > 0
        if not np.any(active_mask):
            return weights / weights.sum()

        learnable_frac = np.zeros(self.num_bins, dtype=np.float64)
        g_copilot = np.zeros(self.num_bins, dtype=np.float64)

        for b in range(self.num_bins):
            if not active_mask[b]:
                continue
            gap = max(self.l_base[b] - self.l_floor[b], 0.0)
            learnable_frac[b] = gap / (self.l_base[b] + self.eps)
            g_copilot[b] = max(self.l_copilot[b] - self.l_floor[b], 0.0)

        for b in range(self.num_bins):
            if not active_mask[b]:
                continue
            weights[b] = (
                (learnable_frac[b] + self.eps) ** self.alpha
                * (g_copilot[b] + self.eps) ** self.beta
            )

        # Bins outside the allowed range get zero weight
        for b in range(self.num_bins):
            bin_start = b * self.bin_size
            bin_end = min((b + 1) * self.bin_size, self.num_total_timesteps)
            if bin_end <= self.min_timestep_id or bin_start >= self.max_timestep_id:
                weights[b] = 0.0

        total = weights.sum()
        if total < self.eps:
            weights = np.ones(self.num_bins, dtype=np.float64)
            for b in range(self.num_bins):
                bin_start = b * self.bin_size
                bin_end = min((b + 1) * self.bin_size, self.num_total_timesteps)
                if bin_end <= self.min_timestep_id or bin_start >= self.max_timestep_id:
                    weights[b] = 0.0
            total = weights.sum()

        return weights / total

    def sample_timestep_id(self) -> int:
        if not self.is_warmed_up:
            return int(np.random.randint(self.min_timestep_id, self.max_timestep_id))

        if self._cache_dirty:
            self._cached_weights = self._compute_weights()
            self._cache_dirty = False

        bin_idx = int(np.random.choice(self.num_bins, p=self._cached_weights))

        bin_start = max(bin_idx * self.bin_size, self.min_timestep_id)
        bin_end = min((bin_idx + 1) * self.bin_size, self.max_timestep_id)
        bin_end = max(bin_end, bin_start + 1)

        return int(np.random.randint(bin_start, bin_end))

    def distribution_entropy(self) -> float:
        if self._cache_dirty:
            self._cached_weights = self._compute_weights()
            self._cache_dirty = False
        w = self._cached_weights
        w_pos = w[w > 0]
        return float(-np.sum(w_pos * np.log(w_pos + 1e-30)))

    def num_active_bins(self) -> int:
        return int(np.sum(self.bin_counts > 0))
