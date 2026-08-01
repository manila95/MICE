# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A small persistent replay buffer for the successor-representation SGD read-out."""

from __future__ import annotations

import torch

from omnisafe.typing import DEVICE_CPU


class ReadoutReplayBuffer:
    r"""FIFO replay buffer of ``(obs, reward, cost)`` for the ``readout='sgd'`` SR read-out.

    On-policy algorithms discard their rollout buffer every epoch, but the successor-representation
    read-out ``r(s) \approx phi(s) . w`` is **policy-independent** -- every past transition is valid
    regression data for it. This buffer keeps that data across epochs so ``w_r`` / ``w_c`` can be
    fit by SGD over the agent's whole history rather than just the current epoch, the on-policy
    analogue of the replay sampling the off-policy path already gets for free.

    Only the immediate reward and cost are stored (not returns): the read-out regresses the
    one-step reward/cost onto ``phi``, exactly as the closed-form ridge solve does. ``phi`` itself
    is recomputed from ``obs`` at sample time so it always reflects the current feature map (a
    no-op when ``phi`` is frozen, which is the intended setting).

    Args:
        obs_dim (int): Observation dimensionality.
        size (int): Maximum number of transitions retained (FIFO once full).
        device (torch.device, optional): Device the buffers live on. Defaults to CPU.
    """

    def __init__(self, obs_dim: int, size: int, device: torch.device = DEVICE_CPU) -> None:
        """Initialize an instance of :class:`ReadoutReplayBuffer`."""
        assert size > 0, 'ReadoutReplayBuffer size must be positive.'
        self._size: int = size
        self._device: torch.device = device
        self._obs: torch.Tensor = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self._reward: torch.Tensor = torch.zeros(size, dtype=torch.float32, device=device)
        self._cost: torch.Tensor = torch.zeros(size, dtype=torch.float32, device=device)
        self._ptr: int = 0
        self._full: bool = False

    def __len__(self) -> int:
        """Number of transitions currently stored."""
        return self._size if self._full else self._ptr

    def add(self, obs: torch.Tensor, reward: torch.Tensor, cost: torch.Tensor) -> None:
        """Append a batch of transitions, overwriting the oldest once the buffer is full.

        Args:
            obs (torch.Tensor): Observations, shape ``(N, obs_dim)``.
            reward (torch.Tensor): Immediate rewards, shape ``(N,)`` or ``(N, 1)``.
            cost (torch.Tensor): Immediate costs, shape ``(N,)`` or ``(N, 1)``.
        """
        obs = obs.to(self._device)
        reward = reward.reshape(-1).to(self._device)
        cost = cost.reshape(-1).to(self._device)
        n = obs.shape[0]
        idx = (torch.arange(n, device=self._device) + self._ptr) % self._size
        self._obs[idx] = obs
        self._reward[idx] = reward
        self._cost[idx] = cost
        if not self._full and self._ptr + n >= self._size:
            self._full = True
        self._ptr = (self._ptr + n) % self._size

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a uniform minibatch (with replacement) of stored transitions.

        Args:
            batch_size (int): Number of transitions to draw.

        Returns:
            ``(obs, reward, cost)`` with shapes ``(batch_size, obs_dim)``, ``(batch_size,)``,
            ``(batch_size,)``.
        """
        high = len(self)
        idx = torch.randint(0, high, (batch_size,), device=self._device)
        return self._obs[idx], self._reward[idx], self._cost[idx]
