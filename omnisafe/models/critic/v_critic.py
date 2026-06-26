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
"""Implementation of VCritic and DistributionalVCritic."""

from __future__ import annotations

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class VCritic(Critic):
    """Implementation of VCritic.

    A V-function approximator that uses a multi-layer perceptron (MLP) to map observations to V-values.
    This class is an inherit class of :class:`Critic`.
    You can design your own V-function approximator by inheriting this class or :class:`Critic`.

    Args:
        obs_dim (int): Observation dimension.
        act_dim (int): Action dimension.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        num_critics (int, optional): Number of critics. Defaults to 1.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
    ) -> None:
        """Initialize an instance of :class:`VCritic`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics,
            use_obs_encoder=False,
        )
        self.net_lst: list[nn.Module]
        self.net_lst = []

        for idx in range(self._num_critics):
            net = build_mlp_network(
                sizes=[self._obs_dim, *self._hidden_sizes, 1],
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
            self.net_lst.append(net)
            self.add_module(f'critic_{idx}', net)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Forward function.

        Specifically, V function approximator maps observations to V-values.

        Args:
            obs (torch.Tensor): Observations from environments.

        Returns:
            The V critic value of observation.
        """
        res = []
        for critic in self.net_lst:
            res.append(torch.squeeze(critic(obs), -1))
        return res


class DistributionalVCritic(Critic):
    """Quantile-regression distributional value critic.

    Instead of predicting E[V(s)], outputs N quantile values q_1,...,q_N at fixed quantile
    levels τ_i = (2i-1)/(2N).  The interface is identical to VCritic — ``forward`` returns
    ``[mean_over_quantiles]`` so the rest of the training pipeline (advantage computation, GAE,
    value targets, diagnostics) is unaffected.  The distributional information is exposed via
    ``quantiles(obs)`` and used only in the critic loss (``_quantile_huber_loss``).

    Args:
        obs_space: Observation space.
        act_space: Action space.
        hidden_sizes: MLP hidden layer sizes.
        activation: Activation function name.
        weight_initialization_mode: Weight init scheme.
        num_critics: Number of ensemble members (typically 1).
        n_quantiles: Number of quantiles N (default 50).
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        n_quantiles: int = 50,
    ) -> None:
        """Initialize an instance of :class:`DistributionalVCritic`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics,
            use_obs_encoder=False,
        )
        self.n_quantiles: int = n_quantiles
        # Fixed quantile levels: midpoints of N equal-probability intervals
        tau = (2.0 * torch.arange(1, n_quantiles + 1) - 1.0) / (2.0 * n_quantiles)
        self.register_buffer('tau', tau)  # shape (N,)

        self.net_lst: list[nn.Module] = []
        for idx in range(self._num_critics):
            net = build_mlp_network(
                sizes=[self._obs_dim, *self._hidden_sizes, n_quantiles],
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
            self.net_lst.append(net)
            self.add_module(f'critic_{idx}', net)

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Return list of per-ensemble mean values, shape (batch,) each — same as VCritic."""
        res = []
        for net in self.net_lst:
            q = net(obs)  # (batch, N)
            res.append(q.mean(dim=-1))  # (batch,)
        return res

    def quantiles(self, obs: torch.Tensor) -> torch.Tensor:
        """Return all quantile predictions from the first ensemble member.

        Returns:
            Tensor of shape (batch, N).
        """
        return self.net_lst[0](obs)
