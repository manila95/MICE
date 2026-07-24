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
"""Implementation of HazardCritic."""

from __future__ import annotations

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class HazardCritic(Critic):
    """Implementation of HazardCritic.

    Unlike :class:`~omnisafe.models.critic.v_critic.VCritic`, which regresses the expected
    discounted cost-to-go, ``HazardCritic`` predicts the probability that the current state
    incurs a nonzero cost. This measures the proximity of the state to a hazardous region under
    the current policy, rather than the discounted sum of future costs. It maps observations to a
    scalar in :math:`[0, 1]` via a sigmoid output head, and is intended to be trained with binary
    cross-entropy against the per-step binary cost indicator :math:`\\mathbb{1}[c_t > 0]`.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
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
        """Initialize an instance of :class:`HazardCritic`."""
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

        Maps observations to the predicted probability of incurring a cost.

        Args:
            obs (torch.Tensor): Observations from environments.

        Returns:
            The predicted hazard probability of each observation, in :math:`[0, 1]`.
        """
        res = []
        for critic in self.net_lst:
            res.append(torch.sigmoid(torch.squeeze(critic(obs), -1)))
        return res
