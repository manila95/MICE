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
"""Implementation of VCritic."""

from __future__ import annotations

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.critic_ensemble import aggregate as aggregate_ensemble
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
        dropout (float, optional): Dropout probability after each hidden layer. ``0.0`` (default)
            adds no dropout module -- see :func:`omnisafe.utils.model.build_mlp_network`.
        use_layer_norm (bool, optional): Whether to insert LayerNorm after each hidden layer's
            affine transform. Defaults to ``False``.
        use_spectral_norm (bool, optional): Whether to spectral-normalize each hidden layer.
            Defaults to ``False``.
        ensemble_method (str, optional): ``'none'`` (default -- single critic, or if
            ``num_critics > 1``, just reads member 0 as before), ``'cdq'``, ``'gpl'``, or
            ``'top'``. See :mod:`omnisafe.utils.critic_ensemble` for what each does and why the
            aggregation direction is tied to ``stream`` rather than independently configurable.
        stream (str, optional): ``'r'`` (reward) or ``'c'`` (cost) -- selects the aggregation
            direction under ``'cdq'``/``'gpl'``/``'top'``. Ignored under ``'none'``.
        beta_init (float, optional): Initial value of the (only meaningful under ``'gpl'``/
            ``'top'``) pessimism/conservatism coefficient. Mutated in place from outside this
            class (see ``PolicyGradient._update_critic_ensemble_beta``) as training progresses.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_spectral_norm: bool = False,
        ensemble_method: str = 'none',
        stream: str = 'r',
        beta_init: float = 0.0,
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
        assert ensemble_method in ('none', 'cdq', 'gpl', 'top')
        assert stream in ('r', 'c')
        self._ensemble_method = ensemble_method
        self._stream = stream
        self.net_lst: list[nn.Module]
        self.net_lst = []

        for idx in range(self._num_critics):
            net = build_mlp_network(
                sizes=[self._obs_dim, *self._hidden_sizes, 1],
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                use_spectral_norm=use_spectral_norm,
            )
            self.net_lst.append(net)
            self.add_module(f'critic_{idx}', net)
        # Mutable in place from outside (see PolicyGradient._update_critic_ensemble_beta) --
        # a plain buffer rather than an nn.Parameter since it is never trained by backprop (see
        # GPLBetaAdapter's docstring for why an in-sample-loss-trained beta would degenerate to
        # 0), but still needs to move with the module across .to(device) calls and show up in
        # state_dict() for checkpointing.
        self.register_buffer('beta', torch.tensor(float(beta_init)))

    def raw_values(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Each ensemble member's own raw prediction, un-aggregated.

        Used by :meth:`PolicyGradient._update_reward_critic`/``_update_cost_critic`` to train
        every member independently against the same regression target -- the pessimism/
        conservatism in :meth:`forward` comes from aggregating *already independently-trained*
        members' predictions, not from routing gradient only through whichever one the
        aggregation function happens to select for a given sample (which would starve the
        other members of training signal and defeat the point of having an ensemble at all).
        """
        return [torch.squeeze(critic(obs), -1) for critic in self.net_lst]

    def forward(
        self,
        obs: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Forward function.

        Specifically, V function approximator maps observations to V-values. Under
        ``ensemble_method != 'none'`` with ``num_critics > 1``, this is the *aggregated*
        (pessimistic for reward, conservative for cost) prediction, not any single member's raw
        one -- see :mod:`omnisafe.utils.critic_ensemble`. Every existing consumer of
        ``reward_critic``/``cost_critic`` (rollout GAE bootstrapping, the actor's advantage
        computation, Eval_s0/Eval_all, the MC value study, the intermediate-state study) reads
        through this method, so aggregation applies everywhere automatically with no other
        call site needing to change.

        Args:
            obs (torch.Tensor): Observations from environments.

        Returns:
            A length-1 list holding the ``(B,)`` value tensor (aggregated, if applicable) --
            preserves the standard :class:`~omnisafe.models.base.Critic` interface exactly.
        """
        raw = self.raw_values(obs)
        if self._ensemble_method == 'none' or len(raw) == 1:
            return [raw[0]]
        stacked = torch.stack(raw, dim=0)
        return [aggregate_ensemble(stacked, self._ensemble_method, self._stream, self.beta.item())]
