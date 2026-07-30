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
"""Implementation of ConstraintActorQCritic."""

from __future__ import annotations

import itertools
from copy import deepcopy

import torch
from torch import nn, optim

from omnisafe.models.actor_critic.actor_q_critic import ActorQCritic
from omnisafe.models.base import Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.models.critic.successor_representation_critic import (
    QSuccessorRepresentationTrunk,
    SuccessorRepresentationQLinearReadout,
    SuccessorRepresentationQReadout,
    TDRidgeSuccessorRepresentationQTrunk,
)
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig


def _dedup_parameters(*modules: nn.Module) -> list[nn.Parameter]:
    """Collect the parameters of ``modules`` in order, keeping only the first of each.

    ``itertools.chain(a.parameters(), b.parameters())`` yields shared parameters twice. For a
    polyak update that is not harmless: applying ``target <- tau * live + (1 - tau) * target``
    twice to the same tensor gives an effective rate of ``1 - (1 - tau)^2 ~= 2 * tau``, so a
    trunk shared between the reward and cost critics would silently track its target at double
    the configured speed.
    """
    seen: set[int] = set()
    unique: list[nn.Parameter] = []
    for param in itertools.chain.from_iterable(m.parameters() for m in modules):
        if id(param) not in seen:
            seen.add(id(param))
            unique.append(param)
    return unique


class ConstraintActorQCritic(ActorQCritic):
    """ConstraintActorQCritic is a wrapper around ActorCritic that adds a cost critic to the model.

    In OmniSafe, we combine the actor and critic into one this class.

    +-----------------+---------------------------------------------------+
    | Model           | Description                                       |
    +=================+===================================================+
    | Actor           | Input is observation. Output is action.           |
    +-----------------+---------------------------------------------------+
    | Reward Q Critic | Input is obs-action pair, Output is reward value. |
    +-----------------+---------------------------------------------------+
    | Cost Q Critic   | Input is obs-action pair. Output is cost value.   |
    +-----------------+---------------------------------------------------+

    .. note::
        When ``model_cfgs.use_successor_representation`` is ``True``, ``reward_critic`` and
        ``cost_critic`` are both read-out heads over a single shared action-conditioned
        successor-representation Q-function (see ``model_cfgs.sr_cfgs.sr_mode``) instead of two
        independent networks. This is the off-policy counterpart of the same option on
        :class:`~omnisafe.models.actor_critic.constraint_actor_critic.ConstraintActorCritic`.
        Every other consumer of ``reward_critic`` / ``cost_critic`` (the TD critic losses, the
        actor loss, the target networks) is unaffected, since both modes still expose the
        standard Q-``Critic`` interface, ``forward(obs, act) -> [value, ...]``.

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        target_actor (Actor): The target actor network.
        reward_critic (Critic): The critic network.
        target_reward_critic (Critic): The target critic network.
        cost_critic (Critic): The critic network.
        target_cost_critic (Critic): The target critic network.
        actor_optimizer (Optimizer): The optimizer for the actor network.
        reward_critic_optimizer (Optimizer): The optimizer for the critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        """Initialize an instance of :class:`ConstraintActorQCritic`."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)

        self._use_sr: bool = bool(model_cfgs.get('use_successor_representation', False))
        self._sr_mode: str | None = (
            model_cfgs.sr_cfgs.get('sr_mode', 'shared_trunk') if self._use_sr else None
        )

        if self._use_sr:
            self._build_successor_representation_critics(obs_space, act_space, model_cfgs)
            return

        self.cost_critic: Critic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        ).build_critic('q')
        self.target_cost_critic: Critic = deepcopy(self.cost_critic)
        for param in self.target_cost_critic.parameters():
            param.requires_grad = False
        self.add_module('cost_critic', self.cost_critic)
        if model_cfgs.critic.lr is not None:
            self.cost_critic_optimizer: optim.Optimizer
            self.cost_critic_optimizer = optim.Adam(
                self.cost_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )

    def _build_successor_representation_critics(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
    ) -> None:
        """Replace the plain reward/cost Q-critics with a shared successor-representation critic.

        Discards the plain ``reward_critic`` (and its target and optimizer) built by
        :class:`ActorQCritic` in favor of a successor-representation-based one, and builds the
        cost critic as a second read-out over the same trunk. ``reward_critic_optimizer`` and
        ``cost_critic_optimizer`` are set to the *same* optimizer instance so the shared trunk's
        Adam momentum state is not split across two independent optimizers (each of
        :meth:`DDPG._update_reward_critic` / :meth:`_update_cost_critic` still does its own
        ``zero_grad`` / ``backward`` / ``step``, so the trunk still only ever moves along one
        loss's gradient at a time).

        The two target critics are produced by a *single* :func:`deepcopy` of the pair, so that
        ``deepcopy``'s memo table preserves the trunk sharing: the targets read from one shared
        target trunk rather than from two independent copies of it.

        Args:
            obs_space (OmnisafeSpace): The observation space.
            act_space (OmnisafeSpace): The action space.
            model_cfgs (ModelConfig): The model configurations.
        """
        sr_cfgs = model_cfgs.sr_cfgs
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        # The reward critic keeps whatever ensemble size the algorithm asked for (2 for SAC/TD3,
        # 1 for DDPG); the cost critic is always single-headed, as in the plain build above.
        num_critics = model_cfgs.critic.num_critics
        # Every hidden layer of the trunk is sized to match sr_dim (keeping the configured
        # depth): the layer that produces phi/psi is a plain linear map with no nonlinearity of
        # its own, so its output can never carry more independent information than the last
        # hidden layer feeding it. A narrower last hidden layer would silently cap psi/phi's
        # effective rank below sr_dim (wasted capacity in shared_trunk mode; a rank-deficient,
        # exactly-singular-without-regularization ridge Gram matrix in td_ridge mode).
        hidden_sizes = [sr_cfgs.sr_dim] * len(sr_cfgs.hidden_sizes)

        trunk: QSuccessorRepresentationTrunk | TDRidgeSuccessorRepresentationQTrunk
        if self._sr_mode == 'shared_trunk':
            trunk = QSuccessorRepresentationTrunk(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_sizes=hidden_sizes,
                sr_dim=sr_cfgs.sr_dim,
                activation=sr_cfgs.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
            )
            self.reward_critic = SuccessorRepresentationQReadout(
                obs_space,
                act_space,
                trunk,
                sr_cfgs.sr_dim,
                num_critics,
                model_cfgs.weight_initialization_mode,
            )
            self.cost_critic: Critic = SuccessorRepresentationQReadout(
                obs_space,
                act_space,
                trunk,
                sr_cfgs.sr_dim,
                1,
                model_cfgs.weight_initialization_mode,
            )
            trainable_params = itertools.chain(
                trunk.parameters(),
                self.reward_critic.heads.parameters(),
                self.cost_critic.heads.parameters(),
            )
        elif self._sr_mode == 'td_ridge':
            trunk = TDRidgeSuccessorRepresentationQTrunk(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_sizes=hidden_sizes,
                sr_dim=sr_cfgs.sr_dim,
                num_psi_heads=num_critics,
                activation=sr_cfgs.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
            )
            self.reward_critic = SuccessorRepresentationQLinearReadout(
                obs_space,
                act_space,
                trunk,
                'w_r',
                list(range(num_critics)),
                model_cfgs.weight_initialization_mode,
            )
            self.cost_critic = SuccessorRepresentationQLinearReadout(
                obs_space,
                act_space,
                trunk,
                'w_c',
                [0],
                model_cfgs.weight_initialization_mode,
            )
            # w_r / w_c are buffers (ridge-solved), so the trunk's own parameters (trunk +
            # phi_head + psi_heads) are the complete set of SGD-trainable SR parameters.
            trainable_params = trunk.parameters()
        else:
            raise NotImplementedError(
                f'Unknown sr_cfgs.sr_mode "{self._sr_mode}". '
                'Available successor-representation modes are: "shared_trunk", "td_ridge".',
            )

        self.sr_trunk = trunk
        self.add_module('reward_critic', self.reward_critic)
        self.add_module('cost_critic', self.cost_critic)
        self.add_module('sr_trunk', self.sr_trunk)

        # One deepcopy of the pair, so the shared trunk is copied once and both targets point
        # at it. Copying them separately would give two target trunks tracking one live trunk.
        self.target_reward_critic, self.target_cost_critic = deepcopy(
            (self.reward_critic, self.cost_critic),
        )
        self.target_sr_trunk = self.target_reward_critic.trunk
        for param in _dedup_parameters(self.target_reward_critic, self.target_cost_critic):
            param.requires_grad = False

        if sr_cfgs.lr is not None:
            sr_optimizer = optim.Adam(list(trainable_params), lr=sr_cfgs.lr)
            self.reward_critic_optimizer: optim.Optimizer = sr_optimizer
            self.cost_critic_optimizer: optim.Optimizer = sr_optimizer
            self.sr_optimizer: optim.Optimizer = sr_optimizer

    def sr_features(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return ``(phi, psi)`` successor-representation features (``td_ridge`` mode only).

        Args:
            obs (torch.Tensor): Observation from environments.
            act (torch.Tensor): Action taken in ``obs``.

        Returns:
            phi: The one-step feature of the observation-action pair.
            psi: The successor feature of the observation-action pair, one entry per ``psi`` head.
        """
        assert self._sr_mode == 'td_ridge', (
            'sr_features() is only available when model_cfgs.sr_cfgs.sr_mode == "td_ridge".'
        )
        with torch.no_grad():
            z = self.sr_trunk.features(obs, act)
            phi = self.sr_trunk.phi(obs, act, z=z)
            psi = self.sr_trunk.psi(obs, act, z=z)
        return phi, psi

    @torch.no_grad()
    def sync_sr_readout_weights(self) -> None:
        """Copy the live trunk's ridge-solved ``w_r`` / ``w_c`` onto the target trunk.

        ``w_r`` / ``w_c`` are buffers, not parameters, so :meth:`polyak_update` does not touch
        them -- and the ridge solve only ever writes the live trunk. Without this sync the
        target critics would keep reading out against the zero-initialized weights they were
        deepcopied with, making every bootstrap target identically zero. The weights are already
        smoothed by ``sr_cfgs.ema_tau`` on the live side, so they are copied rather than
        polyak-averaged a second time.
        """
        assert self._sr_mode == 'td_ridge', 'Ridge read-out weights only exist in td_ridge mode.'
        self.target_sr_trunk.w_r.copy_(self.sr_trunk.w_r)
        self.target_sr_trunk.w_c.copy_(self.sr_trunk.w_c)

    def polyak_update(self, tau: float) -> None:
        """Update the target network with polyak averaging.

        Args:
            tau (float): The polyak averaging factor.
        """
        if self._use_sr:
            # The reward and cost critics share a trunk, so their parameters have to be
            # deduplicated before averaging -- see :func:`_dedup_parameters`.
            for param, target_param in zip(
                _dedup_parameters(self.reward_critic, self.cost_critic),
                _dedup_parameters(self.target_reward_critic, self.target_cost_critic),
            ):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(
                self.actor.parameters(),
                self.target_actor.parameters(),
            ):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            return

        super().polyak_update(tau)
        for target_param, param in zip(
            self.target_cost_critic.parameters(),
            self.cost_critic.parameters(),
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
