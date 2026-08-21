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
"""Implementation of ConstraintActorCritic."""

from __future__ import annotations

import itertools

import torch
from torch import optim

from omnisafe.models.actor_critic.actor_critic import ActorCritic
from omnisafe.models.base import Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.models.critic.successor_representation_critic import (
    SuccessorRepresentationLinearReadout,
    SuccessorRepresentationReadout,
    SuccessorRepresentationTrunk,
    TDRidgeSuccessorRepresentationTrunk,
)
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig


class ConstraintActorCritic(ActorCritic):
    """ConstraintActorCritic is a wrapper around ActorCritic that adds a cost critic to the model.

    In OmniSafe, we combine the actor and critic into one this class.

    +-----------------+-----------------------------------------------+
    | Model           | Description                                   |
    +=================+===============================================+
    | Actor           | Input is observation. Output is action.       |
    +-----------------+-----------------------------------------------+
    | Reward V Critic | Input is observation. Output is reward value. |
    +-----------------+-----------------------------------------------+
    | Cost V Critic   | Input is observation. Output is cost value.   |
    +-----------------+-----------------------------------------------+

    .. note::
        When ``model_cfgs.use_successor_representation`` is ``True``, ``reward_critic`` and
        ``cost_critic`` are both read-out heads over a single shared successor-representation
        value function (see ``model_cfgs.sr_cfgs.sr_mode``) instead of two independent
        networks. Every other consumer of ``reward_critic`` / ``cost_critic`` (rollout, GAE,
        the critic training loop) is unaffected, since both modes still expose the standard
        ``Critic`` interface (``forward(obs) -> [value]``).

        When additionally ``model_cfgs.sr_cfgs.cost_only`` is ``True`` (``sr_mode: 'td_ridge'``
        only), ``reward_critic`` is *not* part of the successor representation at all: it is
        built and trained exactly as it would be with ``use_successor_representation: False``
        (its own network and optimizer), while ``cost_critic`` alone reads out the SR trunk. The
        trunk still fits its reward read-out ``w_r`` every update for diagnostic parity with the
        non-``cost_only`` run; it is simply never consulted by the actual reward critic.

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        reward_critic (Critic): The critic network.
        cost_critic (Critic): The critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        """Initialize an instance of :class:`ConstraintActorCritic`."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)

        self._use_sr: bool = bool(model_cfgs.get('use_successor_representation', False))
        self._sr_mode: str | None = model_cfgs.sr_cfgs.get('sr_mode', 'shared_trunk') if self._use_sr else None
        # sr_cfgs.cost_only: train reward_critic the normal way (a plain critic, no trunk
        # sharing) and give the SR trunk to cost_critic alone. Only meaningful -- and only read
        # -- under sr_mode == 'td_ridge'; see _build_successor_representation_critics.
        self._sr_cost_only: bool = (
            bool(model_cfgs.sr_cfgs.get('cost_only', False)) if self._use_sr else False
        )
        if self._sr_cost_only and self._sr_mode != 'td_ridge':
            raise NotImplementedError(
                'model_cfgs.sr_cfgs.cost_only is only supported under sr_mode == "td_ridge", '
                f'got sr_mode "{self._sr_mode}".',
            )

        if not self._use_sr:
            self.cost_critic: Critic = CriticBuilder(
                obs_space=obs_space,
                act_space=act_space,
                hidden_sizes=model_cfgs.critic.hidden_sizes,
                activation=model_cfgs.critic.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
                num_critics=1,
                use_obs_encoder=False,
            ).build_critic('v')
            self.add_module('cost_critic', self.cost_critic)

            if model_cfgs.critic.lr is not None:
                self.cost_critic_optimizer: optim.Optimizer
                self.cost_critic_optimizer = optim.Adam(
                    self.cost_critic.parameters(),
                    lr=model_cfgs.critic.lr,
                )
            return

        self._build_successor_representation_critics(obs_space, act_space, model_cfgs)

    def _build_successor_representation_critics(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
    ) -> None:
        """Replace the plain reward/cost critics with a shared successor-representation critic.

        Discards the plain ``reward_critic`` built by :class:`ActorCritic` in favor of a
        successor-representation-based one; ``reward_critic_optimizer`` and
        ``cost_critic_optimizer`` are set to the *same* optimizer instance so that the shared
        trunk's Adam momentum state is not split across two independent optimizers (each of
        :meth:`PolicyGradient._update_reward_critic` / :meth:`_update_cost_critic` still does
        its own ``zero_grad`` / ``backward`` / ``step`` call, so the shared trunk still only
        ever moves along one loss's gradient at a time).

        Args:
            obs_space (OmnisafeSpace): The observation space.
            act_space (OmnisafeSpace): The action space.
            model_cfgs (ModelConfig): The model configurations.
        """
        sr_cfgs = model_cfgs.sr_cfgs
        obs_dim = obs_space.shape[0]
        # Every hidden layer of the trunk is sized to match sr_dim (keeping the configured
        # depth): the layer that produces phi/psi is a plain linear map with no nonlinearity of
        # its own, so its output can never carry more independent information than the last
        # hidden layer feeding it. A narrower last hidden layer would silently cap psi/phi's
        # effective rank below sr_dim (wasted capacity in shared_trunk mode; a rank-deficient,
        # exactly-singular-without-regularization ridge Gram matrix in td_ridge mode).
        hidden_sizes = [sr_cfgs.sr_dim] * len(sr_cfgs.hidden_sizes)
        # Same convention for the standalone phi network of sr_cfgs.phi_source='separate': only
        # the configured depth is used, every width is sr_dim. Its depth is independent of the
        # trunk's, which is the point of that mode.
        phi_hidden_sizes = [sr_cfgs.sr_dim] * len(sr_cfgs.get('phi_hidden_sizes', []) or [])

        if self._sr_mode == 'shared_trunk':
            trunk = SuccessorRepresentationTrunk(
                obs_dim=obs_dim,
                hidden_sizes=hidden_sizes,
                sr_dim=sr_cfgs.sr_dim,
                activation=sr_cfgs.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
            )
            self.reward_critic = SuccessorRepresentationReadout(
                obs_space,
                act_space,
                trunk,
                sr_cfgs.sr_dim,
                model_cfgs.weight_initialization_mode,
            )
            self.cost_critic: Critic = SuccessorRepresentationReadout(
                obs_space,
                act_space,
                trunk,
                sr_cfgs.sr_dim,
                model_cfgs.weight_initialization_mode,
            )
            self.sr_trunk: SuccessorRepresentationTrunk | TDRidgeSuccessorRepresentationTrunk = trunk
            trainable_params = itertools.chain(
                trunk.parameters(),
                self.reward_critic.head.parameters(),
                self.cost_critic.head.parameters(),
            )
        elif self._sr_mode == 'td_ridge':
            # Read-out weight learning: 'ridge' (closed-form buffers) or 'sgd' (learned params
            # fit over a persistent replay buffer, see PolicyGradient._sgd_update_readout_weights).
            self._sr_readout: str = sr_cfgs.get('readout', 'ridge')
            learnable_readout = self._sr_readout == 'sgd'
            self._sr_phi_source: str = sr_cfgs.get('phi_source', 'trunk')
            trunk = TDRidgeSuccessorRepresentationTrunk(
                obs_dim=obs_dim,
                hidden_sizes=hidden_sizes,
                sr_dim=sr_cfgs.sr_dim,
                activation=sr_cfgs.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
                phi_source=self._sr_phi_source,
                phi_hidden_sizes=phi_hidden_sizes,
                learnable_readout=learnable_readout,
                phi_orthogonal_init=sr_cfgs.get('phi_orthogonal_init', False),
                phi_rff_bandwidth=sr_cfgs.get('phi_rff_bandwidth', 1.0),
                phi_ensemble_sources=sr_cfgs.get('phi_ensemble_sources', None),
            )
            if self._sr_cost_only:
                # Reward critic is a plain, independent V critic -- see the class docstring's
                # "cost_only" note. The trunk still fits w_r each ridge/sgd update (see
                # PolicyGradient._ridge_update_successor_weights), it just never backs an actual
                # critic.
                self.reward_critic = CriticBuilder(
                    obs_space=obs_space,
                    act_space=act_space,
                    hidden_sizes=model_cfgs.critic.hidden_sizes,
                    activation=model_cfgs.critic.activation,
                    weight_initialization_mode=model_cfgs.weight_initialization_mode,
                    num_critics=1,
                    use_obs_encoder=False,
                ).build_critic('v')
            else:
                self.reward_critic = SuccessorRepresentationLinearReadout(
                    obs_space,
                    act_space,
                    trunk,
                    'w_r',
                    model_cfgs.weight_initialization_mode,
                )
            self.cost_critic = SuccessorRepresentationLinearReadout(
                obs_space,
                act_space,
                trunk,
                'w_c',
                model_cfgs.weight_initialization_mode,
            )
            self.sr_trunk = trunk
            # The trunk parameters (trunk body + phi_head + psi_head) are trained by the value /
            # SR-feature losses through the shared sr_optimizer. Under readout='sgd', w_r / w_c are
            # also parameters but are fit by their own regression loss on a separate optimizer, so
            # they are excluded here. The requires_grad filter additionally drops the frozen phi
            # network under phi_source='random' / 'separate' / 'rff' / 'ensemble' (a no-op under
            # 'trunk'). Under phi_source='contrastive', phi_net is trainable (unlike every other
            # frozen source) but must still be excluded: it is fit by its own InfoNCE loss on its
            # own optimizer (sr_phi_optimizer, below), never by the value/SR-feature losses this
            # optimizer trains against -- so its params would otherwise sit in sr_optimizer's
            # param group forever accumulating unrelated gradients no step of this optimizer
            # should apply.
            readout_weight_ids = (
                {id(trunk.w_r), id(trunk.w_c)} if learnable_readout else set()
            )
            phi_param_ids = (
                {id(p) for p in trunk.phi_net.parameters()}
                if self._sr_phi_source == 'contrastive'
                else set()
            )
            excluded_ids = readout_weight_ids | phi_param_ids
            # Read by PolicyGradient's use_critic_norm loops (_update_reward_critic /
            # _update_cost_critic / _update_successor_features), which otherwise fold phi_net's
            # norm into losses no optimizer of phi_net ever steps from -- see those methods.
            self._sr_critic_norm_excluded_ids: set[int] = excluded_ids
            trainable_params = [
                param
                for param in trunk.parameters()
                if param.requires_grad and id(param) not in excluded_ids
            ]
        else:
            raise NotImplementedError(
                f'Unknown sr_cfgs.sr_mode "{self._sr_mode}". '
                'Available successor-representation modes are: "shared_trunk", "td_ridge".',
            )

        self.add_module('reward_critic', self.reward_critic)
        self.add_module('cost_critic', self.cost_critic)
        self.add_module('sr_trunk', self.sr_trunk)

        if sr_cfgs.lr is not None:
            sr_optimizer = optim.Adam(list(trainable_params), lr=sr_cfgs.lr)
            self.cost_critic_optimizer: optim.Optimizer = sr_optimizer
            self.sr_optimizer: optim.Optimizer = sr_optimizer
            self.reward_critic_optimizer: optim.Optimizer
            if self._sr_cost_only:
                # Trained the normal way, on the standard critic learning rate -- not sr_cfgs.lr,
                # which governs the SR trunk/cost-critic optimizer above.
                if model_cfgs.critic.lr is not None:
                    self.reward_critic_optimizer = optim.Adam(
                        self.reward_critic.parameters(),
                        lr=model_cfgs.critic.lr,
                    )
            else:
                self.reward_critic_optimizer = sr_optimizer
            # Under readout='sgd', w_r / w_c get their own optimizer so their regression loss
            # never shares Adam state with (or steps) the representation parameters. Its
            # weight_decay is the SGD analogue of the ridge kappa -- explicit L2 on w_r / w_c.
            if self._sr_mode == 'td_ridge' and self._sr_readout == 'sgd':
                w_lr = sr_cfgs.get('w_lr', None) or sr_cfgs.lr
                # Separate param groups so w_c can carry its own decay, mirroring
                # ridge_kappa_cost on the closed-form path; None keeps both on one value.
                wd_r = sr_cfgs.get('w_weight_decay', 0.0)
                wd_c = sr_cfgs.get('w_weight_decay_cost', None)
                self.sr_readout_optimizer: optim.Optimizer = optim.Adam(
                    [
                        {'params': [self.sr_trunk.w_r], 'weight_decay': wd_r},
                        {
                            'params': [self.sr_trunk.w_c],
                            'weight_decay': wd_r if wd_c is None else wd_c,
                        },
                    ],
                    lr=w_lr,
                )
            # phi_net's own optimizer (phi_source='contrastive' only): its InfoNCE loss runs on a
            # different cadence (a handful of steps per epoch, see
            # PolicyGradient._contrastive_update_phi) than the per-minibatch value/SR-feature
            # losses sr_optimizer trains against, so it gets independent Adam momentum state
            # rather than sharing sr_optimizer's -- the same reasoning as sr_readout_optimizer
            # above, applied to phi_net instead of w_r / w_c.
            if self._sr_mode == 'td_ridge' and self._sr_phi_source == 'contrastive':
                phi_lr = sr_cfgs.get('phi_lr', None) or sr_cfgs.lr
                self.sr_phi_optimizer: optim.Optimizer = optim.Adam(
                    trunk.phi_net.parameters(),
                    lr=phi_lr,
                )

    def sr_features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(phi, psi)`` successor-representation features (``td_ridge`` mode only).

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            phi: The one-step feature of the observation.
            psi: The successor feature of the observation.
        """
        assert self._sr_mode == 'td_ridge', (
            'sr_features() is only available when model_cfgs.sr_cfgs.sr_mode == "td_ridge".'
        )
        with torch.no_grad():
            z = self.sr_trunk.features(obs)
            phi = self.sr_trunk.phi(obs, z=z)
            psi = self.sr_trunk.psi(obs, z=z)
        return phi, psi

    def step(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_c: The cost value of the observation.
            log_prob: The log probability of the action.
        """
        with torch.no_grad():
            value_r = self.reward_critic(obs)
            value_c = self.cost_critic(obs)

            action = self.actor.predict(obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(action)

        return action, value_r[0], value_c[0], log_prob

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_c: The cost value of the observation.
            log_prob: The log probability of the action.
        """
        return self.step(obs, deterministic=deterministic)
