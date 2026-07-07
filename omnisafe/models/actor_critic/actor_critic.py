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
"""Implementation of ActorCritic."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from torch import nn, optim
from torch.optim.lr_scheduler import ConstantLR, LinearLR

from omnisafe.models.actor import GaussianLearningActor
from omnisafe.models.actor.actor_builder import ActorBuilder
from omnisafe.models.base import Actor, Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig
from omnisafe.utils.schedule import PiecewiseSchedule, Schedule


def _resolve_critic_type(critic_cfg) -> str:
    """Return the critic_type string based on model_cfgs.critic settings."""
    if not getattr(critic_cfg, 'distributional', False):
        return 'v'
    dist_type = getattr(critic_cfg, 'dist_type', 'qr')
    return {'qr': 'v_qr', 'tqc': 'v_tqc', 'iqn': 'v_iqn'}.get(dist_type, 'v_qr')


def _augment_obs_space(obs_space: OmnisafeSpace) -> OmnisafeSpace:
    """Return a copy of ``obs_space`` with one extra feature for the remaining horizon.

    The extra dimension carries the remaining timesteps-to-go (a raw count) so that both the actor
    and the critic can condition on it. Bounds for the new feature are unbounded — the value is
    metadata for the input width only and is not used to clip observations.
    """
    assert isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1, (
        'finite_horizon only supports 1-D Box observation spaces.'
    )
    low = np.append(obs_space.low, -np.inf).astype(obs_space.dtype)
    high = np.append(obs_space.high, np.inf).astype(obs_space.dtype)
    return spaces.Box(low=low, high=high, dtype=obs_space.dtype)


class ActorCritic(nn.Module):
    """Class for ActorCritic.

    In OmniSafe, we combine the actor and critic into one this class.

    +-----------------+-----------------------------------------------+
    | Model           | Description                                   |
    +=================+===============================================+
    | Actor           | Input is observation. Output is action.       |
    +-----------------+-----------------------------------------------+
    | Reward V Critic | Input is observation. Output is reward value. |
    +-----------------+-----------------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        reward_critic (Critic): The critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    std_schedule: Schedule

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        """Initialize an instance of :class:`ActorCritic`."""
        super().__init__()

        # Finite-horizon: the observation is augmented with the remaining timesteps-to-go, seen
        # uniformly by BOTH the actor and the critic (the policy becomes time-aware). We widen the
        # observation space by one feature and build every network against it.
        self._finite_horizon: bool = getattr(model_cfgs.critic, 'finite_horizon', False)
        self._model_obs_space: OmnisafeSpace = (
            _augment_obs_space(obs_space) if self._finite_horizon else obs_space
        )
        model_obs_space = self._model_obs_space

        self.actor: Actor = ActorBuilder(
            obs_space=model_obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.actor.hidden_sizes,
            activation=model_cfgs.actor.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
        ).build_actor(
            actor_type=model_cfgs.actor_type,
        )
        _critic_type = _resolve_critic_type(model_cfgs.critic)
        _n_q   = getattr(model_cfgs.critic, 'n_quantiles', 50)
        _n_tqc = getattr(model_cfgs.critic, 'tqc_n_critics', 2)
        _n_top = getattr(model_cfgs.critic, 'tqc_n_top_to_drop', 2)
        _emb   = getattr(model_cfgs.critic, 'iqn_embed_dim', 64)
        _ncos  = getattr(model_cfgs.critic, 'iqn_n_cos', 64)
        _eval  = getattr(model_cfgs.critic, 'iqn_n_tau_eval', 32)
        self.reward_critic: Critic = CriticBuilder(
            obs_space=model_obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=_n_tqc if _critic_type == 'v_tqc' else 1,
            use_obs_encoder=False,
        ).build_critic(
            critic_type=_critic_type,
            n_quantiles=_n_q,
            n_top_quantiles_to_drop=_n_top,
            iqn_embed_dim=_emb,
            iqn_n_cos=_ncos,
            iqn_n_tau_eval=_eval,
        )
        self.add_module('actor', self.actor)
        self.add_module('reward_critic', self.reward_critic)

        if model_cfgs.actor.lr is not None:
            self.actor_optimizer: optim.Optimizer
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_cfgs.actor.lr)
        if model_cfgs.critic.lr is not None:
            self.reward_critic_optimizer: optim.Optimizer = optim.Adam(
                self.reward_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )
        if model_cfgs.actor.lr is not None:
            self.actor_scheduler: LinearLR | ConstantLR
            if model_cfgs.linear_lr_decay:
                self.actor_scheduler = LinearLR(
                    self.actor_optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=epochs,
                )
            else:
                self.actor_scheduler = ConstantLR(
                    self.actor_optimizer,
                    factor=1.0,
                    total_iters=epochs,
                )

    def augment_obs(
        self,
        obs: torch.Tensor,
        remaining: torch.Tensor | float | None,
    ) -> torch.Tensor:
        """Append the remaining horizon to ``obs`` for a finite-horizon actor-critic.

        When ``finite_horizon`` is disabled this is a no-op and returns ``obs`` unchanged, so it is
        safe to call unconditionally. The remaining horizon is stored/passed as a raw timestep
        count (``H - t``); it is broadcast to the observation's batch shape and concatenated as the
        last feature. Both the actor and the critic consume this augmented observation.

        Args:
            obs: Observation, shape ``(obs_dim,)`` or ``(batch, obs_dim)``.
            remaining: Remaining timesteps-to-go, a scalar or a ``(batch,)`` tensor.

        Returns:
            The (possibly) augmented observation the actor and critic should consume.
        """
        if not self._finite_horizon:
            return obs
        if remaining is None:
            # Graceful fallback (e.g. a caller that does not track the horizon): treat as 0.
            remaining = 0.0
        r = torch.as_tensor(remaining, dtype=obs.dtype, device=obs.device)
        if obs.dim() == 1:  # unbatched (obs_dim,)
            return torch.cat([obs, r.reshape(1)], dim=-1)
        if r.dim() == 0:
            r = r.expand(obs.shape[0])
        return torch.cat([obs, r.reshape(obs.shape[0], 1)], dim=-1)

    def step(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        remaining: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Choose the action based on the observation. used in rollout without gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.
            remaining (torch.Tensor, optional): Remaining timesteps-to-go for a finite-horizon
                actor-critic. Ignored unless ``finite_horizon`` is enabled.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            log_prob: The log probability of the action.
        """
        with torch.no_grad():
            obs = self.augment_obs(obs, remaining)
            value_r = self.reward_critic(obs)
            act = self.actor.predict(obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(act)
        return act, value_r[0], log_prob

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose the action based on the observation. used in training with gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            log_prob: The log probability of the action.
        """
        return self.step(obs, deterministic=deterministic)

    def set_annealing(self, epochs: list[int], std: list[float]) -> None:
        """Set the annealing mode for the actor.

        Args:
            epochs (list of int): The list of epochs.
            std (list of float): The list of standard deviation.
        """
        assert isinstance(
            self.actor,
            GaussianLearningActor,
        ), 'Only GaussianLearningActor support annealing.'
        self.std_schedule = PiecewiseSchedule(
            endpoints=list(zip(epochs, std)),
            outside_value=std[-1],
        )

    def annealing(self, epoch: int) -> None:
        """Set the annealing mode for the actor.

        Args:
            epoch (int): The current epoch.
        """
        assert isinstance(
            self.actor,
            GaussianLearningActor,
        ), 'Only GaussianLearningActor support annealing.'
        self.actor.std = self.std_schedule.value(epoch)
