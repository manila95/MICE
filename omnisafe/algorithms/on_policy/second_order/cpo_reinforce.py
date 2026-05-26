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
"""CPO with pure REINFORCE advantage estimation (no value function)."""

from __future__ import annotations

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.second_order.cpo import CPO
from omnisafe.algorithms.on_policy.pid_lagrange.trpo_pid_reinforce import _ReinforceVectorBuffer


@registry.register
class CPOReinforce(CPO):
    """CPO with pure REINFORCE advantage estimation.

    Uses raw discounted returns (rewards-to-go / costs-to-go) with no value
    function baseline:

    .. math::

        A_t^R = G_t^R = \\sum_{k=0}^{T-t} \\gamma^k r_{t+k}, \\quad
        A_t^C = G_t^C = \\sum_{k=0}^{T-t} \\gamma^k c_{t+k}

    Two differences from :class:`CPO`:

    1. **No bootstrap** — :meth:`finish_path` always uses zero as the terminal
       value, so no value network is queried when a trajectory is truncated.
    2. **No critic training** — :meth:`_update_reward_critic` and
       :meth:`_update_cost_critic` are no-ops; the critic networks are kept in
       the architecture but their weights are never updated.

    The CPO dual-optimization trust-region policy step is unchanged.
    """

    def _init(self) -> None:
        super()._init()
        self._buf = _ReinforceVectorBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        target_value_r: torch.Tensor,
    ) -> None:
        """No-op: pure REINFORCE does not train the reward critic."""

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        target_value_c: torch.Tensor,
    ) -> None:
        """No-op: pure REINFORCE does not train the cost critic."""
