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
"""TRPO-PID with configurable per-signal REINFORCE advantage estimation."""

from __future__ import annotations

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.pid_lagrange.trpo_pid import TRPOPID
from omnisafe.common.buffer import VectorOnPolicyBuffer


class _SelectiveReinforceBuffer(VectorOnPolicyBuffer):
    """VectorOnPolicyBuffer that selectively zeroes bootstrap values per signal.

    For each signal (reward / cost) marked as REINFORCE, the bootstrap value is
    forced to zero so the return G_t = Σγ^k r_{t+k} is computed purely from
    observed rewards/costs, with no critic estimate bleeding in for truncated
    trajectories.  Signals not marked as REINFORCE pass through the real critic
    bootstrap value unchanged.
    """

    def __init__(self, *args, reinforce_reward: bool = True, reinforce_cost: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reinforce_reward = reinforce_reward
        self._reinforce_cost = reinforce_cost

    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_c: torch.Tensor | None = None,
        idx: int = 0,
    ) -> None:
        self.buffers[idx].finish_path(
            last_value_r=torch.zeros(1) if self._reinforce_reward else last_value_r,
            last_value_c=torch.zeros(1) if self._reinforce_cost else last_value_c,
        )


@registry.register
class TRPOPIDReinforce(TRPOPID):
    """TRPO-PID with configurable per-signal REINFORCE advantage estimation.

    Controlled by two ``algo_cfgs`` flags:

    - ``reinforce_reward`` (default ``True``): use raw discounted return
      ``G_t^R = Σγ^k r_{t+k}`` for the reward signal with no value baseline.
    - ``reinforce_cost`` (default ``True``): same for the cost signal.

    When a flag is ``True`` the corresponding critic is not trained and no
    bootstrap value is used at trajectory boundaries.  When ``False``, the
    critic is trained normally (via the parent :class:`TRPOPID` update) and the
    learned value function bootstraps truncated trajectories — equivalent to
    standard TRPO-PID for that signal.

    Note: when disabling REINFORCE for a signal (flag ``False``), also change
    ``adv_estimation_method`` from ``reinforce`` to ``gae`` (or another
    estimator) so the value baseline is actually used in the advantage
    computation.
    """

    def _init(self) -> None:
        super()._init()  # initialises PIDLagrangian + standard VectorOnPolicyBuffer
        reinforce_reward = getattr(self._cfgs.algo_cfgs, 'reinforce_reward', True)
        reinforce_cost = getattr(self._cfgs.algo_cfgs, 'reinforce_cost', True)
        self._reinforce_reward = reinforce_reward
        self._reinforce_cost = reinforce_cost
        self._buf = _SelectiveReinforceBuffer(
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
            reinforce_reward=reinforce_reward,
            reinforce_cost=reinforce_cost,
        )

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        target_value_r: torch.Tensor,
    ) -> None:
        if not self._reinforce_reward:
            super()._update_reward_critic(obs, target_value_r)

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        target_value_c: torch.Tensor,
    ) -> None:
        if not self._reinforce_cost:
            super()._update_cost_critic(obs, target_value_c)


# Backward-compatible alias so existing imports (e.g. CPOReinforce) still work.
_ReinforceVectorBuffer = _SelectiveReinforceBuffer
