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
"""OnPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch
from rich.progress import track

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config


class OnPolicyAdapter(OnlineAdapter):
    """OnPolicy Adapter for OmniSafe.

    :class:`OnPolicyAdapter` is used to adapt the environment to the on-policy training.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._epoch_cost_sum: float = 0.0
        self._finite_horizon: bool = getattr(
            getattr(cfgs.model_cfgs, 'critic', object()), 'finite_horizon', False,
        )
        if self._finite_horizon:
            assert self._max_ep_len is not None, (
                'finite_horizon requires the environment to define max_episode_steps.'
            )
        self._reset_log()

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor , reward critic
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        self._reset_log()
        self._epoch_cost_sum = 0.0

        obs, _ = self.reset()
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            # Remaining timesteps-to-go H - t for the current obs (t = steps already taken this
            # episode, held in _ep_len before _log_value increments it). No-op when finite_horizon
            # is off (remaining is ignored by the critic).
            remaining = self._remaining(obs) if self._finite_horizon else None

            act, value_r, value_c, logp = agent.step(obs, remaining=remaining)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
                remaining_horizon=(
                    remaining if remaining is not None
                    else torch.zeros(self._env.num_envs, device=obs.device)
                ),
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        f'\nWarning: trajectory cut off when rollout by epoch\
                            in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.',
                    )

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        # Bootstrap value for the state after the last stored transition. For a
                        # finite-horizon critic, a true timeout means the horizon was reached, so
                        # there is no future reward within the horizon and the bootstrap stays 0.
                        if self._finite_horizon and time_out:
                            pass  # keep zeros: remaining horizon at the boundary is 0
                        elif epoch_end and not time_out:
                            # Episode cut mid-way by the epoch boundary: bootstrap from next obs.
                            # After _log_value, _ep_len == t+1, so remaining = H - (t+1).
                            boot_rem = (
                                self._remaining(obs[idx], idx) if self._finite_horizon else None
                            )
                            _, last_value_r, last_value_c, _ = agent.step(
                                obs[idx], remaining=boot_rem,
                            )
                            last_value_r = last_value_r.unsqueeze(0)
                            last_value_c = last_value_c.unsqueeze(0)
                        elif time_out:
                            # Standard (non-finite-horizon) timeout: bootstrap from final obs.
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx],
                            )
                            last_value_r = last_value_r.unsqueeze(0)
                            last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0

                    buffer.finish_path(last_value_r, last_value_c, idx)

    def _remaining(
        self,
        obs: torch.Tensor,
        idx: int | None = None,
    ) -> torch.Tensor:
        """Remaining timesteps-to-go ``H - t`` for a finite-horizon critic.

        ``self._ep_len`` holds the number of steps already taken in each episode, so
        ``H - _ep_len`` is the remaining horizon. Call before ``_log_value`` (which increments
        ``_ep_len``) for the current observation; call after it for a bootstrap (next) observation.

        Args:
            obs: Observation, only used to place the result on the right device.
            idx: If given, return the scalar remaining horizon for that single environment;
                otherwise return the ``(num_envs,)`` vector.

        Returns:
            Remaining timesteps-to-go as a raw count (float tensor).
        """
        ep_len = self._ep_len if idx is None else self._ep_len[idx]
        return (self._max_ep_len - ep_len).to(obs.device)

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += 1
        self._epoch_cost_sum += info.get('original_cost', cost).sum().item()

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        if hasattr(self._env, 'spec_log'):
            self._env.spec_log(logger)
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0
