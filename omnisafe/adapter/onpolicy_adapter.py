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
        # Every episode's own cost, for this epoch only -- a tail statistic over the
        # logger's rolling window would select the worst episodes of several epochs ago.
        self._epoch_ep_costs: list[float] = []
        self._reset_log()

    @property
    def _sr_td_ridge(self) -> bool:
        """Whether the ``td_ridge`` successor-representation mode is active.

        .. note::
            Computed on the fly from ``self._cfgs`` (set by :class:`OnlineAdapter`) rather than
            cached at ``__init__`` time, since some subclasses (e.g. :class:`SimmerAdapter`)
            deliberately skip :class:`OnPolicyAdapter`'s own ``__init__`` body via
            ``super(OnPolicyAdapter, self).__init__(...)``.
        """
        use_sr = bool(self._cfgs.model_cfgs.get('use_successor_representation', False))
        return use_sr and self._cfgs.model_cfgs.sr_cfgs.get('sr_mode', 'shared_trunk') == 'td_ridge'

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
        self._epoch_cost_sum = 0.0
        self._epoch_ep_costs = []

        if self._persistent_rollout and getattr(self, '_persist_obs', None) is not None:
            # Continue every env's episode where the last epoch left it (its cut-off segment was
            # already bootstrapped with the critic at epoch end, below).
            obs = self._persist_obs
        else:
            self._reset_log()
            obs, _ = self.reset()
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            act, value_r, value_c, logp = agent.step(obs)
            if self._sr_td_ridge:
                phi, psi = agent.sr_features(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            store_kwargs = {
                'obs': obs,
                'act': act,
                'reward': reward,
                'cost': cost,
                'value_r': value_r,
                'value_c': value_c,
                'logp': logp,
            }
            if self._sr_td_ridge:
                store_kwargs.update(phi=phi, psi=psi)
            buffer.store(**store_kwargs)

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        f'\nWarning: trajectory cut off when rollout by epoch\
                            in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.',
                    )

            # One host transfer per step instead of one sync per env and flag when the env
            # lives on the GPU (identical values; a no-op for CPU tensors).
            for idx, (done, time_out) in enumerate(zip(terminated.tolist(), truncated.tolist())):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    last_psi = torch.zeros(agent.sr_trunk.sr_dim) if self._sr_td_ridge else None
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                            if self._sr_td_ridge:
                                _, last_psi = agent.sr_features(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx],
                            )
                            if self._sr_td_ridge:
                                _, last_psi = agent.sr_features(info['final_observation'][idx])
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0

                    if self._sr_td_ridge:
                        buffer.finish_path(last_value_r, last_value_c, idx, last_psi=last_psi)
                    else:
                        # avoid passing a new kwarg to buffer subclasses (e.g. MICE's
                        # _SelectiveReinforceBuffer) that override finish_path with a narrower
                        # signature and don't expect last_psi.
                        buffer.finish_path(last_value_r, last_value_c, idx)
        self._persist_obs = obs if self._persistent_rollout else None

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
        self._epoch_ep_costs.append(float(self._ep_cost[idx]))
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
            },
        )

    @property
    def _persistent_rollout(self) -> bool:
        """Whether episodes continue across epochs instead of every epoch starting with a reset.

        omnisafe resets all envs at the start of each epoch's rollout. That is harmless when each
        env runs several whole episodes per epoch (5 envs x 4000 steps), but with many parallel
        envs each one runs only ``steps_per_epoch / num_envs`` steps (20 at 1000 envs) and would
        never see past the start of an episode. ``algo_cfgs.persistent_rollout`` turns the reset
        into a one-off; left unset, it is on exactly when training runs on the GPU env.
        """
        flag = self._cfgs.algo_cfgs.get('persistent_rollout', None)
        if flag is not None:
            return bool(flag)
        env_cfgs = getattr(self._cfgs, 'env_cfgs', None)
        cls = env_cfgs.get('class_name') if env_cfgs is not None and hasattr(env_cfgs, 'get') else None
        return cls == 'SafetyGymnasiumGPUEnv'

    def invalidate_persistent_rollout(self) -> None:
        """Forget the carried-over state, e.g. after something else reset or stepped the env."""
        self._persist_obs = None

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
