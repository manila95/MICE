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
from torch import nn
from rich.progress import track

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.utils.risk import (
    RiskReplayBuffer,
    compute_fear,
    fear_to_binary,
    fear_to_quantile,
    quantile_bins,
)
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

    @property
    def _use_risk(self) -> bool:
        """Whether SRPL risk conditioning is active, read from cfgs like :attr:`_sr_td_ridge`."""
        model_cfgs = getattr(self._cfgs, 'model_cfgs', None)
        return bool(model_cfgs is not None and model_cfgs.get('use_risk', False))

    def _risk_setup(self, agent: ConstraintActorCritic) -> None:
        """Allocate the hindsight-label buffers this rollout will fill.

        ``_risk_costs``/``_risk_obs`` accumulate each environment's episode as it happens, since
        the label for a state -- how far it is from the nearest violation -- is only knowable
        once the episode has finished.
        """
        risk_cfgs = self._cfgs.model_cfgs.risk_cfgs
        self._risk_type = str(risk_cfgs.get('risk_type', 'quantile'))
        self._risk_size = int(agent._risk_size)  # pylint: disable=protected-access
        self._risk_bins = quantile_bins(
            int(risk_cfgs.get('quantile_num', 10)),
            int(risk_cfgs.get('quantile_size', 4)),
        )
        self._fear_radius = int(risk_cfgs.get('fear_radius', 5))
        self._fine_tune_risk = bool(risk_cfgs.get('fine_tune_risk', True))
        self._risk_update_period = int(risk_cfgs.get('risk_update_period', 10))
        self._risk_batch_size = int(risk_cfgs.get('risk_batch_size', 5000))
        if not hasattr(self, '_risk_replay'):
            obs_dim = int(self._env.observation_space.shape[0])
            self._risk_replay = RiskReplayBuffer(
                capacity=int(risk_cfgs.get('risk_buffer_size', 100_000)),
                obs_dim=obs_dim,
                risk_size=self._risk_size,
            )
            weight = float(risk_cfgs.get('risk_weight', 1.0))
            weights = torch.ones(self._risk_size)
            # The nearest-to-violation bin is the rare and the important one; the reference
            # up-weights it rather than letting the classifier collapse onto "far from harm".
            weights[0] = weight
            self._risk_criterion = nn.NLLLoss(weight=weights.to(self._device))
            self._risk_step = 0
        self._risk_obs = [[] for _ in range(self._env.num_envs)]
        self._risk_costs = [[] for _ in range(self._env.num_envs)]

    @torch.no_grad()
    def _risk_estimate(self, agent: ConstraintActorCritic, obs: torch.Tensor) -> torch.Tensor:
        """The estimator's current risk distribution for ``obs``, as probabilities.

        ``no_grad`` deliberately: the estimator is a predictor of where violations happen, and
        must not be shaped by the policy's objective. Exponentiated because the module returns
        log-probabilities (the reference does the same before feeding the policy).
        """
        flat = obs.reshape(-1, obs.shape[-1])
        return torch.exp(agent.risk_estimator(flat)).reshape(*obs.shape[:-1], self._risk_size)

    def _risk_finish_episode(self, idx: int) -> None:
        """Label one finished episode in hindsight and file it for training."""
        if not self._risk_costs[idx]:
            return
        costs = torch.stack(self._risk_costs[idx])
        obs = torch.stack(self._risk_obs[idx])
        fear = compute_fear(costs)
        labels = (
            fear_to_quantile(fear, self._risk_bins)
            if self._risk_type == 'quantile'
            else fear_to_binary(fear, self._fear_radius)
        )
        self._risk_replay.add(obs, labels)
        self._risk_obs[idx] = []
        self._risk_costs[idx] = []

    def _update_risk_model(self, agent: ConstraintActorCritic, logger: Logger) -> None:
        """One gradient step on the risk estimator, if it is due and there is data."""
        self._risk_step += 1
        if (
            not self._fine_tune_risk
            or len(self._risk_replay) == 0
            or self._risk_step % self._risk_update_period
        ):
            return
        obs, labels = self._risk_replay.sample(self._risk_batch_size)
        if obs.shape[0] < 2:
            # Batch norm cannot normalize a batch of one.
            return
        agent.risk_estimator.train()
        pred = agent.risk_estimator(obs.to(self._device))
        loss = self._risk_criterion(pred, torch.argmax(labels.to(self._device), dim=1))
        agent.risk_optimizer.zero_grad()
        loss.backward()
        agent.risk_optimizer.step()
        agent.risk_estimator.eval()
        logger.store({'Risk/Loss': loss.item()})

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
        if self._use_risk:
            self._risk_setup(agent)

        obs, _ = self.reset()
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            risk = self._risk_estimate(agent, obs) if self._use_risk else None
            act, value_r, value_c, logp = agent.step(obs, risk=risk)
            if self._sr_td_ridge:
                phi, psi = agent.sr_features(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)
            if self._use_risk:
                for env_idx in range(self._env.num_envs):
                    # Labelled against next_obs, as in the reference: the risk of a state is
                    # read from where the transition landed.
                    self._risk_obs[env_idx].append(next_obs[env_idx].detach().cpu())
                    self._risk_costs[env_idx].append(cost[env_idx].detach().cpu())
                self._update_risk_model(agent, logger)

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
            if self._use_risk:
                # Kept per step so the update re-conditions on the same risk the action was
                # taken under, rather than on a re-estimate from a since-updated estimator.
                store_kwargs['risk'] = risk
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

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    last_psi = torch.zeros(agent.sr_trunk.sr_dim) if self._sr_td_ridge else None
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, _ = agent.step(
                                obs[idx],
                                risk=self._risk_estimate(agent, obs[idx]) if self._use_risk else None,
                            )
                            if self._sr_td_ridge:
                                _, last_psi = agent.sr_features(obs[idx])
                        if time_out:
                            final_obs = info['final_observation'][idx]
                            _, last_value_r, last_value_c, _ = agent.step(
                                final_obs,
                                risk=self._risk_estimate(agent, final_obs) if self._use_risk else None,
                            )
                            if self._sr_td_ridge:
                                _, last_psi = agent.sr_features(info['final_observation'][idx])
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        if self._use_risk:
                            self._risk_finish_episode(idx)
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
