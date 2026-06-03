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
"""SACPID with REINFORCE-style cost signal (no cost critic)."""

from __future__ import annotations

import time

import torch
from tqdm import tqdm

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac_pid import SACPID


@registry.register
class SACPIDReinforce(SACPID):
    """SACPID with REINFORCE-style cost signal in the actor loss.

    Replaces ``Q^C(s, π(s))`` with a policy-gradient estimate of the cost
    using actual observed costs from the replay buffer:

    .. math::

        L_c = \\lambda \\cdot c \\cdot \\log \\pi_\\theta(a_{\\text{stored}} | s)

    Gradient descent on this loss decreases the log-probability of transitions
    that incurred non-zero cost without relying on a learned cost critic.

    Two differences from :class:`SACPID`:

    1. **No cost critic** — :meth:`_update_cost_critic` is a no-op; the cost
       critic network is kept in the architecture but never updated.
    2. **REINFORCE cost gradient** — the actor loss uses
       ``λ * cost * log π(a_stored | s)`` instead of ``λ * Q^C(s, π(s))``.
    """

    def learn(self):
        """Training loop with per-epoch progress bar."""
        self._logger.log('INFO: Start training')
        start_time = time.time()
        step = 0
        for epoch in range(self._epochs):
            self._epoch = epoch
            rollout_time = 0.0
            update_time = 0.0
            epoch_time = time.time()

            pbar = tqdm(
                range(epoch * self._samples_per_epoch, (epoch + 1) * self._samples_per_epoch),
                desc=f'Epoch {epoch}',
                unit='step',
                leave=False,
            )
            for sample_step in pbar:
                step = sample_step * self._update_cycle * self._cfgs.train_cfgs.vector_env_nums

                if self._cfgs.algo_cfgs.use_exploration_noise:
                    self._actor_critic.actor.noise = self._cfgs.algo_cfgs.exploration_noise

                rollout_start = time.time()
                self._env.rollout(
                    rollout_step=self._update_cycle,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                    use_rand_action=(step <= self._cfgs.algo_cfgs.start_learning_steps),
                )
                rollout_time += time.time() - rollout_start

                update_start = time.time()
                if step > self._cfgs.algo_cfgs.start_learning_steps:
                    self._update()
                else:
                    self._log_when_not_update()
                    pbar.set_postfix(phase='warmup')
                update_time += time.time() - update_start

            eval_start = time.time()
            self._env.eval_policy(
                episode=self._cfgs.train_cfgs.eval_episodes,
                agent=self._actor_critic,
                logger=self._logger,
            )
            eval_time = time.time() - eval_start

            self._logger.store({'Time/Update': update_time})
            self._logger.store({'Time/Rollout': rollout_time})
            self._logger.store({'Time/Evaluate': eval_time})

            if (
                step > self._cfgs.algo_cfgs.start_learning_steps
                and self._cfgs.model_cfgs.linear_lr_decay
            ):
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': step + 1,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': self._actor_critic.actor_scheduler.get_last_lr()[0],
                },
            )

            self._logger.dump_tabular()

            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()
        self._env.close()
        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
            )

            self._update_reward_critic(obs, act, reward, done, next_obs)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)  # no-op

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                # Store for use in _loss_pi without changing parent signatures.
                self._reinforce_act = act
                self._reinforce_cost = cost
                self._update_actor(obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
            self._lagrange.pid_update(Jc)
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    def _loss_pi(self, obs: torch.Tensor) -> torch.Tensor:
        r"""Actor loss with REINFORCE cost signal.

        .. math::

            L = \frac{1}{1+\lambda} \left[
                \alpha \log \pi(a'|s) - Q^R(s, a')
                + \lambda \cdot c \cdot \log \pi(a_{\text{stored}} | s)
            \right]

        where :math:`a' \sim \pi(\cdot|s)` is a freshly sampled action used
        for the reward reparameterization gradient, and :math:`a_{\text{stored}}`
        is the replay-buffer action associated with observed cost :math:`c`.

        Args:
            obs (torch.Tensor): Observations sampled from the replay buffer.

        Returns:
            Scalar actor loss.
        """
        act = self._reinforce_act
        cost = self._reinforce_cost

        # Reward term: reparameterization gradient through Q^R (same as SAC/SACPID).
        action_new = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob_new = self._actor_critic.actor.log_prob(action_new)
        q1_r, q2_r = self._actor_critic.reward_critic(obs, action_new)
        loss_r = self._alpha * log_prob_new - torch.min(q1_r, q2_r)

        # Cost term: REINFORCE — evaluate stored action under current policy.
        # forward() sets _current_dist without storing a raw sample, so the
        # subsequent log_prob(act) call evaluates act via TanhNormal.log_prob.
        self._actor_critic.actor.forward(obs)
        log_prob_stored = self._actor_critic.actor.log_prob(act)
        loss_c = self._lagrange.lagrangian_multiplier * cost * log_prob_stored

        return (loss_r + loss_c).mean() / (1 + self._lagrange.lagrangian_multiplier)

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """No-op: REINFORCE does not train the cost critic."""
        self._logger.store(
            {
                'Loss/Loss_cost_critic': 0.0,
                'Value/cost_critic': 0.0,
            },
        )

    def _log_when_not_update(self) -> None:
        super()._log_when_not_update()
        self._logger.store(
            {
                'Loss/Loss_cost_critic': 0.0,
                'Value/cost_critic': 0.0,
            },
        )
