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
"""Implementation of the Lagrangian version of Soft Actor-Critic algorithm."""


import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.common.lagrange import Lagrange


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SACLag(SAC):
    """The Lagrangian version of Soft Actor-Critic (SAC) algorithm.

    References:
        - Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
        - Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.
        - URL: `SAC <https://arxiv.org/abs/1801.01290>`_
    """

    def _init(self) -> None:
        """The initialization of the algorithm.

        Here we additionally initialize the Lagrange multiplier.
        """
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        """Log the SACLag specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')

    def _anneal_cost_limit(self) -> None:
        """Tighten the cost limit over training instead of ramping the multiplier.

        Driving lambda up is a blunt way to buy safety: the multiplier scales the whole
        cost term at once, and past roughly 0.5 on SafetyPointGoal1-v0 the policy stops
        moving rather than learning to move safely. The multiplier is also only able to
        *fall* once cost is under the limit, so a limit the policy cannot yet meet makes
        lambda grow without bound.

        Annealing the limit inverts that. The run starts with a limit the policy can
        actually satisfy, so lambda stays small and reward forms; the limit then tightens
        toward its target, keeping the constraint live but never far out of reach, so
        lambda tracks it instead of ramping. ``cost_limit_start`` is where it begins and
        ``cost_limit_anneal_epochs`` how long it takes to reach the configured
        ``cost_limit``; leaving ``cost_limit_start`` unset keeps the limit fixed.
        """
        cfgs = self._cfgs.algo_cfgs
        start = getattr(cfgs, 'cost_limit_start', 0.0)
        span = getattr(cfgs, 'cost_limit_anneal_epochs', 0)
        if start <= 0.0 or span <= 0:
            return
        target = self._cfgs.lagrange_cfgs.cost_limit
        frac = min(1.0, self._epoch / span)
        self._lagrange.cost_limit = start + frac * (target - start)

    def _lambda_signal(self, Jc: float) -> float | None:
        """The cost estimate the multiplier is actually driven by.

        ``update_cycle: 1`` means this runs once per environment *step*, so the multiplier
        integrates 2000 times per epoch against a ``Metrics/EpCost`` that only refreshes when
        episodes finish -- on a 1000-step task that is a two-episode average. The multiplier
        therefore chases sampling noise and keeps accumulating right up to the moment cost
        crosses the limit, overshooting by the time it gets there.

        Two knobs, both no-ops at their defaults:

        ``lambda_update_every`` skips all but every n-th update, so the multiplier moves on a
        coarser clock (``n = steps_per_epoch`` makes it once per epoch). ``lambda_jc_ema``
        smooths ``Jc`` with an exponential moving average first, so the multiplier responds to
        the trend rather than to which two episodes happened to land in this window.

        Returns:
            The (possibly smoothed) cost estimate, or ``None`` when this call falls on a
            skipped tick. ``None`` means *do not step the optimizer at all*: feeding it the
            cost limit instead would zero the gradient but Adam would still move on stale
            momentum, which is a leaky integrator rather than a skip.
        """
        cfgs = self._cfgs.algo_cfgs
        ema = getattr(cfgs, 'lambda_jc_ema', 0.0)
        if ema > 0.0:
            prev = getattr(self, '_jc_ema', None)
            Jc = Jc if prev is None else ema * prev + (1.0 - ema) * Jc
            self._jc_ema = Jc

        every = getattr(cfgs, 'lambda_update_every', 1)
        if every > 1:
            self._lambda_tick = getattr(self, '_lambda_tick', 0) + 1
            if self._lambda_tick % every:
                return None
        return Jc

    def _update(self) -> None:
        """Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.
        """
        super()._update()
        self._anneal_cost_limit()
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
            signal = self._lambda_signal(Jc)
            if signal is not None:
                self._lagrange.update_lagrange_multiplier(signal)
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing ``pi/actor`` loss.

        The loss function in SACLag is defined as:

        .. math::

            L = -Q^V (s, \pi (s)) + \lambda Q^C (s, \pi (s))

        where :math:`Q^V` is the min value of two reward critic networks outputs, :math:`Q^C` is the
        value of cost critic network, and :math:`\pi` is the policy network.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        loss_q_r_1, loss_q_r_2 = self._actor_critic.reward_critic(obs, action)
        loss_r = self._alpha * log_prob - torch.min(loss_q_r_1, loss_q_r_2)
        loss_q_c = self._actor_critic.cost_critic(obs, action)[0]
        loss_c = self._lagrange.lagrangian_multiplier.item() * loss_q_c

        return (loss_r + loss_c).mean() / (1 + self._lagrange.lagrangian_multiplier.item())

    def _log_when_not_update(self) -> None:
        super()._log_when_not_update()
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )
