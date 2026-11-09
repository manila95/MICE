"""Worst-Case Policy Gradients: a tail constraint instead of a mean one."""

from __future__ import annotations

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag
from omnisafe.models.critic.quantile_v_critic import QuantileVCritic


@registry.register
class WCPG(PPOLag):
    r"""Worst-Case Policy Gradients, in the constrained setting.

    The usual Lagrangian baseline constrains the *mean* episodic cost, which says nothing about
    how bad the bad episodes are: a policy that is safe on average and catastrophic one run in
    ten satisfies it. This constrains the upper tail instead,

    .. math:: \max_\pi \; \mathbb{E}[R] \quad\text{s.t.}\quad \mathrm{CVaR}_\alpha(C) \le d

    where :math:`\mathrm{CVaR}_\alpha` is the mean of the worst :math:`\alpha` fraction. After
    Tang et al., *Worst Cases Policy Gradients* (CoRL 2019), moved to the cost side: that paper
    maximizes the CVaR of the return in an unconstrained problem, whereas here the reward
    objective is untouched and the tail enters the constraint, which is what makes it comparable
    to the other constrained baselines in this codebase.

    Two things change relative to :class:`PPOLag`:

    - **The measured constraint** is the empirical CVaR of the episodic costs, not their mean --
      see :meth:`_constraint_value`. This is the quantity the multiplier is driven by, and it is
      what ``cost_limit`` now means.
    - **The cost critic is distributional**, and reports its upper tail rather than its mean, so
      the cost advantages the policy is pushed by describe tail-cost-to-go. This requires
      ``model_cfgs.critic.num_quantiles_cost`` (or ``num_quantiles``) to be set.

    .. warning::
        A CVaR value function is not a value function in the Bellman sense -- CVaR does not obey
        the expectation recursion that GAE's TD residuals assume -- so the cost advantages here
        are a heuristic, in the same way the original paper's are. It is also why the
        Monte-Carlo value studies cannot score this critic as they stand: they compare against a
        sample *mean* return, which is not what this critic predicts. A tail reference is needed
        for that, and is deliberately left to its own piece of work.
    """

    def _init_model(self) -> None:
        """Build the actor-critic, then point the cost critic at its upper tail."""
        super()._init_model()
        critic = self._actor_critic.cost_critic
        if not isinstance(critic, QuantileVCritic):
            raise ValueError(
                'WCPG constrains the tail of the cost return, so it needs a distributional cost '
                'critic: set model_cfgs.critic.num_quantiles_cost (or num_quantiles) to a '
                'positive number. Got a scalar cost critic, which has no tail to constrain.',
            )
        alpha = float(self._cfgs.algo_cfgs.cvar_alpha)
        if not 0 < alpha <= 1:
            raise ValueError(f'algo_cfgs.cvar_alpha must be in (0, 1], got {alpha}.')
        self._cvar_alpha: float = alpha
        # The critic's own forward() now reports the tail, so every consumer of cost values --
        # the rollout's value_c, the GAE that builds adv_c, the actor's cost surrogate -- works
        # on the tail without any of them needing to know.
        critic._risk_measure = 'cvar'  # pylint: disable=protected-access
        critic._risk_alpha = alpha  # pylint: disable=protected-access

    def _init_log(self) -> None:
        super()._init_log()
        # Both are logged so the gap between them is visible: the tail is what is constrained,
        # the mean is what every other baseline reports, and comparing runs across the two
        # requires seeing both.
        self._logger.register_key('Metrics/EpCostCVaR')
        if getattr(self._cfgs.algo_cfgs, 'eval_critic', False):
            # Said once, loudly: the studies would otherwise report a plausible-looking
            # correlation for a quantity the critic is not predicting.
            self._logger.log(
                'WARNING: the value-evaluation studies score the cost critic against a '
                'Monte-Carlo sample *mean*, but under WCPG that critic predicts the upper '
                f'{self._cvar_alpha:.0%} tail. Its cost-side numbers (MCStudy/*_c, '
                'IntermediateMC/*_c, PooledMC/*_c) compare two different quantities and are '
                'not meaningful here; the reward side is unaffected. Read the *Tail '
                'counterparts instead -- MCStudyTail/, IntermediateMCTail/, PooledMCTail/ -- '
                'which score the same predictions against the Monte-Carlo tail.',
            )

    def _constraint_value(self) -> float:
        """The empirical CVaR of this epoch's episodic costs.

        This epoch's episodes, not the logger's rolling window. The window spans several epochs
        (``logger_cfgs.window_lens`` episodes against ``steps_per_epoch`` worth per epoch), and
        a tail taken over it selects precisely the worst episodes of the *oldest*, least-trained
        policy in the window -- so the measured constraint would barely move as the policy
        improved. A mean survives that mixing far better than a tail does: one bad episode in a
        hundred moves the mean by a percent and the worst-ten tail by a tenth.
        ``cvar_window: logger`` opts back into the windowed version, for comparing like with
        like against the mean-constrained baselines.

        With fewer than ``1 / alpha`` episodes the tail is a single episode -- noisy, and worth
        remembering when reading early epochs or short runs.
        """
        if str(getattr(self._cfgs.algo_cfgs, 'cvar_window', 'epoch')) == 'logger':
            samples = self._logger.get_values('Metrics/EpCost')
        else:
            samples = list(self._env._epoch_ep_costs)  # pylint: disable=protected-access
        costs = np.asarray(samples, dtype=np.float64)
        if costs.size == 0:
            # No episode finished this epoch; leave the multiplier where it is rather than
            # driving it with a statistic of nothing.
            cvar = float(self._logger.get_stats('Metrics/EpCost')[0])
        else:
            k = max(1, int(round(self._cvar_alpha * costs.size)))
            cvar = float(np.sort(costs)[-k:].mean())
        self._logger.store({'Metrics/EpCostCVaR': cvar})
        return cvar
