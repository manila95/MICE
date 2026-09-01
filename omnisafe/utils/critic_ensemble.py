"""Critic-ensemble bias correction: combine several independently-trained critics into one
pessimistic (reward) or conservative (cost) prediction.

Three techniques, adapted from off-policy Q-learning to this repo's on-policy V-critic setting:

- **CDQ** (Clipped Double Q-learning, Fujimoto et al. 2018 / TD3): a fixed rule -- take the min
  of an ensemble (pessimism) or the max (conservatism). No learned/adaptive state.
- **GPL** (Generalized Pessimism Learning, Cetin & Celiktutan 2023): a *learned* pessimism
  coefficient rather than a fixed rule.
- **TOP** (Tactical Optimism/Pessimism, Moskovitz et al. 2021): adaptively selects an
  optimism/pessimism level from a discrete set via a bandit, rather than committing to one level
  a priori (as CDQ does) or learning one continuously (as GPL does).

Adaptation note (read before comparing results against the original papers): all three papers
were built for off-policy, replay-buffer, bootstrapped-max Q-learning (TD3/SAC-family), where the
"exploitable overestimation" mechanism is a literal max_a' Q(s', a') operator in the bootstrap
target. On-policy V-critics here have no such operator -- the policy exploits the critic's noise
more softly, through advantage-weighted policy-gradient reweighting, not an explicit max. CDQ's
fixed min/max rule ports over exactly (it only assumes an ensemble + an aggregation rule, nothing
Q-learning-specific). GPL and TOP are reimplemented here in the same *spirit* -- a learned,
respectively bandit-selected, degree of pessimism/conservatism drawn from a
mean +/- beta * std parametrization of the ensemble -- using this codebase's own on-policy
training signal (the epoch's realized Metrics/EpRet or Metrics/EpCost) as the outcome feedback,
rather than the papers' own precise (off-policy-specific) derivations, which do not have a direct
on-policy analogue. This is a faithful-in-spirit adaptation, not a line-for-line reproduction --
treat GPL/TOP results here as "the same design idea, implemented for this setting," not as a
literal replication of the papers' reported numbers.

The reward/cost asymmetry (the actual point of all three): the reward critic is exploited by
the policy chasing HIGH estimates, so its aggregation is biased toward the ensemble's low end
(pessimism). The cost critic is exploited by the policy (via the Lagrange penalty or trust-region
constraint) chasing LOW estimates -- underestimating cost makes an unsafe action look
constraint-satisfying -- so its aggregation is biased the *opposite* way, toward the ensemble's
high end (conservatism). This module bakes that sign flip in via the ``stream`` argument
('r' vs 'c') rather than leaving it as an independently-configurable choice, since getting it
backwards defeats the entire purpose (see the module using this, model_cfgs/algo_cfgs' shared,
non-"_cost"-overridable critic_ensemble_method).
"""

from __future__ import annotations

import math
import random

import torch


def aggregate(raw: torch.Tensor, method: str, stream: str, beta: float) -> torch.Tensor:
    """Combine an ensemble's raw per-member predictions into one aggregated prediction.

    Args:
        raw: ``(num_critics, B)`` stacked raw predictions, one row per ensemble member.
        method: ``'none'`` (no correction -- just the first member, for parity with the
            single-critic baseline), ``'cdq'``, ``'gpl'``, or ``'top'``.
        stream: ``'r'`` (reward -- biased toward the ensemble's low end) or ``'c'`` (cost --
            biased toward the high end). See the module docstring for why these directions are
            opposite and not independently configurable.
        beta: The current pessimism/conservatism coefficient (only read under ``'gpl'``/``'top'``;
            ignored under ``'none'``/``'cdq'``, which don't have one).

    Returns:
        ``(B,)`` aggregated prediction.
    """
    assert stream in ('r', 'c'), f"stream must be 'r' or 'c', got {stream!r}"
    if method == 'none' or raw.shape[0] == 1:
        return raw[0]
    if method == 'cdq':
        # Reward: pessimistic -> the lower of the ensemble's disagreeing estimates.
        # Cost: conservative -> the higher of the ensemble's disagreeing estimates.
        return raw.min(dim=0).values if stream == 'r' else raw.max(dim=0).values
    if method in ('gpl', 'top'):
        mean = raw.mean(dim=0)
        std = raw.std(dim=0, unbiased=raw.shape[0] > 1)
        sign = -1.0 if stream == 'r' else 1.0
        return mean + sign * float(beta) * std
    raise ValueError(f"unknown critic_ensemble_method {method!r}, expected 'none'/'cdq'/'gpl'/'top'")


class GPLBetaAdapter:
    """Adapts a single scalar pessimism/conservatism coefficient via a simple hill-climbing
    (perturb-and-observe / SPSA-style) rule, driven by the epoch's realized training outcome.

    Why an outcome-driven rule rather than fitting beta to minimize the critic's own regression
    loss: pessimism is a *trade-off* -- shrinking toward the ensemble's low (or high) end always
    costs some in-sample fit accuracy in exchange for robustness against the exploited direction
    of error. A rule that adapts beta to minimize the same in-sample loss the critics themselves
    minimize would simply drive beta -> 0 (the unshrunk ensemble mean always fits its own targets
    best), defeating the entire purpose. So beta must respond to something the critic's own loss
    doesn't see: here, the realized epoch-level training outcome (Metrics/EpRet for reward,
    -Metrics/EpCost for cost -- see PolicyGradient._update_critic_ensemble_beta).

    One step of lag is inherent: the outcome fed into a given ``step()`` call reflects the
    rollout collected under the *previous* epoch's beta (that beta shaped the previous epoch's
    critic training and policy update, which shaped this epoch's rollout) -- standard for this
    kind of online hyperparameter adaptation, not a bug.
    """

    def __init__(self, beta_init: float = 0.0, lr: float = 0.05, beta_max: float = 3.0) -> None:
        self.beta = float(beta_init)
        self.lr = float(lr)
        self.beta_max = float(beta_max)
        self._prev_outcome: float | None = None
        self._prev_delta: float = self.lr

    def step(self, outcome: float) -> float:
        """Update ``beta`` given this epoch's realized outcome; returns the new beta."""
        if self._prev_outcome is not None:
            # Kept improving -> keep moving beta the same direction as last time.
            # Got worse -> reverse direction (classic hill-climbing / coordinate-ascent step).
            delta = self._prev_delta if outcome > self._prev_outcome else -self._prev_delta
            self.beta = min(max(self.beta + delta, 0.0), self.beta_max)
            self._prev_delta = delta
        self._prev_outcome = outcome
        return self.beta


class TOPBanditAdapter:
    """A K-armed bandit over a discrete grid of pessimism/conservatism coefficients, selecting
    the arm with the best recent realized-outcome track record (softmax over running per-arm
    outcome averages) rather than committing to one fixed level (CDQ) or adapting one
    continuously (GPL).

    Same outcome-driven, one-epoch-lagged feedback as :class:`GPLBetaAdapter` -- see its
    docstring for why the training outcome, not the critic's own fit loss, is the right signal.

    Temperature decays exponentially from ``temperature`` toward ``temperature_min`` over
    ``decay_steps`` calls to :meth:`select` -- found empirically necessary, not just a nicety: a
    *fixed* low temperature explores too little and can converge on a middling arm before ever
    sampling the best one enough times to reveal it (measured: at a fixed temperature=0.3 over
    1000 rounds, the bandit locked onto a clearly-suboptimal arm in a synthetic 4-arm test with
    well-separated true means); a fixed high temperature converges reliably but slowly (needed
    ~2000+ rounds in that same test). On-policy training here only gets on the order of a few
    hundred epochs total (one bandit round per epoch -- see
    PolicyGradient._update_critic_ensemble_beta), so exploration has to front-load into that
    budget rather than assume thousands of rounds are available. ``decay_steps`` should be set to
    roughly the run's total epoch count.
    """

    def __init__(
        self,
        beta_grid: list[float],
        bandit_lr: float = 0.1,
        temperature: float = 2.0,
        temperature_min: float = 0.2,
        decay_steps: int = 300,
        seed: int | None = None,
    ) -> None:
        assert len(beta_grid) >= 2, 'TOPBanditAdapter needs at least 2 arms to select among'
        self.beta_grid = list(beta_grid)
        self.bandit_lr = float(bandit_lr)
        self.temperature = float(temperature)
        self.temperature_min = float(temperature_min)
        self.decay_steps = max(int(decay_steps), 1)
        self.q = [0.0] * len(self.beta_grid)
        self.counts = [0] * len(self.beta_grid)
        self.last_arm: int | None = None
        self._rng = random.Random(seed)
        self._n_calls = 0

    def _current_temperature(self) -> float:
        frac = min(self._n_calls / self.decay_steps, 1.0)
        # Exponential interpolation (not linear): spends relatively more of the schedule at
        # higher temperatures, since early rounds -- before any arm's Q estimate is reliable --
        # benefit most from broad exploration; a linear schedule cools too aggressively too soon.
        ratio = self.temperature_min / self.temperature
        return self.temperature * (ratio**frac)

    def select(self) -> float:
        """Pick this epoch's arm (softmax over current per-arm value estimates); returns its beta."""
        self._n_calls += 1
        temperature = self._current_temperature()
        m = max(self.q)
        exps = [math.exp((v - m) / max(temperature, 1e-6)) for v in self.q]
        z = sum(exps)
        probs = [e / z for e in exps]
        draw = self._rng.random()
        cum = 0.0
        arm = len(probs) - 1  # fallback for floating-point roundoff at the tail
        for i, p in enumerate(probs):
            cum += p
            if draw <= cum:
                arm = i
                break
        self.last_arm = arm
        return self.beta_grid[arm]

    def update(self, outcome: float) -> None:
        """Credit the arm selected by the most recent ``select()`` call with this outcome."""
        if self.last_arm is None:
            return
        i = self.last_arm
        self.counts[i] += 1
        self.q[i] += self.bandit_lr * (outcome - self.q[i])
