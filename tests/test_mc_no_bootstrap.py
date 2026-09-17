"""The MC "true" value must be a pure simulated return, with no critic bootstrap folded in.

Run either way -- ``pytest tests/test_mc_no_bootstrap.py`` or ``python tests/test_mc_no_bootstrap.py``.

Why this matters: ``estimate_true_value_same_state_mc`` / ``estimate_value_from_snapshots``
produce the quantity every critic-calibration metric is scored against. If the tail of the
rollout is filled in with the critic's own ``V_c``, that "ground truth" inherits the critic's
artifacts -- most visibly its sign. A discounted cost-to-go is a sum of non-negative per-step
costs and therefore cannot be negative, but an untrained ``V_c`` happily predicts negative values.
Measured on a real 40-repeat study collected under the old default (``bootstrap_tail`` on,
``bootstrap_threshold=0.01``): **14.4% of per-rollout cost returns were negative**, along with 84
of 1350 probe means -- so the pipeline was routinely scoring the critic against impossible
targets.

These tests pin the two halves of the fix:

* the MC return accumulates simulated cost only, so non-negative costs give non-negative returns;
* truncating the rollout without bootstrapping is refused rather than silently dropping the tail.

Both run against a stubbed agent/env rather than a real Safety-Gymnasium rollout, so they are
fast and deterministic; ``tests/test_eval_config_parity.py`` covers the config wiring separately.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from omnisafe.utils.value_eval import (
    _effective_rollout_horizon,
    estimate_true_value_same_state_mc,
)


HORIZON = 40
N_ENVS = 2


class _StubAgent:
    """Critic that always predicts a large NEGATIVE cost value.

    That is the pathological case the fix is about: if the tail is bootstrapped, this value is
    added to the return and drives it negative no matter what the simulated costs were.
    """

    def step(self, obs):  # noqa: D102
        n = obs.shape[0] if obs.ndim > 1 else 1
        return (torch.zeros(n, 1), torch.full((n,), -50.0), torch.full((n,), -50.0),
                torch.zeros(n))


class _StubEnv:
    """Fixed-length episode emitting a constant non-negative cost every step."""

    COST_PER_STEP = 1.0

    def __init__(self, num_envs: int = N_ENVS, max_episode_steps: int = HORIZON) -> None:
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self._t = 0

    def reset(self, seed=None, options=None):  # noqa: D102, ARG002
        self._t = 0
        return torch.zeros(self.num_envs, 3), {}

    def step(self, act):  # noqa: D102, ARG002
        self._t += 1
        done = self._t >= self.max_episode_steps
        flag = torch.full((self.num_envs,), bool(done))
        return (torch.zeros(self.num_envs, 3),
                torch.zeros(self.num_envs),
                torch.full((self.num_envs,), self.COST_PER_STEP),
                torch.zeros(self.num_envs, dtype=torch.bool),
                flag, {})


class _Cfgs:
    class train_cfgs:  # noqa: N801
        device = 'cpu'

    class algo_cfgs:  # noqa: N801
        adv_estimation_method = 'gae'
        cost_adv_estimation_method = None
        lam = 0.95
        lam_c = 0.95
        penalty_coef = 0.0


def _run(**kwargs):
    return estimate_true_value_same_state_mc(
        agent=_StubAgent(), env=_StubEnv(), cfgs=_Cfgs(),
        discount_r=0.99, discount_c=0.99, probe_seeds=[1, 2],
        mc_repeats=2, max_episode_steps=HORIZON, return_raw=True, **kwargs,
    )


def test_cost_returns_are_non_negative_by_default() -> None:
    """The headline invariant: non-negative per-step cost -> non-negative cost-to-go."""
    _, raw = _run()
    returns = np.array(raw['c']['returns'])
    assert (returns >= 0).all(), (
        f'negative cost-to-go with the default (no-bootstrap) settings: min={returns.min()}'
    )


def test_cost_return_matches_the_exact_discounted_sum() -> None:
    """Not merely non-negative -- exactly the geometric sum, i.e. nothing extra folded in."""
    _, raw = _run()
    expected = _StubEnv.COST_PER_STEP * (1 - 0.99 ** HORIZON) / (1 - 0.99)
    returns = np.array(raw['c']['returns'])
    assert np.allclose(returns, expected, rtol=1e-6), (
        f'expected every cost return to be {expected}, got {np.unique(returns)}'
    )


def test_bootstrap_tail_reintroduces_the_negative_value() -> None:
    """The old behaviour is still reachable, and still produces the impossible value.

    Guards against the fix being a no-op: if this passes while the test above also passes, the
    flag genuinely controls what it claims to.
    """
    horizon = _effective_rollout_horizon(HORIZON, 0.99, 0.99, 0.9)
    assert horizon < HORIZON, 'threshold too weak to truncate; pick a larger one'
    _, raw = _run(bootstrap_threshold=0.9, bootstrap_tail=True)
    returns = np.array(raw['c']['returns'])
    assert (returns < 0).any(), 'bootstrap_tail=True should fold the negative V_c into the return'


def test_truncating_without_bootstrapping_is_refused() -> None:
    """Dropping the tail silently would bias every value low -- it must raise instead."""
    with pytest.raises(ValueError, match='bootstrap_tail=False'):
        _run(bootstrap_threshold=0.9, bootstrap_tail=False)


if __name__ == '__main__':
    raise SystemExit(pytest.main([os.path.abspath(__file__), '-q']))
