"""Correctness tests for MICE's algo_cfgs.no_intrinsic_in_ep_costs.

Run either way -- ``pytest tests/test_mice_ep_costs_bias.py`` or
``python tests/test_mice_ep_costs_bias.py``.

Replicates the small arithmetic snippet from ``MICE._update_actor`` under test (the same
"bare-instance, stub logger" approach ``tests/test_cost_bias.py`` uses for CPO's matching
``use_cost_bias`` snippet) rather than invoking the real method, which needs a full actor-critic,
gradients, etc. to run at all -- this isolates exactly the two lines the new flag changes.
"""

from __future__ import annotations

import inspect
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnisafe.utils.config import Config  # noqa: E402


class _StubLogger:
    def __init__(self, ep_cost: float) -> None:
        self._ep_cost = ep_cost
        self.stored: dict[str, float] = {}

    def get_stats(self, key: str) -> tuple[float]:
        assert key == 'Metrics/EpCost'
        return (self._ep_cost,)

    def store(self, data: dict) -> None:
        self.stored.update(data)


def _ep_costs_via_mice(
    no_intrinsic_in_ep_costs: bool,
    ep_discount_ci_value: float,
    cost_limit: float,
    ep_cost: float,
) -> tuple[float, dict]:
    """Replicates exactly the lines under test from mice.py's _update_actor.

    Args:
        no_intrinsic_in_ep_costs: The new flag.
        ep_discount_ci_value: What ``balancing_ep_dicount_ci.mean().item()`` would have been.
        cost_limit: algo_cfgs.cost_limit.
        ep_cost: Metrics/EpCost.

    Returns:
        (self.ep_costs, logger.stored) -- the two observable effects of this snippet.
    """
    cfgs = Config(algo_cfgs=dict(no_intrinsic_in_ep_costs=no_intrinsic_in_ep_costs, cost_limit=cost_limit,
                                  intrinsic_factor=5.0))
    logger = _StubLogger(ep_cost)

    ep_costs = logger.get_stats('Metrics/EpCost')[0] - cfgs.algo_cfgs.cost_limit
    ep_discount_ci = ep_discount_ci_value
    logger.store({'Train/discount_ci': ep_discount_ci, 'Train/intrinsic_factor': cfgs.algo_cfgs.intrinsic_factor})
    if not getattr(cfgs.algo_cfgs, 'no_intrinsic_in_ep_costs', False):
        ep_costs += ep_discount_ci
    return ep_costs, logger.stored


def test_default_reproduces_prior_behavior_bias_always_added() -> None:
    ep_costs, _ = _ep_costs_via_mice(
        no_intrinsic_in_ep_costs=False, ep_discount_ci_value=180.0, cost_limit=25.0, ep_cost=30.0,
    )
    assert ep_costs == 185.0  # (30 - 25) + 180


def test_flag_true_removes_the_bias() -> None:
    ep_costs, _ = _ep_costs_via_mice(
        no_intrinsic_in_ep_costs=True, ep_discount_ci_value=180.0, cost_limit=25.0, ep_cost=30.0,
    )
    assert ep_costs == 5.0  # 30 - 25, ep_discount_ci ignored entirely


def test_discount_ci_is_still_logged_when_flag_is_true() -> None:
    # The bias stops affecting ep_costs, but the diagnostic is still visible either way.
    _, stored = _ep_costs_via_mice(
        no_intrinsic_in_ep_costs=True, ep_discount_ci_value=180.0, cost_limit=25.0, ep_cost=30.0,
    )
    assert stored['Train/discount_ci'] == 180.0


def test_flag_can_flip_the_sign_and_therefore_optim_case() -> None:
    # Same failure mode this flag exists to fix: without it, a large enough ep_discount_ci flips
    # ep_costs from "feasible" (<0) to "infeasible" (>=0) regardless of the real constraint state.
    biased, _ = _ep_costs_via_mice(
        no_intrinsic_in_ep_costs=False, ep_discount_ci_value=50.0, cost_limit=25.0, ep_cost=20.0,
    )
    unbiased, _ = _ep_costs_via_mice(
        no_intrinsic_in_ep_costs=True, ep_discount_ci_value=50.0, cost_limit=25.0, ep_cost=20.0,
    )
    assert unbiased < 0 <= biased


if __name__ == '__main__':
    module = sys.modules[__name__]
    failures = []
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if name.startswith('test_'):
            try:
                fn()
                print(f'PASS {name}')
            except AssertionError as exc:
                failures.append(name)
                print(f'FAIL {name}: {exc}')
    if failures:
        raise SystemExit(f'{len(failures)} test(s) failed: {failures}')
    print('All tests passed.')
