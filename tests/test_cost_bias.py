"""Correctness tests for algo_cfgs.cost_bias / use_cost_bias / use_cost_bias_in_target.

Run either way -- ``pytest tests/test_cost_bias.py`` or ``python tests/test_cost_bias.py``.

Covers three layers:

- :func:`omnisafe.utils.tools.decayed_constant`: the two decay schedules (mirrors
  MICE's ``constant_cost``/``cost_decay_type`` -- see that function's docstring).
- :class:`~omnisafe.common.buffer.onpolicy_buffer.OnPolicyBuffer`'s per-path bias
  tracking (``mean_ep_cost_bias``) and optional target-bias folding
  (``use_cost_bias_in_target``), and :class:`~omnisafe.common.buffer.vector_onpolicy_buffer.VectorOnPolicyBuffer`'s
  pooling of that across parallel envs.
- :class:`~omnisafe.algorithms.on_policy.second_order.cpo.CPO`'s consumption of
  ``mean_ep_cost_bias()`` to bias ``ep_costs`` before optim-case selection, exercised
  directly on a bare (``object.__new__``) instance the same way
  ``tests/test_critic_calibration.py`` does for ``PolicyGradient``.
"""

from __future__ import annotations

import inspect
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from gymnasium.spaces import Box  # noqa: E402

from omnisafe.algorithms.on_policy.second_order.cpo import CPO  # noqa: E402
from omnisafe.common.buffer.onpolicy_buffer import OnPolicyBuffer  # noqa: E402
from omnisafe.common.buffer.vector_onpolicy_buffer import VectorOnPolicyBuffer  # noqa: E402
from omnisafe.utils.config import Config  # noqa: E402
from omnisafe.utils.tools import decayed_constant  # noqa: E402


OBS_SPACE = Box(low=-1, high=1, shape=(3,))
ACT_SPACE = Box(low=-1, high=1, shape=(2,))


# --------------------------------------------------------------------------------------------- #
# decayed_constant
# --------------------------------------------------------------------------------------------- #


def test_decayed_constant_none_is_a_noop() -> None:
    assert decayed_constant(2.0, 0, None, 0.985, 50, 0.4) == 2.0
    assert decayed_constant(2.0, 500, None, 0.985, 50, 0.4) == 2.0


def test_decayed_constant_exponential() -> None:
    assert decayed_constant(2.0, 0, 'exponential', 0.9, 50, 0.4) == 2.0
    val = decayed_constant(2.0, 10, 'exponential', 0.9, 50, 0.4)
    assert abs(val - 2.0 * 0.9**10) < 1e-9


def test_decayed_constant_step() -> None:
    # Within the first interval: no drop yet.
    assert decayed_constant(2.0, 49, 'step', 0.985, 50, 0.4) == 2.0
    # One interval elapsed: exactly one factor of 0.4.
    val = decayed_constant(2.0, 50, 'step', 0.985, 50, 0.4)
    assert abs(val - 2.0 * 0.4) < 1e-9


def test_decayed_constant_unknown_type_raises() -> None:
    try:
        decayed_constant(1.0, 0, 'bogus', 0.985, 50, 0.4)
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for an unrecognized decay_type')


# --------------------------------------------------------------------------------------------- #
# OnPolicyBuffer
# --------------------------------------------------------------------------------------------- #


def _fill_path(buf: OnPolicyBuffer, n_steps: int) -> None:
    for _ in range(n_steps):
        buf.store(
            obs=torch.zeros(3), act=torch.zeros(2), reward=torch.tensor(1.0),
            cost=torch.tensor(0.0), value_r=torch.tensor(0.5), value_c=torch.tensor(0.1),
            logp=torch.tensor(0.0),
        )


def _new_buf(use_cost_bias_in_target: bool) -> OnPolicyBuffer:
    return OnPolicyBuffer(
        OBS_SPACE, ACT_SPACE, size=10, gamma=0.99, lam=0.95, lam_c=0.95,
        advantage_estimator='gae', cost_gamma=0.99,
        use_cost_bias_in_target=use_cost_bias_in_target,
    )


def test_cost_bias_zero_is_an_exact_noop() -> None:
    buf = _new_buf(use_cost_bias_in_target=True)  # bias flag on, but cost_bias itself unset (0.0)
    _fill_path(buf, 5)
    buf.finish_path(last_value_r=torch.tensor([0.0]), last_value_c=torch.tensor([0.0]))
    assert buf.mean_ep_cost_bias() == 0.0

    baseline = _new_buf(use_cost_bias_in_target=False)
    _fill_path(baseline, 5)
    baseline.finish_path(last_value_r=torch.tensor([0.0]), last_value_c=torch.tensor([0.0]))
    assert torch.allclose(buf.data['target_value_c'][:5], baseline.data['target_value_c'][:5])


def test_mean_ep_cost_bias_matches_closed_form() -> None:
    buf = _new_buf(use_cost_bias_in_target=False)
    buf.set_cost_bias_for_epoch(5.0)
    _fill_path(buf, 5)
    buf.finish_path(last_value_r=torch.tensor([0.0]), last_value_c=torch.tensor([0.0]))
    expected = 5.0 * (1 - 0.99**5) / (1 - 0.99)
    assert abs(buf.mean_ep_cost_bias() - expected) < 1e-4


def test_use_cost_bias_in_target_false_leaves_target_untouched() -> None:
    buf = _new_buf(use_cost_bias_in_target=False)
    buf.set_cost_bias_for_epoch(5.0)
    _fill_path(buf, 5)
    buf.finish_path(last_value_r=torch.tensor([0.0]), last_value_c=torch.tensor([0.0]))

    baseline = _new_buf(use_cost_bias_in_target=False)  # cost_bias never set -> stays 0.0
    _fill_path(baseline, 5)
    baseline.finish_path(last_value_r=torch.tensor([0.0]), last_value_c=torch.tensor([0.0]))

    assert torch.allclose(buf.data['target_value_c'][:5], baseline.data['target_value_c'][:5])
    # But the bias is still tracked for mean_ep_cost_bias() regardless of this flag.
    assert buf.mean_ep_cost_bias() > 0.0


def test_use_cost_bias_in_target_true_raises_the_target() -> None:
    buf = _new_buf(use_cost_bias_in_target=True)
    buf.set_cost_bias_for_epoch(5.0)
    _fill_path(buf, 5)
    buf.finish_path(last_value_r=torch.tensor([0.0]), last_value_c=torch.tensor([0.0]))

    baseline = _new_buf(use_cost_bias_in_target=True)  # cost_bias never set -> stays 0.0
    _fill_path(baseline, 5)
    baseline.finish_path(last_value_r=torch.tensor([0.0]), last_value_c=torch.tensor([0.0]))

    assert torch.all(buf.data['target_value_c'][:5] > baseline.data['target_value_c'][:5])


def test_set_cost_bias_for_epoch_resets_the_accumulator() -> None:
    buf = _new_buf(use_cost_bias_in_target=False)
    buf.set_cost_bias_for_epoch(5.0)
    _fill_path(buf, 5)
    buf.finish_path(last_value_r=torch.tensor([0.0]), last_value_c=torch.tensor([0.0]))
    assert buf.mean_ep_cost_bias() > 0.0

    buf.set_cost_bias_for_epoch(5.0)  # new epoch: accumulator must reset even at the same value
    assert buf.mean_ep_cost_bias() == 0.0


# --------------------------------------------------------------------------------------------- #
# VectorOnPolicyBuffer: pooling across parallel envs
# --------------------------------------------------------------------------------------------- #


def test_vector_buffer_pools_by_path_count_not_by_env() -> None:
    vbuf = VectorOnPolicyBuffer(
        OBS_SPACE, ACT_SPACE, size=10, gamma=0.99, lam=0.95, lam_c=0.95,
        advantage_estimator='gae', penalty_coefficient=0.0,
        standardized_adv_r=False, standardized_adv_c=False, num_envs=2, cost_gamma=0.99,
    )
    vbuf.set_cost_bias_for_epoch(3.0)
    _fill_path(vbuf.buffers[0], 4)
    vbuf.finish_path(last_value_r=torch.tensor([0.0]), last_value_c=torch.tensor([0.0]), idx=0)
    _fill_path(vbuf.buffers[1], 6)
    vbuf.finish_path(last_value_r=torch.tensor([0.0]), last_value_c=torch.tensor([0.0]), idx=1)

    e0 = 3.0 * (1 - 0.99**4) / (1 - 0.99)
    e1 = 3.0 * (1 - 0.99**6) / (1 - 0.99)
    expected = (e0 + e1) / 2  # 2 paths total, not a plain average of the two envs' own means
    assert abs(vbuf.mean_ep_cost_bias() - expected) < 1e-4


# --------------------------------------------------------------------------------------------- #
# CPO._update_actor's ep_costs bias
# --------------------------------------------------------------------------------------------- #


class _StubLogger:
    def __init__(self, ep_cost: float) -> None:
        self._ep_cost = ep_cost
        self.stored: dict[str, float] = {}

    def get_stats(self, key: str) -> tuple[float]:
        assert key == 'Metrics/EpCost'
        return (self._ep_cost,)

    def store(self, data: dict) -> None:
        self.stored.update(data)


class _StubBuf:
    def __init__(self, bias: float) -> None:
        self._bias = bias

    def mean_ep_cost_bias(self) -> float:
        return self._bias


def _ep_costs_via_cpo(use_cost_bias: bool, cost_bias_value: float, cost_limit: float, ep_cost: float) -> float:
    """Replicates exactly the two lines of cpo.py under test, on a bare CPO instance.

    Bare (``object.__new__``) rather than a full Agent, matching
    tests/test_critic_calibration.py's ``_bare_policy_gradient`` pattern -- avoids spinning up an
    env + actor-critic just to check this one arithmetic branch.
    """
    cpo = object.__new__(CPO)
    cpo._cfgs = Config(algo_cfgs=dict(use_cost_bias=use_cost_bias, cost_limit=cost_limit))
    cpo._logger = _StubLogger(ep_cost)
    cpo._buf = _StubBuf(cost_bias_value)

    ep_costs = cpo._logger.get_stats('Metrics/EpCost')[0] - cpo._cfgs.algo_cfgs.cost_limit
    if getattr(cpo._cfgs.algo_cfgs, 'use_cost_bias', False):
        ep_cost_bias = cpo._buf.mean_ep_cost_bias()
        ep_costs = ep_costs + ep_cost_bias
        cpo._logger.store({'Misc/EpCostBias': ep_cost_bias})
    return ep_costs


def test_ep_costs_bias_disabled_matches_plain_cpo() -> None:
    ep_costs = _ep_costs_via_cpo(use_cost_bias=False, cost_bias_value=100.0, cost_limit=25.0, ep_cost=30.0)
    assert ep_costs == 5.0  # 30 - 25, cost_bias_value ignored entirely


def test_ep_costs_bias_enabled_adds_the_bias() -> None:
    ep_costs = _ep_costs_via_cpo(use_cost_bias=True, cost_bias_value=100.0, cost_limit=25.0, ep_cost=30.0)
    assert ep_costs == 105.0  # (30 - 25) + 100


def test_ep_costs_bias_can_flip_the_sign_and_therefore_optim_case() -> None:
    # A case CPO's _determine_case would read as "feasible" (ep_costs < 0) flips to "infeasible"
    # (ep_costs >= 0) purely from the bias -- exactly the failure mode surfaced earlier this
    # session for MICE's constant_cost + ep_discount_ci.
    unbiased = _ep_costs_via_cpo(use_cost_bias=False, cost_bias_value=50.0, cost_limit=25.0, ep_cost=20.0)
    biased = _ep_costs_via_cpo(use_cost_bias=True, cost_bias_value=50.0, cost_limit=25.0, ep_cost=20.0)
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
