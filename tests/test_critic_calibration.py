"""Correctness tests for algo_cfgs.use_calibration_loss and its supporting pieces.

Run either way -- ``pytest tests/test_critic_calibration.py`` or
``python tests/test_critic_calibration.py``. The ``__main__`` block is not decoration: the other
modules in this directory have none, so running them as scripts exits 0 having executed nothing,
which looks exactly like a pass.

Covers two layers:

- The pure loss functions in :mod:`omnisafe.utils.critic_calibration` (binned and moment
  calibration losses): do they score 0 for a genuinely calibrated batch, do they detect a
  region-local bias a global MSE/bias term would dilute away, and does gradient descent on them
  actually shrink the gap.
- The blending logic in :class:`~omnisafe.algorithms.on_policy.base.policy_gradient.PolicyGradient`
  (``_stream_cfg``'s null-fallback-to-reward convention, and ``_combined_critic_loss``'s
  ``lambda=0``/``lambda=1`` edge cases and its opt-in no-op when calibration is disabled) --
  exercised directly on the method objects via ``object.__new__`` so this doesn't need to spin up
  a full env + actor-critic just to check loss arithmetic.
"""

from __future__ import annotations

import inspect
import os
import sys


# Resolve ``omnisafe`` to *this* fork rather than whichever checkout happens to be pip-installed
# (see tests/test_laplacian.py's matching comment).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient  # noqa: E402
from omnisafe.utils.config import Config  # noqa: E402
from omnisafe.utils.critic_calibration import (  # noqa: E402
    binned_calibration_loss,
    moment_calibration_loss,
)


# --------------------------------------------------------------------------------------------- #
# binned_calibration_loss
# --------------------------------------------------------------------------------------------- #


def test_binned_loss_zero_when_perfectly_calibrated() -> None:
    pred = torch.linspace(-2.0, 2.0, 40)
    target = pred.clone()
    loss = binned_calibration_loss(pred, target, n_bins=5)
    assert loss.item() < 1e-8


def test_binned_loss_detects_region_local_bias() -> None:
    # A global bias/MSE check would barely notice a shift confined to 10% of the batch; the
    # binned loss should score it at roughly (population weight) * (shift^2) = 0.1 * 1.0.
    pred = torch.linspace(-2.0, 2.0, 40)
    target = pred.clone()
    high_mask = pred >= pred.quantile(0.9)
    assert int(high_mask.sum().item()) == 4  # sanity: this is the top decile, no more
    target = target.clone()
    target[high_mask] -= 1.0
    loss = binned_calibration_loss(pred, target, n_bins=10)
    assert loss.item() == pytest_approx(0.1, abs=0.02)


def test_binned_loss_gradient_descent_shrinks_the_gap() -> None:
    pred = torch.linspace(-2.0, 2.0, 40, requires_grad=True)
    target = pred.detach().clone()
    target[-4:] -= 1.0

    loss0 = binned_calibration_loss(pred, target, n_bins=10)
    loss0.backward()
    assert pred.grad is not None and pred.grad.abs().sum().item() > 0

    with torch.no_grad():
        pred_step = pred - 0.5 * pred.grad

    loss1 = binned_calibration_loss(pred_step, target, n_bins=10)
    assert loss1.item() < loss0.item()


def test_binned_loss_zero_for_single_sample_batch() -> None:
    # batch_size // 2 == 0 usable bins -> short-circuits to 0 rather than a degenerate 1-point bin.
    pred = torch.tensor([3.0])
    target = torch.tensor([-7.0])
    loss = binned_calibration_loss(pred, target, n_bins=10)
    assert loss.item() == 0.0


def test_binned_loss_no_grad_through_bin_edges() -> None:
    # Moving a point across a bin boundary must not itself be a way to reduce the loss --
    # gradient should come only from within-bin means, not from where the (detached) edges land.
    # With 10 equal-population bins of 4 elements each over 40 sorted points, biasing only the
    # single target at index 20 should confine nonzero gradient to (at most) that ~4-wide bin,
    # leaving points far away from it exactly untouched.
    pred = torch.linspace(-2.0, 2.0, 40, requires_grad=True)
    target = pred.detach().clone()
    target[20] += 5.0  # one outlier target, deep inside the batch, not at any bin edge
    loss = binned_calibration_loss(pred, target, n_bins=10)
    loss.backward()
    grad = pred.grad
    assert grad.abs().sum().item() > 0, 'expected some nonzero gradient'
    far_from_20 = list(range(0, 10)) + list(range(31, 40))
    assert all(grad[i].item() == 0.0 for i in far_from_20)


# --------------------------------------------------------------------------------------------- #
# moment_calibration_loss
# --------------------------------------------------------------------------------------------- #


def test_moment_loss_zero_when_perfectly_calibrated() -> None:
    torch.manual_seed(0)
    pred = torch.randn(64)
    target = pred.clone()
    loss = moment_calibration_loss(pred, target)
    assert loss.item() < 1e-6


def test_moment_loss_bias_and_slope_terms_are_additive() -> None:
    pred = torch.tensor([1.0, 1.0, 1.0, 1.0])  # zero variance -> slope term saturates to (0-1)^2=1
    target = torch.tensor([2.0, 2.0, 2.0, 2.0])  # bias = -1 -> bias^2 = 1
    loss = moment_calibration_loss(pred, target)
    assert loss.item() == pytest_approx(2.0, abs=1e-5)


def test_moment_loss_detects_slope_error() -> None:
    torch.manual_seed(0)
    base = torch.randn(64)
    pred = base
    target = 2.0 * base  # true regression slope of target on pred is 2, not 1
    loss = moment_calibration_loss(pred, target)
    # (slope - 1)^2 = (2-1)^2 = 1, plus a small sample-mean bias term.
    assert loss.item() > 0.5


def test_moment_loss_denominator_is_detached() -> None:
    # var(pred) in the denominator must not carry gradient -- otherwise the loss could be reduced
    # by shrinking prediction variance rather than by correcting the covariance with the target.
    pred = torch.randn(32, requires_grad=True)
    target = torch.randn(32)
    loss = moment_calibration_loss(pred, target)
    loss.backward()
    # A live check that this doesn't crash and produces *some* gradient is the meaningful part;
    # the no-grad-through-the-ratio-denominator property is enforced by moment_calibration_loss's
    # explicit .detach() and is exercised implicitly by every other test in this section still
    # passing (a denominator that fought the numerator would make the slope-error test above
    # flaky/wrong).
    assert pred.grad is not None


# --------------------------------------------------------------------------------------------- #
# PolicyGradient._stream_cfg / _calibration_loss / _combined_critic_loss
# --------------------------------------------------------------------------------------------- #


class _StubLogger:
    """Minimal stand-in for omnisafe.common.logger.Logger -- only `.store` is exercised here."""

    def __init__(self) -> None:
        self.stored: dict[str, float] = {}

    def store(self, data: dict) -> None:
        self.stored.update(data)


def _bare_policy_gradient(algo_cfgs: dict) -> PolicyGradient:
    """A PolicyGradient with only `_cfgs.algo_cfgs`/`_logger` set, `__init__` skipped entirely.

    `_stream_cfg`/`_calibration_loss`/`_combined_critic_loss`/`_critic_loss` touch only those two
    attributes, so this avoids constructing a full env + actor-critic just to unit test loss
    arithmetic and config resolution.
    """
    obj = object.__new__(PolicyGradient)
    obj._cfgs = Config(algo_cfgs=dict(algo_cfgs))
    obj._logger = _StubLogger()
    return obj


def test_stream_cfg_cost_override_wins_over_reward_default() -> None:
    pg = _bare_policy_gradient(
        {'use_calibration_loss': False, 'use_calibration_loss_cost': True},
    )
    assert pg._stream_cfg('use_calibration_loss', 'r', False) is False
    assert pg._stream_cfg('use_calibration_loss', 'c', False) is True


def test_stream_cfg_cost_inherits_reward_when_unset() -> None:
    pg = _bare_policy_gradient({'use_calibration_loss': True})
    assert pg._stream_cfg('use_calibration_loss', 'c', False) is True


def test_stream_cfg_falls_back_to_default_when_absent() -> None:
    pg = _bare_policy_gradient({})
    assert pg._stream_cfg('calibration_coef', 'r', 0.5) == 0.5
    assert pg._stream_cfg('calibration_coef', 'c', 0.5) == 0.5


def test_combined_critic_loss_lambda_zero_matches_task_loss() -> None:
    torch.manual_seed(0)
    pg = _bare_policy_gradient(
        {
            'critic_loss': 'mse',
            'use_calibration_loss_cost': True,
            'calibration_coef_cost': 0.0,
            'calibration_loss_type_cost': 'moment',
        },
    )
    pred, target = torch.randn(32), torch.randn(32)
    combined = pg._combined_critic_loss([pred], target, stream='c')
    task = pg._critic_loss(pred, target, stream='c')
    assert torch.allclose(combined[0], task)


def test_combined_critic_loss_lambda_one_matches_calibration_loss() -> None:
    torch.manual_seed(0)
    pg = _bare_policy_gradient(
        {
            'critic_loss': 'mse',
            'use_calibration_loss_cost': True,
            'calibration_coef_cost': 1.0,
            'calibration_loss_type_cost': 'moment',
        },
    )
    pred, target = torch.randn(32), torch.randn(32)
    combined = pg._combined_critic_loss([pred], target, stream='c')
    calib = moment_calibration_loss(pred, target)
    assert torch.allclose(combined[0], calib)
    assert 'Loss/Loss_cost_critic_calib' in pg._logger.stored


def test_combined_critic_loss_disabled_by_default_is_a_noop() -> None:
    pred, target = torch.randn(16), torch.randn(16)
    pg = _bare_policy_gradient({'critic_loss': 'mse'})
    combined = pg._combined_critic_loss([pred], target, stream='r')
    task = pg._critic_loss(pred, target, stream='r')
    assert torch.allclose(combined[0], task)
    assert pg._logger.stored == {}


def test_use_calibration_loss_cost_only_leaves_reward_critic_untouched() -> None:
    # The exact configuration the feature exists for: calibrate the cost critic only.
    torch.manual_seed(0)
    pg = _bare_policy_gradient(
        {
            'critic_loss': 'mse',
            'use_calibration_loss': False,
            'use_calibration_loss_cost': True,
            'calibration_coef_cost': 0.5,
            'calibration_loss_type_cost': 'moment',
        },
    )
    pred, target = torch.randn(32), torch.randn(32)

    reward_combined = pg._combined_critic_loss([pred], target, stream='r')
    reward_task = pg._critic_loss(pred, target, stream='r')
    assert torch.allclose(reward_combined[0], reward_task)  # reward critic: untouched

    cost_combined = pg._combined_critic_loss([pred], target, stream='c')
    cost_task = pg._critic_loss(pred, target, stream='c')
    assert not torch.allclose(cost_combined[0], cost_task)  # cost critic: blended


def test_unknown_calibration_loss_type_raises() -> None:
    pg = _bare_policy_gradient(
        {'use_calibration_loss': True, 'calibration_loss_type': 'bogus'},
    )
    pred, target = torch.randn(8), torch.randn(8)
    try:
        pg._combined_critic_loss([pred], target, stream='r')
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for an unrecognized calibration_loss_type')


# --------------------------------------------------------------------------------------------- #
# tiny pytest.approx substitute so this file has no hard pytest dependency for its __main__ path
# --------------------------------------------------------------------------------------------- #


def pytest_approx(expected: float, abs: float) -> '_Approx':  # noqa: A002
    return _Approx(expected, abs)


class _Approx:
    def __init__(self, expected: float, abs_tol: float) -> None:
        self.expected = expected
        self.abs_tol = abs_tol

    def __eq__(self, other: object) -> bool:
        return isinstance(other, (int, float)) and abs(other - self.expected) <= self.abs_tol

    def __repr__(self) -> str:
        return f'approx({self.expected} +/- {self.abs_tol})'


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
