"""``adv_estimation_method`` used to pick the critic's target as well as the advantage.

Run either way -- ``pytest tests/test_adv_target_split.py`` or
``python tests/test_adv_target_split.py``.

Those are two orthogonal choices, and welding them together had two costs. Only 6 of the 3x3
combinations were reachable -- "TD(0) advantage with a TD(lambda) target" could not be expressed
at all. And, worse for experiments, comparing two estimators silently varied *both* axes:
``plain`` is (TD(0) advantage, MC target) while ``gae`` is (GAE advantage, TD(lambda) target), so
any difference measured between them cannot be attributed to either choice.

The risk in splitting them is a silent change to every existing run, so the first test here pins
each legacy preset against the old hard-coded formulas, written out literally rather than
referenced, and demands exact equality.
"""

from __future__ import annotations

import os

import pytest
import torch

from omnisafe.utils.gae import (
    ADV_TARGET_PRESETS,
    calculate_adv_and_value_targets,
    resolve_adv_and_target,
)
from omnisafe.utils.math import discount_cumsum


GAMMA, LAM = 0.99, 0.95


def _inputs(n: int = 12, seed: int = 0):
    torch.manual_seed(seed)
    return torch.randn(n + 1), torch.randn(n + 1)  # values, rewards (both include the bootstrap)


def _legacy(values, rewards, name):
    """The original hard-coded branches, transcribed verbatim from before the split."""
    if name == 'gae':
        deltas = rewards[:-1] + GAMMA * values[1:] - values[:-1]
        adv = discount_cumsum(deltas, GAMMA * LAM)
        return adv, adv + values[:-1]
    if name == 'gae-rtg':
        deltas = rewards[:-1] + GAMMA * values[1:] - values[:-1]
        return discount_cumsum(deltas, GAMMA * LAM), discount_cumsum(rewards, GAMMA)[:-1]
    if name == 'plain':
        adv = rewards[:-1] + GAMMA * values[1:] - values[:-1]
        return adv, discount_cumsum(rewards, GAMMA)[:-1]
    if name == 'reinforce':
        returns = discount_cumsum(rewards, GAMMA)[:-1]
        return returns, returns
    if name == 'td_zero':
        adv = rewards[:-1] + GAMMA * values[1:] - values[:-1]
        return adv, rewards[:-1] + GAMMA * values[1:]
    if name == 'td_zero_gae':
        deltas = rewards[:-1] + GAMMA * values[1:] - values[:-1]
        return discount_cumsum(deltas, GAMMA * LAM), rewards[:-1] + GAMMA * values[1:]
    raise AssertionError(name)


@pytest.mark.parametrize('name', ['gae', 'gae-rtg', 'plain', 'reinforce', 'td_zero', 'td_zero_gae'])
def test_legacy_presets_are_bit_identical(name: str) -> None:
    """Every pre-existing config must compute exactly what it computed before the split."""
    values, rewards = _inputs()
    adv, target = calculate_adv_and_value_targets(
        values=values, rewards=rewards, lam=LAM, gamma=GAMMA, advantage_estimator=name,
    )
    exp_adv, exp_target = _legacy(values, rewards, name)
    assert torch.equal(adv, exp_adv), f'{name}: advantage changed'
    assert torch.equal(target, exp_target), f'{name}: value target changed'


def test_the_missing_combination_is_now_reachable() -> None:
    """TD(0) advantage with a TD(lambda) target -- previously inexpressible.

    Its advantage must equal ``plain``'s (both TD(0)) and its target must equal ``gae``'s (both
    TD(lambda)), which is precisely what "choose the axes independently" means.
    """
    values, rewards = _inputs()
    adv, target = calculate_adv_and_value_targets(
        values=values, rewards=rewards, lam=LAM, gamma=GAMMA,
        advantage_estimator='td_zero', value_target='td_lambda',
    )
    plain_adv, _ = _legacy(values, rewards, 'plain')
    _, gae_target = _legacy(values, rewards, 'gae')
    assert torch.equal(adv, plain_adv), 'advantage should match the TD(0) one'
    assert torch.equal(target, gae_target), 'target should match the TD(lambda) one'


@pytest.mark.parametrize('adv_m', ['gae', 'td_zero', 'mc'])
@pytest.mark.parametrize('tgt_m', ['td_lambda', 'mc', 'td_zero'])
def test_all_nine_combinations_are_reachable_and_axes_are_independent(adv_m, tgt_m) -> None:
    """The full 3x3 grid works, and each axis depends only on itself.

    Independence is the actual claim: the advantage must not change when only the target axis
    moves, and vice versa. Without that, pinning one axis to isolate the other -- the whole point
    of the split -- would not work.
    """
    values, rewards = _inputs()
    adv, target = calculate_adv_and_value_targets(
        values=values, rewards=rewards, lam=LAM, gamma=GAMMA,
        advantage_estimator=adv_m, value_target=tgt_m,
    )
    ref_adv, _ = calculate_adv_and_value_targets(
        values=values, rewards=rewards, lam=LAM, gamma=GAMMA,
        advantage_estimator=adv_m, value_target='mc',
    )
    _, ref_target = calculate_adv_and_value_targets(
        values=values, rewards=rewards, lam=LAM, gamma=GAMMA,
        advantage_estimator='mc', value_target=tgt_m,
    )
    assert torch.equal(adv, ref_adv), f'advantage {adv_m} changed when the target axis changed'
    assert torch.equal(target, ref_target), f'target {tgt_m} changed when the advantage axis changed'


def test_presets_expand_to_the_documented_pairs() -> None:
    """The preset table is the contract for what each legacy name always meant."""
    assert resolve_adv_and_target('gae') == ('gae', 'td_lambda')
    assert resolve_adv_and_target('plain') == ('td_zero', 'mc')
    assert resolve_adv_and_target('td_zero') == ('td_zero', 'td_zero')
    assert resolve_adv_and_target('gae-rtg') == ('gae', 'mc')
    assert resolve_adv_and_target('reinforce') == ('mc', 'mc')
    assert resolve_adv_and_target('td_zero_gae') == ('gae', 'td_zero')
    assert set(ADV_TARGET_PRESETS) >= {'gae', 'plain', 'td_zero', 'reinforce'}


def test_explicit_axis_overrides_only_that_axis() -> None:
    """Naming a target alongside a preset keeps the preset's advantage."""
    assert resolve_adv_and_target('gae', 'mc') == ('gae', 'mc')
    assert resolve_adv_and_target('plain', 'td_lambda') == ('td_zero', 'td_lambda')


def test_vtrace_cannot_be_split_across_axes() -> None:
    """V-trace derives both quantities from one recursion; half of it would be incoherent."""
    with pytest.raises(NotImplementedError, match='vtrace'):
        resolve_adv_and_target('vtrace', 'mc')
    with pytest.raises(NotImplementedError, match='vtrace'):
        resolve_adv_and_target('gae', 'vtrace')


def test_unknown_axis_values_raise() -> None:
    """A typo must fail loudly rather than silently falling back to a default."""
    with pytest.raises(NotImplementedError):
        resolve_adv_and_target('not_an_estimator')
    with pytest.raises(NotImplementedError):
        resolve_adv_and_target('gae', 'not_a_target')


if __name__ == '__main__':
    raise SystemExit(pytest.main([os.path.abspath(__file__), '-q']))
