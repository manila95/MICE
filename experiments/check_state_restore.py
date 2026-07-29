"""Verify that Safety-Gymnasium state snapshot/restore is exact.

The multi-sample Monte-Carlo value estimator in
:mod:`omnisafe.utils.value_eval` is only meaningful if restoring a snapshot
puts the environment back into *exactly* the state it was captured from — same
physics, same reward-shaping memory, same goal, same remaining horizon. Each
task keeps its own Python-level state (``last_dist_goal``, the buttons timer,
the pushed box distances, ...), so this check replays a fixed action sequence
before and after a restore and asserts the observations, rewards and costs
match bit for bit.

Run from the repository root::

    python -m experiments.check_state_restore
"""

from __future__ import annotations

import numpy as np
import torch

from omnisafe.envs.core import make
from omnisafe.envs.wrapper import (
    ActionScale,
    AutoReset,
    CostNormalize,
    ObsNormalize,
    RewardNormalize,
    TimeLimit,
    Unsqueeze,
)
from omnisafe.utils.value_eval import _frozen_normalizers  # noqa: PLC2701


ENV_IDS = (
    'SafetyPointGoal1-v0',
    'SafetyPointButton1-v0',
    'SafetyPointPush1-v0',
    'SafetyCarGoal2-v0',
    'SafetyPointCircle1-v0',
)

DEVICE = torch.device('cpu')
WARMUP_STEPS = 200
REPLAY_STEPS = 300
TOLERANCE = 1e-5


def build_env(env_id: str, num_envs: int):
    """Build the same wrapper stack that :class:`OnlineAdapter` builds.

    Args:
        env_id (str): Safety-Gymnasium environment id.
        num_envs (int): Number of parallel environments.

    Returns:
        The wrapped environment.
    """
    env = make(env_id, num_envs=num_envs, device=DEVICE, asynchronous=False)
    if env.need_time_limit_wrapper:
        env = TimeLimit(env, time_limit=env.max_episode_steps, device=DEVICE)
    if env.need_auto_reset_wrapper:
        env = AutoReset(env, device=DEVICE)
    env = ObsNormalize(env, device=DEVICE)
    env = RewardNormalize(env, device=DEVICE)
    env = CostNormalize(env, device=DEVICE)
    env = ActionScale(env, low=-1.0, high=1.0, device=DEVICE)
    if env.num_envs == 1:
        env = Unsqueeze(env, device=DEVICE)
    env.set_seed(11)
    return env


def check(env_id: str, num_envs: int) -> None:
    """Snapshot mid-episode, replay a fixed action sequence twice, compare.

    Args:
        env_id (str): Safety-Gymnasium environment id.
        num_envs (int): Number of parallel environments.

    Raises:
        AssertionError: If the restored rollout diverges from the original.
    """
    env = build_env(env_id, num_envs)
    idx = num_envs - 1
    generator = torch.Generator().manual_seed(0)
    actions = [
        torch.tanh(torch.randn((num_envs,) + env.action_space.shape, generator=generator))
        for _ in range(WARMUP_STEPS + REPLAY_STEPS)
    ]

    # Normalizer statistics drift as we step; freeze them so any difference we
    # measure is genuine simulator state, not a moving normalization.
    with _frozen_normalizers(env):
        obs, _ = env.reset()
        for act in actions[:WARMUP_STEPS]:
            obs, *_ = env.step(act)

        snapshot = env.snapshot_state(idx)
        obs_at_snapshot = obs[idx].clone()

        def replay():
            trace = []
            local_obs = obs.clone()
            for act in actions[WARMUP_STEPS:]:
                local_obs, reward, cost, terminated, truncated, _ = env.step(act)
                local_obs = local_obs.clone()
                trace.append(
                    (
                        local_obs[idx].clone(),
                        reward.reshape(num_envs)[idx].item(),
                        cost.reshape(num_envs)[idx].item(),
                    ),
                )
                if (
                    terminated.reshape(num_envs)[idx].bool()
                    | truncated.reshape(num_envs)[idx].bool()
                ).item():
                    break
            return trace

        first = replay()
        restored = env.restore_state(snapshot, idx)
        obs_diff = (restored.reshape(-1) - obs_at_snapshot).abs().max().item()
        obs[idx] = restored.reshape(-1)
        second = replay()

    assert len(first) == len(second), f'{env_id}: episode lengths differ {len(first)} vs {len(second)}'
    d_obs = max((a[0] - b[0]).abs().max().item() for a, b in zip(first, second))
    d_reward = max(abs(a[1] - b[1]) for a, b in zip(first, second))
    d_cost = max(abs(a[2] - b[2]) for a, b in zip(first, second))

    print(
        f'  {env_id:24s} n={num_envs}  len={len(first):4d}  '
        f'restore {obs_diff:.1e}  obs {d_obs:.1e}  reward {d_reward:.1e}  cost {d_cost:.1e}',
    )
    assert obs_diff < TOLERANCE, f'{env_id}: restored observation differs by {obs_diff}'
    assert d_obs < TOLERANCE, f'{env_id}: replayed observations differ by {d_obs}'
    assert d_reward < TOLERANCE, f'{env_id}: replayed rewards differ by {d_reward}'
    assert d_cost < TOLERANCE, f'{env_id}: replayed costs differ by {d_cost}'
    env.close()


def check_remaining_horizon(env_id: str, num_envs: int) -> None:
    """A restore late in an episode must truncate after the *remaining* steps.

    Args:
        env_id (str): Safety-Gymnasium environment id.
        num_envs (int): Number of parallel environments.

    Raises:
        AssertionError: If the restored episode runs for the wrong horizon.
    """
    env = build_env(env_id, num_envs)
    horizon = 1000
    warmup = horizon - 25
    generator = torch.Generator().manual_seed(1)

    obs, _ = env.reset()
    for _ in range(warmup):
        act = torch.tanh(torch.randn((num_envs,) + env.action_space.shape, generator=generator))
        obs, *_ = env.step(act)

    snapshot = env.snapshot_state(0)
    env.set_auto_reset(False)
    try:
        env.restore_state(snapshot, 0)
        steps = 0
        while steps < 2 * horizon:
            act = torch.tanh(torch.randn((num_envs,) + env.action_space.shape, generator=generator))
            _, _, _, terminated, truncated, _ = env.step(act)
            steps += 1
            if (
                terminated.reshape(num_envs)[0].bool() | truncated.reshape(num_envs)[0].bool()
            ).item():
                break
    finally:
        env.set_auto_reset(True)

    print(f'  {env_id:24s} n={num_envs}  truncated after {steps} steps (expected {horizon - warmup})')
    assert steps == horizon - warmup, f'{env_id}: got {steps}, expected {horizon - warmup}'
    env.close()


def main() -> None:
    """Run every check."""
    print('== snapshot / restore fidelity ==')
    for env_id in ENV_IDS:
        check(env_id, num_envs=3)
    check(ENV_IDS[0], num_envs=1)

    print('== remaining horizon after restore ==')
    check_remaining_horizon(ENV_IDS[0], num_envs=3)
    check_remaining_horizon(ENV_IDS[0], num_envs=1)

    print('ALL OK')


if __name__ == '__main__':
    np.seterr(all='raise')
    main()
