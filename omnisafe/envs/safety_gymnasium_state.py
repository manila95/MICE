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
"""Snapshot and restore of Safety-Gymnasium environment state.

Restoring a full environment state lets us re-run the policy from *exactly* the
same state many times, which turns the single-sample discounted cost-to-go into
a proper Monte-Carlo estimate of :math:`V^\\pi(s)`.

A Safety-Gymnasium navigation environment keeps its state in three places, all
of which have to be captured:

1. The MuJoCo ``data`` struct (agent/free-geom positions and velocities).
2. The MuJoCo ``model`` struct — the goal body is *moved* rather than
   re-created (:meth:`Underlying._set_goal`), so ``model.body_pos`` changes
   during an episode.
3. Plain Python attributes on the task, on its obstacle objects and on the
   wrapper chain: reward-shaping memory (``last_dist_goal``, ``last_dist_box``,
   ...), the buttons timer and goal button, the sampled layout, the random
   generator and the elapsed-step counters.

Missing any one of them silently produces a *different* MDP after the restore
(wrong shaping reward, wrong truncation time, wrong goal), so the snapshot
takes all Python attributes of "simple" type generically rather than
enumerating the ones each task happens to use today.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import mujoco
import numpy as np


__all__ = ['snapshot_env', 'restore_env', 'supports_snapshot']


# ``mujoco.mj_getState`` only exists from mujoco 2.3.5 on; these are the fields
# that make up ``mjSTATE_FULLPHYSICS`` plus the applied-force inputs.
_DATA_FIELDS: tuple[str, ...] = (
    'qpos',
    'qvel',
    'act',
    'ctrl',
    'qacc_warmstart',
    'qfrc_applied',
    'xfrc_applied',
    'mocap_pos',
    'mocap_quat',
    'userdata',
)

# Model fields that Safety-Gymnasium mutates in-place during an episode:
# ``_set_goal`` writes ``body_pos``, the vision/fading tasks write ``geom_rgba``.
# The rest are cheap to copy and guard against task-specific tweaks.
_MODEL_FIELDS: tuple[str, ...] = (
    'body_pos',
    'body_quat',
    'geom_pos',
    'geom_quat',
    'geom_size',
    'geom_rgba',
    'site_pos',
    'site_quat',
    'site_rgba',
)

_BUILDER_FIELDS: tuple[str, ...] = ('steps', 'terminated', 'truncated', 'cost', 'first_reset')

# Attribute types worth snapshotting.  Everything else on a task is either a
# config dataclass (immutable during an episode), a back-reference, or the
# MuJoCo/viewer objects we handle explicitly.
_SIMPLE_TYPES = (
    int,
    float,
    bool,
    str,
    bytes,
    type(None),
    np.integer,
    np.floating,
    np.bool_,
    np.ndarray,
)


def _snapshot_simple_attrs(obj: Any) -> dict[str, Any]:
    """Deep-copy every attribute of ``obj`` holding a simple, mutable-state value."""
    return {k: deepcopy(v) for k, v in vars(obj).items() if isinstance(v, _SIMPLE_TYPES)}


def _restore_simple_attrs(obj: Any, saved: dict[str, Any]) -> None:
    """Write back attributes captured by :func:`_snapshot_simple_attrs`."""
    for key, value in saved.items():
        setattr(obj, key, deepcopy(value))


def _obstacle_dicts(task: Any) -> tuple[dict, ...]:
    """The three registries holding a task's obstacle objects."""
    # pylint: disable=protected-access
    return (task._geoms, task._free_geoms, task._mocaps)


def supports_snapshot(env: Any) -> bool:
    """Whether :func:`snapshot_env` can handle this environment.

    Args:
        env (Any): A (possibly wrapped) Gymnasium environment.

    Returns:
        Whether the environment exposes a MuJoCo model/data pair.
    """
    unwrapped = env.unwrapped
    return hasattr(unwrapped, 'task') or hasattr(unwrapped, 'data')


def _snapshot_wrappers(env: Any, unwrapped: Any) -> list[int | None]:
    """Capture the elapsed-step counter of every wrapper between ``env`` and the core env."""
    elapsed: list[int | None] = []
    current = env
    while current is not unwrapped:
        elapsed.append(getattr(current, '_elapsed_steps', None))
        current = current.env
    return elapsed


def _restore_wrappers(env: Any, unwrapped: Any, elapsed: list[int | None]) -> None:
    """Write back the counters captured by :func:`_snapshot_wrappers`."""
    current = env
    for value in elapsed:
        if current is unwrapped:
            break
        if value is not None:
            current._elapsed_steps = value  # noqa: SLF001  # pylint: disable=protected-access
        current = current.env


def snapshot_env(env: Any) -> dict[str, Any]:
    """Capture the complete state of a Safety-Gymnasium environment.

    Args:
        env (Any): A (possibly wrapped) Safety-Gymnasium environment.

    Returns:
        An opaque snapshot dict to hand back to :func:`restore_env`.

    Raises:
        TypeError: If the environment is not MuJoCo-backed.
    """
    unwrapped = env.unwrapped
    if not supports_snapshot(env):
        raise TypeError(f'{type(unwrapped).__name__} does not expose a MuJoCo model/data pair.')

    task = getattr(unwrapped, 'task', unwrapped)
    model, data = task.model, task.data

    snapshot: dict[str, Any] = {
        'data': {field: np.array(getattr(data, field), copy=True) for field in _DATA_FIELDS},
        'time': float(data.time),
        'model': {field: np.array(getattr(model, field), copy=True) for field in _MODEL_FIELDS},
        'wrappers': _snapshot_wrappers(env, unwrapped),
        'is_navigation': hasattr(unwrapped, 'task'),
    }

    if not snapshot['is_navigation']:
        # Velocity envs (plain Gymnasium MuJoCo) keep no extra Python state.
        return snapshot

    snapshot.update(
        {
            'builder': {field: getattr(unwrapped, field) for field in _BUILDER_FIELDS},
            # Copied as one object so the aliasing survives: ``world_info.layout``
            # *is* ``random_generator.layout``, and ``build_goal_position`` relies
            # on a mutation through one being visible through the other.
            'layout_state': deepcopy((task.world_info, task.random_generator)),
            'task': _snapshot_simple_attrs(task),
            'agent': _snapshot_simple_attrs(task.agent),
            'obstacles': {
                name: _snapshot_simple_attrs(obstacle)
                for registry in _obstacle_dicts(task)
                for name, obstacle in registry.items()
            },
        },
    )
    return snapshot


def restore_env(env: Any, snapshot: dict[str, Any], rng_seed: int | None = None) -> np.ndarray:
    """Restore a state captured by :func:`snapshot_env` and return the observation.

    Args:
        env (Any): The environment the snapshot was taken from.
        snapshot (dict[str, Any]): A snapshot from :func:`snapshot_env`.
        rng_seed (int, optional): Re-seed the task's random generator after the
            restore. Pass a different seed per Monte-Carlo sample so that the
            *environment's* stochasticity (goal re-sampling, action noise,
            frame-skip) is resampled too; pass ``None`` to replay the exact
            random stream that followed the snapshot. Defaults to None.

    Returns:
        The observation of the restored state, matching what ``step``/``reset``
        would have returned.
    """
    unwrapped = env.unwrapped
    task = getattr(unwrapped, 'task', unwrapped)
    model, data = task.model, task.data

    for field, value in snapshot['data'].items():
        getattr(data, field)[:] = value
    data.time = snapshot['time']
    for field, value in snapshot['model'].items():
        getattr(model, field)[:] = value

    if snapshot['is_navigation']:
        for field, value in snapshot['builder'].items():
            setattr(unwrapped, field, value)

        task.world_info, task.random_generator = deepcopy(snapshot['layout_state'])
        _restore_simple_attrs(task, snapshot['task'])
        _restore_simple_attrs(task.agent, snapshot['agent'])
        for registry in _obstacle_dicts(task):
            for name, obstacle in registry.items():
                _restore_simple_attrs(obstacle, snapshot['obstacles'][name])
                # The agent and the obstacles hold their own handle on the
                # random generator; re-point them at the restored one.
                obstacle.random_generator = task.random_generator
        task.agent.random_generator = task.random_generator

        if rng_seed is not None:
            task.random_generator.set_random_seed(rng_seed)

    _restore_wrappers(env, unwrapped, snapshot['wrappers'])

    mujoco.mj_forward(model, data)  # pylint: disable=no-member

    if snapshot['is_navigation']:
        return task.obs()
    return unwrapped._get_obs()  # noqa: SLF001  # pylint: disable=protected-access
