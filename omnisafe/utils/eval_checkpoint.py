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
"""Rebuild a training run's agent and eval envs from a saved checkpoint.

This is the half of out-of-process evaluation that has to be *exactly* right. The value studies
compare the critics' predictions against a Monte-Carlo return, so an evaluator that reconstructs
the agent even slightly differently from the one that trained -- a critic left at its random
initialisation, an observation normalizer not applied, a different hidden width read from the
wrong config -- still produces plausible-looking numbers. It just produces wrong ones, silently.

So the contract here is narrow and testable: :func:`load_agent_from_checkpoint` must return an
actor-critic whose every parameter equals the live one at the epoch the checkpoint was written,
and :func:`build_eval_envs` must return envs wrapped exactly as
``PolicyGradient._get_mc_value_study_env`` / ``_get_intermediate_state_env`` build them. Both
claims are asserted directly in ``tests/test_eval_checkpoint.py`` rather than assumed.

Checkpoints must carry ``reward_critic``/``cost_critic`` as well as ``pi`` (see
``PolicyGradient._init_log``'s ``what_to_save``); one written before that change has only the
actor, and :func:`load_agent_from_checkpoint` raises rather than evaluating a randomly
initialised critic.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from omnisafe.envs.core import make as make_env
from omnisafe.envs.wrapper import ActionScale, AutoReset, ObsNormalize, TimeLimit, Unsqueeze
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from omnisafe.utils.state_snapshot import enable_state_snapshots


def load_run_config(run_dir: str) -> Config:
    """Read a finished/running training run's resolved config back into a :class:`Config`.

    ``config.json`` is what the logger dumped at startup, so it is the *resolved* config --
    defaults merged with whatever ``custom_cfgs`` the launch script passed. Reading it (rather
    than re-deriving defaults from the yaml) is what guarantees the evaluator builds the same
    network shapes and the same env the run actually used.

    Args:
        run_dir (str): A run directory -- the one containing ``config.json`` and ``torch_save/``.

    Returns:
        The resolved config.
    """
    with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
        return Config.dict2config(json.load(f))


def checkpoint_epochs(run_dir: str) -> list[int]:
    """Every epoch with a checkpoint on disk, ascending.

    Args:
        run_dir (str): A run directory.

    Returns:
        Sorted epoch numbers parsed from ``torch_save/epoch-<N>.pt``.
    """
    save_dir = os.path.join(run_dir, 'torch_save')
    if not os.path.isdir(save_dir):
        return []
    epochs = []
    for name in os.listdir(save_dir):
        if name.startswith('epoch-') and name.endswith('.pt'):
            try:
                epochs.append(int(name[len('epoch-') : -len('.pt')]))
            except ValueError:
                continue
    return sorted(epochs)


def load_agent_from_checkpoint(
    run_dir: str,
    epoch: int,
    cfgs: Config | None = None,
    device: torch.device | str = 'cpu',
) -> tuple[ConstraintActorCritic, dict[str, Any] | None]:
    """Rebuild the actor-critic that training held at ``epoch``.

    The returned module is in ``eval()`` mode: every read the studies make of it is a prediction,
    never a training step, and leaving it in ``train()`` would let a critic built with
    ``model_cfgs.critic.dropout > 0`` return a *different* value for the same state on every
    call -- the training loop is careful about this too (see ``ActorCritic.__init__``).

    Args:
        run_dir (str): The run directory.
        epoch (int): Which checkpoint to load.
        cfgs (Config or None): Resolved config; read from ``run_dir`` when omitted.
        device (torch.device or str): Where to put the model.

    Returns:
        (agent, obs_normalizer_state): the rebuilt actor-critic, and the saved observation
        normalizer's ``state_dict`` (``None`` when the run had ``obs_normalize: False``).

    Raises:
        FileNotFoundError: If that epoch has no checkpoint.
        KeyError: If the checkpoint predates critics being saved, since evaluating the randomly
            initialised critic it would otherwise leave in place is worse than failing.
    """
    cfgs = cfgs if cfgs is not None else load_run_config(run_dir)
    path = os.path.join(run_dir, 'torch_save', f'epoch-{epoch}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(f'no checkpoint for epoch {epoch}: {path}')
    params = torch.load(path, map_location=device)

    missing = [k for k in ('pi', 'reward_critic') if k not in params]
    if missing:
        raise KeyError(
            f'{path} is missing {missing}. Checkpoints written before reward_critic/cost_critic '
            f'were added to what_to_save cannot be evaluated: the value studies would score a '
            f'randomly initialised critic and report the result as this epoch\'s calibration.',
        )

    # A throwaway env purely to read the spaces the networks were built against. Cheaper and
    # more reliable than reconstructing Box shapes from the config, and it is the same source
    # _init_model reads them from.
    env_cfgs = cfgs.env_cfgs.todict() if getattr(cfgs, 'env_cfgs', None) is not None else {}
    probe = make_env(cfgs.env_id, num_envs=1, device=device, **env_cfgs)
    try:
        obs_space, act_space = probe.observation_space, probe.action_space
    finally:
        probe.close()

    agent = ConstraintActorCritic(
        obs_space=obs_space,
        act_space=act_space,
        model_cfgs=cfgs.model_cfgs,
        epochs=cfgs.train_cfgs.epochs,
    ).to(device)
    agent.actor.load_state_dict(params['pi'])
    agent.reward_critic.load_state_dict(params['reward_critic'])
    if 'cost_critic' in params and getattr(agent, 'cost_critic', None) is not None:
        agent.cost_critic.load_state_dict(params['cost_critic'])
    agent.eval()
    return agent, params.get('obs_normalizer')


def _wrap(env, cfgs: Config, device: torch.device, n_envs: int, auto_reset: bool = True):
    """Apply the training wrapper recipe to a freshly made env.

    Mirrors ``OnlineAdapter._wrapper`` as ``_get_mc_value_study_env`` /
    ``_get_intermediate_state_env`` apply it: ``TimeLimit`` / ``AutoReset`` / ``ObsNormalize``
    (``update_stats=False`` -- an evaluator must never let its own rollouts drift the statistics
    it was handed) / ``ActionScale`` / ``Unsqueeze`` only when ``n_envs == 1``.
    """
    if env.need_time_limit_wrapper:
        env = TimeLimit(env, time_limit=env.max_episode_steps, device=device)
    if auto_reset and env.need_auto_reset_wrapper:
        env = AutoReset(env, device=device)
    if cfgs.algo_cfgs.obs_normalize:
        env = ObsNormalize(env, device=device, update_stats=False)
    env = ActionScale(env, low=-1.0, high=1.0, device=device)
    if n_envs == 1:
        env = Unsqueeze(env, device=device)
    return env


def build_eval_envs(
    cfgs: Config,
    device: torch.device | str = 'cpu',
    which: str = 'both',
) -> dict[str, Any]:
    """Build the eval envs the value studies need, wrapped exactly as training builds them.

    ``enable_state_snapshots()`` runs before the intermediate env is constructed, for the same
    reason ``_get_intermediate_state_env`` does it: the vectorized env forks its subprocess
    workers at construction, and they have to inherit the patched ``Builder.step``.

    Args:
        cfgs (Config): The run's resolved config.
        device (torch.device or str): Torch device.
        which (str): ``'mc'``, ``'intermediate'`` or ``'both'``.

    Returns:
        Dict with any of ``mc_env`` / ``interm_env`` plus ``max_episode_steps``.
    """
    device = torch.device(device)
    env_cfgs = cfgs.env_cfgs.todict() if getattr(cfgs, 'env_cfgs', None) is not None else {}
    out: dict[str, Any] = {}

    probe = make_env(cfgs.env_id, num_envs=1, device=device, **env_cfgs)
    out['max_episode_steps'] = probe.max_episode_steps
    probe.close()

    if which in ('mc', 'both'):
        n = int(getattr(cfgs.algo_cfgs, 'mc_value_study_vector_envs', 1))
        out['mc_env'] = _wrap(
            make_env(cfgs.env_id, num_envs=n, device=device, **env_cfgs), cfgs, device, n,
        )
    if which in ('intermediate', 'both'):
        enable_state_snapshots()
        n = int(getattr(cfgs.algo_cfgs, 'intermediate_state_study_probes', 20))
        out['interm_env'] = _wrap(
            make_env(cfgs.env_id, num_envs=n, device=device, **env_cfgs), cfgs, device, n,
        )
    return out


def apply_obs_normalizer(env, state: dict | None) -> None:
    """Load a saved observation-normalizer ``state_dict`` into ``env``'s ``ObsNormalize``.

    The in-process studies call ``sync_obs_normalizer(eval_env, train_env)`` to copy the *live*
    statistics before probing. An out-of-process evaluator has no live training env to read, so
    it uses the snapshot the checkpoint carries instead -- which is strictly better for
    reproducibility: it is pinned to the epoch being evaluated rather than to whenever the
    evaluator happened to run.

    A no-op when either side is absent (``obs_normalize: False`` runs).
    """
    if state is None:
        return
    node = env
    while node is not None:
        if isinstance(node, ObsNormalize):
            node._obs_normalizer.load_state_dict(state)  # pylint: disable=protected-access
            return
        node = getattr(node, '_env', None)


class eval_rng:  # noqa: N801  # pylint: disable=invalid-name
    """Context manager: run the value studies under a deterministic ``(run seed, epoch)`` RNG.

    Three things this buys, all needed to make evaluation trustworthy:

    * **Reproducible.** The studies roll the *stochastic* policy out, so two evaluations of the
      same checkpoint otherwise differ by Monte-Carlo noise and can only be compared
      statistically. Deriving the seed from ``(seed, epoch)`` alone means the same run config
      evaluated twice -- in-process, out-of-process, today, or re-run from checkpoints next
      month -- produces byte-identical numbers. That is what lets
      ``tests/test_eval_checkpoint.py`` assert exact equality between the two paths rather than
      "close enough given MC noise".
    * **Independent across seeds.** The run's own ``seed`` is folded in, so seed 0 and seed 1 draw
      *different* eval rollouts. Without it every seed of a study would share one eval noise
      realisation, and averaging across seeds would not average that noise away at all -- the
      error bars on a seed-averaged calibration curve would be silently too tight.
    * **No side effect on training.** The global RNG states are saved on entry and restored on
      exit, so seeding here does not perturb the stream training draws from. A run with
      evaluation enabled follows the same trajectory as one without it, which is what makes
      eval-on vs eval-off runs comparable.

    The seed is ``eval_rng_seed_base + seed * 1_000_000 + epoch``. The stride is far larger than
    any plausible epoch count, so no two ``(seed, epoch)`` pairs collide. It shares no namespace
    with the *layout* seeds (``mc_value_study_seed_offset``, the intermediate study's
    ``700_000 + ...``): those are passed to ``env.reset(seed=...)``, this one goes to
    ``torch.manual_seed``, so the values are free to overlap without interacting.

    Examples:
        >>> with eval_rng(cfgs, epoch):
        ...     stats = estimate_true_value_same_state_mc(...)
    """

    #: Multiplier on the run seed; must exceed the largest epoch a run will ever reach.
    SEED_STRIDE = 1_000_000

    def __init__(self, cfgs: Any, epoch: int) -> None:
        """Initialize with the run's resolved config and the epoch being evaluated.

        Args:
            cfgs: The resolved run config; ``cfgs.seed`` and ``cfgs.algo_cfgs.eval_rng_seed_base``
                are read from it.
            epoch (int): The epoch whose checkpoint is being evaluated.
        """
        base = int(getattr(cfgs.algo_cfgs, 'eval_rng_seed_base', 500_000))
        run_seed = int(getattr(cfgs, 'seed', 0) or 0)
        self._seed = base + run_seed * self.SEED_STRIDE + int(epoch)
        self._torch_state: torch.Tensor | None = None
        self._numpy_state: Any = None

    @property
    def seed(self) -> int:
        """The derived seed, exposed so tests and logs can assert on it."""
        return self._seed

    def __enter__(self) -> eval_rng:
        """Save the caller's RNG states, then seed deterministically."""
        self._torch_state = torch.get_rng_state()
        torch.manual_seed(self._seed)
        # Nothing on the eval path draws from numpy today, but seeding and restoring it costs
        # nothing and means a future np.random call cannot quietly break reproducibility.
        try:
            import numpy as np  # noqa: PLC0415

            self._numpy_state = np.random.get_state()
            np.random.seed(self._seed % (2**32 - 1))
        except Exception:  # noqa: BLE001  # pylint: disable=broad-except
            self._numpy_state = None
        return self

    def __exit__(self, *exc: Any) -> None:
        """Restore the caller's RNG states, leaving training's streams untouched."""
        if self._torch_state is not None:
            torch.set_rng_state(self._torch_state)
        if self._numpy_state is not None:
            import numpy as np  # noqa: PLC0415

            np.random.set_state(self._numpy_state)
