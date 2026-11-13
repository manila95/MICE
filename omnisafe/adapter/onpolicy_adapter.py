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
"""OnPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from rich.progress import track

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from omnisafe.utils.state_snapshot import configure_snapshot_triggers


class OnPolicyAdapter(OnlineAdapter):
    """OnPolicy Adapter for OmniSafe.

    :class:`OnPolicyAdapter` is used to adapt the environment to the on-policy training.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._epoch_cost_sum: float = 0.0
        # Off unless an epoch arms it; see arm_probe_capture.
        self._probe_steps: set[int] = set()
        self._probe_records: list[dict] = []
        self._probe_pending: dict[int, dict] = {}
        self._probe_taus: list[list[int]] = []
        self._probe_armed: bool = False
        self._probe_round: int = 0
        self._reset_log()

    @property
    def _sr_td_ridge(self) -> bool:
        """Whether the ``td_ridge`` successor-representation mode is active.

        .. note::
            Computed on the fly from ``self._cfgs`` (set by :class:`OnlineAdapter`) rather than
            cached at ``__init__`` time, since some subclasses (e.g. :class:`SimmerAdapter`)
            deliberately skip :class:`OnPolicyAdapter`'s own ``__init__`` body via
            ``super(OnPolicyAdapter, self).__init__(...)``.
        """
        use_sr = bool(self._cfgs.model_cfgs.get('use_successor_representation', False))
        return use_sr and self._cfgs.model_cfgs.sr_cfgs.get('sr_mode', 'shared_trunk') == 'td_ridge'

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor , reward critic
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        self._reset_log()
        self._epoch_cost_sum = 0.0

        obs, _ = self.reset()
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            act, value_r, value_c, logp = agent.step(obs)
            if self._sr_td_ridge:
                phi, psi = agent.sr_features(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            if self._probe_steps:
                self._capture_probe(
                    obs, act, logp, value_r, value_c, reward, cost, terminated, truncated, info,
                    buffer,
                )

            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            store_kwargs = {
                'obs': obs,
                'act': act,
                'reward': reward,
                'cost': cost,
                'value_r': value_r,
                'value_c': value_c,
                'logp': logp,
            }
            if self._sr_td_ridge:
                store_kwargs.update(phi=phi, psi=psi)
            buffer.store(**store_kwargs)

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        f'\nWarning: trajectory cut off when rollout by epoch\
                            in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.',
                    )

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    last_psi = torch.zeros(agent.sr_trunk.sr_dim) if self._sr_td_ridge else None
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                            if self._sr_td_ridge:
                                _, last_psi = agent.sr_features(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx],
                            )
                            if self._sr_td_ridge:
                                _, last_psi = agent.sr_features(info['final_observation'][idx])
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0

                    if self._sr_td_ridge:
                        buffer.finish_path(last_value_r, last_value_c, idx, last_psi=last_psi)
                    else:
                        # avoid passing a new kwarg to buffer subclasses (e.g. MICE's
                        # _SelectiveReinforceBuffer) that override finish_path with a narrower
                        # signature and don't expect last_psi.
                        buffer.finish_path(last_value_r, last_value_c, idx)

    # --- training-rollout probe capture ------------------------------------------------
    # The value studies score V at states drawn from *dedicated* eval rollouts. A gradient
    # comparison wants something stricter: the (s, a) pairs it averages over should be a
    # subsample of the ones the update itself was computed from, so that swapping the value
    # source is the only thing that differs. These methods capture that subsample -- the probe
    # state, the action the policy actually took there, and the state it actually reached --
    # out of the live training rollout, costing no extra environment steps.
    #
    # Sampling is uniform over the batch's transitions, which is what CPO's own gradient
    # averages over -- NOT d^pi. The policy gradient theorem wants s ~ d^pi (weight gamma^t),
    # but CPO weights all steps_per_epoch transitions equally, so probes drawn gamma^t-weighted
    # would estimate a gradient the algorithm never computes. See the undiscounted-visitation
    # discussion in the paper (Thomas 2014; Nota & Thomas 2020).

    def arm_probe_capture(
        self,
        per_episode: int,
        max_episode_steps: int,
        seed: int,
    ) -> None:
        """Capture ``(s_tau, a_tau, s_tau+1)`` triples from this epoch's training rollout.

        ``per_episode`` step indices are drawn uniformly per episode and **redrawn at every
        episode boundary**, so the batch's trajectories do not all get probed at the same few
        instants. Trigger steps are broadcast to every env slot (the snapshot mechanism cannot
        address slots individually), so within one simultaneous round of episodes the slots
        share their draw; across rounds they differ. With ``N`` episodes in the batch this
        yields ``N * per_episode`` probes.

        Snapshots fire at ``{tau, tau + 1}``: the first gives the probe state, the second the
        state its action led to, and the transition between them is read off the rollout loop
        directly. Pairing is positional, so draws are spaced at least 2 apart -- adjacent ones
        would make the trigger sets overlap and the pairs ambiguous.

        Requires :func:`~omnisafe.utils.state_snapshot.enable_state_snapshots` to have run
        before this adapter's env was constructed, or no snapshot ever arrives.

        Args:
            per_episode: Probe states to draw per episode.
            max_episode_steps: Episode length; draws come from ``[0, max_episode_steps - 3]``
                so that ``tau + 1`` is still inside the same episode.
            seed: Seed for the draws, so an epoch's probe positions are reproducible.
        """
        self._probe_rng = np.random.default_rng(seed)
        self._probe_per_episode = int(per_episode)
        self._probe_max_step = int(max_episode_steps)
        self._probe_records = []
        self._probe_pending = {}
        self._probe_taus = []
        self._probe_armed = True
        self._probe_round = 0
        self._draw_probe_steps()

    def _draw_probe_steps(self) -> None:
        """Draw this episode's probe positions and install them as snapshot triggers."""
        # Rejection-free spacing: draw from a thinned grid, then jitter. Sampling k values from
        # {0, 2, 4, ...} and doubling guarantees the >=2 spacing the pairing needs while
        # keeping the marginal close to uniform over the episode.
        hi = (self._probe_max_step - 2) // 2
        picks = self._probe_rng.choice(hi, size=min(self._probe_per_episode, hi), replace=False)
        steps = sorted(int(2 * p) for p in picks)
        triggers: set[int] = set()
        for t in steps:
            triggers.update({t, t + 1})
        configure_snapshot_triggers(self._env, triggers)
        self._probe_steps = set(steps)
        self._probe_taus.append(steps)

    def disarm_probe_capture(self) -> None:
        """Stop capturing and clear the triggers, so non-eval epochs pay nothing."""
        configure_snapshot_triggers(self._env, None)
        self._probe_steps = set()
        self._probe_pending = {}
        self._probe_armed = False

    def probe_records(self) -> list[dict]:
        """The ``(s, a, s')`` probes captured since the last :meth:`arm_probe_capture`."""
        return self._probe_records

    def probe_taus(self) -> list[list[int]]:
        """The step indices drawn for each round of episodes, in order."""
        return self._probe_taus

    def _capture_probe(  # pylint: disable=too-many-arguments
        self, obs, act, logp, value_r, value_c, reward, cost, terminated, truncated, info,
        buffer=None,
    ) -> None:
        """Pair this step's snapshots with the transition that was just taken.

        Called from :meth:`rollout` *before* ``buffer.store(...)``, so each sub-buffer's write
        pointer is exactly the row this transition is about to occupy -- that pointer is what
        ``'row'`` records below.
        """
        snaps = info.get('state_snapshot')
        mask = info.get('_state_snapshot')
        if snaps is not None:
            # A vectorized env hands back a numpy OBJECT ARRAY of per-slot dicts (with a
            # boolean mask in '_state_snapshot'); a single env hands back the dict itself.
            # Test for the dict, not for list/tuple -- an ndarray is neither, so the old test
            # wrapped the whole array and then indexed it as if it were one snapshot.
            if isinstance(snaps, dict):  # single, non-vectorized env
                snaps, mask = [snaps], None
            for idx in range(len(snaps)):
                if mask is not None and not mask[idx]:
                    continue
                pending = self._probe_pending.pop(idx, None)
                if pending is None:
                    # Snapshot of s_tau. The action out of it is taken on the next iteration,
                    # so only the state is stashed here.
                    self._probe_pending[idx] = snaps[idx]
                else:
                    # This iteration stepped s_tau -> s_tau+1, so obs/act/reward/cost below are
                    # exactly that transition and snaps[idx] is where it landed. tau is read
                    # off the snapshot's own step counter rather than tracked separately.
                    self._probe_records.append({
                        'obs': obs[idx].detach().cpu().clone(),
                        'action': act[idx].detach().cpu().clone(),
                        'logp': float(logp[idx]),
                        'reward': float(reward[idx]),
                        'cost': float(cost[idx]),
                        'value_r': float(value_r[idx]),
                        'value_c': float(value_c[idx]),
                        'terminated': bool(terminated[idx]),
                        'tau': int(pending['steps']),
                        'env_slot': idx,
                        # Which episode of this slot's epoch -- kept for grouping probes by
                        # episode, NOT for locating them in the buffer.
                        'episode': self._probe_round,
                        # The sub-buffer row this transition lands in, read off the write
                        # pointer rather than rebuilt from (episode, tau). The arithmetic
                        # episode * ep_len + tau assumes every episode fills exactly ep_len
                        # rows, and it disagreed with the buffer on ~20% of eval epochs even
                        # with fixed-length episodes, so the pointer is the only source of
                        # truth. Absolute row = env_slot * max_size + row.
                        'row': int(buffer.buffers[idx].ptr) if buffer is not None else -1,
                        'snapshot': pending,
                        'next_snapshot': snaps[idx],
                    })

        # An episode boundary would leave a tau snapshot stranded, and the next episode's tau
        # snapshot would then wrongly complete it -- so drop pending and redraw. Slots run in
        # lockstep here (fixed-length episodes, see collect_on_policy_snapshots's docstring), so
        # one slot finishing means they all did.
        done = [bool(terminated[i]) or bool(truncated[i]) for i in range(len(terminated))]
        if any(done):
            self._probe_pending = {}
            self._probe_round += 1
            if self._probe_armed:
                self._draw_probe_steps()

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += 1
        self._epoch_cost_sum += info.get('original_cost', cost).sum().item()

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        if hasattr(self._env, 'spec_log'):
            self._env.spec_log(logger)
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0
