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
"""Implementation of OnPolicyBuffer."""

from __future__ import annotations

import torch

from omnisafe.common.buffer.base import BaseBuffer
from omnisafe.typing import DEVICE_CPU, AdvatageEstimator, OmnisafeSpace
from omnisafe.utils import distributed
from omnisafe.utils.math import discount_cumsum
from omnisafe.utils.sr_diagnostics import correlation


ADV_NORM_MODES = ('batch', 'timestep')


def grouped_statistics(  # pylint: disable=too-many-arguments
    values: torch.Tensor,
    group_idx: torch.Tensor,
    num_groups: int,
    min_count: int,
    fallback_mean: torch.Tensor,
    fallback_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Per-group mean/std of ``values``, broadcast back to one entry per sample.

    Used to normalize advantages against the statistics of their own episode timestep rather
    than of the whole epoch: ``group_idx`` holds the timestep of each sample and ``num_groups``
    is one past the largest timestep the buffer can hold.

    Groups holding fewer than ``min_count`` samples fall back to ``fallback_mean`` /
    ``fallback_std`` (the statistics over the whole batch).  Without that fallback a group of
    one would normalize its single sample to exactly zero, silently dropping it from the
    gradient, and a group of two or three would divide by a std estimated from noise.

    Statistics are pooled across MPI processes, so every rank normalizes with the same
    constants even though each holds a different slice of the batch.

    Args:
        values (torch.Tensor): The per-sample quantity to compute statistics of, shape ``(N,)``.
        group_idx (torch.Tensor): Integer group of each sample, shape ``(N,)``, in
            ``[0, num_groups)``.
        num_groups (int): Number of groups.
        min_count (int): Minimum samples a group needs before its own statistics are used.
        fallback_mean (torch.Tensor): Mean used for groups below ``min_count``.
        fallback_std (torch.Tensor): Std used for groups below ``min_count``.

    Returns:
        mean (torch.Tensor): The mean to subtract from each sample, shape ``(N,)``.
        std (torch.Tensor): The std to divide each sample by, shape ``(N,)``.
    """
    device = values.device
    zeros = torch.zeros(num_groups, dtype=torch.float32, device=device)
    count = zeros.clone().index_add_(0, group_idx, torch.ones_like(values))
    total = zeros.clone().index_add_(0, group_idx, values)
    count = distributed.dist_sum(count).to(device)
    total = distributed.dist_sum(total).to(device)
    mean = total / count.clamp(min=1.0)

    # Second pass against the pooled mean rather than the E[x^2] - E[x]^2 shortcut, which
    # cancels catastrophically for large tightly-clustered returns -- exactly the regime the
    # early-timestep groups are in when gamma is close to 1.
    total_sq = zeros.clone().index_add_(0, group_idx, (values - mean[group_idx]) ** 2)
    total_sq = distributed.dist_sum(total_sq).to(device)
    # Population std, matching :func:`distributed.dist_statistics_scalar`.
    std = (total_sq / count.clamp(min=1.0)).sqrt()

    enough = count >= min_count
    mean = torch.where(enough, mean, fallback_mean.to(device))
    std = torch.where(enough, std, fallback_std.to(device))
    return mean[group_idx], std[group_idx]


def standardize_advantages(  # pylint: disable=too-many-arguments
    data: dict[str, torch.Tensor],
    standardized_adv_r: bool,
    standardized_adv_c: bool,
    adv_norm_mode: str,
    num_timesteps: int,
    timestep_min_count: int,
) -> None:
    r"""Standardize ``adv_r`` / ``adv_c`` in ``data``, in place.

    Two modes, selected by ``adv_norm_mode``:

    - ``'batch'`` (the original behaviour): one mean/std pair for the whole epoch, so
      :math:`A_r \leftarrow (A_r - \mu) / \sigma` and :math:`A_c \leftarrow A_c - \mu_c`.
    - ``'timestep'``: the statistics are computed separately for each episode timestep
      :math:`t`, over every sample in the batch that sits at that timestep, so
      :math:`A_r[i] \leftarrow (A_r[i] - \mu_{t_i}) / \sigma_{t_i}`.  With an undiscounted-ish
      :math:`\gamma` the return-to-go shrinks monotonically towards the end of an episode, and
      batch statistics therefore encode mostly *when* a transition happened rather than how
      good it was; per-timestep statistics remove that trend and leave only the within-timestep
      ranking.

    In both modes the cost advantage is only centred, never rescaled -- the convention the
    Lagrangian algorithms' multiplier is tuned against.

    Args:
        data (dict[str, torch.Tensor]): Batch to modify, needs ``adv_r``, ``adv_c`` and (for
            ``'timestep'``) ``time_step``.
        standardized_adv_r (bool): Whether to standardize the reward advantage at all.
        standardized_adv_c (bool): Whether to centre the cost advantage at all.
        adv_norm_mode (str): One of :data:`ADV_NORM_MODES`.
        num_timesteps (int): One past the largest episode timestep the buffer can hold.
        timestep_min_count (int): Minimum samples a timestep needs before its own statistics
            are used instead of the whole-batch ones.
    """
    adv_mean, adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'])
    cadv_mean, cadv_std, *_ = distributed.dist_statistics_scalar(data['adv_c'])

    if adv_norm_mode == 'timestep':
        group_idx = data['time_step'].long().clamp_(0, num_timesteps - 1)
        r_mean, r_std = grouped_statistics(
            data['adv_r'], group_idx, num_timesteps, timestep_min_count, adv_mean, adv_std,
        )
        c_mean, _ = grouped_statistics(
            data['adv_c'], group_idx, num_timesteps, timestep_min_count, cadv_mean, cadv_std,
        )
    else:
        r_mean, r_std, c_mean = adv_mean, adv_std, cadv_mean

    if standardized_adv_r:
        data['adv_r'] = (data['adv_r'] - r_mean) / (r_std + 1e-8)
    if standardized_adv_c:
        data['adv_c'] = data['adv_c'] - c_mean


def timestep_baseline_diagnostics(
    data: dict[str, torch.Tensor],
    num_timesteps: int,
    timestep_min_count: int,
) -> dict[str, float]:
    r"""How much of the return-to-go is explained by the episode timestep alone.

    For each signal this takes the per-timestep mean of the observed (Monte-Carlo, un-bootstrapped)
    return-to-go -- the baseline ``adv_norm_mode='timestep'`` subtracts -- and correlates it against
    the individual returns it is subtracted from.  Its square is the fraction of the return variance
    that is pure timing: a correlation near 1 means the returns mostly say *when* a transition
    happened, which is the case ``adv_norm_mode='timestep'`` exists to fix; a correlation near 0
    means the timestep carries no information and the mode has nothing to remove.

    Computed on the full epoch batch in :meth:`get`, before the train/val split and before any
    standardization, so it describes the rollout rather than the normalized advantages.  It is
    computed in both modes -- under ``'batch'`` it measures what is *not* being removed.

    The per-timestep means use the same ``timestep_min_count`` gate as the normalization itself,
    so the logged number describes the baseline actually in use rather than an idealized one.

    Args:
        data (dict[str, torch.Tensor]): The epoch batch, as returned by :meth:`get`.
        num_timesteps (int): One past the largest episode timestep the buffer can hold.
        timestep_min_count (int): Minimum samples a timestep needs before its own mean is used.

    Returns:
        Logger keys mapped to Pearson correlations, ``nan`` where the signal is constant (an
        all-zero cost stream, say) or the batch lacks the fields to compute it.
    """
    stats = {
        'Value/TimestepBaseline/RewardCorr': float('nan'),
        'Value/TimestepBaseline/CostCorr': float('nan'),
    }
    if 'time_step' not in data:
        return stats

    group_idx = data['time_step'].long().clamp(0, num_timesteps - 1)
    for key, label in (('discounted_ret', 'Reward'), ('discounted_cost_ret', 'Cost')):
        if key not in data:
            continue
        returns = data[key]
        mean, std, *_ = distributed.dist_statistics_scalar(returns)
        timestep_mean, _ = grouped_statistics(
            returns, group_idx, num_timesteps, timestep_min_count, mean, std,
        )
        stats[f'Value/TimestepBaseline/{label}Corr'] = correlation(timestep_mean, returns)
    return stats


class OnPolicyBuffer(BaseBuffer):  # pylint: disable=too-many-instance-attributes
    """A buffer for storing trajectories experienced by an agent interacting with the environment.

    Besides, The buffer also provides the functionality of calculating the advantages of
    state-action pairs, ranging from ``GAE``, ``GAE-RTG`` , ``V-trace`` to ``Plain`` method.

    .. warning::
        The buffer only supports Box spaces.

    Compared to the base buffer, the on-policy buffer stores extra data:

    +----------------+---------+---------------+----------------------------------------+
    | Name           | Shape   | Dtype         | Shape                                  |
    +================+=========+===============+========================================+
    | discounted_ret | (size,) | torch.float32 | The discounted sum of return.          |
    +----------------+---------+---------------+----------------------------------------+
    | value_r        | (size,) | torch.float32 | The value estimated by reward critic.  |
    +----------------+---------+---------------+----------------------------------------+
    | value_c        | (size,) | torch.float32 | The value estimated by cost critic.    |
    +----------------+---------+---------------+----------------------------------------+
    | adv_r          | (size,) | torch.float32 | The advantage of the reward.           |
    +----------------+---------+---------------+----------------------------------------+
    | adv_c          | (size,) | torch.float32 | The advantage of the cost.             |
    +----------------+---------+---------------+----------------------------------------+
    | target_value_r | (size,) | torch.float32 | The target value of the reward critic. |
    +----------------+---------+---------------+----------------------------------------+
    | target_value_c | (size,) | torch.float32 | The target value of the cost critic.   |
    +----------------+---------+---------------+----------------------------------------+
    | logp           | (size,) | torch.float32 | The log probability of the action.     |
    +----------------+---------+---------------+----------------------------------------+
    | time_step      | (size,) | torch.float32 | The index of the step in its episode.  |
    +----------------+---------+---------------+----------------------------------------+

    In ``td_ridge`` successor-representation mode (``sr_dim`` not ``None``) four more fields hold
    the vector-valued feature stream, all of shape ``(size, sr_dim)``: ``phi`` and ``psi`` are the
    one-step and successor features as evaluated during the rollout, ``target_sr`` is the
    bootstrapped lambda-target ``psi`` is trained against, and ``discounted_sr`` is the
    Monte-Carlo successor feature -- the vector counterpart of ``discounted_ret``, used only by
    the diagnostics in :mod:`omnisafe.utils.sr_diagnostics`.

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        size (int): The size of the buffer.
        gamma (float): The discount factor.
        lam (float): The lambda factor for calculating the advantages.
        lam_c (float): The lambda factor for calculating the advantages of the critic.
        advantage_estimator (AdvatageEstimator): The advantage estimator.
        penalty_coefficient (float, optional): The penalty coefficient. Defaults to 0.
        standardized_adv_r (bool, optional): Whether to standardize the advantages of the actor.
            Defaults to False.
        standardized_adv_c (bool, optional): Whether to standardize the advantages of the critic.
            Defaults to False.
        device (torch.device, optional): The device to store the data. Defaults to
            ``torch.device('cpu')``.
        adv_norm_mode (str, optional): How the standardization statistics are pooled --
            ``'batch'`` for one mean/std over the whole epoch, ``'timestep'`` for a separate
            mean/std per episode timestep. Defaults to ``'batch'``. See
            :func:`standardize_advantages`.
        adv_norm_timestep_min_count (int, optional): With ``adv_norm_mode='timestep'``, the
            minimum number of samples a timestep needs before its own statistics are used
            instead of the whole-batch ones. Defaults to 4.

    Attributes:
        ptr (int): The pointer of the buffer.
        path_start (int): The start index of the current path.
        max_size (int): The maximum size of the buffer.
        data (dict): The data stored in the buffer.
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        device (torch.device): The device to store the data.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float = 0,
        standardized_adv_r: bool = False,
        standardized_adv_c: bool = False,
        device: torch.device = DEVICE_CPU,
        cost_gamma: float | None = None,
        cost_advantage_estimator: AdvatageEstimator | None = None,
        sr_dim: int | None = None,
        lam_sr: float = 0.95,
        gamma_sr: float | None = None,
        adv_norm_mode: str = 'batch',
        adv_norm_timestep_min_count: int = 4,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyBuffer`."""
        super().__init__(obs_space, act_space, size, device)

        self._standardized_adv_r: bool = standardized_adv_r
        self._standardized_adv_c: bool = standardized_adv_c
        assert adv_norm_mode in ADV_NORM_MODES, f'adv_norm_mode must be one of {ADV_NORM_MODES}!'
        self._adv_norm_mode: str = adv_norm_mode
        self._adv_norm_timestep_min_count: int = adv_norm_timestep_min_count
        self.data['adv_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['discounted_ret'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['discounted_cost_ret'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['value_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['target_value_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['adv_c'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['value_c'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['target_value_c'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['logp'] = torch.zeros((size,), dtype=torch.float32, device=device)
        # Index of each transition within its own episode, filled in by :meth:`finish_path`.
        # Needed by the ``timestep`` advantage-normalization mode, and useful on its own for
        # any diagnostic that wants to condition on how far into an episode a sample is.
        self.data['time_step'] = torch.zeros((size,), dtype=torch.float32, device=device)

        self._gamma: float = gamma
        self._cost_gamma: float = cost_gamma if cost_gamma is not None else gamma
        self._lam: float = lam
        self._lam_c: float = lam_c
        self._penalty_coefficient: float = penalty_coefficient
        self._advantage_estimator: AdvatageEstimator = advantage_estimator
        self._cost_advantage_estimator: AdvatageEstimator = (
            cost_advantage_estimator if cost_advantage_estimator is not None else advantage_estimator
        )
        self.ptr: int = 0
        self.path_start_idx: int = 0
        self.max_size: int = size
        self._episode_slices: list[tuple[int, int]] = []
        self._last_episode_slices: list[tuple[int, int]] = []
        self._timestep_baseline_stats: dict[str, float] = {}

        _valid = ['gae', 'gae-rtg', 'vtrace', 'plain', 'reinforce', 'td_zero', 'td_zero_gae']
        assert self._penalty_coefficient >= 0, 'penalty_coefficient must be non-negative!'
        assert self._advantage_estimator in _valid
        assert self._cost_advantage_estimator in _valid

        # successor-representation (``td_ridge`` mode) extra fields: a d-dimensional feature
        # stream ``phi``/``psi`` trained with the same estimator machinery as the scalar
        # reward/cost streams above (see :meth:`finish_path`).
        self._sr_dim: int | None = sr_dim
        self._lam_sr: float = lam_sr
        self._gamma_sr: float = gamma if gamma_sr is None else gamma_sr
        if self._sr_dim is not None:
            self.data['phi'] = torch.zeros((size, sr_dim), dtype=torch.float32, device=device)
            self.data['psi'] = torch.zeros((size, sr_dim), dtype=torch.float32, device=device)
            self.data['target_sr'] = torch.zeros((size, sr_dim), dtype=torch.float32, device=device)
            # Monte-Carlo successor feature: the vector-valued counterpart of ``discounted_ret``,
            # and the ground truth the bootstrapped ``target_sr`` is scored against by the SR
            # diagnostics. Diagnostic-only -- nothing trains on it.
            self.data['discounted_sr'] = torch.zeros(
                (size, sr_dim),
                dtype=torch.float32,
                device=device,
            )

    @property
    def standardized_adv_r(self) -> bool:
        """Whether to standardize the advantages of the actor."""
        return self._standardized_adv_r

    @property
    def timestep_baseline_stats(self) -> dict[str, float]:
        """Diagnostics from the most recent :meth:`get`, ready to hand to the logger."""
        return self._timestep_baseline_stats

    @property
    def standardized_adv_c(self) -> bool:
        """Whether to standardize the advantages of the critic."""
        return self._standardized_adv_c

    def store(self, **data: torch.Tensor) -> None:
        """Store data into the buffer.

        .. warning::
            The total size of the data must be less than the buffer size.

        Args:
            data (torch.Tensor): The data to store.
        """
        assert self.ptr < self.max_size, 'No more space in the buffer!'
        for key, value in data.items():
            self.data[key][self.ptr] = value
        self.ptr += 1

    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_c: torch.Tensor | None = None,
        last_psi: torch.Tensor | None = None,
    ) -> None:
        """Finish the current path and calculate the advantages of state-action pairs.

        On-policy algorithms need to calculate the advantages of state-action pairs
        after the path is finished. This function calculates the advantages of
        state-action pairs and stores them in the buffer, following the steps:

        .. hint::
            #. Calculate the discounted return.
            #. Calculate the advantages of the reward.
            #. Calculate the advantages of the cost.
            #. (``td_ridge`` successor-representation mode only) Calculate the vector-valued
               successor-representation target, using the same estimator machinery as steps
               2-3 above, applied to the ``phi``/``psi`` feature stream instead of the scalar
               reward/cost stream.

        Args:
            last_value_r (torch.Tensor, optional): The value of the last state of the current path.
                Defaults to torch.zeros(1).
            last_value_c (torch.Tensor, optional): The value of the last state of the current path.
                Defaults to torch.zeros(1).
            last_psi (torch.Tensor, optional): The successor-representation feature of the last
                state of the current path (``td_ridge`` mode only). Defaults to torch.zeros(sr_dim).
        """
        if last_value_r is None:
            last_value_r = torch.zeros(1, device=self._device)
        if last_value_c is None:
            last_value_c = torch.zeros(1, device=self._device)

        path_slice = slice(self.path_start_idx, self.ptr)
        last_value_r = last_value_r.to(self._device)
        last_value_c = last_value_c.to(self._device)

        # A path always starts at an episode boundary -- the adapter resets every environment at
        # the start of an epoch and calls :meth:`finish_path` on every termination, timeout and
        # epoch cut -- so position within the path is the episode timestep.
        self.data['time_step'][path_slice] = torch.arange(
            self.ptr - self.path_start_idx,
            dtype=torch.float32,
            device=self._device,
        )

        self.data['discounted_ret'][path_slice] = discount_cumsum(
            self.data['reward'][path_slice], self._gamma
        )
        self.data['discounted_cost_ret'][path_slice] = discount_cumsum(
            self.data['cost'][path_slice], self._cost_gamma
        )
        rewards = torch.cat([self.data['reward'][path_slice], last_value_r])
        values_r = torch.cat([self.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self.data['cost'][path_slice], last_value_c])
        values_c = torch.cat([self.data['value_c'][path_slice], last_value_c])
        rewards -= self._penalty_coefficient * costs

        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            values_r,
            rewards,
            lam=self._lam,
        )
        adv_c, target_value_c = self._calculate_adv_and_value_targets(
            values_c,
            costs,
            lam=self._lam_c,
            gamma=self._cost_gamma,
            advantage_estimator=self._cost_advantage_estimator,
        )

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_c'][path_slice] = adv_c
        self.data['target_value_c'][path_slice] = target_value_c

        if self._sr_dim is not None:
            if last_psi is None:
                last_psi = torch.zeros(self._sr_dim, device=self._device)
            last_psi = last_psi.to(self._device).reshape(1, self._sr_dim)
            # mirrors the scalar reward/cost case: the bootstrap value is appended as the
            # pseudo-final entry of the "reward" stream too, so gae-rtg/plain/reinforce-style
            # rewards-to-go targets correctly fold in the truncation bootstrap.
            phi_with_boot = torch.cat([self.data['phi'][path_slice], last_psi], dim=0)
            psi_with_boot = torch.cat([self.data['psi'][path_slice], last_psi], dim=0)
            _, target_sr = self._calculate_adv_and_value_targets(
                psi_with_boot,
                phi_with_boot,
                lam=self._lam_sr,
                gamma=self._gamma_sr,
            )
            self.data['target_sr'][path_slice] = target_sr
            # Mirrors ``discounted_ret`` exactly -- a plain Monte-Carlo sum over the path with no
            # truncation bootstrap -- so the two "true" quantities carry the same bias and the SR
            # and value diagnostics stay comparable.
            self.data['discounted_sr'][path_slice] = discount_cumsum(
                self.data['phi'][path_slice],
                self._gamma_sr,
            )

        self._episode_slices.append((self.path_start_idx, self.ptr))
        self.path_start_idx = self.ptr

    def get(self) -> dict[str, torch.Tensor]:
        """Get the data in the buffer.

        .. hint::
            We provide a trick to standardize the advantages of state-action pairs. We calculate the
            mean and standard deviation of the advantages of state-action pairs and then standardize
            the advantages of state-action pairs. You can turn on this trick by setting the
            ``standardized_adv_r`` to ``True``. The same trick is applied to the advantages of the
            cost.

        Returns:
            The data stored and calculated in the buffer.
        """
        self._last_episode_slices = list(self._episode_slices)
        self._episode_slices = []
        self.ptr, self.path_start_idx = 0, 0

        data = {
            'obs': self.data['obs'],
            'act': self.data['act'],
            'target_value_r': self.data['target_value_r'],
            'adv_r': self.data['adv_r'],
            'logp': self.data['logp'],
            'discounted_ret': self.data['discounted_ret'],
            'discounted_cost_ret': self.data['discounted_cost_ret'],
            'value_r': self.data['value_r'],
            'value_c': self.data['value_c'],
            'adv_c': self.data['adv_c'],
            'target_value_c': self.data['target_value_c'],
            'time_step': self.data['time_step'],
        }
        if self._sr_dim is not None:
            data['phi'] = self.data['phi']
            data['target_sr'] = self.data['target_sr']
            data['reward'] = self.data['reward']
            data['cost'] = self.data['cost']
            # ``psi`` is the rollout-time successor feature, the SR counterpart of ``value_r`` /
            # ``value_c``: it gives the diagnostics a pre-update prediction for free, without
            # having to snapshot the network before the gradient loop.
            data['psi'] = self.data['psi']
            data['discounted_sr'] = self.data['discounted_sr']

        self._timestep_baseline_stats = timestep_baseline_diagnostics(
            data,
            num_timesteps=self.max_size,
            timestep_min_count=self._adv_norm_timestep_min_count,
        )
        standardize_advantages(
            data,
            standardized_adv_r=self._standardized_adv_r,
            standardized_adv_c=self._standardized_adv_c,
            adv_norm_mode=self._adv_norm_mode,
            num_timesteps=self.max_size,
            timestep_min_count=self._adv_norm_timestep_min_count,
        )

        return data

    def _calculate_adv_and_value_targets(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        lam: float,
        gamma: float | None = None,
        advantage_estimator: AdvatageEstimator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute the estimated advantage.

        Three methods are supported:

        - GAE (Generalized Advantage Estimation)

            GAE is a variance reduction method for the actor-critic algorithm. It is proposed in the
            paper `High-Dimensional Continuous Control Using Generalized Advantage Estimation <https://arxiv.org/abs/1506.02438>`_.

            GAE calculates the advantage using the following formula:

            .. math::

                A_t = \sum_{k=0}^{n-1} (\lambda \gamma)^k \delta_{t+k}

            where :math:`\delta_{t+k} = r_{t+k} + \gamma*V(s_{t+k+1}) - V(s_{t+k})`. When
            :math:`\lambda =1`, GAE reduces to the Monte Carlo method, which is unbiased but has high
            variance. When :math:`\lambda =0`, GAE reduces to the TD(1) method, which is biased but has
            low variance.

        - V-trace

            V-trace is a variance reduction method for the actor-critic algorithm. It is proposed in
            the paper `IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures <https://arxiv.org/abs/1802.01561>`_.

            V-trace calculates the advantage using the following formula:

            .. math::

                A_t = \sum_{k=0}^{n-1} (\lambda \gamma)^k \delta_{t+k} +
                    (\lambda \gamma)^n \rho_{t+n} (1 - d_{t+n}) (V(x_{t+n}) - b_{t+n})

            where :math:`\delta_{t+k} = r_{t+k} + \gamma*V(s_{t+k+1}) - V(s_{t+k})`,
            :math:`\rho_{t+k} = \frac{\pi(a_{t+k}|s_{t+k})}{b_{t+k}}`, :math:`b_{t+k}` is the
            behavior policy, and :math:`d_{t+k}` is the done flag.

        - Plain

            Plain method is the original actor-critic algorithm. It is unbiased but has high
            variance.

        Args:
            vals (torch.Tensor): The value of states.
            rews (torch.Tensor): The reward of states.
            lam (float): The lambda parameter in GAE formula.

        Returns:
            adv (torch.Tensor): The estimated advantage.
            target_value (torch.Tensor): The target value for the value function.

        Raises:
            NotImplementedError: If the advantage estimator is not supported.
        """  # pylint: disable=line-too-long
        g = gamma if gamma is not None else self._gamma
        estimator = advantage_estimator if advantage_estimator is not None else self._advantage_estimator
        if estimator == 'gae':
            # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
            deltas = rewards[:-1] + g * values[1:] - values[:-1]
            adv = discount_cumsum(deltas, g * lam)
            target_value = adv + values[:-1]

        elif estimator == 'gae-rtg':
            # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
            deltas = rewards[:-1] + g * values[1:] - values[:-1]
            adv = discount_cumsum(deltas, g * lam)
            # compute rewards-to-go, to be targets for the value function update
            target_value = discount_cumsum(rewards, g)[:-1]

        elif estimator == 'vtrace':
            #  v_s = V(x_s) + \sum^{T-1}_{t=s} \gamma^{t-s}
            #                * \prod_{i=s}^{t-1} c_i
            #                 * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
            path_slice = slice(self.path_start_idx, self.ptr)
            action_probs = self.data['logp'][path_slice].exp()
            target_value, adv, _ = self._calculate_v_trace(
                policy_action_probs=action_probs,
                values=values,
                rewards=rewards,
                behavior_action_probs=action_probs,
                gamma=g,
                rho_bar=1.0,
                c_bar=1.0,
            )

        elif estimator == 'plain':
            # A(x, u) = Q(x, u) - V(x) = r(x, u) + gamma V(x+1) - V(x)
            adv = rewards[:-1] + g * values[1:] - values[:-1]
            target_value = discount_cumsum(rewards, g)[:-1]

        elif estimator == 'reinforce':
            # Pure REINFORCE: A_t = G_t (no value baseline subtracted)
            # G_t is the full bootstrapped discounted return (last_value already appended)
            returns = discount_cumsum(rewards, g)[:-1]
            adv = returns
            target_value = returns
        
        elif estimator == "td_zero":
            # TD(0): A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            adv = rewards[:-1] + g * values[1:] - values[:-1]
            target_value = rewards[:-1] + g * values[1:]

        elif estimator == "td_zero_gae":
            # TD(0): A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            deltas = rewards[:-1] + g * values[1:] - values[:-1]
            adv = discount_cumsum(deltas, g * lam)
            target_value = rewards[:-1] + g * values[1:]

        else:
            raise NotImplementedError

        return adv, target_value

    @staticmethod
    # pylint: disable-next=too-many-arguments,too-many-locals
    def _calculate_v_trace(
        policy_action_probs: torch.Tensor,
        values: torch.Tensor,  # including bootstrap
        rewards: torch.Tensor,  # including bootstrap
        behavior_action_probs: torch.Tensor,
        gamma: float = 0.99,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""This function is used to calculate V-trace targets.

        .. math::

            A_t = \sum_{k=0}^{n-1} (\lambda \gamma)^k \delta_{t+k} +
                (\lambda \gamma)^n \rho_{t+n} (1 - d_{t+n}) (V(x_{t+n}) - b_{t+n})

        Calculate V-trace targets for off-policy actor-critic learning recursively. For more
        details, please refer to the paper: `Espeholt et al. 2018, IMPALA <https://arxiv.org/abs/1802.01561>`_.

        Args:
            policy_action_probs (torch.Tensor): Action probabilities of the policy.
            values (torch.Tensor): The value of states.
            rewards (torch.Tensor): The reward of states.
            behavior_action_probs (torch.Tensor): Action probabilities of the behavior policy.
            gamma (float, optional): The discount factor. Defaults to 0.99.
            rho_bar (float, optional): The maximum value of importance weights. Defaults to 1.0.
            c_bar (float, optional): The maximum value of clipped importance weights. Defaults to 1.0.

        Returns:
            V-trace targets, shape=(batch_size, sequence_length)

        Raises:
            AssertionError: If the input tensors are scalars.
            AssertionError: If c_bar is greater than rho_bar.
        """
        assert values.ndim in (1, 2), 'Please provide arrays instead of scalars'
        assert rewards.ndim in (1, 2), 'Please provide arrays instead of scalars'
        assert policy_action_probs.ndim == 1, 'Please provide arrays instead of scalars'
        assert behavior_action_probs.ndim == 1, 'Please provide arrays instead of scalars'
        assert c_bar <= rho_bar, 'c_bar should be less than or equal to rho_bar'

        sequence_length = policy_action_probs.shape[0]
        # pylint: disable-next=assignment-from-no-return
        rhos = torch.div(policy_action_probs, behavior_action_probs)
        clip_rhos = torch.min(
            rhos,
            torch.as_tensor(rho_bar),
        )  # pylint: disable=assignment-from-no-return
        clip_cs = torch.min(
            rhos,
            torch.as_tensor(c_bar),
        )  # pylint: disable=assignment-from-no-return
        if values.ndim == 2:
            # broadcast the per-timestep scalar importance ratio against a (T, D) feature
            # stream (used for the successor-representation vector target).
            clip_rhos = clip_rhos.unsqueeze(-1)
            clip_cs = clip_cs.unsqueeze(-1)
        v_s = values[:-1].clone()  # copy all values except bootstrap value
        last_v_s = values[-1]  # bootstrap from last state

        # calculate v_s
        for index in reversed(range(sequence_length)):
            delta = clip_rhos[index] * (rewards[index] + gamma * values[index + 1] - values[index])
            v_s[index] += delta + gamma * clip_cs[index] * (last_v_s - values[index + 1])
            last_v_s = v_s[index]  # accumulate current v_s for next iteration

        # calculate q_targets
        v_s_plus_1 = torch.cat((v_s[1:], values[-1:]))
        policy_advantage = clip_rhos * (rewards[:-1] + gamma * v_s_plus_1 - values[:-1])

        return v_s, policy_advantage, clip_rhos
