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
"""Implementation of the HL-Gauss VCritic."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class VCriticHLGauss(Critic):
    """V-function approximator trained via classification (HL-Gauss).

    Instead of regressing a scalar value, each network outputs ``num_bins`` logits over a fixed
    set of bins spanning ``[v_min, v_max]``.  A scalar target is projected onto this support by
    integrating a Gaussian (centred at the target, with standard deviation ``sigma``) over each
    bin, and the network is trained with a cross-entropy loss against that soft target
    distribution.  The scalar value is recovered as the expectation of the categorical
    distribution, so this critic is a drop-in replacement for :class:`VCritic`: :meth:`forward`
    still returns scalar V-values.

    References:
        - Title: Stop Regressing: Training Value Functions via Classification for Scalable Deep RL
        - Authors: Jesse Farebrother, Jordi Orbay, Quan Vuong, Adrien Ali Taïga, Yevgen Chebotar,
          Ted Xiao, Alex Irpan, Sergey Levine, Pablo Samuel Castro, Aleksandra Faust,
          Aviral Kumar, Rishabh Agarwal.
        - URL: `HL-Gauss <https://arxiv.org/abs/2403.03950>`_

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        num_critics (int, optional): Number of critics. Defaults to 1.
        v_min (float, optional): Lower edge of the value support. Defaults to ``-10.0``.
        v_max (float, optional): Upper edge of the value support. Defaults to ``10.0``.
        num_bins (int, optional): Number of bins spanning the support. Defaults to ``51``.
        sigma_ratio (float, optional): Gaussian standard deviation as a multiple of the bin width.
            Defaults to ``0.75`` (the value recommended by the paper).
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        v_min: float = -10.0,
        v_max: float = 10.0,
        num_bins: int = 51,
        sigma_ratio: float = 0.75,
    ) -> None:
        """Initialize an instance of :class:`VCriticHLGauss`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics,
            use_obs_encoder=False,
        )
        assert v_max > v_min, 'v_max must be greater than v_min.'
        assert num_bins >= 2, 'num_bins must be at least 2.'
        assert sigma_ratio > 0, 'sigma_ratio must be positive.'

        self._num_bins: int = num_bins
        bin_width = (v_max - v_min) / num_bins
        self._sigma: float = sigma_ratio * bin_width

        # ``num_bins + 1`` edges and ``num_bins`` centres, stored as buffers so they move with
        # ``.to(device)`` and are saved alongside the network parameters.
        edges = torch.linspace(v_min, v_max, num_bins + 1)
        self.register_buffer('edges', edges)
        self.register_buffer('support', (edges[:-1] + edges[1:]) / 2.0)

        self.net_lst: list[nn.Module]
        self.net_lst = []
        for idx in range(self._num_critics):
            net = build_mlp_network(
                sizes=[self._obs_dim, *self._hidden_sizes, num_bins],
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
            self.net_lst.append(net)
            self.add_module(f'critic_{idx}', net)

    def get_logits(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Return the raw bin logits for each critic.

        Args:
            obs (torch.Tensor): Observations from environments.

        Returns:
            A list (one entry per critic) of logit tensors of shape ``(batch, num_bins)``.
        """
        return [critic(obs) for critic in self.net_lst]

    @property
    def sigma(self) -> float:
        """The Gaussian standard deviation used to smear scalar targets onto the support."""
        return self._sigma

    @property
    def num_bins(self) -> int:
        """The number of bins spanning the value support."""
        return self._num_bins

    def get_probs(self, obs: torch.Tensor, idx: int = 0) -> torch.Tensor:
        """Return the predicted categorical distribution over the value support.

        Args:
            obs (torch.Tensor): Observations from environments.
            idx (int, optional): Index of the critic to query. Defaults to 0.

        Returns:
            A tensor of shape ``(batch, num_bins)`` whose last dimension sums to 1.
        """
        return torch.softmax(self.net_lst[idx](obs), dim=-1)

    def distribution_stats(
        self,
        probs: torch.Tensor,
        mode_level: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        r"""Summarise the shape of categorical distributions over the value support.

        These statistics answer whether the critic's output is genuinely spread over the support or
        merely a smeared point estimate.  The natural reference is the training target itself: a
        scalar projected through :meth:`target_to_probs` has a standard deviation of roughly
        ``sigma``, so a prediction whose ``std`` is close to that of its own target carries no
        distributional information beyond the mean.

        ``n_modes`` counts the connected runs of bins holding at least ``mode_level`` of the row's
        peak mass, i.e. how many *separated* regions the distribution concentrates in.  Counting
        bare local maxima does not work here: softmax outputs are rippled, so a flat distribution
        reports as a dozen modes, while tightening the test with a fixed-window prominence rule
        instead erases broad unimodal bumps.  Thresholding at a fraction of the peak is stable
        under both, and gives 1 for anything unimodal (however wide) and 2 for a genuinely split
        distribution.

        Args:
            probs (torch.Tensor): Categorical probabilities of shape ``(batch, num_bins)``.
            mode_level (float, optional): Superlevel-set threshold as a fraction of the row's peak
                mass. Defaults to 0.5.

        Returns:
            A dict of per-row statistics, each of shape ``(batch,)``: ``mean``, ``std``,
            ``entropy``, ``skew``, ``edge_mass_low``, ``edge_mass_high`` and ``n_modes``.
        """
        support = self.support
        mean = (probs * support).sum(dim=-1)
        centred = support.unsqueeze(0) - mean.unsqueeze(-1)
        var = (probs * centred.pow(2)).sum(dim=-1)
        std = var.clamp_min(0.0).sqrt()
        third = (probs * centred.pow(3)).sum(dim=-1)
        skew = third / std.pow(3).clamp_min(1e-8)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

        peak = probs.max(dim=-1, keepdim=True).values
        above = probs >= mode_level * peak
        # One mode per run of ``above`` bins: count the False -> True transitions.
        starts = above.clone()
        starts[:, 1:] &= ~above[:, :-1]
        n_modes = starts.sum(dim=-1).to(probs.dtype)

        return {
            'mean': mean,
            'std': std,
            'entropy': entropy,
            'skew': skew,
            'edge_mass_low': probs[:, 0],
            'edge_mass_high': probs[:, -1],
            'n_modes': n_modes,
        }

    def target_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        r"""Project scalar targets onto the categorical support via the HL-Gauss transform.

        The probability mass assigned to bin :math:`i` (with edges :math:`b_i, b_{i+1}`) is

        .. math::

            p_i = \frac{\Phi\!\left(\frac{b_{i+1} - y}{\sigma}\right)
                        - \Phi\!\left(\frac{b_i - y}{\sigma}\right)}
                       {\Phi\!\left(\frac{b_m - y}{\sigma}\right)
                        - \Phi\!\left(\frac{b_0 - y}{\sigma}\right)}

        where :math:`\Phi` is the standard normal CDF.  Normalising by the mass inside the support
        keeps the target a valid distribution even when :math:`y` sits near (or beyond) the edges.

        Args:
            target (torch.Tensor): Scalar targets of arbitrary shape ``(...)``.

        Returns:
            A tensor of shape ``(..., num_bins)`` whose last dimension sums to 1.
        """
        target = target.unsqueeze(-1)  # (..., 1)
        edges = self.edges  # (num_bins + 1,)
        cdf = 0.5 * (1.0 + torch.erf((edges - target) / (self._sigma * math.sqrt(2.0))))
        probs = cdf[..., 1:] - cdf[..., :-1]  # (..., num_bins)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs

    def loss(self, obs: torch.Tensor, target: torch.Tensor, idx: int = 0) -> torch.Tensor:
        """Cross-entropy classification loss between predicted logits and the HL-Gauss target.

        Args:
            obs (torch.Tensor): Observations from environments.
            target (torch.Tensor): Scalar value targets.
            idx (int, optional): Index of the critic to score. Defaults to 0.

        Returns:
            The mean cross-entropy loss.
        """
        logits = self.net_lst[idx](obs)
        target_probs = self.target_to_probs(target)
        return -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Return scalar V-values as the expectation of the categorical distribution.

        Args:
            obs (torch.Tensor): Observations from environments.

        Returns:
            A list (one entry per critic) of scalar V-value tensors.
        """
        res = []
        for critic in self.net_lst:
            probs = torch.softmax(critic(obs), dim=-1)
            res.append(torch.matmul(probs, self.support))
        return res
