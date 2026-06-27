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
"""Implementation of VCritic, DistributionalVCritic (QR / TQC), and IQNVCritic."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class VCritic(Critic):
    """Standard scalar value critic (MLP → single scalar per state).

    Args:
        obs_space: Observation space.
        act_space: Action space.
        hidden_sizes: List of hidden layer sizes.
        activation: Activation function name.
        weight_initialization_mode: Weight init scheme.
        num_critics: Number of ensemble members (usually 1 for V critics).
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
    ) -> None:
        """Initialize an instance of :class:`VCritic`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics,
            use_obs_encoder=False,
        )
        self.net_lst: list[nn.Module] = []

        for idx in range(self._num_critics):
            net = build_mlp_network(
                sizes=[self._obs_dim, *self._hidden_sizes, 1],
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
            self.net_lst.append(net)
            self.add_module(f'critic_{idx}', net)

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Map observations to V-values.

        Returns:
            List of tensors, one per ensemble member, each shape ``(batch,)``.
        """
        return [torch.squeeze(net(obs), -1) for net in self.net_lst]


class DistributionalVCritic(Critic):
    """Quantile-regression distributional value critic (QR-DQN / TQC style).

    Supports two modes selected by whether ``n_top_quantiles_to_drop > 0``:

    * **QR** (``n_top_quantiles_to_drop=0``, ``num_critics=1``): One network, N fixed quantile
      levels τ_i = (2i-1)/(2N), trained with QR-Huber loss.
    * **TQC** (``n_top_quantiles_to_drop > 0``, ``num_critics ≥ 2``): K independent networks
      each outputting N quantile atoms, total K·N atoms.  During value estimation the top D atoms
      are dropped before averaging, reducing overestimation bias.

    In both modes ``forward(obs)`` returns ``[mean_value]`` (shape ``(batch,)``) — identical to
    the ``VCritic`` interface — so advantage computation, GAE, and buffer code are unchanged.
    Distributional information is exposed via ``quantiles(obs)`` which returns all atoms and their
    quantile levels; this is used exclusively inside the QR-Huber loss.

    Args:
        obs_space: Observation space.
        act_space: Action space.
        hidden_sizes: MLP hidden layer sizes.
        activation: Activation function name.
        weight_initialization_mode: Weight init scheme.
        num_critics: Number of independent networks (1 for QR, ≥2 for TQC).
        n_quantiles: Quantile atoms per network.
        n_top_quantiles_to_drop: Number of top atoms to drop at value-estimation time (TQC).
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        n_quantiles: int = 50,
        n_top_quantiles_to_drop: int = 0,
    ) -> None:
        """Initialize an instance of :class:`DistributionalVCritic`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics,
            use_obs_encoder=False,
        )
        self.n_quantiles: int = n_quantiles
        self.n_top_to_drop: int = n_top_quantiles_to_drop

        # Fixed quantile levels — midpoints of N equal-probability bins, repeated per network
        tau_single = (2.0 * torch.arange(1, n_quantiles + 1) - 1.0) / (2.0 * n_quantiles)
        self.register_buffer('tau', tau_single.repeat(num_critics))  # (K*N,)

        self.net_lst: list[nn.Module] = []
        for idx in range(self._num_critics):
            net = build_mlp_network(
                sizes=[self._obs_dim, *self._hidden_sizes, n_quantiles],
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
            self.net_lst.append(net)
            self.add_module(f'critic_{idx}', net)

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Return ``[mean_value]`` (shape ``(batch,)`` or scalar) — VCritic-compatible interface.

        For TQC, sorts all K·N atoms and averages the bottom (K·N − D) to reduce overestimation.
        """
        unbatched = obs.dim() == 1
        if unbatched:
            obs = obs.unsqueeze(0)
        all_q = torch.cat([net(obs) for net in self.net_lst], dim=-1)  # (B, K*N)
        if self.n_top_to_drop > 0:
            all_q = all_q.sort(dim=-1).values[:, :-self.n_top_to_drop]
        out = all_q.mean(dim=-1)  # (B,)
        return [out.squeeze(0) if unbatched else out]

    def quantiles(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all quantile atoms and their levels for loss computation and logging.

        Returns:
            q:   ``(batch, K*N)`` — all atom values (all critics concatenated).
            tau: ``(K*N,)``       — corresponding fixed quantile levels.
        """
        unbatched = obs.dim() == 1
        if unbatched:
            obs = obs.unsqueeze(0)
        all_q = torch.cat([net(obs) for net in self.net_lst], dim=-1)  # (B, K*N)
        if unbatched:
            all_q = all_q.squeeze(0)
        return all_q, self.tau

    def cvar_value(self, obs: torch.Tensor, alpha: float) -> list[torch.Tensor]:
        """Return CVaR_alpha — mean of the bottom-alpha fraction of quantile atoms.

        CVaR_alpha is the expected value conditional on being in the worst alpha
        fraction of outcomes. alpha=1.0 reduces to the standard mean.

        Args:
            obs: Observation tensor.
            alpha: Risk level in (0, 1].
        """
        unbatched = obs.dim() == 1
        if unbatched:
            obs = obs.unsqueeze(0)
        all_q = torch.cat([net(obs) for net in self.net_lst], dim=-1)  # (B, K*N)
        sorted_q = all_q.sort(dim=-1).values  # ascending
        k = max(1, round(alpha * sorted_q.shape[-1]))
        cvar = sorted_q[:, :k].mean(dim=-1)  # (B,)
        return [cvar.squeeze(0) if unbatched else cvar]


class IQNVCritic(Critic):
    """Implicit Quantile Networks (IQN) value critic.

    Instead of fixed quantile levels, IQN samples τ ~ U(0,1) on each forward pass and learns
    to predict the corresponding return quantile via a cosine-embedding of τ combined with a
    learned observation embedding:

        h(s)    = activation(MLP(s))            shape (B, E)
        φ(τ)    = activation(W · cos(πiτ))     shape (n, E)   for i=1,...,n_cos
        q_τ(s)  = Linear(h(s) ⊙ φ(τ))          shape (B, n)

    ``forward(obs)`` samples ``iqn_n_tau_eval`` quantile levels and returns their mean — a
    better approximation of E[Z(s)] than a fixed-N QR network at the same parameter count.
    ``quantiles(obs)`` samples ``n_quantiles`` fresh levels for QR-Huber loss computation.

    Args:
        obs_space: Observation space.
        act_space: Action space.
        hidden_sizes: Hidden sizes for the observation MLP encoder.
        activation: Activation function name.
        weight_initialization_mode: Weight init scheme.
        num_critics: Ignored (IQN is a single unified network).
        n_quantiles: Quantile samples per train forward pass (used for loss).
        iqn_embed_dim: Dimension of both the obs and quantile embeddings.
        iqn_n_cos: Number of cosine features for quantile encoding.
        iqn_n_tau_eval: Quantile samples at eval / rollout time (more → better mean estimate).
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        n_quantiles: int = 32,
        iqn_embed_dim: int = 64,
        iqn_n_cos: int = 64,
        iqn_n_tau_eval: int = 32,
    ) -> None:
        """Initialize an instance of :class:`IQNVCritic`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            1,  # IQN is a single network
            use_obs_encoder=False,
        )
        self.n_quantiles: int = n_quantiles
        self.n_tau_eval: int = max(iqn_n_tau_eval, n_quantiles)
        self.iqn_n_cos: int = iqn_n_cos
        self.embed_dim: int = iqn_embed_dim

        # Observation encoder: obs_dim → hidden_sizes → embed_dim (activation applied after)
        self.obs_net: nn.Module = build_mlp_network(
            sizes=[self._obs_dim, *self._hidden_sizes, iqn_embed_dim],
            activation=self._activation,
            weight_initialization_mode=self._weight_initialization_mode,
        )
        # Quantile encoder: cosine features → embed_dim (activation applied after)
        self.tau_net: nn.Linear = nn.Linear(iqn_n_cos, iqn_embed_dim)
        # Output head: element-wise product of obs and tau embeddings → scalar
        self.output_net: nn.Linear = nn.Linear(iqn_embed_dim, 1)

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Observation embedding with final activation, shape ``(B, E)``."""
        return F.relu(self.obs_net(obs))

    def _encode_tau(self, tau: torch.Tensor) -> torch.Tensor:
        """Cosine quantile embedding, shape ``(n, E)``.

        Args:
            tau: Quantile levels in (0,1), shape ``(n,)``.
        """
        i = torch.arange(1, self.iqn_n_cos + 1, dtype=tau.dtype, device=tau.device)
        cos_feat = torch.cos(math.pi * torch.outer(tau, i))  # (n, n_cos)
        return F.relu(self.tau_net(cos_feat))  # (n, E)

    def _forward_with_tau(self, obs: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Compute quantile values for given observations and quantile levels.

        Args:
            obs: ``(B, obs_dim)`` or ``(obs_dim,)`` for a single unbatched sample.
            tau: ``(n,)`` quantile levels

        Returns:
            ``(B, n)`` or ``(n,)`` quantile value predictions (matches input batch dims).
        """
        unbatched = obs.dim() == 1
        if unbatched:
            obs = obs.unsqueeze(0)
        h = self._encode_obs(obs)       # (B, E)
        phi = self._encode_tau(tau)     # (n, E)
        combined = h.unsqueeze(1) * phi.unsqueeze(0)  # (B, n, E)
        q = self.output_net(combined).squeeze(-1)      # (B, n)
        return q.squeeze(0) if unbatched else q

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Sample ``iqn_n_tau_eval`` quantile levels and return their mean — VCritic interface.

        Returns:
            ``[mean_value]`` with shape ``(batch,)``.
        """
        tau = torch.rand(self.n_tau_eval, device=obs.device)
        q = self._forward_with_tau(obs, tau)  # (B, n_eval)
        return [q.mean(dim=-1)]  # (B,)

    def quantiles(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``n_quantiles`` fresh levels and return predictions + levels for loss.

        Returns:
            q:   ``(batch, n_quantiles)`` quantile value predictions.
            tau: ``(n_quantiles,)``       the freshly sampled quantile levels.
        """
        tau = torch.rand(self.n_quantiles, device=obs.device)
        q = self._forward_with_tau(obs, tau)  # (B, n)
        return q, tau

    def cvar_value(self, obs: torch.Tensor, alpha: float) -> list[torch.Tensor]:
        """Return CVaR_alpha by sampling tau ~ U(0, alpha).

        IQN's implicit quantile representation lets us query any tau level, so
        CVaR_alpha = E_{tau ~ U(0,alpha)}[Z_tau] is computed directly without sorting.

        Args:
            obs: Observation tensor.
            alpha: Risk level in (0, 1].
        """
        tau = torch.rand(self.n_tau_eval, device=obs.device) * alpha
        q = self._forward_with_tau(obs, tau)  # (B, n_eval) or (n_eval,)
        if q.dim() == 1:
            return [q.mean()]
        return [q.mean(dim=-1)]
