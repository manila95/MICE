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
"""Successor-representation critic architectures shared by reward and cost value functions.

Two modes are provided, selected by ``model_cfgs.sr_cfgs.sr_mode``:

* ``shared_trunk``: a single feature trunk followed by two independent linear read-out heads
  (reward, cost). The whole network is trained end-to-end by the ordinary value-target MSE
  loss -- there is no dedicated successor-representation loss, so this mode reuses the stock
  training loop (:meth:`PolicyGradient._update_reward_critic` /
  :meth:`PolicyGradient._update_cost_critic`) completely unchanged.
* ``td_ridge``: a literal successor representation. A trunk produces a one-step feature
  ``phi(s)`` (l2-normalized linear read-out) and a successor feature ``psi(s)`` (MLP read-out)
  trained by TD to satisfy ``psi(s_t) ~= phi(s_t) + gamma * psi(s_{t+1})``. The reward/cost
  read-out weights ``w_r`` / ``w_c`` are solved in closed form by ridge regression of the
  one-step reward/cost onto ``phi`` once per update (see :meth:`ridge_update`) and are stored
  as buffers, not parameters.

Each mode comes in a state-only V-critic flavor for the on-policy algorithms and an
action-conditioned Q-critic flavor for the off-policy ones (DDPG/TD3/SAC and their Lagrangian
and PID variants). The Q flavors are the same architectures with ``cat([obs, act], -1)`` as
the trunk input, so ``phi(s, a)`` / ``psi(s, a)`` and the read-outs satisfy the standard
``forward(obs, act) -> [value, ...]`` :class:`Critic` contract:

+---------------+-----------------------------------------+---------------------------------------+
| Mode          | V flavor (on-policy)                    | Q flavor (off-policy)                 |
+===============+=========================================+=======================================+
| shared_trunk  | :class:`SuccessorRepresentationTrunk`   | :class:`QSuccessorRepresentationTrunk`|
|               | :class:`SuccessorRepresentationReadout` | :class:`SuccessorRepresentationQReadout`|
+---------------+-----------------------------------------+---------------------------------------+
| td_ridge      | :class:`TDRidgeSuccessorRepresentationTrunk` | :class:`TDRidgeSuccessorRepresentationQTrunk`|
|               | :class:`SuccessorRepresentationLinearReadout`| :class:`SuccessorRepresentationQLinearReadout`|
+---------------+-----------------------------------------+---------------------------------------+

Within ``td_ridge``, ``model_cfgs.sr_cfgs.phi_source`` selects where ``phi`` comes from:

* ``trunk``: ``phi = normalize(phi_head(trunk(s)))`` -- a linear read-out of the same trunk
  ``psi`` is read from. No loss is differentiable in ``phi``, so ``phi_head`` learns nothing, but
  ``phi(s)`` still drifts every update because the trunk beneath it is trained by the ``psi`` TD
  loss and the value losses. ``psi`` is therefore regressed onto a feature stream that moves
  during the very updates that are fitting it, and the ridge basis shifts under ``w_r`` / ``w_c``.
* ``random``: ``phi = normalize(W s + b)`` with ``W``, ``b`` frozen at initialization -- a fixed
  random projection, the classic successor-feature setting. Stationary, but note that an affine
  map cannot raise rank: ``phi``'s effective rank is capped at ``obs_dim + 1`` no matter how large
  ``sr_dim`` is (measured 18 of 64 on a 17-dim observation), and the ridge can only fit
  rewards/costs that are near-affine in the raw observation. Treat it as a lower-bound baseline
  rather than a competitive mode -- ``separate`` is the full-rank nonlinear version of the same
  stationarity idea.
* ``separate``: ``phi = normalize(MLP(s))``, its own network on the raw observation, frozen at
  initialization -- fixed *deep* random features. Stationary like ``random`` while recovering the
  nonlinearity, and its depth is set independently of the trunk, which ``trunk`` mode cannot do
  (there ``phi`` and ``psi`` are both single linear read-outs of one shared body, so they are
  forced to share depth).

The one structural difference in the Q flavors is that ``td_ridge`` carries ``num_psi_heads``
independent ``psi`` heads over the shared trunk rather than one, so that the off-policy
twin-critic (clipped double-Q) machinery in SAC/TD3 has genuinely distinct estimates to take a
minimum over. ``phi`` stays single -- it defines the regression basis that ``w_r`` / ``w_c`` are
solved against, and is what makes the representation shared between reward and cost.
"""

from __future__ import annotations

import torch
from torch import nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network, initialize_layer


class SuccessorRepresentationTrunk(nn.Module):
    """Shared feature trunk producing the successor-representation vector directly.

    Args:
        obs_dim (int): Observation dimension.
        hidden_sizes (list of int): Hidden layer sizes of the trunk.
        sr_dim (int): Dimensionality of the shared feature vector.
        activation (Activation): Activation function.
        weight_initialization_mode (InitFunction): Weight initialization mode.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list[int],
        sr_dim: int,
        activation: Activation,
        weight_initialization_mode: InitFunction,
    ) -> None:
        """Initialize an instance of :class:`SuccessorRepresentationTrunk`."""
        super().__init__()
        self.net = build_mlp_network(
            sizes=[obs_dim, *hidden_sizes, sr_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute the shared feature vector for ``obs``."""
        return self.net(obs)


class SuccessorRepresentationReadout(Critic):
    """A linear read-out head over a shared successor-representation trunk.

    Mirrors the standard :class:`Critic` interface (``forward`` returns a length-1 list holding
    a ``(B,)`` value tensor), so it drops in place of the stock ``reward_critic`` /
    ``cost_critic`` without changing any of their call sites.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        trunk (SuccessorRepresentationTrunk): The shared trunk (same instance passed to both the
            reward and the cost head, so its parameters are trained by both losses).
        sr_dim (int): Dimensionality of the shared feature vector.
        weight_initialization_mode (InitFunction): Weight initialization mode.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        trunk: SuccessorRepresentationTrunk,
        sr_dim: int,
        weight_initialization_mode: InitFunction,
    ) -> None:
        """Initialize an instance of :class:`SuccessorRepresentationReadout`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes=[],
            activation='identity',
            weight_initialization_mode=weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        )
        self.trunk = trunk
        self.head = nn.Linear(sr_dim, 1)
        initialize_layer(weight_initialization_mode, self.head)

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Read out a scalar value from the shared trunk's feature vector."""
        feat = self.trunk(obs)
        value = torch.squeeze(self.head(feat), -1)
        return [value]


class RidgeSolvedReadoutWeights(nn.Module):
    """Owns the ridge-solved ``w_r`` / ``w_c`` read-out weights of a ``td_ridge`` trunk.

    Factored out of the trunks themselves so the state-only (V) and action-conditioned (Q)
    flavors share one implementation of the closed-form solve.

    Args:
        sr_dim (int): Dimensionality of ``phi``, and hence of ``w_r`` / ``w_c``.
    """

    def __init__(self, sr_dim: int) -> None:
        """Initialize an instance of :class:`RidgeSolvedReadoutWeights`."""
        super().__init__()
        self.sr_dim = sr_dim
        # w_r / w_c are solved by closed-form ridge regression (see ridge_update), never by
        # SGD, so they are buffers rather than parameters.
        self.register_buffer('w_r', torch.zeros(sr_dim))
        self.register_buffer('w_c', torch.zeros(sr_dim))

    @torch.no_grad()
    def ridge_update(
        self,
        phi: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        ridge_kappa: float,
        ema_tau: float,
    ) -> dict[str, float]:
        r"""Refresh ``w_r`` and ``w_c`` by closed-form ridge regression on the fresh batch.

        .. math::

            w \leftarrow (1 - \tau) w + \tau (\Phi^T \Phi + \kappa I)^{-1} \Phi^T y

        solved in float64 for numerical stability, then EMA-blended into the stored buffer.

        Args:
            phi (torch.Tensor): One-step features of shape ``(N, sr_dim)`` for the fresh batch.
            reward (torch.Tensor): One-step rewards of shape ``(N,)``.
            cost (torch.Tensor): One-step costs of shape ``(N,)``.
            ridge_kappa (float): Ridge regularization coefficient (scales the mean diagonal of
                the Gram matrix).
            ema_tau (float): EMA blending coefficient; ``1.0`` means no smoothing (replace).

        Returns:
            A dict of diagnostic statistics for logging.
        """
        p = phi.double()
        gram = p.T @ p
        kappa = ridge_kappa * torch.diagonal(gram).mean().clamp(min=1e-12)
        mat = gram + kappa * torch.eye(self.sr_dim, dtype=torch.float64, device=p.device)

        w_r_new = torch.linalg.solve(mat, p.T @ reward.double())
        w_c_new = torch.linalg.solve(mat, p.T @ cost.double())
        self.w_r.mul_(1.0 - ema_tau).add_(ema_tau * w_r_new.float())
        self.w_c.mul_(1.0 - ema_tau).add_(ema_tau * w_c_new.float())

        resid_r = (p @ w_r_new - reward.double()).pow(2).mean().sqrt().item()
        resid_c = (p @ w_c_new - cost.double()).pow(2).mean().sqrt().item()
        return {
            'Misc/RidgeResidualReward': resid_r,
            'Misc/RidgeResidualCost': resid_c,
            'Misc/WrNorm': self.w_r.norm().item(),
            'Misc/WcNorm': self.w_c.norm().item(),
            'Misc/GramCond': torch.linalg.cond(mat).item(),
        }


class FrozenPhiFeatures(nn.Module):
    r"""A one-step feature map ``phi`` held fixed at its initialization.

    Backs the ``phi_source`` settings ``'random'`` (no hidden layers, i.e. a single random linear
    projection) and ``'separate'`` (its own MLP). Both read the *raw* input rather than the trunk
    that ``psi`` is read from, and every parameter is frozen with ``requires_grad_(False)`` and
    left out of the SR optimizer.

    Freezing is the entire point rather than an optimization detail. ``psi`` is *defined* as the
    discounted sum of the ``phi`` stream, so a ``phi`` that moves during training gives ``psi`` a
    target that refers to a feature map which no longer exists, and gives the ridge solve a basis
    that shifts out from under the ``w_r`` / ``w_c`` it fitted. Holding ``phi`` still makes both
    stationary for the whole run.

    Args:
        in_dim (int): Input dimension -- ``obs_dim`` for the V flavor, ``obs_dim + act_dim`` for
            the Q flavor. The Q flavor must include the action: the ridge regresses the immediate
            reward onto ``phi``, and that reward depends on the action taken.
        hidden_sizes (list of int): Hidden layer sizes; empty gives a single linear projection.
        sr_dim (int): Dimensionality of ``phi``.
        activation (Activation): Activation function.
        weight_initialization_mode (InitFunction): Weight initialization mode.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_sizes: list[int],
        sr_dim: int,
        activation: Activation,
        weight_initialization_mode: InitFunction,
    ) -> None:
        """Initialize an instance of :class:`FrozenPhiFeatures`."""
        super().__init__()
        self.net = build_mlp_network(
            sizes=[in_dim, *hidden_sizes, sr_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self.net.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the l2-normalized frozen feature of ``x``."""
        p = self.net(x)
        return p / (p.norm(dim=-1, keepdim=True) + 1e-8)


def build_frozen_phi(
    phi_source: str,
    in_dim: int,
    phi_hidden_sizes: list[int],
    sr_dim: int,
    activation: Activation,
    weight_initialization_mode: InitFunction,
) -> FrozenPhiFeatures | None:
    """Validate ``phi_source`` and build the frozen ``phi`` network it asks for, if any.

    Shared by the V and Q ``td_ridge`` trunks so the three settings are defined in one place.

    Args:
        phi_source (str): One of ``'trunk'``, ``'random'``, ``'separate'``.
        in_dim (int): Input dimension of the frozen network (see :class:`FrozenPhiFeatures`).
        phi_hidden_sizes (list of int): Hidden sizes, used by ``'separate'`` only.
        sr_dim (int): Dimensionality of ``phi``.
        activation (Activation): Activation function.
        weight_initialization_mode (InitFunction): Weight initialization mode.

    Returns:
        The frozen network, or ``None`` for ``'trunk'`` (which reads ``phi`` off the shared trunk
        via a trainable ``phi_head`` instead).
    """
    if phi_source == 'trunk':
        return None
    if phi_source == 'random':
        # Pinned to depth 0 rather than reading phi_hidden_sizes: 'random' names one specific
        # experimental condition (a fixed random linear projection of the raw input), and a
        # stray phi_hidden_sizes in a sweep config should not silently redefine what it means.
        hidden_sizes: list[int] = []
    elif phi_source == 'separate':
        hidden_sizes = list(phi_hidden_sizes)
    else:
        raise NotImplementedError(
            f'Unknown sr_cfgs.phi_source "{phi_source}". '
            'Available phi sources are: "trunk", "random", "separate".',
        )
    return FrozenPhiFeatures(
        in_dim=in_dim,
        hidden_sizes=hidden_sizes,
        sr_dim=sr_dim,
        activation=activation,
        weight_initialization_mode=weight_initialization_mode,
    )


class TDRidgeSuccessorRepresentationTrunk(RidgeSolvedReadoutWeights):
    """phi / psi bundle for the literal successor-representation (``td_ridge``) mode.

    Args:
        obs_dim (int): Observation dimension.
        hidden_sizes (list of int): Hidden layer sizes of the shared trunk.
        sr_dim (int): Dimensionality of ``phi`` and ``psi``.
        activation (Activation): Activation function.
        weight_initialization_mode (InitFunction): Weight initialization mode.
        phi_source (str): Where ``phi`` comes from -- ``'trunk'`` (a trainable linear read-out of
            the trunk shared with ``psi``), or ``'random'`` / ``'separate'`` (a frozen map of the
            raw observation, see :func:`build_frozen_phi`).
        phi_hidden_sizes (list of int): Hidden sizes of the standalone ``phi`` network, used by
            ``phi_source='separate'`` only.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list[int],
        sr_dim: int,
        activation: Activation,
        weight_initialization_mode: InitFunction,
        phi_source: str = 'trunk',
        phi_hidden_sizes: list[int] | None = None,
    ) -> None:
        """Initialize an instance of :class:`TDRidgeSuccessorRepresentationTrunk`."""
        super().__init__(sr_dim)
        trunk_out = hidden_sizes[-1] if hidden_sizes else obs_dim
        self.trunk: nn.Module = (
            build_mlp_network(
                sizes=[obs_dim, *hidden_sizes],
                activation=activation,
                output_activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            if hidden_sizes
            else nn.Identity()
        )
        phi_net = build_frozen_phi(
            phi_source,
            in_dim=obs_dim,
            phi_hidden_sizes=phi_hidden_sizes or [],
            sr_dim=sr_dim,
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self._phi_from_trunk = phi_net is None
        if phi_net is None:
            self.phi_head = nn.Linear(trunk_out, sr_dim)
            initialize_layer(weight_initialization_mode, self.phi_head)
        else:
            self.phi_net = phi_net
        self.psi_head = build_mlp_network(
            sizes=[trunk_out, sr_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        """Shared trunk features."""
        return self.trunk(obs)

    def phi(self, obs: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """L2-normalized one-step feature ``phi(s)``.

        ``z`` is the shared trunk's features, accepted so a caller that already computed them for
        ``psi`` need not pay for them twice. It is ignored when ``phi`` comes from a frozen map of
        the raw observation, which by construction does not read the trunk at all.
        """
        if not self._phi_from_trunk:
            return self.phi_net(obs)
        z = self.features(obs) if z is None else z
        p = self.phi_head(z)
        return p / (p.norm(dim=-1, keepdim=True) + 1e-8)

    def psi(self, obs: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Successor feature ``psi(s)``."""
        z = self.features(obs) if z is None else z
        return self.psi_head(z)


class SuccessorRepresentationLinearReadout(Critic):
    """A read-out head whose weight vector is solved by ridge regression, not by SGD.

    ``value(s) = psi(s)^T w``, where ``psi`` is the trunk's successor-feature output and ``w``
    is a non-learnable buffer refreshed by :meth:`TDRidgeSuccessorRepresentationTrunk.ridge_update`.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        trunk (TDRidgeSuccessorRepresentationTrunk): The shared phi/psi trunk.
        weight_name (str): Name of the trunk buffer to read out (``'w_r'`` or ``'w_c'``).
        weight_initialization_mode (InitFunction): Weight initialization mode.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        trunk: TDRidgeSuccessorRepresentationTrunk,
        weight_name: str,
        weight_initialization_mode: InitFunction,
    ) -> None:
        """Initialize an instance of :class:`SuccessorRepresentationLinearReadout`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes=[],
            activation='identity',
            weight_initialization_mode=weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        )
        self.trunk = trunk
        self._weight_name = weight_name

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Read out a scalar value as ``psi(s)^T w``, with gradient flowing into ``psi`` only."""
        psi = self.trunk.psi(obs)
        weight = getattr(self.trunk, self._weight_name)
        value = (psi * weight).sum(-1)
        return [value]


# ======================================================================================
# Action-conditioned (Q) flavors, for the off-policy algorithms
# ======================================================================================


class QSuccessorRepresentationTrunk(nn.Module):
    """Shared feature trunk over ``(obs, act)`` pairs, for ``shared_trunk`` mode.

    The action-conditioned counterpart of :class:`SuccessorRepresentationTrunk`: identical
    architecture, but the input is ``cat([obs, act], -1)`` so the read-outs above it are
    Q-functions rather than V-functions.

    Args:
        obs_dim (int): Observation dimension.
        act_dim (int): Action dimension.
        hidden_sizes (list of int): Hidden layer sizes of the trunk.
        sr_dim (int): Dimensionality of the shared feature vector.
        activation (Activation): Activation function.
        weight_initialization_mode (InitFunction): Weight initialization mode.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int],
        sr_dim: int,
        activation: Activation,
        weight_initialization_mode: InitFunction,
    ) -> None:
        """Initialize an instance of :class:`QSuccessorRepresentationTrunk`."""
        super().__init__()
        self.net = build_mlp_network(
            sizes=[obs_dim + act_dim, *hidden_sizes, sr_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Compute the shared feature vector for the ``(obs, act)`` pair."""
        return self.net(torch.cat([obs, act], dim=-1))


class SuccessorRepresentationQReadout(Critic):
    """Linear read-out heads over a shared action-conditioned trunk.

    Mirrors the standard Q-:class:`Critic` interface (``forward(obs, act)`` returns a list of
    ``num_critics`` ``(B,)`` value tensors), so it drops in place of the stock ``reward_critic``
    / ``cost_critic`` of :class:`ConstraintActorQCritic` without changing any of their call
    sites -- including SAC's and TD3's twin-critic unpacking, ``q1, q2 = reward_critic(obs, act)``.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        trunk (QSuccessorRepresentationTrunk): The shared trunk (same instance passed to both
            the reward and the cost read-out, so its parameters are trained by both losses).
        sr_dim (int): Dimensionality of the shared feature vector.
        num_critics (int): Number of independent read-out heads.
        weight_initialization_mode (InitFunction): Weight initialization mode.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        trunk: QSuccessorRepresentationTrunk,
        sr_dim: int,
        num_critics: int,
        weight_initialization_mode: InitFunction,
    ) -> None:
        """Initialize an instance of :class:`SuccessorRepresentationQReadout`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes=[],
            activation='identity',
            weight_initialization_mode=weight_initialization_mode,
            num_critics=num_critics,
            use_obs_encoder=False,
        )
        self.trunk = trunk
        self.heads = nn.ModuleList(nn.Linear(sr_dim, 1) for _ in range(num_critics))
        for head in self.heads:
            initialize_layer(weight_initialization_mode, head)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> list[torch.Tensor]:
        """Read out one scalar per head from the shared trunk's feature vector."""
        feat = self.trunk(obs, act)
        return [torch.squeeze(head(feat), -1) for head in self.heads]


class TDRidgeSuccessorRepresentationQTrunk(RidgeSolvedReadoutWeights):
    """phi / psi bundle over ``(obs, act)`` pairs, for ``td_ridge`` mode.

    The action-conditioned counterpart of :class:`TDRidgeSuccessorRepresentationTrunk`. Two
    differences beyond the input being ``cat([obs, act], -1)``:

    * ``psi`` is a list of ``num_psi_heads`` independent heads rather than a single one, so the
      off-policy clipped double-Q machinery has genuinely distinct estimates to minimize over.
      They share the trunk body (and therefore the representation), differing only in their
      read-out MLP and its initialization.
    * ``phi`` stays single. It is the basis that ``w_r`` / ``w_c`` are regressed against, so
      duplicating it would mean duplicating the very thing the two critics are meant to share.

    Args:
        obs_dim (int): Observation dimension.
        act_dim (int): Action dimension.
        hidden_sizes (list of int): Hidden layer sizes of the shared trunk.
        sr_dim (int): Dimensionality of ``phi`` and ``psi``.
        num_psi_heads (int): Number of independent ``psi`` read-out heads.
        activation (Activation): Activation function.
        weight_initialization_mode (InitFunction): Weight initialization mode.
        phi_source (str): Where ``phi`` comes from -- ``'trunk'`` (a trainable linear read-out of
            the trunk shared with ``psi``), or ``'random'`` / ``'separate'`` (a frozen map of the
            raw ``cat([obs, act], -1)``, see :func:`build_frozen_phi`).
        phi_hidden_sizes (list of int): Hidden sizes of the standalone ``phi`` network, used by
            ``phi_source='separate'`` only.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int],
        sr_dim: int,
        num_psi_heads: int,
        activation: Activation,
        weight_initialization_mode: InitFunction,
        phi_source: str = 'trunk',
        phi_hidden_sizes: list[int] | None = None,
    ) -> None:
        """Initialize an instance of :class:`TDRidgeSuccessorRepresentationQTrunk`."""
        super().__init__(sr_dim)
        in_dim = obs_dim + act_dim
        trunk_out = hidden_sizes[-1] if hidden_sizes else in_dim
        self.trunk: nn.Module = (
            build_mlp_network(
                sizes=[in_dim, *hidden_sizes],
                activation=activation,
                output_activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            if hidden_sizes
            else nn.Identity()
        )
        phi_net = build_frozen_phi(
            phi_source,
            in_dim=in_dim,
            phi_hidden_sizes=phi_hidden_sizes or [],
            sr_dim=sr_dim,
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self._phi_from_trunk = phi_net is None
        if phi_net is None:
            self.phi_head = nn.Linear(trunk_out, sr_dim)
            initialize_layer(weight_initialization_mode, self.phi_head)
        else:
            self.phi_net = phi_net
        self.psi_heads = nn.ModuleList(
            build_mlp_network(
                sizes=[trunk_out, sr_dim],
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            for _ in range(num_psi_heads)
        )

    def features(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Shared trunk features of the ``(obs, act)`` pair."""
        return self.trunk(torch.cat([obs, act], dim=-1))

    def phi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """L2-normalized one-step feature ``phi(s, a)``.

        ``z`` is the shared trunk's features, accepted so a caller that already computed them for
        ``psi`` need not pay for them twice. It is ignored when ``phi`` comes from a frozen map of
        the raw input, which by construction does not read the trunk at all.
        """
        if not self._phi_from_trunk:
            return self.phi_net(torch.cat([obs, act], dim=-1))
        z = self.features(obs, act) if z is None else z
        p = self.phi_head(z)
        return p / (p.norm(dim=-1, keepdim=True) + 1e-8)

    def psi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Successor features ``psi_i(s, a)``, one ``(B, sr_dim)`` tensor per head."""
        z = self.features(obs, act) if z is None else z
        return [head(z) for head in self.psi_heads]


class SuccessorRepresentationQLinearReadout(Critic):
    """A Q read-out whose weight vector is solved by ridge regression, not by SGD.

    ``Q_i(s, a) = psi_i(s, a)^T w``, where ``psi_i`` are the trunk's successor-feature heads
    selected by ``head_indices`` and ``w`` is a non-learnable buffer refreshed by
    :meth:`RidgeSolvedReadoutWeights.ridge_update`.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        trunk (TDRidgeSuccessorRepresentationQTrunk): The shared phi/psi trunk.
        weight_name (str): Name of the trunk buffer to read out (``'w_r'`` or ``'w_c'``).
        head_indices (list of int): Which of the trunk's ``psi`` heads this critic reads out,
            one returned value per entry. The reward critic takes all of them (giving SAC/TD3
            their twin estimates); the cost critic takes head ``0`` only.
        weight_initialization_mode (InitFunction): Weight initialization mode.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        trunk: TDRidgeSuccessorRepresentationQTrunk,
        weight_name: str,
        head_indices: list[int],
        weight_initialization_mode: InitFunction,
    ) -> None:
        """Initialize an instance of :class:`SuccessorRepresentationQLinearReadout`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes=[],
            activation='identity',
            weight_initialization_mode=weight_initialization_mode,
            num_critics=len(head_indices),
            use_obs_encoder=False,
        )
        self.trunk = trunk
        self._weight_name = weight_name
        self._head_indices = list(head_indices)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> list[torch.Tensor]:
        """Read out ``psi_i(s, a)^T w``, with gradient flowing into ``psi`` only."""
        psi = self.trunk.psi(obs, act)
        weight = getattr(self.trunk, self._weight_name)
        return [(psi[idx] * weight).sum(-1) for idx in self._head_indices]
