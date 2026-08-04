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

Three modes are provided, selected by ``model_cfgs.sr_cfgs.sr_mode``:

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
* ``fb``: forward-backward representations (Touati & Ollivier, 2021). Two maps ``F`` and ``B``
  jointly factorize the policy's successor *measure* w.r.t. the on-policy state distribution
  ``rho``, ``M^pi(s, ds') ~= F(s)^T B(s') rho(ds')``, trained by a measure-Bellman residual
  plus a ``B``-orthonormality regularizer (see
  :meth:`PolicyGradient._update_fb_representation`). Any state-based signal ``g`` is then read
  out as ``V_g(s) = F(s)^T z_g`` with ``z_g = E_rho[B(s) g(s)]`` -- an expectation, not a
  regression, so unlike ``td_ridge`` there is no ridge residual capping the critic's accuracy.
  ``z_r`` / ``z_c`` are buffers, exactly like ``w_r`` / ``w_c``.

  .. note::
      The measure is taken in the ``t >= 0`` convention
      (``M(s, X) = 1[s in X] + gamma E[M(s+, X)]``), unlike the ``t >= 1`` convention of the
      original paper. That is what makes ``F(s)^T z_g`` *equal* to the value function GAE
      consumes rather than ``(V_g(s) - g(s)) / gamma``.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network, initialize_layer


def _ridge_solve(
    features: torch.Tensor,
    targets: tuple[torch.Tensor, ...],
    ridge_kappa: float,
) -> tuple[list[torch.Tensor], list[float], float]:
    r"""Solve a ridge regression of each target onto ``features`` in float64.

    .. math::

        w = (\Phi^T \Phi + \kappa I)^{-1} \Phi^T y

    Shared by :meth:`TDRidgeSuccessorRepresentationTrunk.ridge_update` (with
    ``features = phi(s)``) and :meth:`ForwardBackwardTrunk.update_task_vectors` (with
    ``features = B(s)``), so the numerics live in exactly one place.

    Args:
        features (torch.Tensor): Design matrix of shape ``(N, d)``.
        targets (tuple of torch.Tensor): One or more target vectors of shape ``(N,)``.
        ridge_kappa (float): Ridge coefficient, scaling the mean diagonal of the Gram matrix.

    Returns:
        A tuple ``(solutions, residuals, cond)``: the float32 solution vector per target, the
        RMS residual per target, and the condition number of the regularized Gram matrix.
    """
    p = features.double()
    dim = p.shape[-1]
    gram = p.T @ p
    kappa = ridge_kappa * torch.diagonal(gram).mean().clamp(min=1e-12)
    mat = gram + kappa * torch.eye(dim, dtype=torch.float64, device=p.device)

    solutions: list[torch.Tensor] = []
    residuals: list[float] = []
    for target in targets:
        y = target.double()
        w = torch.linalg.solve(mat, p.T @ y)
        solutions.append(w.float())
        residuals.append((p @ w - y).pow(2).mean().sqrt().item())

    return solutions, residuals, torch.linalg.cond(mat).item()


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


class TDRidgeSuccessorRepresentationTrunk(nn.Module):
    """phi / psi bundle for the literal successor-representation (``td_ridge``) mode.

    Args:
        obs_dim (int): Observation dimension.
        hidden_sizes (list of int): Hidden layer sizes of the shared trunk.
        sr_dim (int): Dimensionality of ``phi`` and ``psi``.
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
        """Initialize an instance of :class:`TDRidgeSuccessorRepresentationTrunk`."""
        super().__init__()
        self.sr_dim = sr_dim
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
        self.phi_head = nn.Linear(trunk_out, sr_dim)
        initialize_layer(weight_initialization_mode, self.phi_head)
        self.psi_head = build_mlp_network(
            sizes=[trunk_out, sr_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        # w_r / w_c are solved by closed-form ridge regression (see ridge_update), never by
        # SGD, so they are buffers rather than parameters.
        self.register_buffer('w_r', torch.zeros(sr_dim))
        self.register_buffer('w_c', torch.zeros(sr_dim))

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        """Shared trunk features."""
        return self.trunk(obs)

    def phi(self, obs: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """L2-normalized one-step feature ``phi(s)``."""
        z = self.features(obs) if z is None else z
        p = self.phi_head(z)
        return p / (p.norm(dim=-1, keepdim=True) + 1e-8)

    def psi(self, obs: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Successor feature ``psi(s)``."""
        z = self.features(obs) if z is None else z
        return self.psi_head(z)

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
        (w_r_new, w_c_new), (resid_r, resid_c), cond = _ridge_solve(
            phi,
            (reward, cost),
            ridge_kappa,
        )
        self.w_r.mul_(1.0 - ema_tau).add_(ema_tau * w_r_new)
        self.w_c.mul_(1.0 - ema_tau).add_(ema_tau * w_c_new)

        return {
            'Misc/RidgeResidualReward': resid_r,
            'Misc/RidgeResidualCost': resid_c,
            'Misc/WrNorm': self.w_r.norm().item(),
            'Misc/WcNorm': self.w_c.norm().item(),
            'Misc/GramCond': cond,
        }


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


class ForwardBackwardTrunk(nn.Module):
    r"""Forward (``F``) / backward (``B``) map bundle for the ``fb`` mode.

    Together they factorize the successor measure of the current policy with respect to the
    on-policy state distribution ``rho``:

    .. math::

        M^\pi(s, ds') \approx F(s)^T B(s') \rho(ds')

    so that any state-based signal ``g`` is read out as ``V_g(s) = F(s)^T z_g`` with
    ``z_g = E_{s \sim \rho}[B(s) g(s)]``.

    By default ``F`` and ``B`` get *separate* feature bodies: a shared body would tie ``F(s)``
    and ``B(s)`` for the same ``s``, but the two play different roles (``F`` maps a state to
    "where it goes", ``B`` maps a state to "how it is reached"). ``fb_shared_body`` restores the
    tied version for ablation.

    Args:
        obs_dim (int): Observation dimension.
        hidden_sizes (list of int): Hidden layer sizes of the ``F`` / ``B`` bodies.
        sr_dim (int): Dimensionality ``d`` of ``F`` and ``B``.
        activation (Activation): Activation function.
        weight_initialization_mode (InitFunction): Weight initialization mode.
        shared_body (bool): Whether ``F`` and ``B`` share one feature body. Defaults to False.
        normalize_b (bool): Whether to rescale ``B`` to norm ``sqrt(sr_dim)``. Defaults to False,
            since the orthonormality regularizer already controls ``B``'s scale.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_dim: int,
        hidden_sizes: list[int],
        sr_dim: int,
        activation: Activation,
        weight_initialization_mode: InitFunction,
        shared_body: bool = False,
        normalize_b: bool = False,
    ) -> None:
        """Initialize an instance of :class:`ForwardBackwardTrunk`."""
        super().__init__()
        self.sr_dim = sr_dim
        self._shared_body = shared_body
        self._normalize_b = normalize_b
        trunk_out = hidden_sizes[-1] if hidden_sizes else obs_dim

        def _body() -> nn.Module:
            if not hidden_sizes:
                return nn.Identity()
            return build_mlp_network(
                sizes=[obs_dim, *hidden_sizes],
                activation=activation,
                output_activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )

        self.f_body: nn.Module = _body()
        self.b_body: nn.Module = self.f_body if shared_body else _body()

        self.f_head = nn.Linear(trunk_out, sr_dim)
        self.b_head = nn.Linear(trunk_out, sr_dim)
        initialize_layer(weight_initialization_mode, self.f_head)
        initialize_layer(weight_initialization_mode, self.b_head)

        # Target copies used only to build the (detached) measure-Bellman target.
        self.f_body_target = copy.deepcopy(self.f_body).requires_grad_(False)
        self.b_body_target = copy.deepcopy(self.b_body).requires_grad_(False)
        self.f_head_target = copy.deepcopy(self.f_head).requires_grad_(False)
        self.b_head_target = copy.deepcopy(self.b_head).requires_grad_(False)

        # z_r / z_c are Monte-Carlo expectations refreshed once per update (see
        # update_task_vectors), never touched by SGD, so they are buffers rather than parameters
        # -- exactly like w_r / w_c above.
        self.register_buffer('z_r', torch.zeros(sr_dim))
        self.register_buffer('z_c', torch.zeros(sr_dim))

    def _scale_b(self, b: torch.Tensor) -> torch.Tensor:
        """Optionally rescale ``B`` rows to norm ``sqrt(sr_dim)``."""
        if not self._normalize_b:
            return b
        norm = b.norm(dim=-1, keepdim=True) / (self.sr_dim**0.5)
        return b / (norm + 1e-8)

    def forward_map(self, obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        """Forward map ``F(s)``, from the target networks when ``target`` is True."""
        if target:
            return self.f_head_target(self.f_body_target(obs))
        return self.f_head(self.f_body(obs))

    def backward_map(self, obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        """Backward map ``B(s)``, from the target networks when ``target`` is True."""
        if target:
            return self._scale_b(self.b_head_target(self.b_body_target(obs)))
        return self._scale_b(self.b_head(self.b_body(obs)))

    @torch.no_grad()
    def soft_update_targets(self, tau: float) -> None:
        """Polyak-average the online ``F`` / ``B`` weights into their targets.

        Args:
            tau (float): Polyak coefficient; ``1.0`` makes the target an exact copy, i.e. the
                loss degenerates to a plain stop-gradient target.
        """
        pairs = (
            (self.f_body, self.f_body_target),
            (self.b_body, self.b_body_target),
            (self.f_head, self.f_head_target),
            (self.b_head, self.b_head_target),
        )
        for online, target in pairs:
            for param, param_target in zip(online.parameters(), target.parameters()):
                param_target.data.mul_(1.0 - tau).add_(tau * param.data)
            for buf, buf_target in zip(online.buffers(), target.buffers()):
                buf_target.data.copy_(buf.data)

    @torch.no_grad()
    def update_task_vectors(  # pylint: disable=too-many-arguments
        self,
        b: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        ema_tau: float,
        estimator: str = 'expectation',
        ridge_kappa: float = 1e-3,
    ) -> dict[str, float]:
        r"""Refresh the read-out vectors ``z_r`` / ``z_c`` from a fresh batch.

        Two estimators:

        * ``'expectation'`` -- ``z_g = E_{s \sim \rho}[B(s) g(s)]``, the estimator the FB
          factorization actually implies (it assumes ``E_\rho[B B^T] \approx I``, which is what
          the orthonormality regularizer enforces).
        * ``'ridge'`` -- ``z_g = (B^T B + \kappa I)^{-1} B^T g``, i.e. the expectation whitened
          by the empirical covariance of ``B``. Reuses the same float64 solve as ``td_ridge``
          and is the safer choice if ``cov(B)`` drifts away from identity.

        The result is EMA-blended into the stored buffer, mirroring
        :meth:`TDRidgeSuccessorRepresentationTrunk.ridge_update`.

        Args:
            b (torch.Tensor): Backward features ``B(s)`` of shape ``(N, sr_dim)``.
            reward (torch.Tensor): One-step rewards of shape ``(N,)``.
            cost (torch.Tensor): One-step costs of shape ``(N,)``.
            ema_tau (float): EMA blending coefficient; ``1.0`` means replace.
            estimator (str): ``'expectation'`` or ``'ridge'``.
            ridge_kappa (float): Ridge coefficient, used by the ``'ridge'`` estimator only.

        Returns:
            A dict of diagnostic statistics for logging.
        """
        if estimator == 'expectation':
            z_r_new = (b * reward.unsqueeze(-1)).mean(dim=0)
            z_c_new = (b * cost.unsqueeze(-1)).mean(dim=0)
            resid_r = (b @ z_r_new - reward).pow(2).mean().sqrt().item()
            resid_c = (b @ z_c_new - cost).pow(2).mean().sqrt().item()
            cond = float('nan')
        elif estimator == 'ridge':
            (z_r_new, z_c_new), (resid_r, resid_c), cond = _ridge_solve(
                b,
                (reward, cost),
                ridge_kappa,
            )
        else:
            raise NotImplementedError(
                f'Unknown sr_cfgs.fb_z_estimator "{estimator}". '
                'Available estimators are: "expectation", "ridge".',
            )

        self.z_r.mul_(1.0 - ema_tau).add_(ema_tau * z_r_new)
        self.z_c.mul_(1.0 - ema_tau).add_(ema_tau * z_c_new)

        # Deviation of the empirical second moment of B from identity: the quantity L_ortho
        # drives to zero, and the assumption z_g = E[B g] relies on.
        cov = b.T @ b / b.shape[0]
        cov_err = (cov - torch.eye(self.sr_dim, device=b.device)).norm().item()

        return {
            'Misc/RidgeResidualReward': resid_r,
            'Misc/RidgeResidualCost': resid_c,
            'Misc/ZrNorm': self.z_r.norm().item(),
            'Misc/ZcNorm': self.z_c.norm().item(),
            'Misc/BCovErr': cov_err,
            'Misc/GramCond': cond,
        }


class ForwardBackwardReadout(Critic):
    """A read-out head over the forward map: ``value(s) = F(s)^T z``.

    ``z`` is a non-learnable buffer refreshed by
    :meth:`ForwardBackwardTrunk.update_task_vectors`, so gradients flow into ``F`` only.
    Structurally identical to :class:`SuccessorRepresentationLinearReadout`, and like it drops
    into the stock ``reward_critic`` / ``cost_critic`` call sites unchanged.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        trunk (ForwardBackwardTrunk): The shared F/B trunk.
        weight_name (str): Name of the trunk buffer to read out (``'z_r'`` or ``'z_c'``).
        weight_initialization_mode (InitFunction): Weight initialization mode.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        trunk: ForwardBackwardTrunk,
        weight_name: str,
        weight_initialization_mode: InitFunction,
    ) -> None:
        """Initialize an instance of :class:`ForwardBackwardReadout`."""
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
        """Read out a scalar value as ``F(s)^T z``, with gradient flowing into ``F`` only."""
        f = self.trunk.forward_map(obs)
        weight = getattr(self.trunk, self._weight_name)
        value = (f * weight).sum(-1)
        return [value]
