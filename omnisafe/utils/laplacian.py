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
r"""Augmented Lagrangian Laplacian Objective (ALLO) for ``phi_source='laplacian'``.

The second ``td_ridge`` ``phi_source`` trained by a loss of its own (see
:mod:`omnisafe.utils.contrastive` for the first, and the module docstring of
:mod:`omnisafe.models.critic.successor_representation_critic` for how the sources compare).
Where the time-contrastive source asks only that temporally-adjacent states be *close* in
``phi``, this one asks for the strictly stronger property ``td_ridge`` actually needs: that
``phi`` span the ``d`` smoothest functions on the transition graph -- the bottom eigenvectors of
the graph Laplacian -- because those are, by the variational characterization of the eigenvalue
problem, the basis in which an arbitrary state-dependent signal is best approximated linearly for
a given ``d``. The reward and the cost are two such signals, and ``w_r`` / ``w_c`` approximate
them linearly. Nothing in an InfoNCE objective makes that claim; here it is the objective.

**Why the naive objective is not enough.** The classical spectral relaxation

.. math::

    \min_{u} \sum_{i=1}^{d} \langle u_i, \Delta u_i \rangle
    \quad \text{s.t.} \quad \mathbb{E}_{s \sim \rho}[\phi(s) \phi(s)^T] = I

is solved by *any* rotation of the bottom-``d`` eigenvectors, so a penalty-method implementation
(Wu et al., 2019) recovers the right subspace but arbitrary coordinates inside it, and the
generalized version that breaks the tie with fixed decreasing coefficients (Wang et al., 2021)
has an equilibrium that depends on the barrier coefficient -- so the features it converges to are
an artifact of a hyperparameter. ALLO (Gomez, Bowling & Machado, *Proper Laplacian
Representation Learning*, ICLR 2024) fixes both with two changes, and this module implements
exactly those two:

1. **An asymmetric constraint.** The orthogonality constraint between coordinates ``i`` and
   ``j < i`` is enforced by moving ``u_i`` only -- ``u_j`` is stop-gradiented. That imposes a
   deflation order (``u_1`` is free to become the smoothest function, ``u_2`` the smoothest
   function orthogonal to it, and so on), which breaks the rotation symmetry and makes the
   *ordered* eigenvectors, not merely their span, the unique attractor.
2. **Dual variables instead of a fixed penalty.** The constraints are carried by Lagrange
   multipliers updated by dual ascent, with the quadratic barrier retained only for conditioning.
   At a saddle point the constraint is satisfied exactly regardless of the barrier coefficient,
   so the recovered features no longer depend on it.

Kept, like :mod:`omnisafe.utils.contrastive`, as plain tensor-in/tensor-out functions with no
network, config, or logger dependency, so the estimator can be checked against a hand-solved
tabular MDP whose Laplacian eigenvectors are known exactly (``tests/test_laplacian.py``). The
optimizer loop that calls these lives in
:meth:`~omnisafe.algorithms.on_policy.base.policy_gradient.PolicyGradient._laplacian_update_phi`.

Note:
    Unlike every other ``phi_source``, the features this objective trains are deliberately **not**
    l2-normalized -- see
    :class:`~omnisafe.models.critic.successor_representation_critic.LaplacianPhiFeatures`. The
    orthonormality constraint is what sets their scale, and it is unsatisfiable on the unit sphere:
    ``E[phi phi^T] = I`` forces ``E[||phi||^2] = d``, while normalized features have
    ``||phi|| = 1`` identically. Satisfying it instead hands the ridge solve a Gram matrix that is
    ``N * I`` up to estimation error, i.e. the best-conditioned basis any ``phi_source`` can give
    it (watch ``Misc/GramCond`` fall to ~1).
"""

from __future__ import annotations

import torch


def dirichlet_energy(phi_s: torch.Tensor, phi_s_next: torch.Tensor) -> torch.Tensor:
    r"""The graph-drawing term :math:`\sum_i \langle u_i, \Delta u_i \rangle`.

    For the symmetrized transition operator, the Dirichlet form of the Laplacian
    :math:`\Delta = I - \frac{1}{2}(P + P^T)` has the pairwise identity

    .. math::

        \sum_{i=1}^{d} \langle u_i, \Delta u_i \rangle
        = \tfrac{1}{2} \mathbb{E}_{(s, s') \sim \text{edges}} \big[ \|\phi(s) - \phi(s')\|^2 \big],

    so no transition model is needed -- sampled edges suffice. The edges are drawn symmetrically
    (each state paired with its predecessor or its successor with equal probability, see
    :func:`omnisafe.utils.contrastive.sample_temporal_pairs` at ``horizon=1``), which is what
    makes the operator being decomposed the *symmetrized* one the identity above assumes.

    Args:
        phi_s (torch.Tensor): Features of the edge sources, shape ``(M, sr_dim)``.
        phi_s_next (torch.Tensor): Features of the edge targets, shape ``(M, sr_dim)``,
            row-aligned with ``phi_s``.

    Returns:
        The scalar Dirichlet energy, averaged over edges.
    """
    return 0.5 * (phi_s - phi_s_next).pow(2).sum(-1).mean()


def orthogonality_error(
    phi_a: torch.Tensor,
    phi_b: torch.Tensor | None = None,
    asymmetric: bool = True,
) -> torch.Tensor:
    r"""Lower-triangular constraint residual :math:`\mathbb{E}[\phi_i \phi_j] - \delta_{ij}`.

    Only the lower triangle (``j <= i``) is returned: the covariance is symmetric, so enforcing
    both triangles would double-count every off-diagonal constraint and rescale the barrier
    coefficient behind the user's back.

    Args:
        phi_a (torch.Tensor): Features of one state batch, shape ``(N, sr_dim)``.
        phi_b (torch.Tensor or None, optional): Features of a second, *independent* state batch of
            the same width. When given, the estimator is the cross-batch
            ``phi_a^T phi_b / N`` rather than ``phi_a^T phi_a / N``. Defaults to ``None``.
        asymmetric (bool): Apply ALLO's stop-gradient to the ``j`` (column) factor, so the
            constraint on the pair ``(i, j)`` moves coordinate ``i`` only. This is the change that
            breaks the rotation symmetry and orders the recovered eigenvectors; pass ``False`` to
            obtain the plain symmetric residual used for the dual update and for logging.
            Defaults to ``True``.

    Returns:
        A ``(sr_dim, sr_dim)`` lower-triangular tensor; entries above the diagonal are exactly
        zero and carry no gradient.
    """
    left = phi_a
    right = phi_a if phi_b is None else phi_b
    if asymmetric:
        right = right.detach()
    n = left.shape[0]
    cov = left.T @ right / n
    return torch.tril(cov - torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device))


def allo_loss(
    phi_s: torch.Tensor,
    phi_s_next: torch.Tensor,
    phi_rho_a: torch.Tensor,
    phi_rho_b: torch.Tensor,
    dual: torch.Tensor,
    barrier: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    r"""The Augmented Lagrangian Laplacian Objective, as a loss to minimize in ``phi``.

    .. math::

        \mathcal{L}(u, \beta) = \sum_{i=1}^{d} \langle u_i, \Delta u_i \rangle
        + \sum_{j \le i} \beta_{ij} \big( \langle u_i, \bar{u}_j \rangle - \delta_{ij} \big)
        + b \sum_{j \le i} \big( \langle u_i, \bar{u}_j \rangle - \delta_{ij} \big)
          \overline{\big( \langle u_i, u_j \rangle - \delta_{ij} \big)}

    where the overbar is a stop-gradient. Two independent state batches are required, and the
    reason is a bias that a single batch cannot avoid: the barrier is nominally
    :math:`\frac{b}{2} \sum (\cdot)^2`, whose gradient carries the constraint residual as a
    *factor*, and :math:`\mathbb{E}[\hat{x}]^2 \ne \mathbb{E}[\hat{x}^2]` for a minibatch estimate
    :math:`\hat{x}`. Estimating the detached factor on ``phi_rho_b`` and the differentiable one on
    ``phi_rho_a`` makes the product's expectation the true gradient. Written this way the barrier
    term's *value* is not the squared penalty itself, only something with the same gradient, which
    is all that is asked of it.

    Args:
        phi_s (torch.Tensor): Features of the edge sources, shape ``(M, sr_dim)``.
        phi_s_next (torch.Tensor): Features of the edge targets, shape ``(M, sr_dim)``.
        phi_rho_a (torch.Tensor): Features of a state batch drawn from the visitation
            distribution, shape ``(N, sr_dim)``.
        phi_rho_b (torch.Tensor): Features of a second state batch drawn *independently* of
            ``phi_rho_a``, shape ``(N', sr_dim)``. See above for why one batch will not do.
        dual (torch.Tensor): Lower-triangular Lagrange multipliers, shape ``(sr_dim, sr_dim)``.
            Used detached -- this function never updates them; the caller applies dual ascent with
            the ``violation`` it returns.
        barrier (float): Quadratic barrier coefficient ``b``. Unlike the penalty methods ALLO
            replaces, the solution does not depend on it; it only conditions the descent.

    Returns:
        ``(loss, violation, stats)`` -- ``loss`` is the scalar to backward through; ``violation``
        is the detached symmetric lower-triangular constraint residual the caller feeds to dual
        ascent; ``stats`` is a dict of detached floats for logging.
    """
    assert phi_s.shape == phi_s_next.shape, (
        f'phi_s {tuple(phi_s.shape)} and phi_s_next {tuple(phi_s_next.shape)} must match.'
    )
    assert phi_rho_a.shape[-1] == phi_s.shape[-1] and phi_rho_b.shape[-1] == phi_s.shape[-1], (
        'every feature batch must have the same width (sr_dim).'
    )

    energy = dirichlet_energy(phi_s, phi_s_next)
    err_grad = orthogonality_error(phi_rho_a, asymmetric=True)
    err_coef = orthogonality_error(phi_rho_b, asymmetric=True).detach()

    lagrangian = (dual.detach() * err_grad).sum()
    penalty = barrier * (err_grad * err_coef).sum()
    loss = energy + lagrangian + penalty

    with torch.no_grad():
        # Symmetric and pooled over both batches: the dual step should track the constraint the
        # user cares about, not the stop-gradiented surrogate the descent direction is built from.
        violation = 0.5 * (
            orthogonality_error(phi_rho_a, asymmetric=False)
            + orthogonality_error(phi_rho_b, asymmetric=False)
        )
        d = violation.shape[0]
        n_offdiag = d * (d - 1) / 2
        stats = {
            'Loss': loss.item(),
            'Dirichlet': energy.item(),
            # Split because they fail differently: a large diagonal residual means the features
            # have collapsed in scale (or a dimension is dead), a large off-diagonal one means
            # they are correlated and the effective rank is below sr_dim.
            'DiagErr': violation.diagonal().abs().mean().item(),
            'OffDiagErr': (
                (violation.abs().sum() - violation.diagonal().abs().sum()) / n_offdiag
            ).item()
            if n_offdiag > 0
            else 0.0,
            'DualNorm': dual.norm().item(),
        }

    return loss, violation, stats


def dual_ascent_(dual: torch.Tensor, violation: torch.Tensor, dual_lr: float) -> None:
    r"""Take one dual-ascent step on the multipliers, in place.

    .. math::

        \beta_{ij} \leftarrow \beta_{ij} + \eta \big( \langle u_i, u_j \rangle - \delta_{ij} \big)

    Ascent, not descent: the multipliers maximize the Lagrangian, and it is their growth on a
    persistently violated constraint that eventually forces ``phi`` to satisfy it -- which is why
    the equilibrium is exact rather than a barrier-coefficient-dependent compromise.

    Args:
        dual (torch.Tensor): The multipliers to update in place, shape ``(sr_dim, sr_dim)``.
        violation (torch.Tensor): The lower-triangular constraint residual from :func:`allo_loss`.
        dual_lr (float): Dual step size.
    """
    dual.add_(dual_lr * violation)
