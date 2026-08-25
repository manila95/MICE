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
r"""Contrastive successor-measure estimation for ``sr_cfgs.psi_objective='contrastive'``.

The C-learning / contrastive-RL alternative to fitting ``psi`` by TD (Eysenbach et al.,
*C-Learning*, ICLR 2021; *Contrastive Learning as Goal-Conditioned RL*, NeurIPS 2022). Rather than
bootstrapping ``psi(s) ~= phi(s) + gamma * psi(s')`` against a target built from a separately
obtained ``phi``, ``phi`` and ``psi`` are trained **jointly** as the two factors of one bilinear
critic

.. math::

    \rho_\gamma(s, g) := \psi(s)^T \phi(g) \approx
    \frac{\sum_{t \ge 0} \gamma^t P(s_t = g \mid s_0 = s)}{\rho(g)},

the density ratio of the (unnormalized) discounted successor measure against the visitation
distribution. Positives are states drawn from ``s``'s own discounted future, negatives are states
drawn from the batch at large; the objective is a classification-flavored one, with no bootstrap
anywhere. Three things follow, and they are the reason this mode exists:

1. **No moving target.** ``psi``'s TD target refers to ``phi``, so under any ``phi_source`` that
   is not frozen, ``psi`` chases a feature map that is itself still moving -- the pathology the
   frozen sources exist to avoid and the trained ones reintroduce. Here there is no target to be
   stale: one loss, one optimizer, both factors.
2. **The ridge read-out stays exactly as it is, and becomes exactly right.** Writing
   ``V(s) = E[\sum_t \gamma^t r(s_t)]`` as an expectation under ``rho`` weighted by the ratio
   gives ``V(s) = \psi(s)^T E_\rho[\phi(g) r(g)]`` -- a linear read-out of ``psi``, which is
   precisely the ``value(s) = psi(s) . w`` that
   :class:`~omnisafe.models.critic.successor_representation_critic.SuccessorRepresentationLinearReadout`
   already computes and that ``ridge_update`` already solves for. Note the factorization is of the
   ratio in its *linear* scale: an InfoNCE critic with the inner product as its **logit** would
   converge to ``log`` of this ratio, and no linear read-out of ``psi`` would recover the value.
3. **Orthonormal ``phi`` would be the exact case, and nothing here enforces it.** The ridge
   solves ``w = E[\phi \phi^T]^{-1} E[\phi r]`` -- the best linear predictor of the reward from
   ``phi``, which is well-formed whatever ``phi``'s second moment is, but which equals the
   ``E_\rho[\phi r]`` the identity above literally asks for only when ``E[\phi \phi^T] = I``.
   The ``'joint'`` ``phi`` carries no such constraint, so in practice the read-out is the
   projection rather than the expectation, and ``psi`` equals the Monte-Carlo successor feature
   ``\sum_t \gamma^t \phi(s_t)`` only up to that same ``E[\phi \phi^T]^{-1}`` -- which is why
   ``SR/*/MCExplainedVar`` is *not* the metric to judge this mode by (``Misc/CLearningRatioMass``
   is: the fitted ratio must integrate to ``1 / (1 - gamma)`` under ``rho``).

   The obvious fix does not compose the way it looks like it should: ``phi_source='laplacian'``
   enforces exactly that identity, but it fits ``phi`` with a second objective and a second
   optimizer, and this critic's *magnitude* is its estimate -- a ``phi`` rescaled by another loss
   silently rescales the successor measure. Combining them is rejected at construction. Folding
   ALLO's orthonormality term into this loss, as one objective over one set of parameters, is the
   way the two would actually meet; it is not implemented here.

Kept, like :mod:`omnisafe.utils.contrastive` and :mod:`omnisafe.utils.laplacian`, as plain
tensor-in/tensor-out functions with no network, config, or logger dependency, so the estimator can
be checked against a tabular MDP whose successor measure is known exactly by linear solve
(``tests/test_clearning.py``). The gradient loop that calls these lives in
:meth:`~omnisafe.algorithms.on_policy.base.policy_gradient.PolicyGradient._contrastive_update_successor_features`.
"""

from __future__ import annotations

import math

import torch


def sample_geometric_futures(
    lengths: list[int],
    gamma: float,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Sample one future-state row index per row, at a geometrically distributed offset.

    Follows the ``lengths`` convention of
    :func:`omnisafe.utils.sr_diagnostics.discount_cumsum_segments` (concatenated episode segments,
    each contiguous and time-ordered). For every row ``t`` in a segment of length ``L`` draws an
    offset ``k`` from a geometric distribution ``P(k) \propto \gamma^k`` on ``{0, 1, 2, ...}``,
    *truncated* to the window ``0 <= k <= K`` with ``K = L - 1 - t``, and returns ``(t, t + k)``.
    No row is dropped: ``k = 0`` is always available, even for the last row of an episode.

    Geometric, not uniform, and forward-only, not symmetric: ``P(k) \propto \gamma^k`` *is* the
    discounted future-state distribution whose density ratio the loss below estimates. A symmetric
    window (what :func:`omnisafe.utils.contrastive.sample_temporal_pairs` draws, appropriately for
    a time-contrastive objective) would estimate the ratio of a different, time-symmetric measure,
    which is not the successor measure and not what ``psi`` reads out against.

    ``k = 0`` is included, and the inclusion is load-bearing rather than a convention. The
    successor measure is ``\sum_{t \ge 0} \gamma^t P^t``, whose ``t = 0`` term is the identity --
    the same term the recursion ``\psi(s) = \phi(s) + \gamma \psi(s')`` carries as its own
    ``\phi(s)``, and the same one ``V(s) = \sum_t \gamma^t r(s_t)`` carries as ``r(s)``. Excluding
    it fits ``\gamma P (I - \gamma P)^{-1}`` instead: a systematic bias no amount of training
    removes, and one that survives every scale check, since the resulting ratio still integrates
    to very nearly the right mass.

    Truncation is a real approximation, not a formality: conditioning on ``k <= K`` reweights the
    tail mass onto the window that fits, so rows near the end of an episode see a systematically
    nearer future than the true discounted distribution prescribes. It is unavoidable with finite
    episodes and it shrinks as the episode length grows relative to ``1 / (1 - gamma)``; on a
    1000-step episode at ``gamma = 0.99`` (effective horizon 100) it affects the last ~10% of rows.

    Args:
        lengths (list of int): Length of each segment, in the order the corresponding rows appear
            in the flat batch (e.g. ``train_data['_episode_lengths']``).
        gamma (float): Discount of the successor measure, in ``[0, 1)``. Values at or above
            ``1 - 1e-6`` fall back to uniform sampling over ``{0, ..., K}``, which is the limit of
            the truncated geometric as ``gamma -> 1`` and avoids dividing by ``log(gamma) -> 0``.
        generator (torch.Generator or None, optional): Passed to the internal ``torch.rand`` call
            for reproducible sampling. Must live on ``device``. Defaults to ``None``.
        device (torch.device or str or None, optional): Device to build the index tensors on. Pass
            the device of the batch these will index; see
            :func:`omnisafe.utils.contrastive.sample_temporal_pairs` for why it matters.

    Returns:
        ``(anchor_idx, future_idx)``, two ``int64`` tensors of equal shape ``(sum(lengths),)``,
        each indexing directly into the flat batch. ``future_idx >= anchor_idx``, with equality
        whenever ``k = 0`` was drawn.
    """
    assert 0.0 <= gamma < 1.0, f'gamma must lie in [0, 1), got {gamma}.'
    if not lengths:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    n = int(lengths_t.sum().item())
    device = lengths_t.device

    seg_starts = torch.cumsum(lengths_t, dim=0) - lengths_t
    row_start = torch.repeat_interleave(seg_starts, lengths_t)
    row_length = torch.repeat_interleave(lengths_t, lengths_t)
    global_idx = torch.arange(n, device=device)
    # K, the largest offset still inside the episode. Zero on the last row, which is fine --
    # that row's only valid draw is k = 0, and k = 0 is a legitimate draw.
    horizon = row_length - 1 - (global_idx - row_start)
    anchor_idx = global_idx
    horizon_f = horizon.to(torch.float64)

    u = torch.rand(anchor_idx.shape[0], generator=generator, device=device, dtype=torch.float64)
    if gamma <= 1e-12:
        # All the discounted mass sits on t = 0; log(gamma) below would be a domain error.
        offset = torch.zeros_like(anchor_idx)
    elif gamma >= 1.0 - 1e-6:
        offset = (u * (horizon_f + 1.0)).floor().long()
    else:
        # Inverse-CDF of the geometric on {0..K}: F(k) = (1 - gamma^(k+1)) / (1 - gamma^(K+1)),
        # so k = ceil(log(1 - u * (1 - gamma^(K+1))) / log(gamma)) - 1.
        log_gamma = math.log(gamma)
        # gamma^(K+1), via exp/log so a long episode cannot overflow the exponentiation.
        tail = torch.exp((horizon_f + 1.0) * log_gamma)
        offset = torch.ceil(torch.log1p(-u * (1.0 - tail)) / log_gamma).long() - 1
    # Guard the boundaries against float error at both ends rather than trusting the algebra.
    offset = torch.clamp(offset, min=0)
    offset = torch.minimum(offset, horizon)

    return anchor_idx, anchor_idx + offset


def density_ratio_loss(
    psi_anchor: torch.Tensor,
    phi_future: torch.Tensor,
    phi_random: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    r"""Least-squares density-ratio loss fitting ``psi(s)^T phi(g)`` to the successor measure.

    .. math::

        \mathcal{L} = \tfrac{1}{2} \mathbb{E}_{s} \mathbb{E}_{g \sim \rho}
        \big[ (\psi(s)^T \phi(g))^2 \big]
        - \frac{1}{1 - \gamma} \mathbb{E}_{s} \mathbb{E}_{g \sim p_\gamma(\cdot \mid s)}
        \big[ \psi(s)^T \phi(g) \big]

    The unconstrained minimizer of ``E_q[f^2]/2 - E_p[f]`` over functions ``f`` is ``f = p / q``
    (differentiate pointwise), so this fits the ratio directly, in its linear scale, with no link
    function and nothing to exponentiate. The ``1 / (1 - gamma)`` is what makes the target the
    *unnormalized* successor measure rather than the normalized future distribution
    ``p_gamma`` that :func:`sample_geometric_futures` draws from -- and that in turn is what lets
    the existing ``value(s) = psi(s) . w`` read-out be used unchanged, since
    ``sum_t gamma^t r(s_t) = psi(s)^T E_rho[phi(g) r(g)]`` only holds for the unnormalized one.

    The negative term is estimated over *all* anchor-negative pairs in the batch at once via a
    single matmul, so a batch of ``M`` anchors contributes ``M^2`` samples of the ``rho``
    expectation rather than ``M``. The positives are row-aligned, one per anchor.

    Args:
        psi_anchor (torch.Tensor): Successor features of the anchors, shape ``(M, sr_dim)``.
        phi_future (torch.Tensor): One-step features of each anchor's sampled future state, shape
            ``(M, sr_dim)``, row-aligned with ``psi_anchor``.
        phi_random (torch.Tensor): One-step features of states drawn from the visitation
            distribution, shape ``(N, sr_dim)``, independent of the anchors.
        gamma (float): Discount of the successor measure; must match the one
            :func:`sample_geometric_futures` drew the positives with.

    Returns:
        ``(loss, stats)`` -- ``loss`` is the scalar to backward through; ``stats`` is a dict of
        detached floats for logging.
    """
    assert psi_anchor.shape == phi_future.shape, (
        f'psi_anchor {tuple(psi_anchor.shape)} and phi_future {tuple(phi_future.shape)} must match.'
    )
    negative = (psi_anchor @ phi_random.T).pow(2).mean()
    positive = (psi_anchor * phi_future).sum(-1).mean()
    loss = 0.5 * negative - positive / (1.0 - gamma)

    with torch.no_grad():
        # The ratio integrates to the total discounted mass: E_{g~rho}[psi(s)^T phi(g)] should
        # converge to sum_t gamma^t = 1 / (1 - gamma) for every s. Cheap, and the single most
        # informative number here -- a fit that is drifting or collapsing shows up in it long
        # before it shows up in the loss, which has no meaningful scale of its own.
        mass = (psi_anchor @ phi_random.T).mean().item()
        stats = {
            'Loss': loss.item(),
            'PosRatio': positive.item(),
            'NegSq': negative.item(),
            'RatioMass': mass,
            'RatioMassTarget': 1.0 / (1.0 - gamma),
        }

    return loss, stats
