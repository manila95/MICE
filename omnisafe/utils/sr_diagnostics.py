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
r"""Metrics for diagnosing the ``td_ridge`` successor-representation critic.

The SR critic factors the value function through three pieces that are fitted separately::

    phi(s)  ->  psi(s) ~= sum_k gamma^k phi(s_{t+k})  ->  w_r / w_c  ->  V(s) = psi(s) . w
     basis         successor-feature regression           read-out         critic

A poor ``V`` can originate in any of them, and the composite value-critic correlations logged by
:meth:`PolicyGradient._log_critic_diagnostics` cannot tell which. The functions here are the
per-link measurements that can, grouped as:

* **Basis quality** -- :func:`ridge_fit`, :func:`r2_score`, :func:`effective_rank`,
  :func:`stable_rank`, :func:`dead_dim_fraction`. Can ``phi``'s span express the one-step
  reward/cost at all, out of sample, and how many directions does it actually use?
* **Regression quality** -- :func:`explained_variance`, :func:`per_dim_explained_variance`,
  :func:`cosine_norm_stats`. Does ``psi`` match its target? These replace correlation, which is
  undefined for the vector-valued ``psi``; :func:`explained_variance` is the direct analogue and
  :func:`cosine_norm_stats` separates a direction error (fatal) from a scale error (recoverable).
* **Non-stationarity** -- :func:`relative_change`, :func:`mean_cosine`, :func:`subspace_distance`.
  How far did ``phi`` / ``psi`` / ``w`` move under the very fit that assumes them fixed?
  :func:`subspace_distance` is the discriminating one: a rotation *within* ``phi``'s span leaves
  the ridge solution reachable, a change *of* the span does not.

:func:`discount_cumsum_segments` supplies the ground truth the regression metrics are measured
against -- the Monte-Carlo successor feature -- for many episodes at once.

Every function takes and returns plain tensors, and none of them touch a network, a config, or
the logger, so they can be checked against hand-computed values in isolation.
"""

from __future__ import annotations

import torch


def discount_cumsum_segments(
    vector_x: torch.Tensor,
    lengths: list[int],
    discount: float,
) -> torch.Tensor:
    r"""Discounted reverse cumulative sum over many variable-length segments at once.

    The batched counterpart of :func:`omnisafe.utils.math.discount_cumsum`, and the way the
    Monte-Carlo successor feature :math:`\Psi_{MC}(s_t) = \sum_k \gamma^k \phi(s_{t+k})` is built
    for a whole epoch. Applying the scalar helper per episode costs one Python-level iteration per
    *transition*; padding the episodes into a ``(n_seg, T_max, d)`` block costs one per *timestep
    of the longest episode*, which on an epoch of many short episodes is the difference between
    seconds and milliseconds.

    Args:
        vector_x (torch.Tensor): Concatenated segments of shape ``(N, ...)``, where ``N`` is
            ``sum(lengths)`` and the leading dimension is time within each segment.
        lengths (list of int): Length of each segment, in the order they appear in ``vector_x``.
        discount (float): The discount factor.

    Returns:
        A tensor shaped like ``vector_x`` holding the discounted reverse cumulative sum computed
        independently within each segment.
    """
    assert sum(lengths) == vector_x.shape[0], (
        f'segment lengths sum to {sum(lengths)} but vector_x has {vector_x.shape[0]} rows.'
    )
    if not lengths:
        return vector_x.clone()

    device = vector_x.device
    n_seg, t_max = len(lengths), max(lengths)
    feature_shape = vector_x.shape[1:]
    len_t = torch.as_tensor(lengths, device=device).reshape(n_seg, *([1] * len(feature_shape)))

    # scatter the ragged segments into a dense (n_seg, t_max, ...) block
    padded = vector_x.new_zeros((n_seg, t_max, *feature_shape))
    offset = 0
    for i, length in enumerate(lengths):
        padded[i, :length] = vector_x[offset : offset + length]
        offset += length

    out = torch.zeros_like(padded)
    running = padded.new_zeros((n_seg, *feature_shape))
    for t in reversed(range(t_max)):
        # Padding positions (t >= length) reset the accumulator, so each segment starts from zero
        # at its own final step -- they are visited first, before any real step of that segment.
        active = t < len_t
        running = torch.where(active, padded[:, t] + discount * running, torch.zeros_like(running))
        out[:, t] = running

    gathered = vector_x.new_empty(vector_x.shape)
    offset = 0
    for i, length in enumerate(lengths):
        gathered[offset : offset + length] = out[i, :length]
        offset += length
    return gathered


def gae_lambda_targets_segments(
    values: torch.Tensor,
    rewards: torch.Tensor,
    lengths: list[int],
    lam: float,
    gamma: float,
) -> torch.Tensor:
    r"""Recompute GAE-lambda targets per segment, with a zero bootstrap at each segment's end.

    Mirrors the ``'gae'`` branch of :meth:`omnisafe.common.buffer.onpolicy_buffer.OnPolicyBuffer.
    _calculate_adv_and_value_targets` -- ``delta_t = rewards_t + gamma*values_{t+1} - values_t``,
    ``target = discount_cumsum(delta, gamma*lam) + values`` -- applied independently within each
    segment of ``lengths`` rather than to one contiguous path, and reusing
    :func:`discount_cumsum_segments` as the reverse-cumsum subroutine.

    This exists to *relabel* :meth:`~omnisafe.algorithms.on_policy.base.policy_gradient.
    PolicyGradient`'s cached ``target_sr`` (``values=psi``, ``rewards=phi``) after ``phi`` has
    moved by more than a normal epoch's drift -- e.g. the ``phi_source='contrastive'`` epoch-0
    pretraining burst -- so the ``psi`` TD target isn't built against a ``phi`` that no longer
    exists by the time it's used.

    The bootstrap at each segment's end is **always zero**, which is exact for naturally-terminated
    episodes (:meth:`OnPolicyBuffer.finish_path` already defaults ``last_psi`` to zero for those)
    but an approximation for episodes truncated by ``steps_per_epoch``: the true bootstrap value
    used at rollout time depended on the next (discarded, un-stored) observation, which cannot be
    recovered from ``values``/``rewards`` alone after the fact. Callers relabelling ``target_sr``
    should treat this as an accepted, documented bias on truncated segments only.

    Args:
        values (torch.Tensor): Concatenated segments of shape ``(N, d)`` -- e.g. ``psi``.
        rewards (torch.Tensor): Concatenated segments of shape ``(N, d)``, row-aligned with
            ``values`` -- e.g. ``phi``.
        lengths (list of int): Length of each segment, in the order they appear in ``values`` /
            ``rewards``.
        lam (float): The lambda parameter of the GAE-lambda recursion.
        gamma (float): The discount factor.

    Returns:
        A tensor shaped like ``values`` holding the recomputed target within each segment.
    """
    assert values.shape == rewards.shape, (
        f'values {tuple(values.shape)} and rewards {tuple(rewards.shape)} must have the same shape.'
    )
    assert sum(lengths) == values.shape[0], (
        f'segment lengths sum to {sum(lengths)} but values has {values.shape[0]} rows.'
    )
    if not lengths:
        return values.clone()

    feature_shape = values.shape[1:]
    deltas = values.new_empty(values.shape)
    offset = 0
    for length in lengths:
        seg_values = values[offset : offset + length]
        seg_rewards = rewards[offset : offset + length]
        # Bootstrap value at the segment's end is zero -- see the docstring's accepted-bias note.
        next_values = torch.cat(
            [seg_values[1:], seg_values.new_zeros((1, *feature_shape))],
            dim=0,
        )
        deltas[offset : offset + length] = seg_rewards + gamma * next_values - seg_values
        offset += length

    adv = discount_cumsum_segments(deltas, lengths, gamma * lam)
    return adv + values


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    r"""Fraction of the target's total variance captured by ``pred``.

    .. math::

        EV = 1 - \frac{\|pred - target\|_F^2}{\|target - \bar{target}\|_F^2}

    The vector-valued replacement for the Pearson correlation used on the scalar critics: ``1.0``
    is a perfect fit, ``0.0`` is no better than predicting the mean feature, and negative values
    mean worse than the mean. Unlike a raw MSE it is invariant to the scale of ``phi``, which
    drifts freely during training and would otherwise dominate the series.

    Args:
        pred (torch.Tensor): Predictions of shape ``(N, d)`` (or ``(N,)``).
        target (torch.Tensor): Targets of the same shape.

    Returns:
        The explained variance as a float, or ``nan`` if the target has no variance.
    """
    pred, target = pred.double(), target.double()
    resid = (pred - target).pow(2).sum()
    total = (target - target.mean(dim=0, keepdim=True)).pow(2).sum()
    if total <= 0:
        return float('nan')
    return (1.0 - resid / total).item()


def per_dim_explained_variance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Explained variance computed separately for each feature dimension.

    Distinguishes a representation whose accuracy is spread evenly over its dimensions from one
    where a handful of dimensions are fitted well and the rest are noise -- two situations with
    identical aggregate :func:`explained_variance`.

    Args:
        pred (torch.Tensor): Predictions of shape ``(N, d)``.
        target (torch.Tensor): Targets of shape ``(N, d)``.

    Returns:
        A ``(d,)`` tensor of per-dimension explained variances; dimensions with no target variance
        are ``nan``.
    """
    pred, target = pred.double(), target.double()
    resid = (pred - target).pow(2).sum(dim=0)
    total = (target - target.mean(dim=0, keepdim=True)).pow(2).sum(dim=0)
    return torch.where(total > 0, 1.0 - resid / total, torch.full_like(total, float('nan')))


def cosine_norm_stats(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Split the prediction error into a direction part and a magnitude part.

    ``psi`` pointing the right way with the wrong magnitude and ``psi`` pointing the wrong way are
    both bad :func:`explained_variance`, but only the second is a broken representation: a
    uniformly mis-scaled ``psi`` is still a linear re-parameterization that the read-out weights
    can absorb.

    Args:
        pred (torch.Tensor): Predictions of shape ``(N, d)``.
        target (torch.Tensor): Targets of shape ``(N, d)``.

    Returns:
        The mean row-wise cosine similarity, and the mean row-wise norm ratio
        ``||pred|| / ||target||``.
    """
    pred, target = pred.double(), target.double()
    pred_norm = pred.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    denom = (pred_norm * target_norm).clamp(min=1e-12)
    cos = ((pred * target).sum(-1) / denom).mean().item()
    ratio = (pred_norm / target_norm.clamp(min=1e-12)).mean().item()
    return cos, ratio


def _singular_values(mat: torch.Tensor) -> torch.Tensor:
    """Singular values of the mean-centered matrix, in float64, descending."""
    mat = mat.double()
    return torch.linalg.svdvals(mat - mat.mean(dim=0, keepdim=True))


def effective_rank(mat: torch.Tensor) -> float:
    r"""Entropy-based effective rank of a feature matrix.

    .. math::

        \mathrm{erank} = \exp\left(-\sum_i p_i \log p_i\right),
        \qquad p_i = \sigma_i / \sum_j \sigma_j

    A soft count of how many directions of the ``sr_dim``-dimensional feature space are actually
    in use. It is bounded above by the algebraic rank but, unlike it, degrades continuously, so a
    representation quietly collapsing onto a few directions shows up as a falling curve rather
    than a step at the point of exact singularity.

    Args:
        mat (torch.Tensor): Feature matrix of shape ``(N, d)``.

    Returns:
        The effective rank, between ``1`` and ``min(N, d)``.
    """
    svals = _singular_values(mat)
    total = svals.sum()
    if total <= 0:
        return 0.0
    probs = svals / total
    entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum()
    return torch.exp(entropy).item()


def stable_rank(mat: torch.Tensor) -> float:
    r"""Stable rank :math:`\|M\|_F^2 / \|M\|_2^2` of a feature matrix.

    A second, cheaper collapse measure that weights the spectrum differently from
    :func:`effective_rank` (by squared rather than log-scaled singular values), so it reacts to
    one direction dominating in energy rather than to the spread of the whole spectrum.

    Args:
        mat (torch.Tensor): Feature matrix of shape ``(N, d)``.

    Returns:
        The stable rank, between ``1`` and ``min(N, d)``.
    """
    svals = _singular_values(mat)
    top = svals[0] if svals.numel() else torch.zeros((), dtype=torch.float64)
    if top <= 0:
        return 0.0
    return (svals.pow(2).sum() / top.pow(2)).item()


def dead_dim_fraction(mat: torch.Tensor, threshold: float = 1e-6) -> float:
    """Fraction of feature dimensions whose variance across the batch is negligible.

    Coordinates that never move carry no information but still consume ``sr_dim``, and -- unlike
    the rank measures, which are basis-independent -- this one names the failure in the
    coordinates the ridge solve actually works in.

    Args:
        mat (torch.Tensor): Feature matrix of shape ``(N, d)``.
        threshold (float): Variance below this counts as dead, relative to the mean variance
            across dimensions. Defaults to ``1e-6``.

    Returns:
        The fraction of dead dimensions, between ``0`` and ``1``.
    """
    var = mat.double().var(dim=0, unbiased=False)
    scale = var.mean().clamp(min=1e-12)
    return (var / scale < threshold).double().mean().item()


def _orthonormal_basis(mat: torch.Tensor, tol: float) -> torch.Tensor:
    """Orthonormal basis of the column space of ``mat``, dropping near-null directions."""
    u_mat, svals, _ = torch.linalg.svd(mat.double(), full_matrices=False)
    if svals.numel() == 0 or svals[0] <= 0:
        return u_mat[:, :0]
    keep = int((svals > tol * svals[0]).sum().item())
    return u_mat[:, :keep]


def subspace_distance(mat_a: torch.Tensor, mat_b: torch.Tensor, tol: float = 1e-3) -> float:
    """Principal-angle distance between the column spaces of two feature matrices.

    The measurement that decides whether ``phi``'s drift matters. ``w_r`` / ``w_c`` are solved
    against ``phi``'s *span*, so a drift that merely rotates the features inside a fixed span
    leaves the ridge solution reachable (the fitted ``w`` is stale but a re-solve recovers it),
    whereas a drift that tilts the span itself invalidates the regression that ``psi`` and ``w``
    were both built on. :func:`relative_change` and :func:`mean_cosine` cannot separate the two;
    this can.

    Both matrices must share the same number of rows -- they are ``phi`` evaluated on one probe
    set of observations at two different times.

    Args:
        mat_a (torch.Tensor): First feature matrix of shape ``(N, d)``.
        mat_b (torch.Tensor): Second feature matrix of shape ``(N, d)``.
        tol (float): Relative singular-value threshold for deciding which directions are part of
            the column space. Defaults to ``1e-3``.

    Returns:
        ``1 - mean(cos(theta_i))`` over the principal angles: ``0`` for identical spans, rising to
        ``1`` as they become orthogonal.
    """
    assert mat_a.shape[0] == mat_b.shape[0], 'subspace_distance needs matching row counts.'
    basis_a = _orthonormal_basis(mat_a, tol)
    basis_b = _orthonormal_basis(mat_b, tol)
    rank = min(basis_a.shape[1], basis_b.shape[1])
    if rank == 0:
        return float('nan')
    cos = torch.linalg.svdvals(basis_a[:, :rank].T @ basis_b[:, :rank])
    return (1.0 - cos.clamp(max=1.0).mean()).item()


def relative_change(new: torch.Tensor, old: torch.Tensor) -> float:
    """Relative Frobenius-norm change ``||new - old|| / ||old||``.

    Args:
        new (torch.Tensor): The later value.
        old (torch.Tensor): The earlier value, of the same shape.

    Returns:
        The relative change, or ``nan`` if ``old`` is zero.
    """
    new, old = new.double(), old.double()
    denom = old.norm()
    if denom <= 0:
        return float('nan')
    return ((new - old).norm() / denom).item()


def mean_cosine(new: torch.Tensor, old: torch.Tensor) -> float:
    """Mean row-wise cosine similarity between two feature matrices, or between two vectors.

    Args:
        new (torch.Tensor): The later value, of shape ``(N, d)`` or ``(d,)``.
        old (torch.Tensor): The earlier value, of the same shape.

    Returns:
        The mean cosine similarity.
    """
    new, old = new.double(), old.double()
    if new.dim() == 1:
        new, old = new.reshape(1, -1), old.reshape(1, -1)
    denom = (new.norm(dim=-1) * old.norm(dim=-1)).clamp(min=1e-12)
    return ((new * old).sum(-1) / denom).mean().item()


def ridge_fit(
    features: torch.Tensor,
    targets: torch.Tensor,
    ridge_kappa: float = 1e-3,
) -> torch.Tensor:
    r"""Closed-form ridge regression of ``targets`` onto ``features``.

    .. math::

        w = (\Phi^T \Phi + \kappa I)^{-1} \Phi^T y

    Matches :meth:`RidgeSolvedReadoutWeights.ridge_update` -- same float64 solve and the same
    ``kappa`` scaling by the mean Gram diagonal -- so an oracle weight fitted here is directly
    comparable to the weights the critic actually uses.

    Args:
        features (torch.Tensor): Design matrix of shape ``(N, d)``.
        targets (torch.Tensor): Targets of shape ``(N,)``.
        ridge_kappa (float): Regularization coefficient, scaling the mean diagonal of the Gram
            matrix. Defaults to ``1e-3``.

    Returns:
        The fitted weight vector of shape ``(d,)``, in the dtype of ``features``.
    """
    feat = features.double()
    gram = feat.T @ feat
    kappa = ridge_kappa * torch.diagonal(gram).mean().clamp(min=1e-12)
    mat = gram + kappa * torch.eye(feat.shape[1], dtype=torch.float64, device=feat.device)
    weight = torch.linalg.solve(mat, feat.T @ targets.double().reshape(-1))
    return weight.to(features.dtype)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    r"""Coefficient of determination for a scalar prediction.

    The scale-free counterpart of the raw RMSE residuals reported by ``ridge_update``: an RMSE
    rises and falls with the magnitude of ``phi`` and of the rewards, so it cannot be compared
    across epochs or between the training and validation splits, which is precisely the
    comparison the generalization question needs.

    Args:
        pred (torch.Tensor): Predictions of shape ``(N,)``.
        target (torch.Tensor): Targets of shape ``(N,)``.

    Returns:
        The :math:`R^2` score, or ``nan`` if the target is constant.
    """
    pred, target = pred.double().reshape(-1), target.double().reshape(-1)
    resid = (pred - target).pow(2).sum()
    total = (target - target.mean()).pow(2).sum()
    if total <= 0:
        return float('nan')
    return (1.0 - resid / total).item()


def correlation(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Pearson correlation between two 1-d tensors, ``nan`` if either is constant.

    A guarded :func:`torch.corrcoef`, used so a degenerate batch produces a ``nan`` entry in the
    logs instead of aborting the whole diagnostic pass.

    Args:
        vec_a (torch.Tensor): First tensor of shape ``(N,)``.
        vec_b (torch.Tensor): Second tensor of shape ``(N,)``.

    Returns:
        The Pearson correlation coefficient.
    """
    vec_a, vec_b = vec_a.double().reshape(-1), vec_b.double().reshape(-1)
    if vec_a.numel() < 2 or vec_a.std() <= 0 or vec_b.std() <= 0:
        return float('nan')
    return torch.corrcoef(torch.stack([vec_a, vec_b]))[0, 1].item()
