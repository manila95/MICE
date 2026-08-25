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
r"""Time-contrastive pair sampling and InfoNCE loss for ``phi_source='contrastive'``.

Backs ``td_ridge``'s time-contrastive ``phi_source`` (see
:mod:`omnisafe.models.critic.successor_representation_critic`'s module docstring and
:class:`~omnisafe.models.critic.successor_representation_critic.ContrastivePhiFeatures`), rather
than either drifting implicitly under the ``psi``/value loss (``'trunk'``) or staying fixed at
init (every other source). The idea is a Time-Contrastive Network: states visited close together
in time within one rollout episode are pulled together in ``phi``-space, temporally distant or
other-episode states are pushed apart -- an inductive bias that a temporally-smooth,
slowly-evolving state space should also be smooth in the representation used to regress reward
and cost onto.

Deliberately kept separate from :mod:`omnisafe.utils.sr_diagnostics`, whose docstring scopes it to
*measuring* an already-fitted critic. What lives here is the training algorithm itself: sampling
the (anchor, positive) pairs the loss is computed over, and the loss function. Both are plain
tensor-in/tensor-out functions with no network, config, or logger dependency, so they can be
checked against hand-computed values in isolation -- the same testability goal
:mod:`sr_diagnostics` states for itself. The optimizer step / gradient loop that calls these lives
in :meth:`~omnisafe.algorithms.on_policy.base.policy_gradient.PolicyGradient._contrastive_update_phi`.

:func:`sample_temporal_pairs` is additionally reused at ``horizon=1`` by ``phi_source='laplacian'``
(:mod:`omnisafe.utils.laplacian`) to draw transition-graph edges: pairing each state with its
predecessor or its successor with equal probability is exactly what makes the operator that
objective decomposes the *symmetrized* one its Dirichlet form assumes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_temporal_pairs(
    lengths: list[int],
    horizon: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Sample one (anchor, positive) row-index pair per valid row of a segmented batch.

    Given the same ``lengths`` convention :func:`omnisafe.utils.sr_diagnostics.discount_cumsum_segments`
    uses (concatenated episode segments, each contiguous and time-ordered), for every row ``t``
    within a segment of length ``L`` draws one offset ``k`` uniformly from the valid window
    ``{-min(horizon, t), ..., -1} \cup {1, ..., min(horizon, L-1-t)}`` -- i.e. a state up to
    ``horizon`` steps in the past *or* future, within the same episode, never ``t`` itself. Rows
    with an empty window (only possible when ``L == 1``) are dropped.

    Fully vectorized: no per-row Python loop (only one per-segment loop over ``lengths`` to derive
    each row's local position and segment length, cheap since the number of episodes per epoch is
    small).

    Args:
        lengths (list of int): Length of each segment, in the order the corresponding rows appear
            in the flat batch (e.g. ``train_data['_episode_lengths']``).
        horizon (int): Maximum ``|offset|`` a positive may be sampled at. Must be ``>= 1``.
        generator (torch.Generator or None, optional): Passed to the internal ``torch.rand`` call
            for reproducible sampling. Must live on ``device``. Defaults to ``None`` (the global
            RNG).
        device (torch.device or str or None, optional): Device to build the index tensors on. Pass
            the device of the batch these indices will be used against: PyTorch permits CPU indices
            into a CUDA tensor but not the reverse, so indices left on the CPU raise as soon as a
            caller draws a minibatch *of* them with a CUDA index tensor. Defaults to ``None``
            (CPU).

    Returns:
        ``(anchor_idx, positive_idx)``, two ``int64`` tensors of equal shape ``(M,)`` with
        ``M &lt;= sum(lengths)``, each indexing directly into the flat batch (e.g. ``obs``/``phi``).
    """
    assert horizon >= 1, f'horizon must be >= 1, got {horizon}.'
    if not lengths:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    n = int(lengths_t.sum().item())
    device = lengths_t.device

    # Per-row local position t_i within its segment, and that segment's length L_i -- both
    # derived without a per-row loop via repeat_interleave over the per-segment lengths.
    seg_starts = torch.cumsum(lengths_t, dim=0) - lengths_t
    row_start = torch.repeat_interleave(seg_starts, lengths_t)
    row_length = torch.repeat_interleave(lengths_t, lengths_t)
    global_idx = torch.arange(n, device=device)
    t = global_idx - row_start

    n_neg = torch.clamp(torch.clamp(t, max=horizon), min=0)
    n_pos = torch.clamp(torch.clamp(row_length - 1 - t, max=horizon), min=0)
    total = n_neg + n_pos
    valid = total > 0
    if not bool(valid.any()):
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    anchor_idx = global_idx[valid]
    n_neg, total = n_neg[valid], total[valid]

    # u ~ Uniform{0, ..., total-1}, then map to the signed offset window: the first n_neg values
    # of u give the past offsets -n_neg..-1, the rest give the future offsets 1..n_pos.
    u = (torch.rand(anchor_idx.shape[0], generator=generator, device=device) * total).floor().long()
    u = torch.clamp(u, max=total - 1)
    k = torch.where(u < n_neg, -(n_neg - u), 1 + (u - n_neg))

    positive_idx = anchor_idx + k
    return anchor_idx, positive_idx


def info_nce_loss(
    anchor_feat: torch.Tensor,
    positive_feat: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    r"""Symmetric InfoNCE / NT-Xent loss with in-batch negatives.

    ``anchor_feat`` / ``positive_feat`` are assumed already l2-normalized (true of
    :class:`~omnisafe.models.critic.successor_representation_critic.ContrastivePhiFeatures`'s
    output), so ``anchor_feat @ positive_feat.T`` is a matrix of cosine similarities. Row ``i``'s
    positive is column ``i``; every other column is an implicit negative -- no explicit negative
    mining is needed since a batch spanning multiple episodes (the usual case once
    :func:`sample_temporal_pairs` has drawn from more than one segment) already supplies
    temporally-distant and cross-episode negatives "for free."

    Args:
        anchor_feat (torch.Tensor): Anchor features, shape ``(M, sr_dim)``.
        positive_feat (torch.Tensor): Positive features, shape ``(M, sr_dim)``, row-aligned with
            ``anchor_feat``.
        temperature (float): Softmax temperature scaling the similarity matrix before the
            cross-entropy. Lower is sharper.

    Returns:
        ``(loss, stats)`` -- ``loss`` is the scalar tensor to backward through; ``stats`` is a
        plain ``{'Loss': ..., 'PosSim': ..., 'NegSim': ...}`` dict of detached floats (raw,
        un-scaled cosine similarities) for logging.
    """
    assert anchor_feat.shape == positive_feat.shape, (
        f'anchor_feat {tuple(anchor_feat.shape)} and positive_feat {tuple(positive_feat.shape)} '
        'must have the same shape.'
    )
    m = anchor_feat.shape[0]
    sim = (anchor_feat @ positive_feat.T) / temperature
    labels = torch.arange(m, device=sim.device)
    loss = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))

    with torch.no_grad():
        diag = sim.diagonal()
        pos_sim = (diag.mean() * temperature).item()
        if m > 1:
            neg_sim = ((sim.sum() - diag.sum()) / (m * (m - 1)) * temperature).item()
        else:
            neg_sim = float('nan')

    return loss, {'Loss': loss.item(), 'PosSim': pos_sim, 'NegSim': neg_sim}
