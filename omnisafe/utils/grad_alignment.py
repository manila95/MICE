r"""Alignment between the reward and the cost policy gradients of a constrained update.

Notation (CPO's): ``g`` is the gradient of the reward surrogate (the direction that raises reward),
``b`` the gradient of the cost surrogate (the direction that raises cost), ``H`` the Fisher matrix.
``cos(g, b) > 0`` means the two objectives *conflict*: following the reward gradient also raises
cost; ``cos(g, b) < 0`` means they cooperate.

A cosine between two minibatch estimates is biased toward 0 by sampling noise: with
``g_hat = g + e_g`` the noise inflates ``|g_hat|`` but not, on average, ``g_hat . b_hat`` taken over
independent samples. :func:`alignment_stats` therefore splits the batch into ``K`` groups of whole
environments (independent given the policy) and uses the cross-group products

.. math::

    \widehat{g \cdot b} = \operatorname{mean}_{i \neq j} g_i \cdot b_j, \qquad
    \widehat{|g|^2} = \operatorname{mean}_{i \neq j} g_i \cdot g_j,

which are unbiased for the population quantities, giving a noise-corrected cosine. The same is done
in the Fisher metric (``u . H^{-1} v``), the geometry CPO's step actually lives in. The
"signal fraction" ``mean_{i!=j} g_i.g_j / mean_i |g_i|^2`` is the share of a group gradient's
squared norm that is signal rather than noise (1 = noiseless, 0 = pure noise).
"""

from __future__ import annotations

from typing import Callable

import torch


def _cos(u: torch.Tensor, v: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(u, v, dim=0).item()


def _cross(a: list[torch.Tensor], b: list[torch.Tensor], metric=None) -> torch.Tensor:
    """mean over i != j of <a_i, b_j>, with <u, v> = u . metric(v) (Euclidean when None)."""
    mb = [metric(v) if metric is not None else v for v in b]
    k = len(a)
    tot = sum(a[i].dot(mb[j]) for i in range(k) for j in range(k) if i != j)
    return tot / (k * (k - 1))


def _corrected_cos(gb: torch.Tensor, gg: torch.Tensor, bb: torch.Tensor) -> float:
    # A noisy estimate can make gg or bb <= 0 (signal swamped by noise): report nan then.
    if gg <= 0 or bb <= 0:
        return float('nan')
    return float((gb / torch.sqrt(gg * bb)).clamp(-1.0, 1.0))


def alignment_stats(
    flat_grad: Callable[[slice, str], torch.Tensor],
    n_rows: int,
    n_envs: int,
    groups: int,
    g: torch.Tensor,
    b: torch.Tensor,
    hinv: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    p: torch.Tensor,
    step: torch.Tensor | None = None,
) -> dict[str, float]:
    """Reward/cost gradient alignment for one update.

    Args:
        flat_grad: ``flat_grad(rows, which)`` returns the flat gradient of the reward (``'r'``,
            ascent direction) or cost (``'c'``) surrogate over the given buffer rows.
        n_rows: Number of rows in the update batch.
        n_envs: Number of environments; the batch is ``n_envs`` contiguous equal blocks.
        groups: Number of independent groups to split the environments into (>= 2).
        g, b: Full-batch reward and cost gradients.
        hinv: Approximate ``H^{-1} v`` (conjugate gradients).
        x, p: ``H^{-1} g`` and ``H^{-1} b`` already computed by the update.
        step: The accepted parameter step (after line search), if any.
    """
    out: dict[str, float] = {
        'Align/Cos': _cos(g, b),
        'Align/FisherCos': float(g.dot(p) / torch.sqrt(g.dot(x) * b.dot(p)).clamp_min(1e-30)),
        # cos between the natural gradients H^-1 g and H^-1 b in plain parameter space
        'Align/NatGradCos': _cos(x, p),
    }
    moved = step is not None and step.norm() > 0
    out['Align/StepCosReward'] = _cos(step, g) if moved else float('nan')
    out['Align/StepCosNegCost'] = _cos(step, -b) if moved else float('nan')
    if groups >= 2 and n_envs % groups == 0 and n_rows % n_envs == 0:
        per = n_rows // groups
        sl = [slice(i * per, (i + 1) * per) for i in range(groups)]
        gs = [flat_grad(s, 'r') for s in sl]
        bs = [flat_grad(s, 'c') for s in sl]
        gb, gg, bb = _cross(gs, bs), _cross(gs, gs), _cross(bs, bs)
        # symmetrize: mean_{i!=j} g_i.b_j already averages both orders over all ordered pairs
        out['Align/CosCorrected'] = _corrected_cos(gb, gg, bb)
        out['Align/GroupCos'] = sum(_cos(gs[i], bs[i]) for i in range(groups)) / groups
        out['Align/SignalFracR'] = float(gg / (sum(v.dot(v) for v in gs) / groups))
        out['Align/SignalFracC'] = float(bb / (sum(v.dot(v) for v in bs) / groups))
        hgs = [hinv(v) for v in gs]
        hbs = [hinv(v) for v in bs]
        fgb = _cross(gs, hbs)
        fgg = _cross(gs, hgs)
        fbb = _cross(bs, hbs)
        out['Align/FisherCosCorrected'] = _corrected_cos(fgb, fgg, fbb)
    return out


ALIGN_KEYS = (
    'Align/Cos', 'Align/FisherCos', 'Align/NatGradCos', 'Align/StepCosReward',
    'Align/StepCosNegCost', 'Align/CosCorrected', 'Align/GroupCos', 'Align/SignalFracR',
    'Align/SignalFracC', 'Align/FisherCosCorrected',
)
