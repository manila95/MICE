r"""Direction diagnostics for the policy gradient.

The formal justification of TRPO/CPO runs through per-state value accuracy, but the
optimizer only ever sees a gradient.  Two advantage constructions that disagree
pointwise can still yield the same update direction, and that is the quantity a claim
about "the impact of the cost policy gradient" has to be about.

These helpers build the score-function gradient

.. math::
    g \;=\; \frac{1}{N} \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A_t

for several choices of :math:`A_t` over **one fixed batch** of ``(s, a)`` pairs, so the
resulting gradients differ only in the advantage and never in which samples were drawn,
and then report the angles between them.  Holding the samples fixed is the whole point:
an unpaired comparison confounds "the critic points the update elsewhere" with "a
smaller batch points anywhere".

Two metrics are reported for every pair.

* **Euclidean cosine** -- the quantity Ilyas et al. (2020), *A Closer Look at Deep
  Policy Gradients* (arXiv:1811.02553), report.

* **Fisher cosine.**  TRPO/CPO move inside :math:`\{\Delta\theta : \tfrac12
  \Delta\theta^\top H \Delta\theta \le \delta\}` and step along :math:`H^{-1}g`, so
  :math:`H` is the metric on parameter increments and the angle between two *steps* is

  .. math::
      \cos_H(H^{-1}g_1, H^{-1}g_2)
      = \frac{g_1^\top H^{-1} g_2}
             {\sqrt{g_1^\top H^{-1} g_1 \; g_2^\top H^{-1} g_2}},

  i.e. the :math:`H^{-1}`-weighted cosine of the raw gradients.  Only solves against
  :math:`H` are needed, so this reuses the algorithm's own Fisher-vector product and
  conjugate-gradient solver rather than ever forming :math:`H`.

  Caveat to report alongside the number: ``_fvp`` returns :math:`Hv + \lambda v` with
  :math:`\lambda =` ``cg_damping``, so the metric is really :math:`(H + \lambda I)^{-1}`.
  With ~2e4 samples and a policy of ~1e4 parameters :math:`H` is badly rank-deficient and
  the damping does real work, so the damping belongs in any caption quoting these numbers.

Conjugate gradients solves only approximately, which makes :math:`g_1^\top H^{-1} g_2`
and :math:`g_2^\top H^{-1} g_1` differ although they are equal in exact arithmetic.
:func:`fisher_cosines` averages the two and returns the worst relative gap it saw as a
free convergence diagnostic -- if it is not small, raise ``cg_iters``.  Its floor is not
zero: :func:`~omnisafe.utils.math.conjugate_gradients` carries ``eps=1e-6`` in its
denominators, so a fully converged solve still reports an asymmetry of order 1e-5, and
the cosines themselves are only good to about that.  That is five orders of magnitude
below the differences these diagnostics are meant to resolve; treat anything above ~1e-3
as a real convergence failure.
"""

from __future__ import annotations

import torch

from omnisafe.utils.math import conjugate_gradients


def score_gradients(
    actor,
    obs: torch.Tensor,
    act: torch.Tensor,
    advantages: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    r"""Flat score-function gradients for several advantages over one fixed batch.

    One forward pass is shared by every gradient, so the policy distribution (and hence
    :math:`\nabla \log \pi`) is bit-for-bit identical across them; only the weighting
    differs.  Advantages are detached -- this is the REINFORCE estimator, where only
    :math:`\log \pi` is differentiated.

    Args:
        actor: Policy network with this codebase's actor interface: ``__call__(obs)``
            caches the distribution that ``log_prob(act)`` then reads.
        obs: Observations, shape ``(N, obs_dim)``.
        act: Actions actually taken, shape ``(N, act_dim)``.
        advantages: Name -> advantage tensor of shape ``(N,)``.

    Returns:
        Name -> flat gradient vector, one per entry of ``advantages``.
    """
    params = [p for p in actor.parameters() if p.requires_grad]
    actor(obs)
    logp = actor.log_prob(act)

    names = list(advantages)
    out: dict[str, torch.Tensor] = {}
    for i, name in enumerate(names):
        loss = -(logp * advantages[name].detach()).mean()
        grads = torch.autograd.grad(
            loss, params, retain_graph=i < len(names) - 1, allow_unused=True,
        )
        out[name] = torch.cat([
            g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=obs.device)
            for g, p in zip(grads, params)
        ])
    return out


def euclidean_cosines(grads: dict[str, torch.Tensor]) -> dict[str, float]:
    """Pairwise Euclidean cosines, keyed ``'<a>|<b>'`` over the pairs in input order."""
    names = list(grads)
    out: dict[str, float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            na, nb = grads[a].norm(), grads[b].norm()
            out[f'{a}|{b}'] = (
                float('nan') if na == 0 or nb == 0
                else (grads[a].dot(grads[b]) / (na * nb)).item()
            )
    return out


def fisher_cosines(
    grads: dict[str, torch.Tensor],
    fvp,
    cg_iters: int = 10,
) -> tuple[dict[str, float], float]:
    r"""Pairwise cosines in the :math:`H^{-1}` metric, plus a CG-convergence diagnostic.

    Args:
        grads: Name -> flat gradient vector, all in the same parameter layout.
        fvp: Callable ``v -> (H + damping I) v``; the algorithm's own ``_fvp``.  Its
            ``_fvp_obs`` must already be set for the policy these gradients came from.
        cg_iters: Conjugate-gradient iterations per solve, one solve per gradient.

    Returns:
        ``(cosines, asymmetry)`` where ``cosines`` is keyed ``'<a>|<b>'`` and
        ``asymmetry`` is the largest relative gap between :math:`g_a^\top H^{-1} g_b`
        and :math:`g_b^\top H^{-1} g_a` over all pairs -- exactly zero in exact
        arithmetic, so it measures how far CG is from converged.
    """
    names = list(grads)
    sol = {k: conjugate_gradients(fvp, grads[k], cg_iters) for k in names}
    quad = {k: grads[k].dot(sol[k]).item() for k in names}

    out: dict[str, float] = {}
    asymmetry = 0.0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ab, ba = grads[a].dot(sol[b]).item(), grads[b].dot(sol[a]).item()
            scale = abs(ab) + abs(ba)
            if scale > 0:
                asymmetry = max(asymmetry, abs(ab - ba) / scale)
            denom = quad[a] * quad[b]
            out[f'{a}|{b}'] = (
                0.5 * (ab + ba) / denom ** 0.5 if denom > 0 else float('nan')
            )
    return out, asymmetry
