"""Scoring a risk-sensitive critic against the Monte-Carlo *tail* rather than the mean.

The value studies compare a critic's prediction against the mean of the realized returns, which
is the right reference for a critic that predicts a mean. A tail-constrained algorithm's cost
critic does not: under :class:`~omnisafe.algorithms.on_policy.worst_case.wcpg.WCPG` it predicts
:math:`\\mathrm{CVaR}_\\alpha` of the cost return, so the usual numbers compare two different
quantities and a good score there would mean nothing.

This recomputes the reference from the same rollouts the value studies already collected --
``raw[stream]['returns']`` holds each probe's ``mc_repeats`` realized returns -- taking their
empirical tail instead of their mean. No extra environment interaction: the data to answer the
question was always there, only the reduction was wrong.

.. warning::
    An empirical tail is a far noisier statistic than an empirical mean. At ``alpha=0.1`` with
    ``mc_repeats=10`` the reference is the single worst of ten rollouts; at ``mc_repeats=5`` it
    is the worst of five. The reported ``TailSamples`` is how many repeats actually enter each
    probe's tail, and a value of 1 or 2 should be read as "this correlation is mostly noise".
    Meaningful tail evaluation wants ``mc_repeats`` an order of magnitude above ``1 / alpha``,
    which costs rollouts in direct proportion.
"""

from __future__ import annotations

import numpy as np
import torch


def empirical_cvar(values, alpha: float = 0.1, upper: bool = True) -> float:
    """Mean of the worst ``alpha`` fraction of ``values``.

    Args:
        values: The samples to take the tail of.
        alpha (float): Tail fraction, in ``(0, 1]``. ``1.0`` recovers the plain mean.
        upper (bool): Take the upper tail (the risk-relevant end for a cost) or the lower one.

    Returns:
        The tail mean, or ``nan`` when there is nothing to take it of.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float('nan')
    if not 0 < alpha <= 1:
        raise ValueError(f'alpha must be in (0, 1], got {alpha}.')
    k = tail_sample_count(arr.size, alpha)
    ordered = np.sort(arr)
    return float(ordered[-k:].mean() if upper else ordered[:k].mean())


def tail_sample_count(n_samples: int, alpha: float) -> int:
    """How many of ``n_samples`` enter the ``alpha`` tail -- at least one."""
    return max(1, int(round(alpha * n_samples)))


def tail_stats(
    raw: dict,
    alpha: float,
    stream: str = 'c',
    prefix: str = '',
    upper: bool = True,
) -> tuple[dict, dict]:
    """Score ``raw[stream]['pred']`` against the per-probe Monte-Carlo tail.

    Mirrors the value study's statistics so the two are read the same way, with ``MeanTrue``
    now meaning the mean *tail* across probes rather than the mean return.

    Args:
        raw: A ``return_raw=True`` dict from either value study, needing
            ``raw[stream]['pred']`` and ``raw[stream]['returns']``.
        alpha (float): Tail fraction, matching the critic's own.
        stream (str): ``'c'`` (the cost side, where this matters) or ``'r'``.
        prefix (str): Prepended to every statistic's name.
        upper (bool): Upper tail for cost, lower for reward.

    Returns:
        ``(stats, raw)``, the latter carrying the per-probe predictions and tails so the
        comparison can be redone or plotted offline.
    """
    per_probe_returns = raw[stream].get('returns')
    if not per_probe_returns:
        return {}, {}
    pred = np.asarray(raw[stream]['pred'], dtype=np.float64)
    tails = np.asarray(
        [empirical_cvar(reps, alpha, upper=upper) for reps in per_probe_returns],
        dtype=np.float64,
    )
    n = min(pred.size, tails.size)
    pred, tails = pred[:n], tails[:n]
    repeats = min((len(reps) for reps in per_probe_returns), default=0)

    pred_t = torch.tensor(pred, dtype=torch.float32)
    true_t = torch.tensor(tails, dtype=torch.float32)
    if pred_t.numel() < 2 or pred_t.std() <= 0 or true_t.std() <= 0:
        correlation = float('nan')
    else:
        correlation = torch.corrcoef(torch.stack([pred_t, true_t]))[0, 1].item()

    stats = {
        f'{prefix}EstimationError_{stream}': float(np.mean(tails - pred)),
        f'{prefix}Correlation_{stream}': correlation,
        f'{prefix}MeanTrue_{stream}': float(np.mean(tails)),
        f'{prefix}MeanPred_{stream}': float(np.mean(pred)),
        f'{prefix}NumProbes': float(n),
        # How many repeats back each probe's tail. 1 means the reference is a single rollout.
        f'{prefix}TailSamples': float(tail_sample_count(repeats, alpha) if repeats else 0),
    }
    out_raw = {
        stream: {
            'pred': pred.tolist(),
            'mc_tail': tails.tolist(),
            'mc_mean': [float(np.mean(reps)) for reps in per_probe_returns][:n],
        },
        'alpha': alpha,
    }
    return stats, out_raw
