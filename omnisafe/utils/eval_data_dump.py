"""Persist per-eval-epoch value-study data (raw per-probe arrays + aggregate stats) to disk for
later offline analysis, and render a quick-look scatter-plot grid alongside it.

Complements the online logging (``progress.csv``/wandb/tensorboard, which only ever see the
aggregate stats -- ``MCStudy/*``, ``IntermediateMC/pos*/*``): those numbers are enough to watch
trends live, but reconstructing the underlying per-probe (predicted, MC-true) pairs after the fact
-- e.g. to look for outlier probes, refit a different correlation estimator, or build a custom
plot -- requires the raw arrays, which ``estimate_true_value_same_state_mc`` and
``estimate_value_from_snapshots`` only produce with ``return_raw=True`` and never persist
themselves.
"""

from __future__ import annotations

import os
import pickle


def save_eval_data(log_dir: str, epoch: int, bundle: dict) -> str:
    """Pickle ``bundle`` to ``<log_dir>/eval_data/epoch_{epoch:05d}.pkl``.

    Args:
        log_dir: The run's log directory (e.g. ``self._logger.log_dir``).
        epoch: Current epoch, used for both the filename and included in ``bundle`` for
            self-description (so a lone pickle file, moved elsewhere, still identifies itself).
        bundle: Arbitrary picklable data -- expected shape is a dict with keys like
            ``mc_study: {'stats': ..., 'raw': ...}`` and
            ``intermediate_study: {pos: {'stats': ..., 'raw': ...}, ...}``, but this function
            doesn't inspect or require any particular structure.

    Returns:
        The path written to.
    """
    out_dir = os.path.join(log_dir, 'eval_data')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'epoch_{epoch:05d}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(bundle, f)
    return path


def save_scatter_grid(log_dir: str, epoch: int, series: list[tuple[str, dict]]) -> str | None:
    """Render a grid of (predicted vs. MC-true) scatter plots, one row per series, reward and cost
    as the two columns, to ``<log_dir>/eval_data/epoch_{epoch:05d}_scatter.png``.

    Args:
        log_dir: The run's log directory.
        epoch: Current epoch, for the filename.
        series: ``[(label, raw), ...]`` -- ``label`` names the row (e.g. ``'s0'``,
            ``'pos300'``), ``raw`` is one of ``estimate_true_value_same_state_mc`` /
            ``estimate_value_from_snapshots``'s ``return_raw=True`` dicts (has ``'r'``/``'c'``
            keys, each with ``'pred'``/``'mc_mean'`` lists). A series with too few probes for a
            meaningful scatter (< 2) is skipped rather than erroring.

    Returns:
        The path written to, or ``None`` if there was nothing plottable (empty ``series``, or
        every series had < 2 probes).
    """
    plottable = [(label, raw) for label, raw in series if len(raw['r']['pred']) >= 2]
    if not plottable:
        return None

    # Local import: matplotlib is not a hot-path dependency of the rest of this module (or of
    # value_eval.py/state_snapshot.py, which this function's callers sit alongside) -- keep it
    # out of the module-level import graph so nothing pays its (real, if modest) import cost
    # unless a scatter plot is actually being rendered.
    import matplotlib  # noqa: PLC0415

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # noqa: PLC0415

    n_rows = len(plottable)
    fig, axes = plt.subplots(n_rows, 2, figsize=(9, 3.4 * n_rows), squeeze=False)
    for row, (label, raw) in enumerate(plottable):
        for col, stream in enumerate(('r', 'c')):
            ax = axes[row][col]
            pred = raw[stream]['pred']
            true = raw[stream]['mc_mean']
            ax.scatter(true, pred, s=18, alpha=0.65, color='#2a78d6' if stream == 'r' else '#d9541c')
            lo = min(min(true), min(pred))
            hi = max(max(true), max(pred))
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='#8c8c8c', linewidth=1, linestyle='--')
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            stream_name = 'reward' if stream == 'r' else 'cost'
            ax.set_title(f'{label}: {stream_name}', fontsize=10)
            ax.set_xlabel('MC-true value', fontsize=9)
            ax.set_ylabel('critic prediction', fontsize=9)
            ax.tick_params(labelsize=8)
    fig.suptitle(f'Epoch {epoch}: predicted vs. MC-true value', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir = os.path.join(log_dir, 'eval_data')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'epoch_{epoch:05d}_scatter.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
