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


def save_eval_data(log_dir: str, epoch: int, bundle: dict, subdir: str = 'eval_data') -> str:
    """Pickle ``bundle`` to ``<log_dir>/<subdir>/epoch_{epoch:05d}.pkl``.

    Args:
        log_dir: The run's log directory (e.g. ``self._logger.log_dir``).
        epoch: Current epoch, used for both the filename and included in ``bundle`` for
            self-description (so a lone pickle file, moved elsewhere, still identifies itself).
        bundle: Arbitrary picklable data -- expected shape is a dict with keys like
            ``mc_study: {'stats': ..., 'raw': ...}`` and
            ``intermediate_study: {pos: {'stats': ..., 'raw': ...}, ...}``, but this function
            doesn't inspect or require any particular structure.
        subdir: Which subdirectory under ``log_dir`` to write to -- ``'eval_data'`` (default) for
            the MC-value-study bundle, ``'scatter_data'`` for the raw arrays behind
            ``Logger.log_scatter_image`` calls (see ``Logger.pop_scatter_raw_data``). Kept as one
            function rather than two near-duplicates since the pickling logic is identical either
            way; only the destination and the bundle's shape differ.

    Returns:
        The path written to.
    """
    out_dir = os.path.join(log_dir, subdir)
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


def log_scatter_to_wandb(png_path: str | None, epoch: int) -> None:
    """Push a scatter-grid PNG (see :func:`save_scatter_grid`) to the active wandb run, if any.

    ``save_scatter_grid`` only writes to disk -- unlike the numeric eval metrics (``MCStudy/*``,
    ``IntermediateMC/pos*/*``, ``PooledMC/*``, ...), which reach wandb automatically via
    ``Logger.store``/``dump_tabular``, nothing pushes these images there on its own. This is a
    thin, decoupled wrapper (rather than routing through ``Logger``) so callers don't need to
    reach into ``Logger``'s private ``_use_wandb``/``_maste_proc`` state: ``wandb.run`` is
    ``None`` on any process that never called ``wandb.init()`` (``use_wandb=False``, or a
    non-master process under ``train_cfgs.parallel > 1``), which is exactly the set of cases
    where nothing should be logged, so a no-op naturally follows the same wandb.init gating the
    numeric metrics already use without needing to duplicate it.

    Args:
        png_path: Path returned by ``save_scatter_grid`` (``None`` if it had nothing plottable).
        epoch: Current epoch, used as the wandb step so this lands on the same x-axis position
            as that epoch's numeric eval metrics.
    """
    if png_path is None:
        return
    import wandb  # noqa: PLC0415 -- see save_scatter_grid's matplotlib import for why this is local

    if wandb.run is None:
        return
    wandb.log({'eval_data/scatter': wandb.Image(png_path)}, step=epoch)


def log_eval_data_to_wandb(
    pkl_path: str,
    epoch: int,
    name_prefix: str = 'eval-data',
    artifact_type: str = 'eval_data',
    description: str | None = None,
) -> None:
    """Sync a per-epoch data pickle (see :func:`save_eval_data`) to the active wandb run, so the
    raw arrays are downloadable from the run page without SSH access to whatever machine produced
    them -- previously only the aggregate stats (via ``Logger.store``/``dump_tabular``) and quick-
    look images (:func:`log_scatter_to_wandb`, ``Logger.log_scatter_image``) ever left local disk;
    the raw pickle itself was local-only.

    Uses a per-epoch :class:`wandb.Artifact` (one version per epoch) rather than ``wandb.save`` --
    an Artifact gets its own content-addressed version history and survives independently of the
    run's live file sync, so a pickle from epoch 100 stays fetchable (``wandb.Api().artifact(...)``)
    even long after the run itself has finished or if the plain run-files view gets pruned. The
    per-epoch granularity mirrors ``save_eval_data``'s own filename scheme (``epoch_{epoch:05d}.pkl``)
    precisely so a downloaded artifact identifies itself the same way the local copy does.

    Args:
        pkl_path: Path returned by :func:`save_eval_data`.
        epoch: Current epoch -- used for both the artifact's name suffix and the wandb step, so
            this lands on the same x-axis position as that epoch's numeric eval metrics.
        name_prefix: Artifact name becomes ``f'{name_prefix}-epoch-{epoch:05d}'`` -- override to
            distinguish multiple data streams (e.g. ``'scatter-data'`` for the
            ``Logger.pop_scatter_raw_data`` bundle) sharing this same upload path.
        artifact_type: wandb Artifact ``type``, distinct per data stream for the same reason.
        description: Optional artifact description; a generic one is used if omitted.
    """
    import wandb  # noqa: PLC0415 -- see save_scatter_grid's matplotlib import for why this is local

    if wandb.run is None:
        return
    artifact = wandb.Artifact(
        name=f'{name_prefix}-epoch-{epoch:05d}',
        type=artifact_type,
        description=description or f'Raw per-probe arrays + aggregate stats, epoch {epoch}.',
    )
    artifact.add_file(pkl_path, name=os.path.basename(pkl_path))
    wandb.log_artifact(artifact)
