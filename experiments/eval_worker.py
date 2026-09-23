"""Evaluate a training run's checkpoints out of process, while the run is still training.

Eval, not training, is what a run's wall-clock is made of: measured on this box an eval epoch
costs ~241 s against a ~9 s training epoch, and the two are almost perfectly balanced over a
500-epoch run (~4.6 ks each). Running them in the same process serialises them, so the run takes
their *sum*. This worker takes the other half off the critical path -- training writes a
checkpoint at each eval epoch and moves straight on, this process evaluates whatever checkpoints
exist -- so the run takes their *max* instead.

Why a separate process rather than a thread inside the training loop:

* Training never executes eval code, so eval can't corrupt it. There is no shared actor to
  snapshot, no torn read of parameters mid-update, no eval env owned by two threads.
* Backpressure is free. If evaluation falls behind (it will: in the dense early window evals are
  ~46 s apart and take ~241 s), pending work is just files on disk, not an in-memory queue.
* Evaluations are re-runnable. A diagnostic added later can be applied to every past checkpoint
  without retraining -- which for a calibration study is worth more than the speedup.
* No GIL contention with the training loop's own Python-level rollout.

The cost is that eval numbers no longer stream into the training process's ``progress.csv`` or
its wandb row. They go to ``<run_dir>/eval_progress.csv``, keyed by the epoch they belong to
rather than the epoch that happened to be current when they finished, plus the usual
``eval_data/epoch_XXXXX.pkl``.

Usage::

    python experiments/eval_worker.py <run_dir>              # evaluate, then follow the run
    python experiments/eval_worker.py <run_dir> --once       # evaluate what exists, then exit
    python experiments/eval_worker.py <run_dir> --epochs 1,5 # only these

``<run_dir>`` is the directory holding ``config.json`` and ``torch_save/`` (i.e.
``runs/<exp>/seed-000-.../``).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

# Must run before any parallel torch work, and therefore before the omnisafe imports below --
# see the comment in algorithms/algo_wrapper.py, which does the same thing for training. The
# inter-op pool silently defaults to the machine's core count and can only be set once, at which
# point it is fixed for the life of the process. This worker is *designed* to share a box with a
# training run, which is precisely the situation that comment describes: leaving it at the
# default made one eval epoch take 1018 s instead of 155 s at 16 cores (32 inter-op + 32 intra-op
# threads against 22 env subprocesses), and inflated the concurrent training epoch 4.9x.
try:
    torch.set_num_interop_threads(1)
except RuntimeError:  # pragma: no cover - only if parallel work already ran
    pass

from omnisafe.utils.eval_checkpoint import (  # noqa: E402
    apply_obs_normalizer,
    build_eval_envs,
    checkpoint_epochs,
    eval_rng,
    load_agent_from_checkpoint,
    load_run_config,
)
from omnisafe.utils.eval_data_dump import save_eval_data  # noqa: E402
from omnisafe.utils.state_snapshot import collect_on_policy_snapshots  # noqa: E402
from omnisafe.utils.value_eval import (  # noqa: E402
    compute_gradient_alignment,
    estimate_true_value_same_state_mc,
    estimate_value_from_snapshots,
    pool_correlation_stats,
)


def evaluate_epoch(run_dir: str, epoch: int, cfgs, envs: dict) -> dict:
    """Run the value studies for one checkpoint, returning flat ``{stat_key: value}``.

    Deliberately calls the *same* study functions the in-process path calls
    (``estimate_true_value_same_state_mc`` / ``collect_on_policy_snapshots`` +
    ``estimate_value_from_snapshots`` / ``pool_correlation_stats``), with the same arguments
    derived from the same resolved config, inside the same
    :class:`~omnisafe.utils.eval_checkpoint.eval_rng` scope. That shared-call-path property is
    what ``tests/test_eval_checkpoint.py`` leans on: if this drifts from the in-process path, the
    equality assertion there fails rather than the two silently reporting different numbers.

    Args:
        run_dir (str): The run directory.
        epoch (int): Checkpoint epoch to evaluate.
        cfgs: The run's resolved config.
        envs (dict): Output of :func:`~omnisafe.utils.eval_checkpoint.build_eval_envs`.

    Returns:
        Flat dict of statistics, plus ``{'epoch': epoch}``.
    """
    agent, norm_state = load_agent_from_checkpoint(run_dir, epoch, cfgs=cfgs)
    a = cfgs.algo_cfgs
    gamma_c = getattr(a, 'cost_gamma', a.gamma)
    max_eps = envs['max_episode_steps']
    flat: dict = {'epoch': epoch}
    bundle: dict = {'epoch': epoch}

    with eval_rng(cfgs, epoch):
        if getattr(a, 'mc_value_study', False):
            n = int(getattr(a, 'mc_value_study_probes', 17))
            off = int(getattr(a, 'mc_value_study_seed_offset', 100_000))
            apply_obs_normalizer(envs['mc_env'], norm_state)
            stats, raw = estimate_true_value_same_state_mc(
                agent=agent, env=envs['mc_env'], cfgs=cfgs,
                discount_r=a.gamma, discount_c=gamma_c,
                probe_seeds=list(range(off, off + n)),
                mc_repeats=int(getattr(a, 'mc_value_study_repeats', 10)),
                epoch=epoch, max_episode_steps=max_eps, return_raw=True,
                bootstrap_threshold=getattr(a, 'mc_eval_bootstrap_threshold', None),
                tail_mode=getattr(a, 'mc_eval_tail', None),
            )
            flat.update(stats)
            bundle['mc_study'] = {'stats': stats, 'raw': raw}

        if getattr(a, 'intermediate_state_study', False):
            positions = list(getattr(a, 'intermediate_state_study_positions', [100, 300, 500, 700, 900]))
            repeats = int(getattr(a, 'intermediate_state_study_repeats', 10))
            n_probes = int(getattr(a, 'intermediate_state_study_probes', 17))
            apply_obs_normalizer(envs['interm_env'], norm_state)
            # base_seed mirrors the in-process call site exactly (700_000 + epoch * n_probes) so
            # the snapshot layouts for a given epoch are the same ones training would have used.
            collected = collect_on_policy_snapshots(
                agent, envs['interm_env'], positions, base_seed=700_000 + epoch * n_probes,
            )
            bundle['intermediate_study'] = {}
            for pos in positions:
                stats, raw = estimate_value_from_snapshots(
                    agent=agent, env=envs['interm_env'], cfgs=cfgs,
                    discount_r=a.gamma, discount_c=gamma_c,
                    snapshots=collected[pos], horizon=max_eps, mc_repeats=repeats,
                    epoch=epoch, return_raw=True,
                    bootstrap_threshold=getattr(a, 'mc_eval_bootstrap_threshold', None),
                    tail_mode=getattr(a, 'mc_eval_tail', None),
                )
                # Same key shape the in-process path logs, so eval_progress.csv columns line up
                # with progress.csv's rather than needing a translation table.
                flat.update({f'IntermediateMC/pos{pos}/{k}': v for k, v in stats.items()})
                bundle['intermediate_study'][pos] = {'stats': stats, 'raw': raw}

            # Pooled over s0 *plus* every intermediate position -- not the positions alone. The
            # whole point of pooling is "how accurate is the critic across the diversity of
            # states we evaluate on", and s0 is one of those categories; dropping it changes
            # which states the single pooled correlation is computed over. `prefix='PooledMC/'`
            # matches the in-process call exactly. Both of these were wrong here until the
            # bundle-level comparison in tests/test_eval_checkpoint.py caught them.
            pooled_sources = []
            if 'mc_study' in bundle:
                pooled_sources.append(bundle['mc_study']['raw'])
            pooled_sources.extend(d['raw'] for d in bundle['intermediate_study'].values())
            if pooled_sources:
                pooled_stats, pooled_raw = pool_correlation_stats(pooled_sources, prefix='PooledMC/')
                flat.update(pooled_stats)
                for stream in ('r', 'c'):
                    flat[f'PooledMC/GradientAlignment_{stream}'] = compute_gradient_alignment(
                        agent.actor, pooled_raw, stream,
                    )
                bundle['pooled'] = {'stats': pooled_stats, 'raw': pooled_raw}

    save_eval_data(run_dir, epoch, bundle)
    return flat


def append_row(run_dir: str, row: dict) -> None:
    """Append one evaluation to ``eval_progress.csv``, keyed by its own epoch.

    A separate file rather than the training ``progress.csv`` because that one is written a row
    per epoch and flushed as the epoch ends -- by the time an out-of-process evaluation of epoch N
    finishes, that row is long gone. Here the epoch is a *column*, so rows may legitimately arrive
    out of order and still describe the right checkpoint.
    """
    path = os.path.join(run_dir, 'eval_progress.csv')
    exists = os.path.exists(path)
    keys = sorted(row.keys(), key=lambda k: (k != 'epoch', k))
    with open(path, 'a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir')
    p.add_argument('--once', action='store_true', help='evaluate what exists now, then exit')
    p.add_argument('--epochs', default=None, help='comma-separated epochs (default: all found)')
    p.add_argument('--poll', type=float, default=5.0, help='seconds between scans when following')
    args = p.parse_args()

    cfgs = load_run_config(args.run_dir)
    # Match the training process's intra-op cap (train_cfgs.torch_threads). Without it torch
    # opens one thread per core on top of the env subprocesses, which is self-defeating for a
    # workload that is latency-bound rather than compute-bound.
    threads = int(getattr(cfgs.train_cfgs, 'torch_threads', 4) or 4)
    torch.set_num_threads(max(1, threads))
    print(f'[eval_worker] torch threads: intra-op={torch.get_num_threads()} '
          f'inter-op={torch.get_num_interop_threads()}', flush=True)
    which = 'both' if getattr(cfgs.algo_cfgs, 'intermediate_state_study', False) else 'mc'
    envs = build_eval_envs(cfgs, which=which)
    done: set[int] = set()

    # Anything already in eval_progress.csv was evaluated by a previous invocation; skipping it
    # makes the worker restartable mid-run rather than redoing hours of work.
    csv_path = os.path.join(args.run_dir, 'eval_progress.csv')
    if os.path.exists(csv_path):
        with open(csv_path, encoding='utf-8') as f:
            done = {int(float(r['epoch'])) for r in csv.DictReader(f) if r.get('epoch')}
        print(f'[eval_worker] {len(done)} epoch(s) already done, skipping those', flush=True)

    wanted = {int(x) for x in args.epochs.split(',')} if args.epochs else None
    try:
        while True:
            todo = [e for e in checkpoint_epochs(args.run_dir) if e not in done]
            if wanted is not None:
                todo = [e for e in todo if e in wanted]
            for epoch in todo:
                t0 = time.time()
                try:
                    row = evaluate_epoch(args.run_dir, epoch, cfgs, envs)
                except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
                    # One unreadable checkpoint (e.g. still being written) must not kill a worker
                    # that has hours of queued work behind it.
                    print(f'[eval_worker] epoch {epoch} FAILED: {exc}', flush=True)
                    continue
                append_row(args.run_dir, row)
                done.add(epoch)
                print(f'[eval_worker] epoch {epoch} done in {time.time() - t0:.0f}s', flush=True)
            if args.once or (wanted is not None and not set(wanted) - done):
                break
            # Training writes TRAINING_COMPLETE when its loop ends. Re-scan *after* seeing it
            # rather than exiting straight away: a checkpoint can land between the scan above and
            # the sentinel appearing, and dropping it would silently lose that epoch's
            # evaluation. Only an empty rescan with the sentinel present means genuinely done.
            if os.path.exists(os.path.join(args.run_dir, 'TRAINING_COMPLETE')):
                remaining = [e for e in checkpoint_epochs(args.run_dir) if e not in done]
                if wanted is not None:
                    remaining = [e for e in remaining if e in wanted]
                if not remaining:
                    print('[eval_worker] training complete and backlog drained', flush=True)
                    break
                continue
            time.sleep(args.poll)
    finally:
        for key in ('mc_env', 'interm_env'):
            if key in envs:
                envs[key].close()
    print('[eval_worker] EVAL_WORKER_DONE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
