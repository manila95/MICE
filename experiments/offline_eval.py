"""Offline-evaluate every checkpoint a wandb run pushed, and push each epoch's results back into
that same run -- evaluating independent epochs in parallel worker processes.

Built for runs trained under ``algo_cfgs.eval_critic: False`` (so no evaluation ever ran live and
the run's ``eval-data-epoch_*.pkl`` files don't exist yet), but works on any run: it finds every
``actor-snapshot-epoch-{N}.pt`` the run has pushed, skips epochs that already have a matching
``eval-data-epoch_{N:05d}.pkl`` with real content (unless ``--force``), and for the rest, loads
the checkpoint into an agent built from the run's own recorded config (via
``wandb.Api().run(...).config``, so no algo/env-id/seed need to be passed by hand) and calls
:meth:`~omnisafe.algorithms.on_policy.base.policy_gradient.PolicyGradient._run_eval_studies`
directly -- the exact production path a live ``eval_critic: True`` run would have taken -- with
``algo_cfgs.eval_critic`` forced on regardless of what the run itself used.

**Correctness note, not just a performance one: every epoch gets its own freshly-constructed
agent.** ``BaseAlgo.__init__`` calls ``seed_all(seed)`` unconditionally, which is what makes one
epoch's stochastic MC-rollout results independent of whatever other epoch happened to be evaluated
before it in the same process. An earlier version of this script built one agent and reused it
across a sequential epoch loop -- epoch 5's ``MCStudy/MeanTrue_c`` measurably differed depending on
whether epoch 1 had been evaluated first in that same process (0.66 vs. 1.16 in one measured case),
because both epochs' stochastic rollouts drew from one continuously-advancing RNG stream. A fresh
agent per epoch removes that dependency entirely -- which is also exactly what makes running
epochs concurrently in separate worker processes safe: nothing about parallelizing changes any
epoch's result, since each was already independent of every other by construction.

Two things are deliberately NOT re-saved or re-pushed, since they already exist on the run:
* The checkpoint itself (``Logger.torch_save`` is no-op'd inside each worker, so the
  ``actor-snapshot-epoch-{N}.pt`` push inside ``_run_eval_studies`` -- gated on that file
  existing -- never fires).
* Anything from a *previous* run of this same script (the content-based epoch skip-list).

**Parallelism**: each epoch's evaluation already runs its own internally-vectorized environments
(``algo_cfgs.mc_value_study_vector_envs`` and ``algo_cfgs.intermediate_state_study_probes``, each a
subprocess per env slot) -- so stacking one worker process per *epoch* on top multiplies that
subprocess count again. Workers never touch wandb (built with ``use_wandb: False``, and no worker
ever calls ``wandb.init`` itself, so ``_run_eval_studies``'s internal push sees ``wandb.run is
None`` and no-ops); the main process holds the single resumed wandb session and pushes each
worker's resulting local pickle serially once it's ready -- avoiding concurrent writers to one run
entirely, not just avoiding a race we didn't want to think about.

**Watch mode** (``--watch``) runs the discover-evaluate-push cycle in a loop instead of once, so
evaluation happens *concurrently* with a still-running training process instead of only after the
fact. Training's own checkpoint save is already unconditional on the eval-epoch schedule (see
``PolicyGradient._init_log``'s ``what_to_save`` comment) regardless of ``algo_cfgs.eval_critic``,
so a training run started with ``eval_critic: False`` never blocks on evaluation at all -- it just
checkpoints and moves on -- while this process, running independently as its own OS process (no
shared Python state whatsoever, a strictly stronger isolation than the in-process ``isolated_rng``
guard training's own live evaluation needs), watches for each new checkpoint and evaluates it as it
appears. The loop stops once the run reaches a terminal wandb state (finished/crashed/failed) and
one more pass finds nothing new to do, so a checkpoint written right before training exits still
gets picked up.

To avoid this competing with training's own rollout collection for cores on the same machine, the
default worker-count heuristic reserves ``train_cfgs.vector_env_nums`` (read from the run's own
recorded config, so it matches whatever training is actually running) off the top of
``cpu_count()`` before sizing eval workers, on top of the existing per-epoch peak-env-slots
capping. Pass ``--workers`` to override this entirely (e.g. the watcher is on a different machine).

Usage::

    python experiments/offline_eval.py liam-paull/omnisafe/6jvsugv6
    python experiments/offline_eval.py 6jvsugv6 --project omnisafe --entity liam-paull
    python experiments/offline_eval.py <run> --epochs 1,5 --force
    python experiments/offline_eval.py <run> --workers 4
    python experiments/offline_eval.py <run> --watch --poll-interval 60
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import wandb

import omnisafe
from omnisafe.utils.eval_data_dump import log_eval_data_to_wandb
from omnisafe.utils.tools import update_dict
from omnisafe.utils.value_eval import _find_obs_normalizer


#: wandb run states that mean training will never produce another checkpoint.
TERMINAL_RUN_STATES = frozenset({'finished', 'crashed', 'failed', 'killed'})


ACTOR_SNAPSHOT_RE = re.compile(r'^actor-snapshot-epoch-(\d+)\.pt$')
EVAL_DATA_RE = re.compile(r'^eval-data-epoch_(\d+)\.pkl$')

# Config sub-trees that genuinely exist in the on-policy yaml schema (and so pass
# recursive_check_config as custom_cfgs). 'algo'/'env_id'/'seed'/'exp_name'/'exp_increment_cfgs'
# are runtime-computed metadata on the Config object, not overridable defaults -- passing those
# back as custom_cfgs would raise "Invalid key".
CUSTOM_CFGS_SUBTREES = ('algo_cfgs', 'model_cfgs', 'train_cfgs', 'env_cfgs', 'logger_cfgs')

# wandb's config storage round-trips through JSON/YAML, which does not distinguish a whole-number
# float (0.0, 40.0) from an int -- any such key comes back as a bare int, and check_all_configs's
# isinstance(x, float) asserts (entropy_coef, penalty_coef, max_grad_norm, etc.) then reject it
# outright. Recast the known float-typed algo_cfgs keys explicitly.
_FLOAT_ALGO_CFGS = (
    'target_kl', 'entropy_coef', 'max_grad_norm', 'critic_norm_coef', 'gamma', 'cost_gamma',
    'lam', 'lam_c', 'clip', 'penalty_coef',
)
# A stub eval-data-epoch_*.pkl from an eval_critic=False run (just {'epoch': N}, no studies)
# pickles to ~25-40 bytes -- checking existence alone would treat that as "already evaluated" and
# skip it forever. Any real study output (even a single probe's raw arrays) is orders of magnitude
# bigger, so a generous size floor distinguishes the two without downloading every file to check.
STUB_SIZE_THRESHOLD = 200


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'run', help="wandb run to evaluate: 'entity/project/run_id', or bare run_id with "
        '--project/--entity supplying the rest.',
    )
    parser.add_argument('--project', default='omnisafe')
    parser.add_argument('--entity', default=None)
    parser.add_argument(
        '--epochs', default=None,
        help='comma-separated epochs to (re-)evaluate; default: every checkpoint the run has.',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='re-evaluate and overwrite epochs that already have real eval-data pushed.',
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='epochs to evaluate concurrently; default: cpu_count() // (peak env-slots used by '
        'one epoch), so stacking epoch-parallelism on top of each epoch\'s own internal env '
        'vectorization does not oversubscribe the machine. Pass 1 to force fully sequential.',
    )
    parser.add_argument(
        '--mc-value-study-probes', type=int, default=None,
        help='override the run-recorded value (smaller = faster, for a quick check).',
    )
    parser.add_argument('--mc-value-study-repeats', type=int, default=None)
    parser.add_argument('--intermediate-state-study-probes', type=int, default=None)
    parser.add_argument('--intermediate-state-study-repeats', type=int, default=None)
    parser.add_argument(
        '--device', default=None, help='override the run-recorded device (e.g. force "cpu").',
    )
    parser.add_argument(
        '--cache-dir', default=None,
        help='local dir to download checkpoints/write scratch logs into; default: a temp dir.',
    )
    parser.add_argument(
        '--watch', action='store_true',
        help='run continuously, evaluating each new checkpoint as it appears, instead of once '
        'against whatever the run has right now. Stops once the run finishes and one more pass '
        'finds nothing new.',
    )
    parser.add_argument(
        '--poll-interval', type=float, default=60.0,
        help='seconds between checks for new checkpoints in --watch mode.',
    )
    return parser.parse_args()


def _build_base_custom_cfgs(cfg: dict, args: argparse.Namespace) -> dict:
    """Turn a run's downloaded ``wandb.Api()`` config into valid ``omnisafe.Agent`` custom_cfgs.

    Shared by the epoch-count/worker-count heuristic (needs to know the eval studies' env-slot
    settings before dispatching anything) and every worker (needs the same base config, only
    differing per epoch in ``logger_cfgs.log_dir``).
    """
    custom_cfgs = {k: cfg[k] for k in CUSTOM_CFGS_SUBTREES if k in cfg}
    # 'epochs' is computed at runtime (total_steps // steps_per_epoch) by algo_wrapper.py's
    # _init_config, not part of the yaml defaults schema -- recorded in the run's config anyway
    # since it lives on the same Config object, but invalid to pass back as a custom_cfgs override.
    custom_cfgs.get('train_cfgs', {}).pop('epochs', None)
    for key in _FLOAT_ALGO_CFGS:
        if key in custom_cfgs.get('algo_cfgs', {}) and custom_cfgs['algo_cfgs'][key] is not None:
            custom_cfgs['algo_cfgs'][key] = float(custom_cfgs['algo_cfgs'][key])
    # Force the full study machinery on regardless of what the run itself used -- that is the
    # entire point of this script -- and keep every worker's own Logger off wandb entirely: no
    # worker ever calls wandb.init, so _run_eval_studies's internal log_eval_data_to_wandb call
    # (which checks only the global wandb.run, not this flag) correctly no-ops there. The main
    # process pushes each worker's result itself, once, after the fact.
    update_dict(custom_cfgs, {'algo_cfgs': {'eval_critic': True}})
    update_dict(custom_cfgs, {'logger_cfgs': {'use_wandb': False}})
    if args.device:
        update_dict(custom_cfgs, {'train_cfgs': {'device': args.device}})
    for flag, value in [
        ('mc_value_study_probes', args.mc_value_study_probes),
        ('mc_value_study_repeats', args.mc_value_study_repeats),
        ('intermediate_state_study_probes', args.intermediate_state_study_probes),
        ('intermediate_state_study_repeats', args.intermediate_state_study_repeats),
    ]:
        if value is not None:
            update_dict(custom_cfgs, {'algo_cfgs': {flag: value}})
    return custom_cfgs


def _peak_env_slots(custom_cfgs: dict) -> int:
    """Most subprocess env slots any single epoch's evaluation uses at once.

    The MC value study and the intermediate-state study run sequentially within one epoch (not
    concurrently with each other), so the peak is the larger of the two, not their sum.
    """
    algo_cfgs = custom_cfgs.get('algo_cfgs', {})
    mc_envs = int(algo_cfgs.get('mc_value_study_vector_envs', 1) or 1)
    intermediate_envs = int(algo_cfgs.get('intermediate_state_study_probes', 20) or 20)
    return max(mc_envs, intermediate_envs, 1)


def _evaluate_one_epoch(
    epoch: int, algo: str, env_id: str, seed: int, custom_cfgs: dict, ckpt_path: str,
) -> tuple[int, str | None, str | None]:
    """Worker entry point. Builds a FRESH agent (see the module docstring for why that is required
    for correctness, not merely convenient for parallelism), loads the checkpoint into it, runs
    the real eval-studies path, and returns the local eval-data pickle path -- never touching
    wandb itself.

    Returns:
        ``(epoch, pkl_path, error)`` -- exactly one of ``pkl_path``/``error`` is ``None``.
    """
    try:
        agent = omnisafe.Agent(algo, env_id, seed, custom_cfgs=custom_cfgs)
        pg = agent.agent
        # Nothing new to checkpoint here -- this epoch's actor-snapshot already exists on the
        # run. No-op so _run_eval_studies's internal torch_save()/push never fires (that push is
        # itself gated on the file torch_save would have written existing).
        pg._logger.torch_save = lambda: None  # noqa: SLF001

        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        pg._actor_critic.actor.load_state_dict(state['pi'])
        pg._actor_critic.reward_critic.load_state_dict(state['reward_critic'])
        if 'cost_critic' in state and getattr(pg._actor_critic, 'cost_critic', None) is not None:
            pg._actor_critic.cost_critic.load_state_dict(state['cost_critic'])
        if 'obs_normalizer' in state:
            norm = _find_obs_normalizer(pg._env._env)
            if norm is not None:
                norm.load_state_dict(state['obs_normalizer'])

        pg._run_eval_studies(epoch)
        pkl_path = os.path.join(pg._logger.log_dir, 'eval_data', f'epoch_{epoch:05d}.pkl')
        if not os.path.exists(pkl_path):
            return epoch, None, f'expected eval-data pickle missing: {pkl_path}'
        return epoch, pkl_path, None
    except Exception:  # noqa: BLE001 -- report the failure to the parent, don't lose it in a dead worker
        return epoch, None, traceback.format_exc()


def _list_run_files(run) -> list:
    """``list(run.files())``, tolerating wandb's own file-listing API against a run so fresh it
    hasn't uploaded anything yet.

    Reproduced directly in ``--watch`` mode: connecting within a few seconds of ``wandb.init()``
    on the training side, before even its first file (``config.yaml``/``wandb-metadata.json``) has
    landed, gets back a GraphQL response with a null ``files.pageInfo`` -- wandb's own paginator
    (``wandb/apis/public/files.py``) doesn't guard that case and raises
    ``TypeError: 'NoneType' object is not subscriptable`` instead of treating it as an empty page.
    Since "no files yet" is exactly the state a watcher polling a just-started run needs to handle
    as "nothing to evaluate yet, not an error," this is the one place that translates the crash.
    """
    try:
        return list(run.files())
    except TypeError:
        return []


def _discover_target_epochs(run, args: argparse.Namespace) -> tuple[list[int], bool]:
    """Which epochs to (re-)evaluate right now, given the run's *current* file listing.

    Returns ``(target_epochs, had_any_checkpoint)`` -- the second value lets watch mode
    distinguish "nothing new yet, keep polling" from "this run will never have anything to
    evaluate," though both currently print the same way; kept separate in case that changes.
    """
    files = _list_run_files(run)
    ckpt_epochs = sorted(
        int(m.group(1)) for f in files if (m := ACTOR_SNAPSHOT_RE.match(f.name))
    )
    have_eval_data = {
        int(m.group(1)) for f in files
        if (m := EVAL_DATA_RE.match(f.name)) and f.size > STUB_SIZE_THRESHOLD
    }

    if args.epochs:
        requested = [int(e) for e in args.epochs.split(',')]
        missing_ckpt = [e for e in requested if e not in ckpt_epochs]
        if missing_ckpt:
            print(f'no checkpoint on this run for epochs {missing_ckpt}, skipping those.')
        target_epochs = [e for e in requested if e in ckpt_epochs]
    else:
        target_epochs = ckpt_epochs

    if not args.force:
        already_done = [e for e in target_epochs if e in have_eval_data]
        if already_done:
            print(f'eval-data already exists for epochs {already_done}; skipping '
                  f'(pass --force to redo).')
        target_epochs = [e for e in target_epochs if e not in have_eval_data]

    return target_epochs, bool(ckpt_epochs)


def _compute_n_workers(
    base_custom_cfgs: dict, cfg: dict, args: argparse.Namespace, n_targets: int,
) -> int:
    """Worker-count heuristic, optionally reserving cores for a still-running training process.

    In ``--watch`` mode, ``train_cfgs.vector_env_nums`` (read from the run's own recorded config,
    so it matches what training is actually running -- not guessed) is reserved off the top of
    ``cpu_count()`` before sizing eval workers, so evaluating concurrently with training doesn't
    compete with its own rollout collection for cores. Batch mode (no ``--watch``) skips this --
    nothing else is running concurrently with it, so the full core count is fair game.
    """
    if args.workers is not None:
        return max(1, min(args.workers, n_targets))
    peak_slots = _peak_env_slots(base_custom_cfgs)
    available = os.cpu_count() or 1
    if args.watch:
        reserved = int(cfg.get('train_cfgs', {}).get('vector_env_nums', 0) or 0)
        available = max(1, available - reserved)
    return max(1, min(available // peak_slots, n_targets))


def _evaluate_and_push(
    run, target_epochs: list[int], algo: str, env_id: str, seed: int,
    base_custom_cfgs: dict, cache_dir: str, n_workers: int,
) -> list[tuple[int, str]]:
    """One discover-download-evaluate-push pass over ``target_epochs``. Assumes a wandb session
    is already active (``wandb.init`` called by the caller) -- every worker's own
    ``log_eval_data_to_wandb`` call pushes into that session, never its own.

    Returns the ``(epoch, error)`` pairs for any epoch whose evaluation raised, if any.
    """
    # Download every checkpoint up front, serially, in the main process -- simple, avoids any
    # concern about concurrent wandb API reads, and is a small fraction of total time next to the
    # eval studies themselves.
    payloads = []
    for epoch in target_epochs:
        ckpt_name = f'actor-snapshot-epoch-{epoch}.pt'
        run.file(ckpt_name).download(root=cache_dir, replace=True)
        ckpt_path = os.path.join(cache_dir, ckpt_name)
        epoch_custom_cfgs = copy.deepcopy(base_custom_cfgs)
        update_dict(
            epoch_custom_cfgs,
            {'logger_cfgs': {'log_dir': os.path.join(cache_dir, f'epoch_{epoch}')}},
        )
        payloads.append((epoch, algo, env_id, seed, epoch_custom_cfgs, ckpt_path))

    failures: list[tuple[int, str]] = []
    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_evaluate_one_epoch, *payload): payload[0] for payload in payloads}
        for future in as_completed(futures):
            epoch = futures[future]
            epoch, pkl_path, error = future.result()
            if error is not None:
                print(f'-- epoch {epoch}: FAILED --\n{error}')
                failures.append((epoch, error))
                continue
            print(f'-- epoch {epoch}: eval studies done, pushing eval-data-epoch_{epoch:05d}.pkl --')
            log_eval_data_to_wandb(pkl_path)
            print(f'-- epoch {epoch}: pushed to {run.id} --')
    return failures


def main() -> None:
    args = _parse_args()

    api = wandb.Api()
    run_path = args.run if args.run.count('/') == 2 else f'{args.entity}/{args.project}/{args.run}'
    run = api.run(run_path)
    entity, project, run_id = run.entity, run.project, run.id
    print(f'evaluating run: {entity}/{project}/{run_id} ({run.name})'
          + (' [watch mode]' if args.watch else ''))

    # Same race as _list_run_files, one field over: connecting within the first moment of
    # wandb.init() can catch the run before its config has synced from the training process,
    # so cfg comes back {} rather than raising -- retry briefly rather than KeyError on 'algo'.
    cfg = dict(run.config)
    for _ in range(10):
        if 'algo' in cfg:
            break
        time.sleep(1.0)
        run = api.run(run_path)
        cfg = dict(run.config)
    algo, env_id, seed = cfg['algo'], cfg['env_id'], cfg['seed']
    base_custom_cfgs = _build_base_custom_cfgs(cfg, args)
    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix='offline_eval_')
    os.makedirs(cache_dir, exist_ok=True)

    # Deliberately NOT held open across the whole --watch loop: a resumed wandb session is an
    # attached client that sends its own heartbeats, so if held continuously it makes run.state
    # read 'running' and run.heartbeatAt stay fresh *because of the watcher's own presence* --
    # indistinguishable from training itself still being alive. Measured directly: a held-open
    # session kept a finished training run reading 'running' 10+ minutes after its process had
    # already exited, since the watcher's own connection was the only thing still heartbeating.
    # Attaching only for the few seconds it takes to push means the run's state between polls
    # reflects training's own connection (or lack of one), not ours.
    all_failures: list[tuple[int, str]] = []
    while True:
        target_epochs, _ = _discover_target_epochs(run, args)
        if target_epochs:
            n_workers = _compute_n_workers(base_custom_cfgs, cfg, args, len(target_epochs))
            print(f'will evaluate epochs: {target_epochs} '
                  f'({n_workers} concurrent worker(s), '
                  f'peak env-slots/epoch: {_peak_env_slots(base_custom_cfgs)}, '
                  f'cpu_count: {os.cpu_count()})')
            wandb.init(entity=entity, project=project, id=run_id, resume='must')
            try:
                all_failures.extend(_evaluate_and_push(
                    run, target_epochs, algo, env_id, seed, base_custom_cfgs, cache_dir, n_workers,
                ))
            finally:
                wandb.finish()
        elif not args.watch:
            print('nothing to do.')

        if not args.watch:
            break

        # Re-fetch (not just re-read the cached object) so .state and .files() reflect what's
        # actually happened on the run since the last pass -- and, per the note above, is only
        # meaningful because no wandb session of ours is attached at this point.
        run = api.run(run_path)
        if run.state in TERMINAL_RUN_STATES and not target_epochs:
            print(f'run is {run.state} and nothing new to evaluate -- stopping watch.')
            break
        # Always sleep the same interval here, even when the run just went terminal with work
        # still pending this pass: the file we just pushed needs a moment to be visible to a
        # fresh run.files() call, and one more pass follows either way (the loop-top discovery +
        # the terminal-and-empty check above) -- looping immediately here would risk hammering
        # the wandb API in a tight loop if that visibility lags.
        print(f'sleeping {args.poll_interval}s before next check...')
        time.sleep(args.poll_interval)

    if all_failures:
        print(f'done, with {len(all_failures)} failure(s): {[e for e, _ in all_failures]}')
        sys.exit(1)
    print('done.')


if __name__ == '__main__':
    main()
