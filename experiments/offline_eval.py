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

Usage::

    python experiments/offline_eval.py liam-paull/omnisafe/6jvsugv6
    python experiments/offline_eval.py 6jvsugv6 --project omnisafe --entity liam-paull
    python experiments/offline_eval.py <run> --epochs 1,5 --force
    python experiments/offline_eval.py <run> --workers 4
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import re
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import wandb

import omnisafe
from omnisafe.utils.eval_data_dump import log_eval_data_to_wandb
from omnisafe.utils.tools import update_dict
from omnisafe.utils.value_eval import _find_obs_normalizer


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


def main() -> None:
    args = _parse_args()

    api = wandb.Api()
    run_path = args.run if args.run.count('/') == 2 else f'{args.entity}/{args.project}/{args.run}'
    run = api.run(run_path)
    entity, project, run_id = run.entity, run.project, run.id
    print(f'evaluating run: {entity}/{project}/{run_id} ({run.name})')

    files = list(run.files())
    ckpt_epochs = sorted(
        int(m.group(1)) for f in files if (m := ACTOR_SNAPSHOT_RE.match(f.name))
    )
    have_eval_data = {
        int(m.group(1)) for f in files
        if (m := EVAL_DATA_RE.match(f.name)) and f.size > STUB_SIZE_THRESHOLD
    }
    if not ckpt_epochs:
        print('no actor-snapshot-epoch-*.pt files on this run -- nothing to evaluate.')
        return

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

    if not target_epochs:
        print('nothing to do.')
        return
    print(f'will evaluate epochs: {target_epochs}')

    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix='offline_eval_')
    os.makedirs(cache_dir, exist_ok=True)

    cfg = dict(run.config)
    algo, env_id, seed = cfg['algo'], cfg['env_id'], cfg['seed']
    base_custom_cfgs = _build_base_custom_cfgs(cfg, args)

    if args.workers is not None:
        n_workers = max(1, args.workers)
    else:
        peak_slots = _peak_env_slots(base_custom_cfgs)
        n_workers = max(1, (os.cpu_count() or 1) // peak_slots)
    n_workers = min(n_workers, len(target_epochs))
    print(f'running {len(target_epochs)} epoch(s) with {n_workers} concurrent worker(s) '
          f'(peak env-slots/epoch: {_peak_env_slots(base_custom_cfgs)}, '
          f'cpu_count: {os.cpu_count()})')

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

    # Attach to the ORIGINAL run for the rest of the process -- the only wandb session active
    # anywhere in this script, main process included; no worker ever calls wandb.init.
    wandb.init(entity=entity, project=project, id=run_id, resume='must')

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
            print(f'-- epoch {epoch}: pushed to {run_id} --')

    wandb.finish()
    if failures:
        print(f'done, with {len(failures)} failure(s): {[e for e, _ in failures]}')
        sys.exit(1)
    print('done.')


if __name__ == '__main__':
    main()
