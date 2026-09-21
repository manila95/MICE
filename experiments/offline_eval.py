"""Offline-evaluate every checkpoint a wandb run pushed, and push each epoch's results back into
that same run.

Built for runs trained under ``algo_cfgs.eval_critic: False`` (so no evaluation ever ran live and
the run's ``eval-data-epoch_*.pkl`` files don't exist yet), but works on any run: it finds every
``actor-snapshot-epoch-{N}.pt`` the run has pushed, skips epochs that already have a matching
``eval-data-epoch_{N:05d}.pkl`` (unless ``--force``), and for the rest, loads the checkpoint into
an agent built from the run's own recorded config (via ``wandb.Api().run(...).config``, so no
algo/env-id/seed need to be passed by hand) and calls
:meth:`~omnisafe.algorithms.on_policy.base.policy_gradient.PolicyGradient._run_eval_studies`
directly -- the exact production path a live ``eval_critic: True`` run would have taken -- with
``algo_cfgs.eval_critic`` forced on regardless of what the run itself used.

Two things are deliberately NOT re-saved or re-pushed, since they already exist on the run:
* The checkpoint itself (``Logger.torch_save`` is no-op'd for the duration of the call, so the
  ``actor-snapshot-epoch-{N}.pt`` push inside ``_run_eval_studies`` -- gated on that file
  existing -- never fires).
* Anything from a *previous* run of this same script (the epoch skip-list above).

Usage::

    python experiments/offline_eval.py liam-paull/omnisafe/6jvsugv6
    python experiments/offline_eval.py 6jvsugv6 --project omnisafe --entity liam-paull
    python experiments/offline_eval.py <run> --epochs 1,5 --force
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile

import torch
import wandb

import omnisafe
from omnisafe.utils.tools import update_dict
from omnisafe.utils.value_eval import _find_obs_normalizer


ACTOR_SNAPSHOT_RE = re.compile(r'^actor-snapshot-epoch-(\d+)\.pt$')
EVAL_DATA_RE = re.compile(r'^eval-data-epoch_(\d+)\.pkl$')

# Config sub-trees that genuinely exist in the on-policy yaml schema (and so pass
# recursive_check_config as custom_cfgs). 'algo'/'env_id'/'seed'/'exp_name'/'exp_increment_cfgs'
# are runtime-computed metadata on the Config object, not overridable defaults -- passing those
# back as custom_cfgs would raise "Invalid key".
CUSTOM_CFGS_SUBTREES = ('algo_cfgs', 'model_cfgs', 'train_cfgs', 'env_cfgs', 'logger_cfgs')


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
        help='re-evaluate and overwrite epochs that already have eval-data pushed.',
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
    # A stub eval-data-epoch_*.pkl from an eval_critic=False run (just {'epoch': N}, no studies)
    # pickles to ~25-40 bytes -- checking existence alone would treat that as "already evaluated"
    # and skip it forever. Any real study output (even a single probe's raw arrays) is orders of
    # magnitude bigger, so a generous size floor distinguishes the two without downloading every
    # file just to check.
    STUB_SIZE_THRESHOLD = 200
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
    custom_cfgs = {k: cfg[k] for k in CUSTOM_CFGS_SUBTREES if k in cfg}
    # 'epochs' is computed at runtime (total_steps // steps_per_epoch) by algo_wrapper.py's
    # _init_config, not part of the yaml defaults schema -- recorded in the run's config anyway
    # since it lives on the same Config object, but invalid to pass back as a custom_cfgs override.
    custom_cfgs.get('train_cfgs', {}).pop('epochs', None)
    # wandb's config storage round-trips through JSON/YAML, which does not distinguish a
    # whole-number float (0.0, 40.0) from an int -- any such key comes back as a bare int, and
    # check_all_configs's isinstance(x, float) asserts (entropy_coef, penalty_coef, max_grad_norm,
    # etc.) then reject it outright. Recast the known float-typed algo_cfgs keys explicitly.
    _FLOAT_ALGO_CFGS = (
        'target_kl', 'entropy_coef', 'max_grad_norm', 'critic_norm_coef', 'gamma', 'cost_gamma',
        'lam', 'lam_c', 'clip', 'penalty_coef',
    )
    for key in _FLOAT_ALGO_CFGS:
        if key in custom_cfgs.get('algo_cfgs', {}) and custom_cfgs['algo_cfgs'][key] is not None:
            custom_cfgs['algo_cfgs'][key] = float(custom_cfgs['algo_cfgs'][key])
    # Force the full study machinery on regardless of what the run itself used -- that is the
    # entire point of this script -- and keep the Agent's own Logger off wandb entirely: this
    # process attaches to the run's wandb session itself (below), and _run_eval_studies's own
    # log_eval_data_to_wandb calls check the global wandb.run, not this flag, so they still push
    # correctly once that session is active.
    update_dict(custom_cfgs, {'algo_cfgs': {'eval_critic': True}})
    update_dict(custom_cfgs, {'logger_cfgs': {'use_wandb': False}})
    update_dict(custom_cfgs, {'logger_cfgs': {'log_dir': cache_dir}})
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

    agent = omnisafe.Agent(algo, env_id, seed, custom_cfgs=custom_cfgs)
    pg = agent.agent
    # Nothing new to checkpoint in this flow -- every epoch's actor-snapshot already exists on
    # the run. No-op for the whole script so _run_eval_studies's internal torch_save()/push never
    # fires (its actor-snapshot push is itself gated on the file torch_save would have written).
    pg._logger.torch_save = lambda: None  # noqa: SLF001

    # Attach to the ORIGINAL run for the rest of the process. Every _run_eval_studies call below
    # checks only the global wandb.run (not this Agent's own use_wandb=False), so its internal
    # log_eval_data_to_wandb push lands directly on this resumed run -- no separate manual push
    # step needed.
    wandb.init(entity=entity, project=project, id=run_id, resume='must')

    for epoch in target_epochs:
        ckpt_name = f'actor-snapshot-epoch-{epoch}.pt'
        run.file(ckpt_name).download(root=cache_dir, replace=True)
        ckpt_path = os.path.join(cache_dir, ckpt_name)

        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        pg._actor_critic.actor.load_state_dict(state['pi'])
        pg._actor_critic.reward_critic.load_state_dict(state['reward_critic'])
        if 'cost_critic' in state and getattr(pg._actor_critic, 'cost_critic', None) is not None:
            pg._actor_critic.cost_critic.load_state_dict(state['cost_critic'])
        if 'obs_normalizer' in state:
            norm = _find_obs_normalizer(pg._env._env)
            if norm is not None:
                norm.load_state_dict(state['obs_normalizer'])

        print(f'-- epoch {epoch}: running eval studies --')
        pg._run_eval_studies(epoch)
        print(f'-- epoch {epoch}: pushed eval-data-epoch_{epoch:05d}.pkl to {run_id} --')

    wandb.finish()
    print('done.')


if __name__ == '__main__':
    sys.exit(main())
