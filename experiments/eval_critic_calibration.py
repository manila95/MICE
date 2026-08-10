"""Critic-calibration / sample-complexity experiment.

Question this answers: given a policy frozen after k epochs of on-policy training, how many
freshly-collected on-policy samples M does a critic *trained from scratch* need to recover a
well-calibrated V^pi (and, if ``algo_cfgs.use_cost``, V^pi_c)?

The experiment runs in two phases:

  Phase 1 (pretrain): train --algo on --env-id for --k-epochs epochs, exactly like a normal run.
      This is skipped if --checkpoint-path is given.

  Phase 2 (critic calibration sweep): build a fresh agent, load the phase-1 actor (+ obs
      normalizer) into it and freeze it, then for each requested M (and each requested critic
      target):

        1. reinitialize the reward/cost critics from scratch (fresh weights, fresh optimizers);
        2. collect exactly M on-policy transitions under the frozen policy, letting every episode
           finish naturally so the Monte-Carlo return is unbiased for every collected transition;
        3. train the critics on that fixed batch of M transitions;
        4. evaluate calibration by rolling out fresh episodes and comparing predicted vs. true
           (Monte-Carlo) value, at s0 and at every visited state (reusing
           ``omnisafe.utils.value_eval.estimate_true_value`` -- the same routine normal training
           uses for its periodic value-calibration checks).

Phase 1's checkpoint is a normal omnisafe ``torch_save`` checkpoint. Its path is printed at the
end of phase 1 (and can be passed back in via --checkpoint-path) so that sweeping many M values
against the same frozen policy never has to repeat the k-epoch training.

Example:

    python experiments/eval_critic_calibration.py --k-epochs 200 --M 2000 5000 20000 50000 \\
        --env-id SafetyPointGoal1-v0 --device cuda:0

    # reuse the same frozen policy for a second sweep, e.g. with the MC target:
    python experiments/eval_critic_calibration.py --k-epochs 200 --M 2000 5000 20000 50000 \\
        --critic-target mc --checkpoint-path runs/CPO-{SafetyPointGoal1-v0}/seed-.../torch_save/epoch-200.pt

Any additional ``--key value`` (or ``--key=value``) is forwarded as a config override exactly like
``experiments/train_mice.py`` does, e.g. ``--model_cfgs.critic.lr 0.0005``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import torch

import omnisafe
from omnisafe.utils.critic_calibration import (
    assert_plain_critic,
    collect_fixed_policy_rollout,
    reinit_critics,
    train_critic,
)
from omnisafe.utils.tools import custom_cfgs_to_dict, get_device, update_dict
from omnisafe.utils.value_eval import estimate_true_value


try:
    import wandb
except ImportError:  # pragma: no cover - wandb is an optional dependency
    wandb = None


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse this script's own flags; anything left over is a generic config override."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--algo', type=str, default='CPO', choices=omnisafe.ALGORITHMS['all'])
    parser.add_argument('--env-id', type=str, default='SafetyPointGoal1-v0')
    parser.add_argument(
        '--k-epochs',
        type=int,
        required=True,
        help='epochs to pretrain the policy for before freezing it (phase 1)',
    )
    parser.add_argument(
        '--M',
        type=int,
        nargs='+',
        required=True,
        metavar='M',
        help='one or more on-policy sample budgets to evaluate against the frozen policy '
        '(collected + trained + evaluated independently, reusing the same frozen policy)',
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=None,
        help='skip phase-1 training and load a previously-saved epoch-*.pt checkpoint '
        '(actor + obs_normalizer) instead',
    )
    parser.add_argument('--seed', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--vector-env-nums', type=int, default=5)
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument(
        '--steps-per-epoch',
        type=int,
        default=20000,
        help='must match algo_cfgs.steps_per_epoch (also forces it via config override); '
        'used to translate --k-epochs into a total_steps budget for phase-1 training',
    )
    parser.add_argument(
        '--critic-target',
        type=str,
        default='bootstrap',
        choices=['bootstrap', 'mc', 'both'],
        help="'bootstrap' regresses onto the algorithm's normal target_value_r/c (e.g. GAE, same "
        "as PolicyGradient._update); 'mc' regresses onto the unbiased Monte-Carlo return "
        "discounted_ret/discounted_cost_ret; 'both' runs each M through both, reusing the same "
        'collected samples for a fair comparison',
    )
    parser.add_argument(
        '--stopping',
        type=str,
        default='fixed',
        choices=['fixed', 'early_stop'],
        help="'fixed' runs --critic-train-epochs passes over all of M; 'early_stop' holds out "
        '--critic-val-frac of the episodes and stops on held-out-loss plateau',
    )
    parser.add_argument('--critic-train-epochs', type=int, default=200)
    parser.add_argument(
        '--critic-batch-size',
        type=int,
        default=None,
        help='defaults to algo_cfgs.batch_size',
    )
    parser.add_argument('--critic-val-frac', type=float, default=0.1)
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--early-stop-min-delta', type=float, default=1e-4)
    parser.add_argument('--early-stop-max-epochs', type=int, default=2000)
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=100,
        help='fresh episodes rolled out under the frozen policy to score final critic calibration',
    )
    parser.add_argument(
        '--max-episode-len',
        type=int,
        default=1000,
        help='safety bound used to size the collection buffer and to force-finish '
        'pathologically long episodes',
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='where to dump the compact per-(M, mode) summary; defaults next to the phase-2 run logs',
    )
    return parser.parse_known_args()


def build_custom_cfgs(args: argparse.Namespace, unparsed_args: list[str]) -> dict[str, Any]:
    """Turn leftover ``--key value`` tokens into a nested config-override dict.

    Mirrors ``experiments/train_mice.py``'s override parsing exactly, so any config path
    (``.``/``:`` separated, ``-``/``_`` interchangeable) can be overridden the same way here.
    """
    custom_cfgs: dict[str, Any] = {}
    i = 0
    while i < len(unparsed_args):
        tok = unparsed_args[i]
        if tok.startswith('--'):
            if '=' in tok:
                key, val = tok[2:].split('=', 1)
                i += 1
            elif i + 1 < len(unparsed_args) and not unparsed_args[i + 1].startswith('--'):
                key, val = tok[2:], unparsed_args[i + 1]
                i += 2
            else:
                i += 1
                continue
            update_dict(custom_cfgs, custom_cfgs_to_dict(key.replace('.', ':').replace('-', '_'), val))
        else:
            i += 1
    # Canonical source of truth for steps_per_epoch is --steps-per-epoch: force it so phase 1's
    # total_steps -> epochs division always lands on exactly --k-epochs.
    update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:steps_per_epoch', str(args.steps_per_epoch)))
    return custom_cfgs


def run_phase1_pretrain(args: argparse.Namespace, custom_cfgs: dict[str, Any]) -> str:
    """Train ``--algo`` for ``--k-epochs`` epochs and return the saved checkpoint's path."""
    total_steps = args.k_epochs * args.steps_per_epoch
    print(
        f'\n=== Phase 1: training {args.algo} on {args.env_id} for {args.k_epochs} epochs '
        f'({total_steps} steps) before freezing the policy ===\n',
    )
    pretrain = omnisafe.Agent(
        args.algo,
        args.env_id,
        args.seed,
        train_terminal_cfgs={
            'total_steps': total_steps,
            'device': args.device,
            'vector_env_nums': args.vector_env_nums,
            'parallel': args.parallel,
            'torch_threads': args.torch_threads,
        },
        custom_cfgs=custom_cfgs,
    )
    assert_plain_critic(pretrain.cfgs.model_cfgs)
    pretrain.learn()

    final_epoch = pretrain.agent.logger.current_epoch
    log_dir = pretrain.agent.logger.log_dir
    checkpoint_path = os.path.join(log_dir, 'torch_save', f'epoch-{final_epoch}.pt')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f'Expected phase-1 checkpoint at {checkpoint_path} but it is missing -- '
            f'check logger_cfgs.save_model_freq.',
        )
    print(
        f'\nPhase 1 complete. Checkpoint saved to:\n  {checkpoint_path}\n'
        f'Reuse it later with --checkpoint-path {checkpoint_path} to skip retraining.\n',
    )
    del pretrain
    return checkpoint_path


def build_phase2_agent(args: argparse.Namespace, custom_cfgs: dict[str, Any]) -> omnisafe.Agent:
    """Construct a fresh agent (env + model + logger) for the critic-calibration sweep.

    ``.learn()`` is never called on this one -- it exists purely to host the frozen policy, its
    (soon to be repeatedly reinitialized) critics, and the logger/wandb run for phase 2.
    """
    eval_wrapper = omnisafe.Agent(
        args.algo,
        args.env_id,
        args.seed,
        train_terminal_cfgs={
            'total_steps': args.steps_per_epoch,  # epochs=1; irrelevant since .learn() never runs
            'device': args.device,
            'vector_env_nums': args.vector_env_nums,
            'parallel': args.parallel,
            'torch_threads': args.torch_threads,
        },
        custom_cfgs=custom_cfgs,
    )
    assert_plain_critic(eval_wrapper.cfgs.model_cfgs)
    return eval_wrapper


def load_checkpoint(algo_instance: Any, checkpoint_path: str, device: torch.device) -> None:
    """Load a phase-1 actor (+ obs normalizer, if any) into ``algo_instance`` in place."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    algo_instance._actor_critic.actor.load_state_dict(ckpt['pi'])  # pylint: disable=protected-access
    if 'obs_normalizer' in ckpt:
        saved = algo_instance._env.save()  # pylint: disable=protected-access
        assert 'obs_normalizer' in saved, (
            'checkpoint carries an obs_normalizer but the current config has '
            'algo_cfgs.obs_normalize=False; the two configurations must match.'
        )
        saved['obs_normalizer'].load_state_dict(ckpt['obs_normalizer'])
    elif algo_instance._cfgs.algo_cfgs.obs_normalize:  # pylint: disable=protected-access
        print(
            'WARNING: checkpoint has no obs_normalizer but algo_cfgs.obs_normalize=True; the '
            'frozen policy will see differently-normalized observations than it was trained on.',
        )


def run_sweep(  # pylint: disable=too-many-locals
    args: argparse.Namespace,
    algo_instance: Any,
) -> list[dict[str, Any]]:
    """Run the (M x critic_target) sweep against the frozen policy and return summary records."""
    device = get_device(args.device)
    cfgs = algo_instance._cfgs  # pylint: disable=protected-access
    modes = ['bootstrap', 'mc'] if args.critic_target == 'both' else [args.critic_target]
    m_values = sorted(set(args.M))

    actor_critic = algo_instance._actor_critic  # pylint: disable=protected-access
    adapter = algo_instance._env  # pylint: disable=protected-access
    obs_space, act_space = adapter.observation_space, adapter.action_space
    gamma = cfgs.algo_cfgs.gamma
    cost_gamma = getattr(cfgs.algo_cfgs, 'cost_gamma', None) or gamma
    use_wandb = bool(cfgs.logger_cfgs.use_wandb) and wandb is not None

    records: list[dict[str, Any]] = []
    sweep_idx = 0
    for m_target in m_values:
        for mode in modes:
            print(f'\n--- M={m_target}  target={mode} ---')
            reinit_critics(actor_critic, obs_space, act_space, cfgs.model_cfgs, device)

            t0 = time.time()
            data, episode_slices, n_collected = collect_fixed_policy_rollout(
                adapter, actor_critic, cfgs, m_target, args.max_episode_len, device,
            )
            collect_time = time.time() - t0
            print(
                f'  collected {n_collected} transitions over {len(episode_slices)} episodes '
                f'({collect_time:.1f}s)',
            )

            t0 = time.time()
            train_stats = train_critic(
                actor_critic,
                cfgs,
                data,
                episode_slices,
                target_mode=mode,
                stopping=args.stopping,
                batch_size=args.critic_batch_size or cfgs.algo_cfgs.batch_size,
                fixed_epochs=args.critic_train_epochs,
                val_frac=args.critic_val_frac,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
                max_epochs=args.early_stop_max_epochs,
            )
            train_time = time.time() - t0
            print(
                f"  trained {train_stats['epochs_run']} epochs ({train_time:.1f}s); "
                f"final train loss r={train_stats['final_train_loss_r']:.4g} "
                f"c={train_stats['final_train_loss_c']:.4g}",
            )

            (
                s0_c_error, _s0_true_c_m, _s0_est_c_m, s0_corr_c,
                s0_r_error, _s0_true_r_m, _s0_est_r_m, s0_corr_r,
                all_c_error, _all_true_c_m, _all_est_c_m, all_corr_c,
                all_r_error, _all_true_r_m, _all_est_r_m, all_corr_r,
            ) = estimate_true_value(
                agent=actor_critic,
                env=adapter._env,  # pylint: disable=protected-access
                cfgs=cfgs,
                discount_r=gamma,
                discount_c=cost_gamma,
                eval_episodes=args.eval_episodes,
                epoch=sweep_idx,
            )

            record = {
                'k_epochs': args.k_epochs,
                'M': m_target,
                'n_collected': n_collected,
                'n_episodes_collected': len(episode_slices),
                'target_mode': mode,
                'stopping': args.stopping,
                'env_id': args.env_id,
                'algo': args.algo,
                'seed': args.seed,
                'collect_time_s': collect_time,
                'train_time_s': train_time,
                **{f'critic_train/{k}': v for k, v in train_stats.items()},
                'eval/s0_corr_r': s0_corr_r.item(),
                'eval/s0_corr_c': s0_corr_c.item(),
                'eval/s0_error_r': s0_r_error.item(),
                'eval/s0_error_c': s0_c_error.item(),
                'eval/all_corr_r': all_corr_r.item(),
                'eval/all_corr_c': all_corr_c.item(),
                'eval/all_error_r': all_r_error.item(),
                'eval/all_error_c': all_c_error.item(),
            }
            records.append(record)
            print(
                f"  eval: s0 corr_r={record['eval/s0_corr_r']:.3f} "
                f"corr_c={record['eval/s0_corr_c']:.3f} | "
                f"all corr_r={record['eval/all_corr_r']:.3f} "
                f"corr_c={record['eval/all_corr_c']:.3f}",
            )

            if use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        'CriticCalib/M': m_target,
                        'CriticCalib/n_collected': n_collected,
                        'CriticCalib/n_episodes': len(episode_slices),
                        'CriticCalib/epochs_run': train_stats['epochs_run'],
                        'CriticCalib/final_train_loss_r': train_stats['final_train_loss_r'],
                        'CriticCalib/final_train_loss_c': train_stats['final_train_loss_c'],
                        'CriticCalib/final_val_loss_r': train_stats['final_val_loss_r'],
                        'CriticCalib/final_val_loss_c': train_stats['final_val_loss_c'],
                    },
                    step=sweep_idx,
                )

            sweep_idx += 1

    return records


def main() -> None:
    args, unparsed_args = parse_args()
    assert args.parallel == 1, (
        'eval_critic_calibration only supports --parallel 1 -- the collection and critic-'
        'training loops do not average gradients/samples across MPI workers.'
    )
    for m_target in args.M:
        assert m_target > 0, f'--M values must be positive, got {m_target}'
    if args.stopping == 'early_stop':
        assert 0 < args.critic_val_frac < 1, '--critic-val-frac must be in (0, 1) for early_stop'

    custom_cfgs = build_custom_cfgs(args, unparsed_args)

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = run_phase1_pretrain(args, custom_cfgs)
    else:
        print(f'\n=== Skipping phase-1 training; will load checkpoint from {checkpoint_path} ===\n')

    print(
        f'\n=== Phase 2: building a fresh {args.algo} agent, loading the frozen policy, and '
        f'sweeping critic calibration over M={sorted(set(args.M))} '
        f'(critic_target={args.critic_target}) ===\n',
    )
    eval_wrapper = build_phase2_agent(args, custom_cfgs)
    algo_instance = eval_wrapper.agent
    load_checkpoint(algo_instance, checkpoint_path, get_device(args.device))

    records = run_sweep(args, algo_instance)

    output_json = args.output_json or os.path.join(
        algo_instance.logger.log_dir, 'critic_calibration_summary.json',
    )
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    print(f'\nSaved summary for {len(records)} (M, mode) point(s) to:\n  {output_json}\n')

    algo_instance._env.close()  # pylint: disable=protected-access
    if bool(algo_instance._cfgs.logger_cfgs.use_wandb) and wandb is not None and wandb.run is not None:  # pylint: disable=protected-access
        wandb.finish()


if __name__ == '__main__':
    main()
