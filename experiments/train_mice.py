import os
import sys

import argparse

import omnisafe
from omnisafe.algorithms import ALGORITHM2TYPE
from omnisafe.utils.config import get_default_kwargs_yaml
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict


def _typed_flags(parser, argv):
    """Which dests the user actually typed -- argparse cannot tell a default from a given value."""
    typed = set()
    for action in parser._actions:  # pylint: disable=protected-access
        for opt in action.option_strings:
            if any(a == opt or a.startswith(opt + '=') for a in argv):
                typed.add(action.dest)
    return typed


def _env_block(algo, env_id):
    """The per-environment block for this algo/env from its yaml, or {} if there is none."""
    import glob as _glob  # noqa: PLC0415
    import os as _os  # noqa: PLC0415

    import yaml as _yaml  # noqa: PLC0415

    here = _os.path.dirname(_os.path.abspath(__file__))
    for path in _glob.glob(_os.path.join(here, '..', 'omnisafe', 'configs', '*', f'{algo}.yaml')):
        with open(path, encoding='utf-8') as handle:
            return (_yaml.safe_load(handle) or {}).get(env_id) or {}
    return {}


def _check_env_block_applied(cfgs, block, typed):
    """Report any key the per-environment block asked for that did not survive resolution."""
    mismatches = []

    def walk(want, got, path=''):
        for key, value in want.items():
            if key not in got:
                mismatches.append((f'{path}{key}', value, '<missing>'))
            elif isinstance(value, dict):
                walk(value, got[key], f'{path}{key}.')
            elif str(got[key]) != str(value):
                mismatches.append((f'{path}{key}', value, got[key]))

    walk(block, cfgs)
    return [m for m in mismatches if m[0].split('.')[-1] not in typed]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        metavar='ALGO',
        default='MICE',
        help='algorithm to train',
        choices=omnisafe.ALGORITHMS['all'],
    )
    parser.add_argument(
        '--env-id',
        type=str,
        metavar='ENV',
        default='SafetyPointGoal1-v0',
        help='the name of test environment',
    )
    parser.add_argument(
        '--parallel',
        default=1,
        type=int,
        metavar='N',
        help='number of paralleled progress for calculations.',
    )
    parser.add_argument(
        '--total-steps',
        type=int,
        default=None,
        metavar='STEPS',
        help=(
            'total number of steps to train for algorithm; unset leaves it to the config, '
            "including any per-environment override (e.g. the SafeMetaDrive: block)"
        ),
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:2',
        metavar='DEVICES',
        help='device to use for training',
    )
    parser.add_argument(
        '--vector-env-nums',
        type=int,
        default=5,
        metavar='VECTOR-ENV',
        help='number of vector envs to use for training',
    )
    parser.add_argument(
        '--torch-threads',
        type=int,
        default=4,
        metavar='THREADS',
        help='number of threads to use for torch',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=16,
        metavar='THREADS',
        help='number of threads to use for torch',
    )
    parser.add_argument(
        '--constant-cost',
        type=float,
        default=None,
        metavar='COST',
        help='constant intrinsic cost added to every transition (None uses KNN-based intrinsic cost)',
    )
    parser.add_argument(
        '--cost-decay-type',
        type=str,
        default=None,
        choices=['exponential', 'step'],
        help='decay schedule for constant_cost: exponential (C*rate^epoch) or step (C*factor^(epoch//interval))',
    )
    parser.add_argument(
        '--cost-decay-rate',
        type=float,
        default=None,
        metavar='RATE',
        help='per-epoch decay rate for exponential schedule (e.g. 0.985 → ~1%% at epoch 300)',
    )
    parser.add_argument(
        '--cost-decay-step-interval',
        type=int,
        default=None,
        metavar='INTERVAL',
        help='number of epochs between decay steps for step schedule',
    )
    parser.add_argument(
        '--cost-decay-factor',
        type=float,
        default=None,
        metavar='FACTOR',
        help='multiplicative factor applied at each step for step schedule (e.g. 0.4)',
    )
    parser.add_argument(
        '--steps-per-epoch',
        type=int,
        default=None,
        metavar='STEPS',
        help='number of environment steps per epoch (overrides the value in the algo config yaml)',
    )
    parser.add_argument(
        '--early-eval-freq',
        type=int,
        default=None,
        metavar='FREQ',
        help='eval frequency (in epochs) for the first 100 epochs (default: 5)',
    )
    parser.add_argument(
        '--lidar-bins',
        type=int,
        default=None,
        metavar='BINS',
        help='number of lidar angular bins (overrides lidar_conf.num_bins, default: 16)',
    )
    parser.add_argument(
        '--hazards-num',
        type=int,
        default=None,
        metavar='N',
        help=(
            'number of hazards, for the *GoalHazards* envs only '
            '(e.g. --env-id SafetyPointGoalHazards1-v0 --hazards-num 4). Must be >= 1. '
            'Unlike --lidar-bins this travels in the env config, so it reaches async vector '
            'workers and does not force synchronous envs.'
        ),
    )
    parser.add_argument(
        '--use-env-cfgs',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        metavar='BOOL',
        help=(
            'honour the per-environment block in the algorithm yaml (e.g. SafeMetaDrive). '
            'This script hands argparse defaults to omnisafe as train_terminal_cfgs, which are '
            'applied after the yaml, so a flag left at its default would otherwise silently '
            'override whatever that block sets. With this on, any train_cfgs key the block '
            'defines and you did not actually type is left to the block, and the resolved '
            'config is checked against it afterwards. Off reproduces the old behaviour: the '
            "script's own defaults win."
        ),
    )
    parser.add_argument(
        '--no-intrinsic-in-deltas',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,
        metavar='BOOL',
        help='if true, intrinsic costs are zeroed out in the deltas_n TD-error computation',
    )
    parser.add_argument(
        '--cost-limit',
        type=float,
        default=None,
        metavar='COST_LIMIT',
        help='constraint cost limit (overrides the value in the algo config yaml)',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=None,
        metavar='GAMMA',
        help='discount factor for the reward critic (defaults to algo_cfgs.gamma in yaml)',
    )
    parser.add_argument(
        '--cost-gamma',
        type=float,
        default=None,
        metavar='COST_GAMMA',
        help='discount factor for the cost critic (defaults to algo_cfgs.gamma if not set)',
    )
    parser.add_argument(
        '--target-kl',
        type=float,
        default=None,
        metavar='TARGET_KL',
        help='KL divergence trust-region size (overrides the value in the algo config yaml)',
    )
    parser.add_argument(
        '--lagrangian-multiplier-init',
        type=float,
        default=None,
        metavar='LAMBDA_INIT',
        help='initial value of the Lagrange multiplier (for Lagrangian algorithms)',
    )
    parser.add_argument(
        '--pid-kp',
        type=float,
        default=None,
        metavar='KP',
        help='proportional gain of the PID Lagrangian controller',
    )
    parser.add_argument(
        '--reinforce-reward',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=None,
        metavar='BOOL',
        help='if false, use value-function bootstrap for reward (TRPOPIDReinforce only)',
    )
    parser.add_argument(
        '--lam',
        type=float,
        default=None,
        metavar='LAM',
        help='GAE lambda for reward advantage (on-policy algorithms)',
    )
    parser.add_argument(
        '--lam-c',
        type=float,
        default=None,
        metavar='LAM_C',
        help='GAE lambda for cost advantage (on-policy algorithms)',
    )
    parser.add_argument(
        '--n-val-episodes',
        type=int,
        default=None,
        metavar='N',
        help='number of episodes per epoch held out for critic calibration validation (0 = no split)',
    )
    parser.add_argument(
        '--adv-estimation-method',
        type=str,
        default=None,
        metavar='METHOD',
        choices=['td_zero', 'td_zero_gae', 'gae', 'gae-rtg', 'vtrace', 'plain', 'reinforce'],
        help='Advantage estimation method for reward (gae, gae-rtg, vtrace, plain, reinforce)',
    )
    parser.add_argument(
        '--cost-adv-estimation-method',
        type=str,
        default=None,
        metavar='METHOD',
        choices=['td_zero', 'td_zero_gae', 'gae', 'gae-rtg', 'vtrace', 'plain', 'reinforce'],
        help='Advantage estimation method for cost; defaults to --adv-estimation-method if unset',
    )
    args, unparsed_args = parser.parse_known_args()

    if str(args.device).startswith('cpu'):
        # torch.cuda.device_count() caches its result ONLY when it finds >0 GPUs, so on a CPU
        # node every call re-probes -- and the probe runs ctypes.util.find_library, which forks
        # gcc. torch.optim's _cuda_graph_capture_health_check calls it once per optimizer step,
        # i.e. ~628 times per epoch here. Profiled on rorqual: 101 s of a 105 s _update(), with
        # 86 s inside _findLib_gcc and 121 s of self time reading the subprocess pipe, against
        # ~1.2 s of actual tensor math. It is node-dependent (harmless where cvmfs is warm,
        # catastrophic where it is not), which is why it hid for so long.
        #
        # Pinning the count to 0 for CPU runs makes the probe happen never instead of per step.
        # Guarded on --device cpu, so a GPU run is untouched.
        import torch  # noqa: PLC0415

        torch.cuda.device_count = lambda: 0

    # Handle both '--key value' and '--key=value' (wandb sweep format).
    # Keys may use '.' or ':' as hierarchy separator, and '-' or '_' between words.
    _cfg_overrides = {}
    _i = 0
    while _i < len(unparsed_args):
        tok = unparsed_args[_i]
        if tok.startswith('--'):
            if '=' in tok:
                k, v = tok[2:].split('=', 1)
                _i += 1
            elif _i + 1 < len(unparsed_args) and not unparsed_args[_i + 1].startswith('--'):
                k, v = tok[2:], unparsed_args[_i + 1]
                _i += 2
            else:
                _i += 1
                continue
            _cfg_overrides[k.replace('.', ':').replace('-', '_')] = v
        else:
            _i += 1
    unparsed_args = _cfg_overrides

    seed = args.seed
    constant_cost = args.constant_cost
    cost_decay_type = args.cost_decay_type
    cost_decay_rate = args.cost_decay_rate
    cost_decay_step_interval = args.cost_decay_step_interval
    cost_decay_factor = args.cost_decay_factor
    no_intrinsic_in_deltas = args.no_intrinsic_in_deltas
    cost_limit = args.cost_limit
    gamma = args.gamma
    cost_gamma = args.cost_gamma
    target_kl = args.target_kl
    lagrangian_multiplier_init = args.lagrangian_multiplier_init
    pid_kp = args.pid_kp
    reinforce_reward = args.reinforce_reward
    lam = args.lam
    lam_c = args.lam_c
    adv_estimation_method = args.adv_estimation_method
    cost_adv_estimation_method = args.cost_adv_estimation_method
    n_val_episodes = args.n_val_episodes
    steps_per_epoch = args.steps_per_epoch
    early_eval_freq = args.early_eval_freq
    lidar_bins = args.lidar_bins
    hazards_num = args.hazards_num
    opt = vars(args)
    del opt["seed"]
    del opt["constant_cost"]
    del opt["cost_decay_type"]
    del opt["cost_decay_rate"]
    del opt["cost_decay_step_interval"]
    del opt["cost_decay_factor"]
    del opt["no_intrinsic_in_deltas"]
    del opt["cost_limit"]
    del opt["gamma"]
    del opt["cost_gamma"]
    del opt["target_kl"]
    del opt["lagrangian_multiplier_init"]
    del opt["pid_kp"]
    del opt["reinforce_reward"]
    del opt["lam"]
    del opt["lam_c"]
    del opt["adv_estimation_method"]
    del opt["cost_adv_estimation_method"]
    del opt["n_val_episodes"]
    del opt["steps_per_epoch"]
    del opt["early_eval_freq"]
    del opt["lidar_bins"]
    del opt["hazards_num"]
    use_env_cfgs = opt.pop("use_env_cfgs")
    # vars(args) is handed over as train_terminal_cfgs, which is applied *after* the yaml --
    # so leaving an argparse default in here silently overrides the per-environment block
    # (SafeMetaDrive's total_steps, say). Drop it when the flag was not actually passed.
    if opt.get("total_steps") is None:
        del opt["total_steps"]
    import sys as _sys

    typed = _typed_flags(parser, _sys.argv[1:])
    # Held onto here: algo_wrapper pops 'env_id'/'algo' out of train_terminal_cfgs, which is
    # this same vars(args) dict, so args loses them once the Agent is built.
    env_id_name = args.env_id
    env_block = _env_block(args.algo, env_id_name) if use_env_cfgs else {}
    if env_block:
        # Leave every train_cfgs key the block sets to the block, unless it was actually typed.
        deferred = [k for k in (env_block.get("train_cfgs") or {}) if k not in typed and k in opt]
        for key in deferred:
            del opt[key]
        if deferred:
            print(f'[use-env-cfgs] deferring to the {env_id_name} block for: {sorted(deferred)}')
    custom_cfgs = {}
    for k, v in unparsed_args.items():
        update_dict(custom_cfgs, custom_cfgs_to_dict(k, v))
    if constant_cost is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:constant_cost', str(constant_cost)))
    if cost_decay_type is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:cost_decay_type', cost_decay_type))
    if cost_decay_rate is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:cost_decay_rate', str(cost_decay_rate)))
    if cost_decay_step_interval is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:cost_decay_step_interval', str(cost_decay_step_interval)))
    if cost_decay_factor is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:cost_decay_factor', str(cost_decay_factor)))
    if no_intrinsic_in_deltas:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:no_intrinsic_in_deltas', 'True'))
    if target_kl is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:target_kl', str(target_kl)))
    if cost_limit is not None:
        # cost_limit lives under algo_cfgs for trust-region algorithms (e.g. CPO) but under
        # lagrange_cfgs for Lagrangian ones (e.g. TRPOPID) -- ask the algorithm's own default
        # yaml which one it actually has, rather than guessing. The previous try/except here
        # never worked: update_dict() merges dicts unconditionally and never raises, so the
        # 'except' branch (which also passed a nonexistent allow_new_key kwarg) was unreachable;
        # the actual KeyError only surfaced later, at Agent construction, for any algorithm whose
        # cost_limit isn't under algo_cfgs.
        algo_type = ALGORITHM2TYPE.get(args.algo, '')
        default_cfgs = get_default_kwargs_yaml(args.algo, args.env_id, algo_type)
        cost_limit_block = (
            'algo_cfgs' if 'cost_limit' in default_cfgs.algo_cfgs.todict() else 'lagrange_cfgs'
        )
        update_dict(custom_cfgs, custom_cfgs_to_dict(f'{cost_limit_block}:cost_limit', str(cost_limit)))
    if gamma is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:gamma', str(gamma)))
    if cost_gamma is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:cost_gamma', str(cost_gamma)))
    elif gamma is not None:
        # if only --gamma was set, mirror it to cost_gamma unless explicitly overridden
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:cost_gamma', str(gamma)))
    if lagrangian_multiplier_init is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('lagrange_cfgs:lagrangian_multiplier_init', str(lagrangian_multiplier_init)))
    if pid_kp is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('lagrange_cfgs:pid_kp', str(pid_kp)))
    if reinforce_reward is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:reinforce_reward', str(reinforce_reward)))
    if lam is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:lam', str(lam)))
    if lam_c is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:lam_c', str(lam_c)))
    elif lam is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:lam_c', str(lam)))
    if adv_estimation_method is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:adv_estimation_method', adv_estimation_method))
    if cost_adv_estimation_method is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:cost_adv_estimation_method', cost_adv_estimation_method))
    elif adv_estimation_method is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:cost_adv_estimation_method', adv_estimation_method))
    if n_val_episodes is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:n_val_episodes', str(n_val_episodes)))
    if steps_per_epoch is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:steps_per_epoch', str(steps_per_epoch)))
    if early_eval_freq is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:early_eval_freq', str(early_eval_freq)))
    if lidar_bins is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('env_cfgs:lidar_num_bins', str(lidar_bins)))
    if hazards_num is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('env_cfgs:hazards_num', str(hazards_num)))

    agent = omnisafe.Agent(
        args.algo,
        args.env_id,
        seed,
        train_terminal_cfgs=vars(args),
        custom_cfgs=custom_cfgs,
    )
    import json
    print('\n=== Resolved config ===')
    print(json.dumps(agent.cfgs.todict(), indent=2, default=str))
    print('=======================\n')
    if env_block:
        unapplied = _check_env_block_applied(agent.cfgs.todict(), env_block, typed)
        if unapplied:
            print(f'[use-env-cfgs] WARNING: the {env_id_name} block asked for values that did '
                  'not survive resolution, and they were not overridden on the command line:')
            for key, want, got in unapplied:
                print(f'    {key}: block={want!r} resolved={got!r}')
        else:
            print(f'[use-env-cfgs] every {env_id_name} block value is applied.')
    agent.learn()
