import os
import sys

import argparse

import omnisafe
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict


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
        default= 10240000,  
        metavar='STEPS',
        help='total number of steps to train for algorithm',
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
    args, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))

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
    steps_per_epoch = args.steps_per_epoch
    early_eval_freq = args.early_eval_freq
    lidar_bins = args.lidar_bins
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
    del opt["steps_per_epoch"]
    del opt["early_eval_freq"]
    del opt["lidar_bins"]
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
        try:
            update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:cost_limit', str(cost_limit)))
        except:
            update_dict(custom_cfgs, custom_cfgs_to_dict('lagrange_cfgs:cost_limit', str(cost_limit)), allow_new_key=True)
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
    if steps_per_epoch is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:steps_per_epoch', str(steps_per_epoch)))
    if early_eval_freq is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('algo_cfgs:early_eval_freq', str(early_eval_freq)))
    if lidar_bins is not None:
        update_dict(custom_cfgs, custom_cfgs_to_dict('env_cfgs:lidar_num_bins', str(lidar_bins)))

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
    agent.learn()
