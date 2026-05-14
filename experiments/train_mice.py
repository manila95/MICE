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
        default=10,
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
        '--no-intrinsic-in-deltas',
        action='store_true',
        default=False,
        help='if set, intrinsic costs are zeroed out in the deltas_n TD-error computation',
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
    opt = vars(args)
    del opt["seed"]
    del opt["constant_cost"]
    del opt["cost_decay_type"]
    del opt["cost_decay_rate"]
    del opt["cost_decay_step_interval"]
    del opt["cost_decay_factor"]
    del opt["no_intrinsic_in_deltas"]
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

    agent = omnisafe.Agent(
        args.algo,
        args.env_id,
        seed,
        train_terminal_cfgs=vars(args),
        custom_cfgs=custom_cfgs,
    )
    agent.learn()
