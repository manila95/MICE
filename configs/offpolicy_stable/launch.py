"""Reproduce the stable SACLag runs on SafetyPointGoal1-v0.

    python launch.py <cost_limit> <seed> [device]

Only `cost_limit` and the step budget are set; every other hyperparameter comes from
OmniSafe's SafetyPointGoal1-v0 env block and must not be overridden. See README.md.
"""
import sys

import omnisafe

cost_limit = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
device = sys.argv[3] if len(sys.argv) > 3 else 'cuda:0'

omnisafe.Agent('SACLag', 'SafetyPointGoal1-v0', seed, custom_cfgs={
    'train_cfgs': {'total_steps': 2_000_000, 'vector_env_nums': 1,
                   'device': device, 'torch_threads': 2},
    'lagrange_cfgs': {'cost_limit': cost_limit},
    'logger_cfgs': {'use_wandb': False, 'use_tensorboard': True},
}).learn()
