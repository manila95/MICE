"""CPO on SafeMetaDrive with value-function evaluation on.

Run inside the mice:metadrive image:
    IMAGE=mice:metadrive docker/run.sh python experiments/run_metadrive_eval.py
"""
import omnisafe


def main() -> None:
    omnisafe.Agent(
        'CPO',
        'SafeMetaDrive',
        0,
        custom_cfgs={
            'train_cfgs': {
                'total_steps': 6000,
                'vector_env_nums': 1,   # MetaDrive is single-env only
                'device': 'cpu',
                'torch_threads': 4,
            },
            'algo_cfgs': {
                'steps_per_epoch': 2000,
                'eval_critic': True,
                'isolated_eval_episode_cap': 1000,   # episodes here have no time limit
                'mc_value_study_probes': 12,
                'mc_value_study_repeats': 3,
                'intermediate_state_study_probes': 12,
            },
            'logger_cfgs': {'use_wandb': False, 'use_tensorboard': False},
        },
    ).learn()


# Required: the eval studies spawn a subprocess, which re-imports this module.
if __name__ == '__main__':
    main()
