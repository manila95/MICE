"""Utility for evaluating true vs. estimated value functions."""

from __future__ import annotations

import torch

from omnisafe.envs.core import make


def estimate_true_value(agent, env_id, num_envs, seed, cfgs, discount, eval_episodes=100):
    """Estimate true V(s) vs. critic estimate by rolling out full episodes.

    For each episode: sample an initial state, record the critic's estimate,
    then run the policy to episode end computing the actual discounted return.

    Returns:
        (c_error, true_c, estimate_c, r_error, true_r, estimate_r)
    """
    eval_env = make(env_id, num_envs=num_envs, device=cfgs.train_cfgs.device)

    true_cvalues, true_rvalues = [], []
    estimate_rvalues, estimate_cvalues = [], []

    for _ in range(eval_episodes):
        obs0, _ = eval_env.reset()
        _, estimate_rvalue, estimate_cvalue, _ = agent.step(obs0)

        obs = obs0
        true_cvalue = 0.0
        true_rvalue = 0.0
        step = 0
        while True:
            act, _, _, _ = agent.step(obs)
            next_obs, r, c, terminated, truncated, _ = eval_env.step(act)
            true_cvalue += c * (discount ** step)
            true_rvalue += r * (discount ** step)
            step += 1
            obs = next_obs
            if terminated or truncated:
                break

        true_cvalues.append(true_cvalue)
        true_rvalues.append(true_rvalue)
        estimate_cvalues.append(estimate_cvalue)
        estimate_rvalues.append(estimate_rvalue)

    c_error = torch.mean(torch.stack(true_cvalues) - torch.stack(estimate_cvalues))
    r_error = torch.mean(torch.stack(true_rvalues) - torch.stack(estimate_rvalues))
    true_c = torch.mean(torch.stack(true_cvalues))
    true_r = torch.mean(torch.stack(true_rvalues))
    estimate_c = torch.mean(torch.stack(estimate_cvalues))
    estimate_r = torch.mean(torch.stack(estimate_rvalues))

    if cfgs.logger_cfgs.use_wandb:
        import matplotlib.pyplot as plt
        import wandb

        true_c_vals = torch.stack(true_cvalues).detach().cpu().numpy()
        est_c_vals = torch.stack(estimate_cvalues).detach().cpu().numpy()
        true_r_vals = torch.stack(true_rvalues).detach().cpu().numpy()
        est_r_vals = torch.stack(estimate_rvalues).detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, tx, ex, label, color in [
            (axes[0], true_c_vals, est_c_vals, 'C', 'steelblue'),
            (axes[1], true_r_vals, est_r_vals, 'R', 'darkorange'),
        ]:
            ax.scatter(tx, ex, alpha=0.5, s=10, color=color)
            lo, hi = min(tx.min(), ex.min()), max(tx.max(), ex.max())
            ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, label='ideal')
            ax.set_xlabel(f'True {label}')
            ax.set_ylabel(f'Estimated {label}')
            ax.set_title(f'{label}-Values: True vs Estimated')
            ax.legend()

        plt.tight_layout()
        wandb.log({'scatter/c_and_r_values': wandb.Image(fig)})
        plt.close(fig)

    return c_error, true_c, estimate_c, r_error, true_r, estimate_r
