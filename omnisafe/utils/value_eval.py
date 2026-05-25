"""Utility for evaluating true vs. estimated value functions."""

from __future__ import annotations

import torch

from omnisafe.envs.core import make


def estimate_true_value(agent, env_id, num_envs, seed, cfgs, discount, eval_episodes=100, epoch=None):
    """Estimate true V(s) vs. critic estimate by rolling out full episodes.

    For each episode: sample an initial state, record the critic's estimate,
    then run the policy to episode end computing the actual discounted return.

    Returns:
        (c_error, true_c, estimate_c, r_error, true_r, estimate_r)
    """
    env_cfgs = {}
    if hasattr(cfgs, 'env_cfgs') and cfgs.env_cfgs is not None:
        env_cfgs = cfgs.env_cfgs.todict()
    eval_env = make(env_id, num_envs=num_envs, device=cfgs.train_cfgs.device, **env_cfgs)

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

    true_cvalues_t = torch.stack(true_cvalues)
    true_rvalues_t = torch.stack(true_rvalues)
    estimate_cvalues_t = torch.stack(estimate_cvalues)
    estimate_rvalues_t = torch.stack(estimate_rvalues)

    c_error = torch.mean(true_cvalues_t - estimate_cvalues_t)
    r_error = torch.mean(true_rvalues_t - estimate_rvalues_t)
    true_c = torch.mean(true_cvalues_t)
    true_r = torch.mean(true_rvalues_t)
    estimate_c = torch.mean(estimate_cvalues_t)
    estimate_r = torch.mean(estimate_rvalues_t)

    corr_c = torch.corrcoef(torch.stack([true_cvalues_t, estimate_cvalues_t]))[0, 1]
    corr_r = torch.corrcoef(torch.stack([true_rvalues_t, estimate_rvalues_t]))[0, 1]

    if cfgs.logger_cfgs.use_wandb:
        import matplotlib.pyplot as plt
        import wandb

        true_c_vals = true_cvalues_t.detach().cpu().numpy()
        est_c_vals = estimate_cvalues_t.detach().cpu().numpy()
        true_r_vals = true_rvalues_t.detach().cpu().numpy()
        est_r_vals = estimate_rvalues_t.detach().cpu().numpy()

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
        wandb.log({
            'scatter/c_and_r_values': wandb.Image(fig),
            'Eval/Correlation_c': corr_c.item(),
            'Eval/Correlation_r': corr_r.item(),
            'Eval/EstimationError_c': c_error.item(),
            'Eval/true_value_c': true_c.item(),
            'Eval/estimate_value_c': estimate_c.item(),
            'Eval/EstimationError_r': r_error.item(),
            'Eval/true_value_r': true_r.item(),
            'Eval/estimate_value_r': estimate_r.item(),
        }, step=epoch)
        plt.close(fig)

    return c_error, true_c, estimate_c, corr_c, r_error, true_r, estimate_r, corr_r
