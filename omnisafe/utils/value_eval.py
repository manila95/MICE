"""Utility for evaluating true vs. estimated value functions."""

from __future__ import annotations

import torch
from rich.progress import Progress

from omnisafe.envs.core import make


def estimate_true_value(agent, env_id, num_envs, seed, cfgs, discount, eval_episodes=100, epoch=None):
    """Estimate true V(s) vs. critic estimate by rolling out full episodes.

    Runs `num_envs` parallel environments, collecting `eval_episodes` total
    completed episodes. For each episode: records the critic's estimate at the
    initial state, then computes the actual discounted return via rollout.

    Returns:
        (c_error, true_c, estimate_c, corr_c, r_error, true_r, estimate_r, corr_r)
    """
    env_cfgs = {}
    if hasattr(cfgs, 'env_cfgs') and cfgs.env_cfgs is not None:
        env_cfgs = cfgs.env_cfgs.todict()
    eval_env = make(env_id, num_envs=num_envs, device=cfgs.train_cfgs.device, **env_cfgs)
    device = torch.device(cfgs.train_cfgs.device)

    obs, _ = eval_env.reset()  # (num_envs, obs_dim)
    act, init_est_r, init_est_c, _ = agent.step(obs)  # values: (num_envs,)

    running_r = torch.zeros(num_envs, device=device)
    running_c = torch.zeros(num_envs, device=device)
    step_counts = torch.zeros(num_envs, device=device)

    true_cvalues, true_rvalues = [], []
    estimate_rvalues, estimate_cvalues = [], []
    episodes_done = 0

    with Progress() as progress:
        task = progress.add_task('Evaluating value function...', total=eval_episodes)
        while episodes_done < eval_episodes:
            next_obs, r, c, terminated, truncated, _ = eval_env.step(act)

            discount_factors = discount ** step_counts
            running_r += r.squeeze(-1) * discount_factors
            running_c += c.squeeze(-1) * discount_factors
            step_counts += 1

            done = (terminated.bool() | truncated.bool()).squeeze(-1)  # (num_envs,)

            newly_done = 0
            for i in done.nonzero(as_tuple=False).flatten().tolist():
                if episodes_done < eval_episodes:
                    true_rvalues.append(running_r[i].clone())
                    true_cvalues.append(running_c[i].clone())
                    estimate_rvalues.append(init_est_r[i].clone())
                    estimate_cvalues.append(init_est_c[i].clone())
                    episodes_done += 1
                    newly_done += 1
                running_r[i] = 0.0
                running_c[i] = 0.0
                step_counts[i] = 0.0

            progress.update(task, advance=newly_done)

            obs = next_obs
            # One agent.step call: gets action for next step AND new init estimates for just-reset envs
            act, new_est_r, new_est_c, _ = agent.step(obs)
            if done.any():
                init_est_r = torch.where(done, new_est_r, init_est_r)
                init_est_c = torch.where(done, new_est_c, init_est_c)

    eval_env.close()

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
