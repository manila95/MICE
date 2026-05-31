"""Utility for evaluating true vs. estimated value functions."""

from __future__ import annotations

import torch
from rich.progress import Progress

from omnisafe.envs.core import make


def estimate_true_value(agent, env_id, num_envs, seed, cfgs, discount_r, discount_c, eval_episodes=100, epoch=None):
    """Estimate true V(s) vs. critic estimate by rolling out full episodes.

    Runs `num_envs` parallel environments, collecting `eval_episodes` total
    completed episodes.

    Two evaluation regimes:
    - Eval_s0: initial state of each episode — V(s_0) vs. G_0.
    - Eval_all: every visited state — V(s_t) vs. G_t (MC return from step t).
    """
    env_cfgs = {}
    if hasattr(cfgs, 'env_cfgs') and cfgs.env_cfgs is not None:
        env_cfgs = cfgs.env_cfgs.todict()
    eval_env = make(env_id, num_envs=num_envs, device=cfgs.train_cfgs.device, **env_cfgs)
    device = torch.device(cfgs.train_cfgs.device)

    obs, _ = eval_env.reset()
    act, cur_est_r, cur_est_c, _ = agent.step(obs)

    # Per-env episode history: each entry is (est_r, est_c, reward, cost) at step t
    ep_history = [[] for _ in range(num_envs)]

    # s0 collected data
    s0_true_r, s0_true_c = [], []
    s0_est_r,  s0_est_c  = [], []
    # all-states collected data
    all_true_r, all_true_c = [], []
    all_est_r,  all_est_c  = [], []

    episodes_done = 0

    with Progress() as progress:
        task = progress.add_task('Evaluating value function...', total=eval_episodes)
        while episodes_done < eval_episodes:
            next_obs, r, c, terminated, truncated, _ = eval_env.step(act)

            r_sq = r.squeeze(-1)
            c_sq = c.squeeze(-1)

            # Record V(s_t), V_c(s_t), r_t, c_t BEFORE moving to next state
            for i in range(num_envs):
                ep_history[i].append((
                    cur_est_r[i].item(),
                    cur_est_c[i].item(),
                    r_sq[i].item(),
                    c_sq[i].item(),
                ))

            done = (terminated.bool() | truncated.bool()).squeeze(-1)

            newly_done = 0
            for i in done.nonzero(as_tuple=False).flatten().tolist():
                if episodes_done < eval_episodes:
                    hist = ep_history[i]
                    T = len(hist)

                    # Backward scan: G_t = r_t + gamma * G_{t+1}
                    G_r, G_c = 0.0, 0.0
                    step_true_r = [0.0] * T
                    step_true_c = [0.0] * T
                    for t in range(T - 1, -1, -1):
                        G_r = hist[t][2] + discount_r * G_r
                        G_c = hist[t][3] + discount_c * G_c
                        step_true_r[t] = G_r
                        step_true_c[t] = G_c

                    # s0
                    s0_true_r.append(step_true_r[0])
                    s0_true_c.append(step_true_c[0])
                    s0_est_r.append(hist[0][0])
                    s0_est_c.append(hist[0][1])

                    # all states
                    for t in range(T):
                        all_true_r.append(step_true_r[t])
                        all_true_c.append(step_true_c[t])
                        all_est_r.append(hist[t][0])
                        all_est_c.append(hist[t][1])

                    episodes_done += 1
                    newly_done += 1

                ep_history[i] = []

            progress.update(task, advance=newly_done)

            obs = next_obs
            act, cur_est_r, cur_est_c, _ = agent.step(obs)

    eval_env.close()

    def _to_tensor(lst):
        return torch.tensor(lst, device=device, dtype=torch.float32)

    s0_true_r_t  = _to_tensor(s0_true_r)
    s0_true_c_t  = _to_tensor(s0_true_c)
    s0_est_r_t   = _to_tensor(s0_est_r)
    s0_est_c_t   = _to_tensor(s0_est_c)

    all_true_r_t = _to_tensor(all_true_r)
    all_true_c_t = _to_tensor(all_true_c)
    all_est_r_t  = _to_tensor(all_est_r)
    all_est_c_t  = _to_tensor(all_est_c)

    def _stats(true_t, est_t):
        error    = torch.mean(true_t - est_t)
        true_m   = torch.mean(true_t)
        est_m    = torch.mean(est_t)
        corr     = torch.corrcoef(torch.stack([true_t, est_t]))[0, 1]
        return error, true_m, est_m, corr

    s0_c_error,  s0_true_c_m,  s0_est_c_m,  s0_corr_c  = _stats(s0_true_c_t,  s0_est_c_t)
    s0_r_error,  s0_true_r_m,  s0_est_r_m,  s0_corr_r  = _stats(s0_true_r_t,  s0_est_r_t)
    all_c_error, all_true_c_m, all_est_c_m, all_corr_c = _stats(all_true_c_t, all_est_c_t)
    all_r_error, all_true_r_m, all_est_r_m, all_corr_r = _stats(all_true_r_t, all_est_r_t)

    if cfgs.logger_cfgs.use_wandb:
        import matplotlib.pyplot as plt
        import wandb

        def _scatter_fig(true_vals, est_vals, label, color, title):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(true_vals, est_vals, alpha=0.3, s=8, color=color)
            lo = min(true_vals.min(), est_vals.min())
            hi = max(true_vals.max(), est_vals.max())
            ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, label='ideal')
            ax.set_xlabel(f'True {label}')
            ax.set_ylabel(f'Estimated {label}')
            ax.set_title(title)
            ax.legend()
            plt.tight_layout()
            return fig

        s0_c_np  = s0_true_c_t.cpu().numpy();  s0_ec_np  = s0_est_c_t.cpu().numpy()
        s0_r_np  = s0_true_r_t.cpu().numpy();  s0_er_np  = s0_est_r_t.cpu().numpy()
        all_c_np = all_true_c_t.cpu().numpy(); all_ec_np = all_est_c_t.cpu().numpy()
        all_r_np = all_true_r_t.cpu().numpy(); all_er_np = all_est_r_t.cpu().numpy()

        fig_s0_c   = _scatter_fig(s0_c_np,  s0_ec_np,  'C', 'steelblue',  'C-Values: True vs Estimated (s0)')
        fig_s0_r   = _scatter_fig(s0_r_np,  s0_er_np,  'R', 'darkorange', 'R-Values: True vs Estimated (s0)')
        fig_all_c  = _scatter_fig(all_c_np, all_ec_np, 'C', 'steelblue',  'C-Values: True vs Estimated (all states)')
        fig_all_r  = _scatter_fig(all_r_np, all_er_np, 'R', 'darkorange', 'R-Values: True vs Estimated (all states)')

        wandb.log({
            # Scatter plots
            'scatter/s0_c_values':  wandb.Image(fig_s0_c),
            'scatter/s0_r_values':  wandb.Image(fig_s0_r),
            'scatter/all_c_values': wandb.Image(fig_all_c),
            'scatter/all_r_values': wandb.Image(fig_all_r),
            # Eval_s0 stats
            'Eval_s0/Correlation_c':      s0_corr_c.item(),
            'Eval_s0/Correlation_r':      s0_corr_r.item(),
            'Eval_s0/EstimationError_c':  s0_c_error.item(),
            'Eval_s0/true_value_c':       s0_true_c_m.item(),
            'Eval_s0/estimate_value_c':   s0_est_c_m.item(),
            'Eval_s0/EstimationError_r':  s0_r_error.item(),
            'Eval_s0/true_value_r':       s0_true_r_m.item(),
            'Eval_s0/estimate_value_r':   s0_est_r_m.item(),
            # Eval_all stats
            'Eval_all/Correlation_c':     all_corr_c.item(),
            'Eval_all/Correlation_r':     all_corr_r.item(),
            'Eval_all/EstimationError_c': all_c_error.item(),
            'Eval_all/true_value_c':      all_true_c_m.item(),
            'Eval_all/estimate_value_c':  all_est_c_m.item(),
            'Eval_all/EstimationError_r': all_r_error.item(),
            'Eval_all/true_value_r':      all_true_r_m.item(),
            'Eval_all/estimate_value_r':  all_est_r_m.item(),
        }, step=epoch)

        plt.close(fig_s0_c)
        plt.close(fig_s0_r)
        plt.close(fig_all_c)
        plt.close(fig_all_r)

    return (
        s0_c_error,  s0_true_c_m,  s0_est_c_m,  s0_corr_c,
        s0_r_error,  s0_true_r_m,  s0_est_r_m,  s0_corr_r,
        all_c_error, all_true_c_m, all_est_c_m, all_corr_c,
        all_r_error, all_true_r_m, all_est_r_m, all_corr_r,
    )
