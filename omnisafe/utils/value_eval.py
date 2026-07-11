"""Utility for evaluating true vs. estimated value functions."""

from __future__ import annotations

import numpy as np
import torch
from rich.progress import Progress


_HORIZONS = [1, 5, 10, 25, 50]   # H-step horizon analysis
_PROX_HORIZONS = [1, 5, 10]       # binary "cost in next H steps" AUC


def _auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via trapz. Returns NaN when only one class is present."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    idx = np.argsort(-scores)
    labels_s = labels[idx]
    tpr = np.cumsum(labels_s) / n_pos
    fpr = np.cumsum(1 - labels_s) / n_neg
    return float(np.trapz(tpr, fpr))


def estimate_true_value(
    agent, env, cfgs, discount_r, discount_c, eval_episodes=100, epoch=None,
    finite_horizon=False, max_ep_len=None,
):
    """Estimate true V(s) vs. critic estimate by rolling out full episodes.

    Uses the provided (already-wrapped) environment directly so that observation
    and reward normalizers are identical to those seen during training.

    Two evaluation regimes:
    - Eval_s0: initial state of each episode — V(s_0) vs. G_0.
    - Eval_all: every visited state — V(s_t) vs. G_t (MC return from step t).

    Additionally logs three near-term diagnostic analyses for the cost critic:
    - Horizon-stratified correlation: corr(V_c, G_c^H) for H in _HORIZONS.
    - Cost-proximity correlation: corr(V_c, 1/(1+steps_to_next_cost)).
    - Binary AUC: how well V_c predicts "cost in next H steps" for H in _PROX_HORIZONS.
    - Per-episode-timestep mean V_c vs. mean G_c^10 profile (wandb figure).

    When ``finite_horizon`` is enabled the critic is a function of ``(obs, remaining)``, so the
    per-env timestep is tracked and ``remaining = max_ep_len - t`` is fed to ``agent.step``. The
    Monte-Carlo targets are already finite-horizon-correct because each episode ends at the horizon.
    """
    device = torch.device(cfgs.train_cfgs.device)
    num_envs = env.num_envs

    t_env = [0] * num_envs

    def _remaining():
        if not finite_horizon:
            return None
        return torch.tensor([max_ep_len - t for t in t_env], dtype=torch.float32, device=device)

    obs, _ = env.reset()
    act, cur_est_r, cur_est_c, _ = agent.step(obs, remaining=_remaining())

    # Per-env episode history: each entry is (est_r, est_c, reward, cost) at step t
    ep_history = [[] for _ in range(num_envs)]

    # --- existing accumulators ---
    s0_true_r, s0_true_c = [], []
    s0_est_r,  s0_est_c  = [], []
    all_true_r, all_true_c = [], []
    all_est_r,  all_est_c  = [], []

    # --- near-term diagnostic accumulators ---
    # H-horizon cost returns: H -> list of G_c^H values (paired with all_est_c)
    all_hcost: dict[int, list] = {H: [] for H in _HORIZONS}
    # proximity score 1/(1+steps_to_next_cost), paired with all_est_c
    all_proximity: list = []
    # binary "cost in next H steps", H -> list of 0/1 labels, paired with all_est_c
    all_in_next: dict[int, list] = {H: [] for H in _PROX_HORIZONS}
    # per-episode-timestep sums for profile plot: t -> [V_c values], [G_c^10 values]
    vc_by_t:   dict[int, list] = {}
    gc10_by_t: dict[int, list] = {}

    episodes_done = 0

    with Progress() as progress:
        task = progress.add_task('Evaluating value function...', total=eval_episodes)
        while episodes_done < eval_episodes:
            next_obs, r, c, terminated, truncated, _ = env.step(act)

            r_sq = r.squeeze(-1)
            c_sq = c.squeeze(-1)

            for i in range(num_envs):
                ep_history[i].append((
                    cur_est_r[i].item(),
                    cur_est_c[i].item(),
                    r_sq[i].item(),
                    c_sq[i].item(),
                ))

            done = (terminated.bool() | truncated.bool()).squeeze(-1)

            done_list = done.flatten().tolist()
            for i in range(num_envs):
                t_env[i] = 0 if done_list[i] else t_env[i] + 1

            newly_done = 0
            for i in done.nonzero(as_tuple=False).flatten().tolist():
                if episodes_done < eval_episodes:
                    hist = ep_history[i]
                    T = len(hist)

                    costs = np.array([hist[t][3] for t in range(T)], dtype=np.float64)
                    vc    = np.array([hist[t][1] for t in range(T)], dtype=np.float64)

                    # --- full MC returns (backward scan) ---
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

                    # --- horizon-stratified G_c^H ---
                    # Precompute a discount vector for the longest horizon
                    max_H = max(_HORIZONS)
                    disc_vec = discount_c ** np.arange(max_H)
                    for H in _HORIZONS:
                        gc_H = np.zeros(T)
                        for t in range(T):
                            end = min(t + H, T)
                            k   = end - t
                            gc_H[t] = float(np.dot(disc_vec[:k], costs[t:end]))
                        all_hcost[H].extend(gc_H.tolist())

                    # --- proximity to next cost event (including current step) ---
                    # steps_to_next_cost[t] = 0 if c[t]>0 else 1+steps_to_next_cost[t+1]
                    steps = np.full(T, float(T))
                    for t in range(T - 1, -1, -1):
                        if costs[t] > 0:
                            steps[t] = 0.0
                        elif t < T - 1 and steps[t + 1] < T:
                            steps[t] = steps[t + 1] + 1.0
                    proximity = 1.0 / (1.0 + steps)
                    all_proximity.extend(proximity.tolist())

                    # --- binary labels: was cost incurred in the next H steps? ---
                    for H in _PROX_HORIZONS:
                        labels = np.zeros(T)
                        for t in range(T):
                            labels[t] = float(np.any(costs[t:t + H] > 0))
                        all_in_next[H].extend(labels.tolist())

                    # --- per-timestep profile (G_c^10) ---
                    gc10 = np.zeros(T)
                    for t in range(T):
                        end = min(t + 10, T)
                        k   = end - t
                        gc10[t] = float(np.dot(disc_vec[:k], costs[t:end]))
                    for t in range(T):
                        vc_by_t.setdefault(t, []).append(float(vc[t]))
                        gc10_by_t.setdefault(t, []).append(float(gc10[t]))

                    episodes_done += 1
                    newly_done += 1

                ep_history[i] = []

            progress.update(task, advance=newly_done)

            obs = next_obs
            act, cur_est_r, cur_est_c, _ = agent.step(obs, remaining=_remaining())

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
        error  = torch.mean(true_t - est_t)
        true_m = torch.mean(true_t)
        est_m  = torch.mean(est_t)
        corr   = torch.corrcoef(torch.stack([true_t, est_t]))[0, 1]
        return error, true_m, est_m, corr

    s0_c_error,  s0_true_c_m,  s0_est_c_m,  s0_corr_c  = _stats(s0_true_c_t,  s0_est_c_t)
    s0_r_error,  s0_true_r_m,  s0_est_r_m,  s0_corr_r  = _stats(s0_true_r_t,  s0_est_r_t)
    all_c_error, all_true_c_m, all_est_c_m, all_corr_c = _stats(all_true_c_t, all_est_c_t)
    all_r_error, all_true_r_m, all_est_r_m, all_corr_r = _stats(all_true_r_t, all_est_r_t)

    # --- near-term diagnostic metrics ---
    vc_np  = np.array(all_est_c, dtype=np.float64)
    prox_np = np.array(all_proximity, dtype=np.float64)

    def _np_corr(x, y):
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return float('nan')
        return float(np.corrcoef(x, y)[0, 1])

    horizon_corrs = {}
    for H in _HORIZONS:
        gc_H_np = np.array(all_hcost[H], dtype=np.float64)
        horizon_corrs[H] = _np_corr(vc_np, gc_H_np)

    proximity_corr = _np_corr(vc_np, prox_np)

    auc_by_H = {}
    for H in _PROX_HORIZONS:
        labels_np = np.array(all_in_next[H], dtype=np.float64)
        auc_by_H[H] = _auc_roc(vc_np, labels_np)

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

        # --- horizon-corr curve figure ---
        fig_hcorr, ax_hc = plt.subplots(figsize=(6, 4))
        hs = sorted(_HORIZONS)
        cs = [horizon_corrs[H] for H in hs]
        ax_hc.plot(hs, cs, 'o-', color='steelblue', markersize=6)
        ax_hc.axhline(all_corr_c.item(), color='gray', linestyle='--', linewidth=1,
                      label=f'full-episode corr ({all_corr_c.item():.3f})')
        ax_hc.set_xlabel('Horizon H (steps)')
        ax_hc.set_ylabel('Pearson r  (V_c vs G_c^H)')
        ax_hc.set_title('Near-term diagnostic: V_c correlation vs cost horizon')
        ax_hc.legend(fontsize=8)
        plt.tight_layout()

        # --- per-timestep profile figure ---
        max_t = max(vc_by_t.keys()) + 1
        ts         = sorted(vc_by_t.keys())
        mean_vc    = [np.mean(vc_by_t[t])   for t in ts]
        mean_gc10  = [np.mean(gc10_by_t[t]) for t in ts]
        fig_prof, ax_p = plt.subplots(figsize=(8, 4))
        ax_p.plot(ts, mean_vc,   color='steelblue',  label='mean V_c(s_t)')
        ax_p.plot(ts, mean_gc10, color='darkorange', linestyle='--',
                  label='mean G_c^10(s_t)  (near-term cost)')
        ax_p.set_xlabel('Timestep within episode')
        ax_p.set_ylabel('Value')
        ax_p.set_title('V_c profile over episode vs. near-term cost density')
        ax_p.legend(fontsize=8)
        plt.tight_layout()

        # --- build scalar wandb log ---
        log_dict: dict = {
            # existing scatter plots
            'scatter/s0_c_values':  wandb.Image(fig_s0_c),
            'scatter/s0_r_values':  wandb.Image(fig_s0_r),
            'scatter/all_c_values': wandb.Image(fig_all_c),
            'scatter/all_r_values': wandb.Image(fig_all_r),
            # existing scalar metrics
            'Eval_s0/Correlation_c':      s0_corr_c.item(),
            'Eval_s0/Correlation_r':      s0_corr_r.item(),
            'Eval_s0/EstimationError_c':  s0_c_error.item(),
            'Eval_s0/true_value_c':       s0_true_c_m.item(),
            'Eval_s0/estimate_value_c':   s0_est_c_m.item(),
            'Eval_s0/EstimationError_r':  s0_r_error.item(),
            'Eval_s0/true_value_r':       s0_true_r_m.item(),
            'Eval_s0/estimate_value_r':   s0_est_r_m.item(),
            'Eval_all/Correlation_c':     all_corr_c.item(),
            'Eval_all/Correlation_r':     all_corr_r.item(),
            'Eval_all/EstimationError_c': all_c_error.item(),
            'Eval_all/true_value_c':      all_true_c_m.item(),
            'Eval_all/estimate_value_c':  all_est_c_m.item(),
            'Eval_all/EstimationError_r': all_r_error.item(),
            'Eval_all/true_value_r':      all_true_r_m.item(),
            'Eval_all/estimate_value_r':  all_est_r_m.item(),
            # near-term: horizon-stratified correlation scalars
            **{f'Eval_all/CostCorr_H{H}': horizon_corrs[H] for H in _HORIZONS},
            # near-term: proximity correlation
            'Eval_all/CostProximityCorr': proximity_corr,
            # near-term: binary AUC per horizon
            **{f'Eval_all/CostAUC_H{H}': auc_by_H[H] for H in _PROX_HORIZONS},
            # near-term: figures
            'near_term/CostCorrVsHorizon': wandb.Image(fig_hcorr),
            'near_term/VcProfileVsGc10':   wandb.Image(fig_prof),
        }

        # wandb Table for the horizon-corr curve (makes it queryable)
        table = wandb.Table(columns=['horizon', 'pearson_r'])
        for H in sorted(_HORIZONS):
            table.add_data(H, horizon_corrs[H])
        log_dict['near_term/CostCorrVsHorizonTable'] = table

        wandb.log(log_dict, step=epoch)

        plt.close(fig_s0_c)
        plt.close(fig_s0_r)
        plt.close(fig_all_c)
        plt.close(fig_all_r)
        plt.close(fig_hcorr)
        plt.close(fig_prof)

    return (
        s0_c_error,  s0_true_c_m,  s0_est_c_m,  s0_corr_c,
        s0_r_error,  s0_true_r_m,  s0_est_r_m,  s0_corr_r,
        all_c_error, all_true_c_m, all_est_c_m, all_corr_c,
        all_r_error, all_true_r_m, all_est_r_m, all_corr_r,
    )
