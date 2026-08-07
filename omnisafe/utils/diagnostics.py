#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_cost_critic.py
=======================

Fresh-eyes diagnostic for the "near-zero estimation error, near-zero
MC-correlation" cost-critic pathology in on-policy Safe RL (PPO-Lag / CPO
stack, OmniSafe + Safety Gym).

The whole thing is built around one identity. Over the visitation distribution,
for critic V_hat and MC target G:

    MSE(V_hat, G) = var(V_hat) + var(G) - 2*rho*std(V_hat)*std(G) + bias^2
    R^2 = 1 - MSE/var(G)  <=  rho^2

so "rho ~ 0" already caps R^2 near zero, and a genuinely small MSE-against-MC
is then only possible if var(G) is itself tiny. That forks the problem into
three worlds:

  World 1  (degenerate estimand):  var(G_c) ~ 0. The near-zero error is a scale
           artifact and 'correlation' is meaningless on a near-constant. There
           is no critic bug -- the target is the problem.

  World 2  (self-consistent collapse):  the reported near-zero error is the
           TD / Bellman residual, not the MC error. The critic solves its own
           bootstrap equation whose fixed point is near-constant because the
           effective bootstrap horizon is shorter than the hazard spacing.
           1-step Bellman residual small; MC error ~ var(G_c) large; deciles FLAT.

  World 3  (imbalanced-regression tail collapse):  MC error is smallish only
           because the ~97.5% near-zero-cost bulk dominates the mean. The critic
           nails the bulk and flatlines the near-hazard tail. Correlation dies
           in the spread; deciles show a HINGE (saturating predictions), error
           explodes only in the top deciles.

This script computes the five numbers, the decile/hinge curve, the time
regression (T-t explanatory power), the correlation ceiling, the conditional
Spearman at the constraint boundary, and a truncation-bias check, then prints a
verdict. It has three entry points:

  1. --self-test         : validate the numeric core on synthetic World 1/2/3
                           (no OmniSafe needed). Run this first.
  2. --run-dir DIR       : best-effort load of a saved OmniSafe run, roll out,
                           diagnose. Replace `load_omnisafe_run` if you use the
                           FH-SR critics (cost_value_fn should return w_c . psi).
  3. import + call        : `run_diagnostics(batch)` on your own array dict.

A "batch" is a dict of 1-D numpy arrays, all length N (concatenated over
episodes), plus episode bookkeeping:
    obs        : (N, obs_dim) float
    cost       : (N,)  instantaneous cost c_t
    v_hat_c    : (N,)  cost-critic prediction at s_t
    t          : (N,)  int, timestep within episode (0-indexed)
    ep_id      : (N,)  int, episode index (contiguous blocks)
    ep_len     : (N,)  int, length T of the owning episode
    truncated  : (N,)  bool, True on the LAST step of a timed-out episode
    boot_v     : (N,)  float, V_hat_c(final_obs) for the owning episode
                       (only used on truncated last steps; 0 elsewhere ok)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, Optional

import numpy as np

# coral is reserved EXCLUSIVELY for cost-critic pathology elements.
MIDNIGHT = "#21295C"   # neutral / structure
DEEPBLUE = "#065A82"   # secondary neutral (predictions, fits)
CORAL    = "#C0392B"   # pathology only: tail error, flat/hinge collapse region
GREY     = "#9AA0B4"


# ---------------------------------------------------------------------------
# numeric helpers (numpy-only; scipy used only if present, never required)
# ---------------------------------------------------------------------------
def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling (needed: cost-to-go has heavy 0-mass)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)  # 0-indexed average rank
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _avg_ranks(a), _avg_ranks(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:  # all ties -> undefined
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(a, b)[0, 1])


def _quantile_bin_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Dedup'd quantile edges; robust to heavy point-mass at zero."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) < 2:  # fully degenerate
        edges = np.array([x.min(), x.max() + 1e-9])
    return edges


# ---------------------------------------------------------------------------
# targets: MC cost-to-go and GAE(lambda_c) advantage / residuals
# ---------------------------------------------------------------------------
def returns_to_go(costs: np.ndarray, gamma: float, bootstrap: float = 0.0) -> np.ndarray:
    """Per-episode discounted return-to-go with optional terminal bootstrap."""
    G = np.zeros(len(costs))
    running = bootstrap
    for t in range(len(costs) - 1, -1, -1):
        running = costs[t] + gamma * running
        G[t] = running
    return G


def gae_advantage(costs: np.ndarray, values: np.ndarray, boot_v: float,
                  gamma: float, lam: float):
    """GAE(lambda) cost advantage and the 1-step Bellman residual delta_t."""
    T = len(costs)
    adv = np.zeros(T)
    delta = np.zeros(T)
    last = 0.0
    for t in range(T - 1, -1, -1):
        v_next = values[t + 1] if (t + 1) < T else boot_v
        d = costs[t] + gamma * v_next - values[t]
        delta[t] = d
        last = d + gamma * lam * last
        adv[t] = last
    return adv, delta


def build_targets(batch: Dict[str, np.ndarray], gamma_c: float, lambda_c: float):
    """Assemble per-transition MC target, GAE advantage, and 1-step residual,
    iterating episode-by-episode so boundaries and truncation are exact.

    Returns dict with:
       g_mc      : MC cost-to-go, NO bootstrap (finite remaining sum). This is
                   the empirical 'discounted cost-to-go' the critic should track.
                   For gamma_c=1 it is the count of remaining cost events.
       g_target  : training target, WITH truncation bootstrap (what the critic
                   is actually regressed toward). Differs from g_mc only near
                   truncated episode ends -> this is the truncation-bias handle.
       gae_adv   : GAE(lambda_c) cost advantage. TD-target = v_hat + gae_adv,
                   so MSE(v_hat, TD-target) = mean(gae_adv^2).
       delta1    : 1-step Bellman residual c + gamma*V' - V. Its RMS is the
                   cleanest candidate for the 'near-zero estimation error'.
    """
    N = len(batch["cost"])
    g_mc = np.zeros(N)
    g_target = np.zeros(N)
    gae_adv = np.zeros(N)
    delta1 = np.zeros(N)

    ep_ids = batch["ep_id"]
    for ep in np.unique(ep_ids):
        idx = np.where(ep_ids == ep)[0]
        idx = idx[np.argsort(batch["t"][idx])]  # ensure time order
        costs = batch["cost"][idx]
        vals = batch["v_hat_c"][idx]
        trunc_last = bool(batch["truncated"][idx][-1])
        boot = float(batch["boot_v"][idx][-1]) if trunc_last else 0.0

        g_mc[idx] = returns_to_go(costs, gamma_c, bootstrap=0.0)
        g_target[idx] = returns_to_go(costs, gamma_c, bootstrap=boot)
        adv, d1 = gae_advantage(costs, vals, boot, gamma_c, lambda_c)
        gae_adv[idx] = adv
        delta1[idx] = d1

    return dict(g_mc=g_mc, g_target=g_target, gae_adv=gae_adv, delta1=delta1)


# ---------------------------------------------------------------------------
# the diagnostics
# ---------------------------------------------------------------------------
@dataclass
class FiveNumbers:
    var_vhat: float
    var_gmc: float
    mse_td: float          # mean(gae_adv^2) == MSE(V, TD-target)
    bellman1_rms: float    # sqrt(mean(delta1^2)), the clean 'near-zero error' probe
    mse_mc: float          # MSE(V, g_mc)
    r2_mc: float           # 1 - mse_mc/var_gmc
    pearson_mc: float
    bias: float            # mean(V - g_mc)
    frac_zero_gmc: float   # point-mass at zero cost-to-go


def five_numbers(v_hat, g_mc, gae_adv, delta1) -> FiveNumbers:
    var_v = float(np.var(v_hat))
    var_g = float(np.var(g_mc))
    mse_td = float(np.mean(gae_adv ** 2))
    b1 = float(np.sqrt(np.mean(delta1 ** 2)))
    mse_mc = float(np.mean((v_hat - g_mc) ** 2))
    r2 = float(1.0 - mse_mc / var_g) if var_g > 1e-12 else float("nan")
    rho = pearson(v_hat, g_mc)
    bias = float(np.mean(v_hat - g_mc))
    fz = float(np.mean(g_mc <= 1e-9))
    return FiveNumbers(var_v, var_g, mse_td, b1, mse_mc, r2, rho, bias, fz)


def decile_analysis(v_hat, g_mc, n_bins=10):
    """Bin by TRUE cost-to-go; report per-bin mean prediction, mean truth, RMSE.
    Flat curve => World 2. Saturating hinge => World 3."""
    edges = _quantile_bin_edges(g_mc, n_bins)
    which = np.clip(np.digitize(g_mc, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = which == b
        if not np.any(m):
            continue
        rows.append(dict(
            bin=b,
            lo=float(edges[b]), hi=float(edges[b + 1]),
            n=int(m.sum()),
            true_mean=float(g_mc[m].mean()),
            pred_mean=float(v_hat[m].mean()),
            rmse=float(np.sqrt(np.mean((v_hat[m] - g_mc[m]) ** 2))),
        ))
    return rows


def time_regression(g_mc, t, ep_len):
    """Explanatory power of remaining-time (T-t) alone for cost-to-go.
    High R^2 + timestep absent from obs => representability wall (Pardo lever)."""
    rem = (ep_len - t).astype(float)
    X = np.column_stack([np.ones_like(rem), rem])
    beta, *_ = np.linalg.lstsq(X, g_mc, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((g_mc - pred) ** 2))
    ss_tot = float(np.sum((g_mc - g_mc.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return dict(r2_time=r2, slope=float(beta[1]), intercept=float(beta[0]))


def correlation_ceiling(obs, g_mc, k=16, n_anchor=1500, n_bg=4000, seed=0, feat=None):
    """Irreducible-noise estimate: within-neighborhood variance of cost-to-go
    vs total variance. ceiling_rho ~ sqrt(1 - E[within]/total). If low, Pearson
    against MC is the wrong scalar to chase -- the ceiling is structural.

    CAVEAT: this is a kNN estimate and is sensitive to the metric. In raw obs
    with many uninformative dims it UNDER-estimates the ceiling (distractors
    swamp the signal in Euclidean distance). Pass `feat` = a learned feature
    space (e.g. the SR psi(obs)) for a metric where distance is meaningful; that
    is both more reliable and directly relevant to the SR ablation."""
    rng = np.random.default_rng(seed)
    N = len(g_mc)
    total = float(np.var(g_mc))

    def _ceil(Z):
        bg = rng.choice(N, size=min(n_bg, N), replace=False)
        an = rng.choice(bg, size=min(n_anchor, len(bg)), replace=False)
        Zbg = Z[bg]
        within = []
        for i in an:
            d = np.sum((Zbg - Z[i]) ** 2, axis=1)
            nn = np.argpartition(d, min(k, len(d) - 1))[:k]
            within.append(np.var(g_mc[nn]))
        e_w = float(np.mean(within))
        return (float(np.sqrt(max(0.0, 1.0 - e_w / total))) if total > 1e-12 else float("nan")), e_w

    if feat is not None:
        Z = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)
        c, ew = _ceil(Z)
        return dict(ceiling_rho=c, ceiling_rho_raw=c, e_within=ew, total_var=total, k=k)

    # no learned metric: bracket a raw (unweighted, pessimistic under distractors)
    # against a Spearman-relevance-weighted (optimistic) Euclidean metric.
    Zraw = (obs - obs.mean(0)) / (obs.std(0) + 1e-8)
    c_raw, ew_raw = _ceil(Zraw)
    w = np.array([abs(spearman(obs[:, d], g_mc)) for d in range(obs.shape[1])])
    w = np.nan_to_num(w)
    Zw = Zraw * w[None, :]
    c_hi, ew_hi = _ceil(Zw)
    return dict(ceiling_rho=float(max(c_raw, c_hi)), ceiling_rho_raw=c_raw,
                ceiling_rho_weighted=c_hi, e_within=ew_raw, total_var=total, k=k)


def conditional_spearman(v_hat, g_mc, top_q=(0.90, 0.99)):
    """Directional fidelity at the boundary: does the critic ORDER the near-
    hazard states? Sensitive exactly where global Pearson is blind."""
    out = {}
    for q in top_q:
        thr = np.quantile(g_mc, q)
        m = g_mc >= thr
        if m.sum() >= 3:
            out[f"spearman_top{int((1-q)*100)}pct"] = spearman(v_hat[m], g_mc[m])
            out[f"n_top{int((1-q)*100)}pct"] = int(m.sum())
    out["spearman_overall"] = spearman(v_hat, g_mc)
    return out


def truncation_bias_check(v_hat, g_mc, g_target, t, ep_len, n_steps=8):
    """Their flagged sanity check: late-episode states have artificially small
    cost-to-go under truncation. Sweep how much of the episode tail we trim and
    watch correlation move. Big movement => results were truncation-inflated.
    Also reports correlation against the bootstrapped target for contrast."""
    T = ep_len.max()
    fracs = np.linspace(0.0, 0.5, n_steps)
    rows = []
    for f in fracs:
        keep = t < (ep_len * (1.0 - f))
        if keep.sum() < 10:
            continue
        rows.append(dict(
            trim_frac=float(f),
            n=int(keep.sum()),
            pearson_mc=pearson(v_hat[keep], g_mc[keep]),
            spearman_mc=spearman(v_hat[keep], g_mc[keep]),
        ))
    delta_pearson = (rows[-1]["pearson_mc"] - rows[0]["pearson_mc"]) if len(rows) > 1 else float("nan")
    return dict(
        sweep=rows,
        delta_pearson_over_trim=float(delta_pearson),
        pearson_vs_bootstrapped_target=pearson(v_hat, g_target),
        pearson_vs_pure_mc=pearson(v_hat, g_mc),
    )


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
def interpret(f: FiveNumbers, deciles, timeR, ceil, cond, trunc) -> str:
    L = []
    L.append("=" * 70)
    L.append("VERDICT")
    L.append("=" * 70)

    # World 1 test: is the estimand degenerate?
    # heuristic: MC-target std small relative to a single cost event's scale.
    if f.var_gmc < 0.25:  # std(G_c) < 0.5 events -> essentially constant
        L.append("[World 1  DEGENERATE ESTIMAND]  var(G_c) is tiny "
                 f"(std={np.sqrt(f.var_gmc):.3f}). 'Correlation' is meaningless on a "
                 "near-constant and small MSE is a scale artifact. The target, not "
                 "the critic, is the object to reconsider (discounting / units).")
        return "\n".join(L)

    # not degenerate -> near-zero MSE-against-MC is impossible; error must be TD.
    L.append(f"var(G_c) is non-trivial (std={np.sqrt(f.var_gmc):.3f}), so a near-zero "
             "error against MC is mathematically impossible "
             f"(R^2_mc={f.r2_mc:.3f} <= rho^2={f.pearson_mc**2:.3f}).")
    L.append(f"  1-step Bellman residual RMS = {f.bellman1_rms:.4f}  "
             f"(this is the plausible 'near-zero estimation error')")
    L.append(f"  MSE vs TD-target           = {f.mse_td:.4f}")
    L.append(f"  MSE vs MC cost-to-go       = {f.mse_mc:.4f}  <-- the real error")

    # World 2 vs 3: flat deciles vs hinge.
    pred_span = max(r["pred_mean"] for r in deciles) - min(r["pred_mean"] for r in deciles)
    true_span = max(r["true_mean"] for r in deciles) - min(r["true_mean"] for r in deciles)
    slope_ratio = pred_span / true_span if true_span > 1e-9 else 0.0
    L.append(f"  decile prediction span / truth span = {slope_ratio:.3f}  "
             f"(1.0 = perfectly tracking, 0.0 = flat)")
    if slope_ratio < 0.15:
        L.append("[World 2  SELF-CONSISTENT COLLAPSE]  predictions are ~flat across "
                 "true-cost deciles while the 1-step residual is small: the critic "
                 "sits at a near-constant bootstrap fixed point. Densifying targets "
                 "(cost-to-go / MC) is necessary; representation alone will not move it.")
    elif slope_ratio < 0.6:
        L.append("[World 3  TAIL COLLAPSE]  predictions rise then SATURATE through the "
                 "top deciles (hinge). The near-zero-cost bulk "
                 f"(frac_zero={f.frac_zero_gmc:.3f}) dominates the mean error while the "
                 "near-hazard tail is flatlined. Imbalanced-regression fix indicated "
                 "(HL-Gauss / target densification), orthogonal to SR.")
    else:
        L.append("[MIXED / HEALTHIER]  predictions track truth substantially across "
                 "deciles; the failure is milder than a pure flat/hinge collapse.")

    # representability wall
    if timeR["r2_time"] > 0.5:
        L.append(f"[TIME WALL]  (T-t) alone explains R^2={timeR['r2_time']:.2f} of "
                 "cost-to-go. If the timestep is absent from obs, the cost critic is "
                 "MISSPECIFIED, not underfit -> add time-to-go / partial-episode "
                 "bootstrap before spending on representation.")

    # ceiling
    if not np.isnan(ceil["ceiling_rho"]):
        note = "" if ceil.get("metric") == "psi(feat)" else \
               " (raw-obs metric; likely an UNDER-estimate -- rerun with feat=psi)"
        L.append(f"[CEILING]  irreducible-noise ceiling on Pearson ~ "
                 f"{ceil['ceiling_rho']:.2f}{note}. Interpret the achieved "
                 "correlation relative to this, not to 1.0.")

    # boundary ordering
    keys = [k for k in cond if k.startswith("spearman_top")]
    if keys:
        L.append("[BOUNDARY ORDERING]  " + ", ".join(f"{k}={cond[k]:.3f}" for k in keys) +
                 f" (overall spearman={cond['spearman_overall']:.3f}). This is the "
                 "operationally binding quantity for CPO's constraint gradient.")

    # truncation
    dp = trunc["delta_pearson_over_trim"]
    if not np.isnan(dp) and abs(dp) > 0.1:
        L.append(f"[TRUNCATION FLAG]  Pearson moves by {dp:+.3f} as the episode tail is "
                 "trimmed -> late-episode small-cost-to-go states are materially "
                 "inflating/deflating the number. Report the trimmed version.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# plotting (4-panel). coral == pathology only.
# ---------------------------------------------------------------------------
def make_plots(batch, targ, f, deciles, timeR, ceil, cond, trunc, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = batch["v_hat_c"]
    g = targ["g_mc"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Cost-critic diagnostic", color=MIDNIGHT, fontsize=14, fontweight="bold")

    # (1) decile calibration: predicted vs true, y=x reference
    a = ax[0, 0]
    tm = np.array([r["true_mean"] for r in deciles])
    pm = np.array([r["pred_mean"] for r in deciles])
    hi = max(tm.max(), pm.max()) * 1.05 + 1e-6
    a.plot([0, hi], [0, hi], "--", color=GREY, lw=1, label="perfect (y=x)")
    # top-two deciles flagged as pathology region
    npath = min(2, len(tm))
    a.plot(tm[:-npath], pm[:-npath], "o-", color=DEEPBLUE, label="prediction")
    a.plot(tm[-npath:], pm[-npath:], "o-", color=CORAL, lw=2,
           label="near-hazard tail")
    a.set_xlabel("true cost-to-go (decile mean)")
    a.set_ylabel("predicted (decile mean)")
    a.set_title("Calibration by decile  (flat=W2, hinge=W3)", color=MIDNIGHT)
    a.legend(fontsize=8)

    # (2) per-decile RMSE, tail in coral
    a = ax[0, 1]
    rmse = np.array([r["rmse"] for r in deciles])
    colors = [DEEPBLUE] * len(rmse)
    for i in range(max(0, len(rmse) - npath), len(rmse)):
        colors[i] = CORAL
    a.bar(np.arange(len(rmse)), rmse, color=colors)
    a.set_xlabel("cost-to-go decile (low -> high)")
    a.set_ylabel("RMSE within decile")
    a.set_title("Where the error lives", color=MIDNIGHT)

    # (3) scatter V vs G (subsample), plus time-R2 annotation
    a = ax[1, 0]
    rng = np.random.default_rng(0)
    s = rng.choice(len(v), size=min(3000, len(v)), replace=False)
    a.scatter(g[s], v[s], s=6, alpha=0.25, color=MIDNIGHT, edgecolors="none")
    hi2 = g[s].max() * 1.05 + 1e-6
    a.plot([0, hi2], [0, hi2], "--", color=GREY, lw=1)
    a.set_xlabel("true cost-to-go G_c")
    a.set_ylabel("predicted V_hat_c")
    a.set_title(f"pearson={f.pearson_mc:.3f}  R2={f.r2_mc:.3f}  "
                f"ceil~{ceil['ceiling_rho']:.2f}\n"
                f"(T-t) explains R2={timeR['r2_time']:.2f}", color=MIDNIGHT)

    # (4) truncation sweep: correlation vs tail trim
    a = ax[1, 1]
    sw = trunc["sweep"]
    if sw:
        tf = [r["trim_frac"] for r in sw]
        pc = [r["pearson_mc"] for r in sw]
        sc = [r["spearman_mc"] for r in sw]
        a.plot(tf, pc, "o-", color=DEEPBLUE, label="pearson")
        a.plot(tf, sc, "s--", color=MIDNIGHT, label="spearman")
        moved = abs(trunc["delta_pearson_over_trim"]) > 0.1
        a.set_title("Truncation-bias check"
                    + ("  [FLAGGED]" if moved else ""),
                    color=CORAL if moved else MIDNIGHT)
        a.set_xlabel("fraction of episode tail trimmed")
        a.set_ylabel("correlation with G_c")
        a.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# top-level driver
# ---------------------------------------------------------------------------
def run_diagnostics(batch: Dict[str, np.ndarray], gamma_c=1.0, lambda_c=0.95,
                    out_png="cost_critic_diagnostic.png", verbose=True):
    targ = build_targets(batch, gamma_c, lambda_c)
    v = batch["v_hat_c"]
    f = five_numbers(v, targ["g_mc"], targ["gae_adv"], targ["delta1"])
    deciles = decile_analysis(v, targ["g_mc"])
    timeR = time_regression(targ["g_mc"], batch["t"], batch["ep_len"])
    ceil = correlation_ceiling(batch["obs"], targ["g_mc"], feat=batch.get("feat"))
    ceil["metric"] = "psi(feat)" if batch.get("feat") is not None else "raw_obs"
    cond = conditional_spearman(v, targ["g_mc"])
    trunc = truncation_bias_check(v, targ["g_mc"], targ["g_target"],
                                  batch["t"], batch["ep_len"])
    verdict = interpret(f, deciles, timeR, ceil, cond, trunc)

    report = dict(five_numbers=asdict(f), deciles=deciles, time_regression=timeR,
                  correlation_ceiling=ceil, conditional_spearman=cond,
                  truncation_check={k: v for k, v in trunc.items() if k != "sweep"},
                  gamma_c=gamma_c, lambda_c=lambda_c)
    if verbose:
        print(json.dumps(report, indent=2, default=float))
        print("\n" + verdict + "\n")
    png = make_plots(batch, targ, f, deciles, timeR, ceil, cond, trunc, out_png)
    report["plot"] = png
    report["verdict"] = verdict
    return report


# ---------------------------------------------------------------------------
# OmniSafe glue  (best-effort; SWAP for FH-SR critics)
# ---------------------------------------------------------------------------
def load_omnisafe_run(run_dir: str, epoch: Optional[int] = None, device="cpu"):
    """Rebuild env + actor-critic from a saved OmniSafe run.

    Returns (env, actor_fn, cost_value_fn).

    ---- FH-SR NOTE -------------------------------------------------------
    For the SR-factored critics, this default reconstruction of the vanilla
    ConstraintActorCritic will not match your registry. Replace the body with
    your constructor and set:
        cost_value_fn = lambda obs_t: (psi(obs_t) @ w_c).squeeze(-1)
    Everything downstream (collect_rollouts + run_diagnostics) is unchanged.
    -----------------------------------------------------------------------
    """
    import torch
    from omnisafe.envs.core import make as make_env
    from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
    from omnisafe.utils.config import Config

    with open(os.path.join(run_dir, "config.json")) as fh:
        cfg = json.load(fh)
    env_id = cfg["env_id"]
    env = make_env(env_id, num_envs=1, device=device)

    save_dir = os.path.join(run_dir, "torch_save")
    ckpts = sorted(p for p in os.listdir(save_dir) if p.endswith(".pt"))
    ckpt = f"epoch-{epoch}.pt" if epoch is not None else ckpts[-1]
    params = torch.load(os.path.join(save_dir, ckpt), map_location=device)

    model_cfgs = Config.dict2config(cfg["model_cfgs"]) if isinstance(cfg["model_cfgs"], dict) else cfg["model_cfgs"]
    ac = ConstraintActorCritic(
        obs_space=env.observation_space,
        act_space=env.action_space,
        model_cfgs=model_cfgs,
        epochs=1,
    ).to(device)
    ac.load_state_dict(params["pi"])
    ac.eval()

    def actor_fn(obs_t):
        with torch.no_grad():
            act = ac.actor.predict(obs_t, deterministic=False)
        return act

    def cost_value_fn(obs_t):
        with torch.no_grad():
            vc = ac.cost_critic(obs_t)
        if isinstance(vc, (list, tuple)):
            vc = vc[0]
        return vc.squeeze(-1)

    return env, actor_fn, cost_value_fn


def collect_rollouts(env, actor_fn, cost_value_fn, n_steps=20000, device="cpu",
                     feat_fn=None):
    """Roll out on-policy and assemble the batch dict the diagnostics expect.
    Collects whole episodes so boundaries/truncation are exact. Single-env
    (num_envs=1) by design: manual reset gives the true terminal observation,
    so the truncation bootstrap is exact and autoreset can't corrupt it.

    feat_fn (optional): obs_t -> feature tensor (e.g. SR psi). If given, features
    are stored under batch['feat'] and used for the correlation-ceiling metric."""
    import torch
    obs_buf, cost_buf, v_buf, t_buf, ep_buf, len_buf, trunc_buf, boot_buf = \
        [], [], [], [], [], [], [], []
    feat_buf = [] if feat_fn is not None else None
    steps = 0
    ep = 0

    def _prep(o):
        ot = o if torch.is_tensor(o) else torch.as_tensor(o, dtype=torch.float32, device=device)
        ot = ot.to(device)
        if ot.ndim == 1:
            ot = ot.unsqueeze(0)
        return ot.float()

    while steps < n_steps:
        obs, _ = env.reset()
        ep_obs, ep_cost, ep_v, ep_feat = [], [], [], []
        done = False
        last_trunc = False
        while not done:
            ot = _prep(obs)
            vc = float(cost_value_fn(ot).reshape(-1)[0].detach().cpu())
            if feat_fn is not None:
                ep_feat.append(np.asarray(feat_fn(ot).reshape(-1).detach().cpu(), dtype=float))
            act = actor_fn(ot)
            act_step = act.reshape(-1) if torch.is_tensor(act) else act
            nobs, _rew, cost, term, trunc, info = env.step(act_step)
            ep_obs.append(np.asarray(ot.reshape(-1).detach().cpu(), dtype=float))
            ep_cost.append(float(np.asarray(cost).reshape(-1)[0]))
            ep_v.append(vc)
            term_b = bool(np.asarray(term).reshape(-1)[0])
            trunc_b = bool(np.asarray(trunc).reshape(-1)[0])
            done = term_b or trunc_b
            last_trunc = trunc_b and not term_b
            # autoreset-safe terminal obs: prefer info['final_observation'] if the
            # env auto-reset; else the returned nobs is the true terminal obs.
            final_obs = None
            if done and isinstance(info, dict) and info.get("final_observation") is not None:
                final_obs = info["final_observation"]
            obs = nobs if not done else (final_obs if final_obs is not None else nobs)
        boot_v = float(cost_value_fn(_prep(obs)).reshape(-1)[0].detach().cpu()) if last_trunc else 0.0

        T = len(ep_cost)
        obs_buf.extend(ep_obs); cost_buf.extend(ep_cost); v_buf.extend(ep_v)
        t_buf.extend(range(T)); ep_buf.extend([ep] * T); len_buf.extend([T] * T)
        trunc_flags = [False] * T
        if last_trunc:
            trunc_flags[-1] = True
        trunc_buf.extend(trunc_flags); boot_buf.extend([boot_v] * T)
        if feat_buf is not None:
            feat_buf.extend(ep_feat)
        steps += T
        ep += 1

    batch = dict(
        obs=np.asarray(obs_buf, dtype=float),
        cost=np.asarray(cost_buf, dtype=float),
        v_hat_c=np.asarray(v_buf, dtype=float),
        t=np.asarray(t_buf, dtype=int),
        ep_id=np.asarray(ep_buf, dtype=int),
        ep_len=np.asarray(len_buf, dtype=int),
        truncated=np.asarray(trunc_buf, dtype=bool),
        boot_v=np.asarray(boot_buf, dtype=float),
    )
    if feat_buf is not None:
        batch["feat"] = np.asarray(feat_buf, dtype=float)
    return batch


# ---------------------------------------------------------------------------
# IN-TRAINING HOOK  (the single function to call from your training loop)
# ---------------------------------------------------------------------------
_DIAG_ENV_CACHE: Dict[str, object] = {}   # keyed by env_id, reused across calls


def _derive_fns(actor_critic, cost_value_fn, act_fn, feat_fn):
    """Auto-wire action + cost-value (+ optional feature) callables from the AC.
    Priority: explicit user callables > ac.step 4-tuple > actor.predict/cost_critic.
    This makes the hook work unchanged for the vanilla ConstraintActorCritic and
    for FH-SR critics whose .step already returns the SR cost value in slot [2]."""
    import torch
    ac = actor_critic

    if cost_value_fn is not None and act_fn is not None:
        return act_fn, cost_value_fn, feat_fn

    has_step = hasattr(ac, "step")

    def _via_step(obs_t):
        out = ac.step(obs_t)                 # (act, value_r, value_c, logp)
        return out[0], out[2]

    def _default_act(obs_t):
        if act_fn is not None:
            return act_fn(obs_t)
        if has_step:
            return _via_step(obs_t)[0]
        return ac.actor.predict(obs_t, deterministic=False)

    def _default_cost_value(obs_t):
        if cost_value_fn is not None:
            return cost_value_fn(obs_t)
        if has_step:
            vc = _via_step(obs_t)[1]
        else:
            vc = ac.cost_critic(obs_t)
        if isinstance(vc, (list, tuple)):
            vc = vc[0]
        return vc.squeeze(-1)

    # sanity: if ac.step exists but doesn't return >=3 items, the call-time
    # _via_step will raise; callers hitting a non-standard .step should pass
    # cost_value_fn explicitly (documented in diagnose_during_training).
    return _default_act, _default_cost_value, feat_fn


COST_CRITIC_DIAG_KEYS = [
    "CostCritic/std_Gc",
    "CostCritic/bellman1_rms",
    "CostCritic/mse_td",
    "CostCritic/mse_mc",
    "CostCritic/R2_mc",
    "CostCritic/pearson_mc",
    "CostCritic/decile_span_ratio",
    "CostCritic/spearman_overall",
    "CostCritic/spearman_top10pct",
    "CostCritic/spearman_top1pct",
    "CostCritic/r2_time",
    "CostCritic/ceiling_rho",
    "CostCritic/trunc_delta_pearson",
    "CostCritic/frac_zero_Gc",
]
"""Keys `diagnose_during_training` stores via `logger.store`. Callers must
register these on the OmniSafe Logger (e.g. with a window so they hold their
last value on epochs the diagnostic doesn't run) before passing `logger=`."""


def _flatten_scalars(report) -> Dict[str, float]:
    """The scalars worth logging every period so you can watch the pathology
    evolve. decile_span_ratio is the money metric: it climbs out of collapse."""
    f = report["five_numbers"]
    dec = report["deciles"]
    pred_span = max(r["pred_mean"] for r in dec) - min(r["pred_mean"] for r in dec)
    true_span = max(r["true_mean"] for r in dec) - min(r["true_mean"] for r in dec)
    span_ratio = float(pred_span / true_span) if true_span > 1e-9 else 0.0
    cond = report["conditional_spearman"]
    scal = {
        "CostCritic/std_Gc": float(np.sqrt(f["var_gmc"])),
        "CostCritic/bellman1_rms": f["bellman1_rms"],
        "CostCritic/mse_td": f["mse_td"],
        "CostCritic/mse_mc": f["mse_mc"],
        "CostCritic/R2_mc": f["r2_mc"],
        "CostCritic/pearson_mc": f["pearson_mc"],
        "CostCritic/decile_span_ratio": span_ratio,
        "CostCritic/spearman_overall": cond.get("spearman_overall", float("nan")),
        "CostCritic/spearman_top10pct": cond.get("spearman_top9pct", float("nan")),
        "CostCritic/spearman_top1pct": cond.get("spearman_top1pct", float("nan")),
        "CostCritic/r2_time": report["time_regression"]["r2_time"],
        "CostCritic/ceiling_rho": report["correlation_ceiling"]["ceiling_rho"],
        "CostCritic/trunc_delta_pearson": report["truncation_check"]["delta_pearson_over_trim"],
        "CostCritic/frac_zero_Gc": f["frac_zero_gmc"],
    }
    return scal


def diagnose_during_training(
    actor_critic,
    *,
    env=None,
    env_id: Optional[str] = None,
    gamma_c: float = 1.0,
    lambda_c: float = 0.95,
    n_steps: int = 8000,
    epoch: Optional[int] = None,
    logger=None,
    out_dir: str = ".",
    cost_value_fn: Optional[Callable] = None,
    act_fn: Optional[Callable] = None,
    feat_fn: Optional[Callable] = None,
    device=None,
    make_plot: bool = True,
    verbose: bool = False,
) -> Dict:
    """Run the full cost-critic diagnostic from inside a training loop.

    THE ONE FUNCTION TO CALL. Pass your actor-critic object; everything else is
    auto-wired. It is non-intrusive: eval mode is set and restored, grads are
    disabled, and the torch (+cuda) RNG state is saved and restored so training
    stays bit-reproducible. It never touches your training env -- it uses a
    separate, cached eval env built from `env_id` (or an `env` you pass in that
    you guarantee is safe to step).

    Typical use, inside your OmniSafe algorithm's epoch loop:

        from diagnose_cost_critic import diagnose_during_training
        if self._epoch % 20 == 0:
            diagnose_during_training(
                self._actor_critic,
                env_id=self._cfgs.env_id,
                gamma_c=1.0, lambda_c=self._cfgs.algo_cfgs.lambda_c,
                n_steps=8000, epoch=self._epoch,
                logger=self._logger,               # optional
                out_dir=self._logger.log_dir,
                feat_fn=lambda o: self._actor_critic.cost_critic.psi(o),  # FH-SR ceiling
            )

    For the vanilla ConstraintActorCritic you can drop `feat_fn` (the ceiling
    then falls back to the raw-obs bracket). For FH-SR critics, pass `feat_fn`
    (psi) and, if your `.step` does not already return the SR cost value in slot
    [2], pass `cost_value_fn=lambda o: (self._actor_critic.cost_critic.psi(o) @ w_c)`.

    Returns the full report dict (also written to JSONL for offline tracking).
    """
    import torch

    # device
    if device is None:
        try:
            device = next(actor_critic.parameters()).device
        except Exception:
            device = "cpu"

    # isolated eval env (never the training env)
    if env is None:
        if env_id is None:
            raise ValueError("Provide env_id (preferred, builds an isolated eval "
                             "env) or an env you guarantee is safe to step.")
        if env_id not in _DIAG_ENV_CACHE:
            from omnisafe.envs.core import make as make_env
            _DIAG_ENV_CACHE[env_id] = make_env(env_id, num_envs=1, device=device)
        env = _DIAG_ENV_CACHE[env_id]

    a_fn, c_fn, ft_fn = _derive_fns(actor_critic, cost_value_fn, act_fn, feat_fn)

    # save state we must not perturb
    was_training = actor_critic.training
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        actor_critic.eval()
        with torch.no_grad():
            batch = collect_rollouts(env, a_fn, c_fn, n_steps=n_steps,
                                     device=device, feat_fn=ft_fn)
    finally:
        # restore training state and RNG so the run is unaffected
        if was_training:
            actor_critic.train()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    tag = f"_epoch{epoch}" if epoch is not None else ""
    out_png = os.path.join(out_dir, f"cost_critic_diag{tag}.png")
    os.makedirs(out_dir, exist_ok=True)

    report = run_diagnostics(batch, gamma_c=gamma_c, lambda_c=lambda_c,
                             out_png=out_png if make_plot else "/tmp/_diag_tmp.png",
                             verbose=verbose)
    report["epoch"] = epoch
    scal = _flatten_scalars(report)
    report["scalars"] = scal

    # 1) robust offline trace: one JSON line per call, always written
    try:
        with open(os.path.join(out_dir, "cost_critic_diag_metrics.jsonl"), "a") as fh:
            fh.write(json.dumps({"epoch": epoch, **scal}, default=float) + "\n")
    except Exception:
        pass

    # 2) best-effort logger integration (OmniSafe Logger needs pre-registered
    #    keys; we try/except so an unregistered key never crashes training).
    if logger is not None and hasattr(logger, "store"):
        try:
            logger.store(**scal)
        except Exception:
            for k, v in scal.items():
                try:
                    logger.store({k: v})
                except Exception:
                    pass

    # 3) push the diagnostic plot itself to tensorboard/wandb, if the logger
    #    supports it (added for image logging; older loggers just skip this).
    if make_plot and logger is not None and hasattr(logger, "log_image"):
        try:
            logger.log_image("CostCritic/diagnostic_plot", out_png, step=epoch)
        except Exception:
            pass

    if verbose:
        print(f"[cost-critic diag @ epoch {epoch}] "
              f"span_ratio={scal['CostCritic/decile_span_ratio']:.3f} "
              f"R2_mc={scal['CostCritic/R2_mc']:.3f} "
              f"bellman1={scal['CostCritic/bellman1_rms']:.4f} "
              f"std_Gc={scal['CostCritic/std_Gc']:.3f}")
    return report



def plot_metric_history(jsonl_path: str, out_png: str = "cost_critic_history.png"):
    """Plot the periodic diagnostic trace (from the JSONL the hook appends).
    The panel to watch is decile_span_ratio: 0 = collapsed critic, ->1 = the
    critic climbing out and tracking cost-to-go across the whole range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    if not rows:
        return None
    ep = [r.get("epoch") for r in rows]

    def col(k):
        return [r.get(k, float("nan")) for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    a = ax[0]
    a.plot(ep, col("CostCritic/decile_span_ratio"), "o-", color=MIDNIGHT, label="decile span ratio")
    a.plot(ep, col("CostCritic/pearson_mc"), "s--", color=DEEPBLUE, label="pearson(V,G)")
    a.axhline(0.15, color=CORAL, ls=":", lw=1, label="collapse threshold")
    a.set_ylim(-0.1, 1.05); a.set_xlabel("epoch"); a.set_ylabel("value")
    a.set_title("Climbing out of collapse", color=MIDNIGHT); a.legend(fontsize=8)
    a = ax[1]
    a.plot(ep, col("CostCritic/mse_mc"), "o-", color=CORAL, label="MSE vs MC (real)")
    a.plot(ep, col("CostCritic/mse_td"), "s--", color=DEEPBLUE, label="MSE vs TD")
    a.plot(ep, [x ** 2 for x in col("CostCritic/bellman1_rms")], "^:", color=MIDNIGHT, label="1-step Bellman MSE")
    a.set_xlabel("epoch"); a.set_ylabel("mean sq error")
    a.set_title("Error decomposition", color=MIDNIGHT); a.legend(fontsize=8)
    a = ax[2]
    a.plot(ep, col("CostCritic/spearman_top10pct"), "o-", color=MIDNIGHT, label="spearman top-10%")
    a.plot(ep, col("CostCritic/ceiling_rho"), "--", color=GREY, label="ceiling")
    a.plot(ep, col("CostCritic/std_Gc"), "s:", color=DEEPBLUE, label="std(G_c)")
    a.set_xlabel("epoch"); a.set_ylabel("value")
    a.set_title("Boundary fidelity & estimand scale", color=MIDNIGHT); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def _synth_batch(world: str, n_ep=120, T=200, spacing=40, n_distract=3, seed=0):
    """Hazard field with mean event spacing ~= `spacing`; gamma_c=1 so
    ground-truth cost-to-go is the count of remaining events.

    obs = [noisy proxy of G_c] + a few distractor dims, giving a realistic
    correlation ceiling < 1. World 2 is a SINGLE GLOBAL fixed point (not a
    per-episode mean) -- that is the faithful near-zero-fixed-point mechanism."""
    rng = np.random.default_rng(seed)
    p = 1.0 / spacing
    gamma_used = 0.5 if world == "world1" else 1.0

    # pass 1: costs, targets, obs
    COSTS, GREF, OBS = [], [], []
    for _ in range(n_ep):
        costs = (rng.random(T) < p).astype(float)
        gref = returns_to_go(costs, gamma_used, 0.0)
        signal = gref + rng.normal(0, 0.7, size=T)
        obs = np.column_stack([signal] + [rng.normal(size=T) for _ in range(n_distract)])
        COSTS.append(costs); GREF.append(gref); OBS.append(obs)
    global_mean = float(np.mean(np.concatenate(GREF)))

    # pass 2: critic predictions per world
    OB, C, V, TT, EP, LN, TR, BV = [], [], [], [], [], [], [], []
    for ep in range(n_ep):
        costs, gref, obs = COSTS[ep], GREF[ep], OBS[ep]
        if world == "world1":            # degenerate estimand: critic ~ const, var(G)~0
            v = global_mean + rng.normal(0, 0.01, size=T)
        elif world == "world2":          # global near-constant fixed point
            v = np.full(T, global_mean) + rng.normal(0, 0.02, size=T)
        elif world == "world3":          # tail collapse: track bulk, saturate tail
            v = np.minimum(gref, 2.5) + rng.normal(0, 0.05, size=T)
        else:
            raise ValueError(world)
        OB.append(obs); C.append(costs); V.append(v)
        TT.append(np.arange(T)); EP.append(np.full(T, ep)); LN.append(np.full(T, T))
        tr = np.zeros(T, bool); tr[-1] = True; TR.append(tr)
        BV.append(np.full(T, v[-1]))
    return dict(
        obs=np.vstack(OB), cost=np.concatenate(C), v_hat_c=np.concatenate(V),
        t=np.concatenate(TT), ep_id=np.concatenate(EP), ep_len=np.concatenate(LN),
        truncated=np.concatenate(TR), boot_v=np.concatenate(BV),
    )


def self_test():
    print("### SELF-TEST: synthetic World 1 / 2 / 3 (gamma_c=1, spacing~40) ###\n")
    for w in ("world1", "world2", "world3"):
        print("#" * 72)
        print(f"# {w}")
        print("#" * 72)
        b = _synth_batch(w, seed=hash(w) % 2**32)
        rep = run_diagnostics(b, gamma_c=(0.5 if w == "world1" else 1.0),
                              lambda_c=0.95, out_png=f"selftest_{w}.png",
                              verbose=False)
        f = rep["five_numbers"]
        print(f"  std(G_c)={np.sqrt(f['var_gmc']):.3f}  "
              f"bellman1_rms={f['bellman1_rms']:.4f}  "
              f"mse_td={f['mse_td']:.4f}  mse_mc={f['mse_mc']:.4f}  "
              f"R2_mc={f['r2_mc']:.3f}  pearson={f['pearson_mc']:.3f}")
        print(rep["verdict"].split("VERDICT")[-1].strip().split("\n")[1] if "VERDICT" in rep["verdict"] else "")
        # print just the world tag lines
        for line in rep["verdict"].splitlines():
            if line.strip().startswith("[World") or line.strip().startswith("[TIME") \
               or line.strip().startswith("[BOUNDARY") or line.strip().startswith("[CEILING"):
                print("   ", line.strip())
        print(f"   plot -> {rep['plot']}\n")
    print("Self-test complete. Expected: world1->W1, world2->W2 (flat, tiny "
          "bellman1), world3->W3 (hinge, tail RMSE).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="validate the numeric core on synthetic worlds (no OmniSafe)")
    ap.add_argument("--run-dir", type=str, default=None,
                    help="saved OmniSafe run directory (contains config.json, torch_save/)")
    ap.add_argument("--epoch", type=int, default=None, help="checkpoint epoch (default: latest)")
    ap.add_argument("--steps", type=int, default=20000, help="rollout steps to collect")
    ap.add_argument("--gamma-c", type=float, default=1.0)
    ap.add_argument("--lambda-c", type=float, default=0.95)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="cost_critic_diagnostic.png")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.run_dir is None:
        ap.error("provide --run-dir, or --self-test")

    env, actor_fn, cost_value_fn = load_omnisafe_run(args.run_dir, args.epoch, args.device)
    batch = collect_rollouts(env, actor_fn, cost_value_fn, args.steps, args.device)
    run_diagnostics(batch, gamma_c=args.gamma_c, lambda_c=args.lambda_c, out_png=args.out)


if __name__ == "__main__":
    main()