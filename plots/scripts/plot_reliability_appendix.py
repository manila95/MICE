"""Appendix: why the correlations of Figure 1 can be trusted.

The reference Gbar(s) is the mean of K=10 rollouts from probe state s, so correlating a
critic against it rather than against V attenuates the correlation by
    rho_max(K) = sqrt(Var[V] / (Var[V] + E[sigma^2] / K)),
estimated by one-way ANOVA over the probes.  Cross-check that assumes only independent
rollouts: split each state's rollouts into two halves of 5 and correlate the half-means
across states, r = Corr(Gbar_A, Gbar_B) = rho_max(5)^2, so rho_max(10)^2 = 2r / (1 + r)
(Spearman-Brown), averaged over random splits.

Sweep lsaldj7r, whose references are simulation-only (no critic value folded into the
tail); panels, Figure-1 names and colours (CPO blue, PIDLag orange), 5 seeds, 102 probes:
  (a)/(b) Correlation (reward / cost): critic vs reference (seed mean +- sd) with the ceiling
          rho_max(10) from ANOVA (dashed) and from split halves (dotted)
  (c)/(d) Split-half agreement of the reference (reward / cost): r (seed mean +- sd) with the
          ANOVA prediction rho_max(5)^2 (dashed)

Usage:
    uv run --python 3.12 --with "numpy<2" --with torch --with matplotlib \
        python MICE/plots/scripts/plot_reliability_appendix.py
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # plots/ -> import common.*

from common.paths import fig_path
from common.sweep_data import load_sweep

CELL = "gae_iters2_lr0.0001"          # the configuration plotted in Figure 1
ALGOS = [("CPO", "eval_data_lsaldj7r_cpo", "#4c72b0"),
         ("PIDLag", "eval_data_lsaldj7r_pidlag", "#dd8452")]
SPE = 20000
TICKS = [3e4, 1e5, 3e5, 1e6, 3e6, 1e7]
TICKLABELS = ["30k", "100k", "300k", "1M", "3M", "10M"]


def stats(pred: np.ndarray, G: np.ndarray, n_rep: int = 200, seed: int = 0) -> dict[str, float]:
    n, K = G.shape
    gbar = G.mean(1)
    sigma2 = G.var(1, ddof=1).mean()
    s2b = gbar.var(ddof=1)
    var_v = max(s2b - sigma2 / K, 0.0)
    rho_max = np.sqrt(var_v / s2b) if s2b > 0 else np.nan
    pred_half = var_v / (var_v + sigma2 / (K // 2)) if var_v + sigma2 > 0 else np.nan

    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((n_rep, n, K)), axis=2)
    Gp = np.take_along_axis(np.broadcast_to(G, (n_rep, n, K)), order, axis=2)
    A, B = Gp[:, :, :K // 2].mean(2), Gp[:, :, K // 2:].mean(2)
    Ac, Bc = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
    den = np.sqrt((Ac ** 2).sum(1) * (Bc ** 2).sum(1))
    r = float(np.mean((Ac * Bc).sum(1)[den > 0] / den[den > 0])) if (den > 0).any() else np.nan
    return {"corr": float(np.corrcoef(pred, gbar)[0, 1]) if pred.std() > 0 else np.nan,
            "rho_max": float(rho_max), "split_r": r, "split_r_pred": float(pred_half),
            "rho_max_split": float(np.sqrt(2 * r / (1 + r))) if r > 0 else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=fig_path("reliability_appendix"))
    args = ap.parse_args()

    D = {}
    for name, root, _ in ALGOS:
        seeds = load_sweep(root, "pooled")[CELL].seeds
        for head in ("r", "c"):
            per = [[stats(p.pred[head], p.G[head]) for p in pts] for _, pts in sorted(seeds.items())]
            x = np.array([(p.epoch + 1) * SPE for p in next(iter(seeds.values()))], float)
            D[(name, head)] = {"x": x, **{k: np.array([[d[k] for d in s] for s in per])
                                           for k in per[0][0]}}

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.2), constrained_layout=True)
    titles = ["Correlation (reward)", "Correlation (cost)",
              "Split-half agreement (reward)", "Split-half agreement (cost)"]
    for j, head in enumerate(("r", "c")):
        for name, _, col in ALGOS:
            d = D[(name, head)]
            ax = axes[j]
            m, sd = np.nanmean(d["corr"], 0), np.nanstd(d["corr"], 0, ddof=1)
            ax.plot(d["x"], m, color=col, lw=1.6)
            ax.fill_between(d["x"], m - sd, m + sd, color=col, alpha=0.18, lw=0)
            ax.plot(d["x"], np.nanmean(d["rho_max"], 0), color=col, lw=1.3, ls="--")
            ax.plot(d["x"], np.nanmean(d["rho_max_split"], 0), color=col, lw=1.3, ls=":")
            ax = axes[2 + j]
            m, sd = np.nanmean(d["split_r"], 0), np.nanstd(d["split_r"], 0, ddof=1)
            ax.plot(d["x"], m, color=col, lw=1.6)
            ax.fill_between(d["x"], m - sd, m + sd, color=col, alpha=0.18, lw=0)
            ax.plot(d["x"], np.nanmean(d["split_r_pred"], 0), color=col, lw=1.3, ls="--")
    for j, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_xlim(2e4, 1.03e7)
        ax.xaxis.set_major_locator(FixedLocator(TICKS))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(TICKLABELS)
        ax.set_xlabel("Total env steps")
        ax.set_title(titles[j])
        ax.set_ylim(0.0, 1.05) #if j < 2 else ax.set_ylim(0.5, 1.02)
        ax.grid(alpha=0.3, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].text(0.04, 0.70, "CPO", color=ALGOS[0][2], transform=axes[0].transAxes, fontsize=9, fontweight="bold")
    axes[0].text(0.04, 0.60, "PIDLag", color=ALGOS[1][2], transform=axes[0].transAxes, fontsize=9, fontweight="bold")
    handles = [Line2D([], [], color="0.3", lw=1.6, label="Correlation: critic vs reference; Split-half: half vs half (band: ±1 sd over 5 seeds)"),
               Line2D([], [], color="0.3", lw=1.3, ls="--", label=r"ANOVA: $\rho_{\max}(10)$ (Correlation), $\rho_{\max}(5)^2$ (Split-half)"),
               Line2D([], [], color="0.3", lw=1.3, ls=":", label=r"split-half estimate of $\rho_{\max}(10)$")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.13))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")

    with open(f"{args.out}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["algo", "head", "steps", "corr_mean", "rho_max", "rho_max_split", "split_r", "split_r_pred"])
        for (name, head), d in D.items():
            for t, x in enumerate(d["x"]):
                w.writerow([name, head, int(x)] + [float(np.nanmean(d[k][:, t])) for k in
                                                   ("corr", "rho_max", "rho_max_split", "split_r", "split_r_pred")])

    print("seed-mean curves, over all checkpoints (and checkpoints <= 2M steps)")
    for (name, head), d in D.items():
        rm, rs = np.nanmean(d["rho_max"], 0), np.nanmean(d["rho_max_split"], 0)
        r, rp, c = np.nanmean(d["split_r"], 0), np.nanmean(d["split_r_pred"], 0), np.nanmean(d["corr"], 0)
        e = d["x"] <= 2e6
        per_run_min = np.nanmin(d["rho_max"])
        print(f"{name:>6} {head}: rho_max median {np.median(rm):.3f} min {rm.min():.3f} (early min {rm[e].min():.3f}); "
              f"per-(seed,ckpt) min {per_run_min:.3f} | split-half rho_max median {np.median(rs):.3f}; "
              f"max |ANOVA - split| {np.max(np.abs(rm - rs)):.3f} | r median {np.median(r):.3f} min {r.min():.3f}, "
              f"ANOVA-predicted {np.median(rp):.3f}, max |r - pred| {np.max(np.abs(r - rp)):.3f} | "
              f"critic corr final {c[-1]:.2f}, max {c.max():.2f}; ceiling at final {rm[-1]:.3f}")
    print(f"\nwrote {args.out}.pdf/.png/.csv")


if __name__ == "__main__":
    main()
