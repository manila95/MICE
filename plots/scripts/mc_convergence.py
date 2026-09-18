"""How many Monte Carlo rollouts per probe state?  Convergence of the value estimate, run 1ijr3962.

CPO on PointGoal1, seed 0, 9 checkpoints; 150 probe states (25 at each of six within-episode
steps) with 40 rollouts each.  Quality of the K-rollout value estimate only (no critic):
  (a) error: RMS over states of (mean of K random rollouts - mean of all 40), divided by the sd
      of the 40-rollout estimates across states; median over 200 draws, then median over
      checkpoints (band: range).  Nested subsets, so the expected squared error is
      sigma^2 (1/K - 1/40), smaller than against the true value
  (b) reliability rho_max(K) = Corr(K-rollout estimate, V) = sqrt(K lam / (K lam + 1)),
      lam = Var[V] / E[sigma^2] from a one-way ANOVA on all 40 rollouts
  (d) split-half correlation: correlation across states between two estimates built from
      disjoint random sets of K rollouts each (K <= 20; 200 random splits), against its ANOVA
      prediction rho_max(K)^2 -- a check that does not rely on the variance model
  (c) split-half correlation across training for two rollout budgets: 5 vs 5 (a 10-rollout
      budget) and 20 vs 20 (all 40 rollouts)

Usage:
    uv run --python 3.12 --with "numpy<2" --with torch --with matplotlib \
        python MICE/plots/scripts/mc_convergence.py
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # plots/ -> import common.*

from common.paths import fig_path
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

ROOT = "eval_data_1ijr3962/seed0"
SPE = 20000
K_ALL = np.arange(1, 41)
KS = [1, 2, 3, 5, 10, 15, 20, 30, 40]
KS_SPLIT = [1, 2, 3, 5, 10, 15, 20]
DRAWS = 200
HEADS = [("r", "reward", "#1f77b4"), ("c", "cost", "#c44e52")]
KTICKS = [1, 2, 5, 10, 20, 40]
TICKS, TICKLABELS = [3e4, 1e5, 3e5, 1e6, 3e6, 1e7], ["30k", "100k", "300k", "1M", "3M", "10M"]


def perm_idx(rng, n, K):
    return np.argsort(rng.random((DRAWS, n, K)), axis=2)


def corr_rows(A, B):
    """Row-wise correlation of (draws, n) arrays across the n states."""
    Ac, Bc = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
    return (Ac * Bc).sum(1) / np.sqrt((Ac ** 2).sum(1) * (Bc ** 2).sum(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=fig_path("mc_convergence"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    C = []
    for f in sorted(glob.glob(os.path.join(ROOT, "epoch_*.pkl")), key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1))):
        b = pickle.load(open(f, "rb"))
        row = {"steps": (int(b["epoch"]) + 1) * SPE}
        for h, _, _ in HEADS:
            G = np.asarray(b["pooled"]["raw"][h]["returns"], float)
            n, K = G.shape
            Gb = np.broadcast_to(G, (DRAWS, n, K))
            sigma2 = G.var(1, ddof=1).mean()
            lam = max(G.mean(1).var(ddof=1) - sigma2 / K, 0.0) / sigma2
            d = {"rho": np.sqrt(K_ALL * lam / (K_ALL * lam + 1)), "split": {}, "relerr": {},
                 "err_true10": float(1.0 / np.sqrt(10 * lam)) if lam > 0 else np.nan}
            for k in KS_SPLIT:
                idx = perm_idx(rng, n, K)
                A = np.take_along_axis(Gb, idx[:, :, :k], 2).mean(2)
                B = np.take_along_axis(Gb, idx[:, :, k:2 * k], 2).mean(2)
                d["split"][k] = float(np.nanmean(corr_rows(A, B)))
            gfull = G.mean(1)
            for k in KS:
                if k == K:
                    d["relerr"][k] = 0.0
                    continue
                M = np.take_along_axis(Gb, perm_idx(rng, n, K)[:, :, :k], 2).mean(2)
                d["relerr"][k] = float(np.median(np.sqrt(((M - gfull) ** 2).mean(1))) / gfull.std(ddof=1))
            row[h] = d
        C.append(row)

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.6), constrained_layout=True)
    steps = np.array([c["steps"] for c in C], float)
    ax = axes[0]
    for h, lab, col in HEADS:
        E = np.array([[c[h]["relerr"][k] for k in KS] for c in C])
        ax.plot(KS, np.median(E, 0), color=col, lw=1.7, marker="o", ms=3, label=lab)
        ax.fill_between(KS, E.min(0), E.max(0), color=col, alpha=0.15, lw=0)
    ax.set_ylim(0, None); ax.set_ylabel("RMS error vs 40 rollouts / spread across states")
    ax.set_title("Error of a K-rollout estimate")

    ax = axes[1]
    for h, lab, col in HEADS:
        R = np.array([c[h]["rho"] for c in C])
        ax.plot(K_ALL, np.median(R, 0), color=col, lw=1.7, label=lab)
        ax.fill_between(K_ALL, R.min(0), R.max(0), color=col, alpha=0.15, lw=0)
    ax.set_ylim(0.5, 1.005); ax.set_ylabel(r"$\rho_{\max}(K)$ = Corr(estimate, true value)")
    ax.set_title("Reliability of a K-rollout estimate")

    ax = axes[2]
    for h, lab, col in HEADS:
        ax.plot(steps, [c[h]["split"][10] for c in C], color=col, lw=1.7, marker="o", ms=3, label=lab)
        ax.plot(steps, [c[h]["split"][20] for c in C], color=col, lw=1.2, ls="--")
    ax.set_ylim(0.8, 1.005)
    ax.set_ylabel("Corr(two disjoint estimates)")
    ax.set_title("Split-half correlation across training")
    ax.set_xscale("log"); ax.set_xlim(1.5e4, 1.1e7)
    ax.xaxis.set_major_locator(FixedLocator(TICKS)); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(TICKLABELS); ax.set_xlabel("Total env steps")
    ax.legend(handles=[Line2D([], [], color="0.3", lw=1.7, marker="o", ms=3, label="10 vs 10 (budget K = 10)"),
                       Line2D([], [], color="0.3", lw=1.2, ls="--", label="20 vs 20 (budget K = 40)")],
              fontsize=8, frameon=False, loc="lower right")

    ax = axes[3]
    for h, lab, col in HEADS:
        S = np.array([[c[h]["split"][k] for k in KS_SPLIT] for c in C])
        ax.plot(KS_SPLIT, np.median(S, 0), color=col, lw=1.7, marker="o", ms=3, label=lab)
        ax.fill_between(KS_SPLIT, S.min(0), S.max(0), color=col, alpha=0.15, lw=0)
        P = np.array([c[h]["rho"][np.array(KS_SPLIT) - 1] ** 2 for c in C])
        ax.plot(KS_SPLIT, np.median(P, 0), color=col, lw=1.1, ls=":")
    ax.set_ylim(0.5, 1.005); ax.set_ylabel("Corr(two disjoint K-rollout estimates)")
    ax.set_title("Split-half correlation")

    for ax in (axes[0], axes[1], axes[3]):
        ax.axvline(10, color="0.3", ls="--", lw=1)
        ticks = [k for k in KTICKS if k <= 20] if ax is axes[3] else KTICKS
        ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels([str(k) for k in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("rollouts per half K" if ax is axes[3] else "rollouts per probe state K")
    for ax in axes:
        ax.grid(alpha=0.3, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.legend(handles=[Line2D([], [], color=col, lw=1.7, label=lab) for _, lab, col in HEADS] +
               [Line2D([], [], color="0.3", ls=":", lw=1.1, label=r"ANOVA prediction $\rho_{\max}(K)^2$ (split-half)"),
                Line2D([], [], color="0.3", ls="--", lw=1, label="K = 10")],
               loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.12))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")

    for h, lab, _ in HEADS:
        R = np.array([c[h]["rho"] for c in C])
        E = np.array([[c[h]["relerr"][k] for k in KS] for c in C])
        S = np.sqrt(np.clip([[c[h]["split"][k] for k in KS_SPLIT] for c in C], 0, None))
        print(f"\n{lab}: rho_max(K) median [min]: " +
              "  ".join(f"K={k}: {np.median(R[:, k - 1]):.3f} [{R[:, k - 1].min():.3f}]" for k in (1, 2, 5, 10, 20, 40)))
        print(f"  gain 10->40: median {np.median(R[:, 39] - R[:, 9]):.3f} max {np.max(R[:, 39] - R[:, 9]):.3f}; "
              f"loss 10->5: median {np.median(R[:, 9] - R[:, 4]):.3f} max {np.max(R[:, 9] - R[:, 4]):.3f}")
        print("  relative error median [min, max]: " +
              "  ".join(f"K={k}: {np.median(E[:, i]):.3f} [{E[:, i].min():.3f}, {E[:, i].max():.3f}]" for i, k in enumerate(KS) if k < 40))
        Sr = np.array([[c[h]["split"][k] for k in KS_SPLIT] for c in C])
        print("  split-half corr median [min]: " +
              "  ".join(f"K={k}: {np.median(Sr[:, j]):.3f} [{Sr[:, j].min():.3f}]" for j, k in enumerate(KS_SPLIT)))
        print("  split-half across training: 5v5 (K=10 budget) median "
              f"{np.median([c[h]['split'][5] for c in C]):.3f} [{min(c[h]['split'][5] for c in C):.3f}, "
              f"{max(c[h]['split'][5] for c in C):.3f}]; 20v20 (K=40) median "
              f"{np.median([c[h]['split'][20] for c in C]):.3f} [{min(c[h]['split'][20] for c in C):.3f}, "
              f"{max(c[h]['split'][20] for c in C):.3f}]")
        print("  per checkpoint 5v5 / 20v20: " + "  ".join(
            f"{c['steps']/1e6:.2f}M: {c[h]['split'][5]:.3f}/{c[h]['split'][20]:.3f}" for c in C))
        print("  max |split-half corr - ANOVA rho^2| by K: " +
              "  ".join(f"K={k}: {np.max(np.abs(Sr[:, j] - R[:, k - 1] ** 2)):.3f}" for j, k in enumerate(KS_SPLIT)))
    print(f"\nwrote {args.out}.pdf/.png")


if __name__ == "__main__":
    main()
