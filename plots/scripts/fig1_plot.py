"""Figure 1: episodic return, episodic cost and critic-vs-reference correlations.

Sweep liam-paull/calibration_rl/lsaldj7r (CPO and TRPOPID on PointGoal1, 512
epochs of 20k steps, 5 seeds per cell), the re-run in which the reference
rollouts carry no critic output at their last step.

Data layout, as written by download_sweep.py:
    eval_data_lsaldj7r_<tag>/history.csv                  -- EpRet / EpCost curves
    eval_data_lsaldj7r_<tag>/<cell>/seed<N>/epoch_*.pkl   -- probe predictions + rollouts

Panels: (a) episode return, (b) episode cost with the limit, (c) Corr(V_hat_r, Gbar^r),
(d) Corr(V_hat_c, Gbar^c).  Lines are the mean over seeds, bands +-1 sd.

Each algorithm is shown at its best cell by (final episodic return) / (total cost),
the final return being the mean over the last FINAL_EPOCHS epochs and the total cost
the summed EpCost over training; --cell overrides the choice.

Usage:
    uv run --python 3.12 --with "numpy<2" --with torch --with matplotlib \
        python MICE/plots/scripts/fig1_plot.py [--cell gae_iters2_lr0.0003]

(torch is needed only to unpickle this sweep's eval_data.)
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

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

ALGOS = [("CPO", "eval_data_lsaldj7r_cpo", "#4c72b0"),
         ("PIDLag", "eval_data_lsaldj7r_pidlag", "#dd8452")]
COST_LIMIT = 10.0
SMOOTH = 5          # running-mean window (epochs) for the return / cost curves
M = 1e6             # x axis is in millions of environment steps
FINAL_EPOCHS = 10   # epochs averaged into the "final" episodic return
EP_PER_EPOCH = 20   # 20k steps per epoch / 1000-step episodes


def history(root: str) -> dict[str, dict[str, np.ndarray]]:
    """Per-cell, per-seed EpRet / EpCost against TotalEnvSteps, from history.csv."""
    per: dict[str, dict[int, list[tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(list))
    with open(os.path.join(root, "history.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            if not row["TotalEnvSteps"]:
                continue
            per[row["cell"]][int(row["seed"])].append((float(row["TotalEnvSteps"]),
                                                       float(row["Metrics/EpRet"]),
                                                       float(row["Metrics/EpCost"])))
    out = {}
    for cell, seeds in per.items():
        runs = [np.array(sorted(v)) for _, v in sorted(seeds.items())]
        n = min(len(r) for r in runs)
        a = np.stack([r[:n] for r in runs])                   # (seeds, epochs, 3)
        out[cell] = {"x": a[0, :, 0], "ret": a[:, :, 1], "cost": a[:, :, 2]}
    return out


def rank(hist: dict[str, dict[str, np.ndarray]]) -> list[tuple[str, float, float, float]]:
    """(cell, final return, total cost, ratio) over the sweep's cells, best ratio first."""
    rows = []
    for cell, h in hist.items():
        ret = h["ret"][:, -FINAL_EPOCHS:].mean()
        tot = h["cost"].sum(1).mean() * EP_PER_EPOCH
        rows.append((cell, ret, tot, ret / (tot / 1e3)))
    return sorted(rows, key=lambda r: -r[3])


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    """Running mean along the last axis, normalised at the edges."""
    if w <= 1:
        return y
    k = np.ones(w)
    num = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), -1, y)
    return num / np.convolve(np.ones(y.shape[-1]), k, "same")


def correlations(root: str, cell: str, block: str) -> dict[str, np.ndarray]:
    """Corr(critic, K-rollout reference) per seed and checkpoint, for r and c."""
    seeds = load_sweep(root, block)[cell].seeds
    out = {"x": np.array([(p.epoch + 1) * 20000 for p in next(iter(seeds.values()))], float)}
    for head in ("r", "c"):
        out[head] = np.array([[np.corrcoef(p.pred[head], p.G[head].mean(1))[0, 1]
                               if p.pred[head].std() > 0 else np.nan for p in pts]
                              for _, pts in sorted(seeds.items())])
    return out


def band(ax, x, y, col, lw=1.6):
    m, sd = np.nanmean(y, 0), np.nanstd(y, 0, ddof=1)
    ax.plot(x, m, color=col, lw=lw)
    ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.18, lw=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=None,
                    help="default: the cell with the best (final EpRet)/(total cost), "
                         "chosen per algorithm")
    ap.add_argument("--block", default="pooled")
    ap.add_argument("--out", default=fig_path("fig1"))
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.0), constrained_layout=True)
    for name, root, col in ALGOS:
        hist = history(root)
        ranked = rank(hist)
        cell = args.cell or ranked[0][0]
        print(f"{name}: {cell}" + ("" if args.cell else
              "  (final return %.2f, total cost %.1fk, ratio %.4f)"
              % (ranked[0][1], ranked[0][2] / 1e3, ranked[0][3])))
        h = hist[cell]
        band(axes[0], h["x"] / M, smooth(h["ret"], SMOOTH), col)
        band(axes[1], h["x"] / M, smooth(h["cost"], SMOOTH), col)
        d = correlations(root, cell, args.block)
        band(axes[2], d["x"] / M, d["r"], col)
        band(axes[3], d["x"] / M, d["c"], col)

    axes[0].set_ylabel("Episode return")
    axes[1].set_ylabel("Episode cost")
    axes[1].axhline(COST_LIMIT, color="0.4", ls="--", lw=1.0)
    axes[1].text(0.03, COST_LIMIT, "limit", color="0.4", fontsize=8,
                 ha="left", va="bottom", transform=axes[1].get_yaxis_transform())
    axes[2].set_ylabel(r"Corr($\hat V_r$, $\bar G^r$)")
    axes[3].set_ylabel(r"Corr($\hat V_c$, $\bar G^c$)")
    for ax in axes[2:]:
        ax.axhline(0.0, color="0.4", ls="--", lw=0.9)
        ax.set_ylim(-0.25, 1.0)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(2e4 / M, 1.05e7 / M)
        ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["0.1", "1", "10"])
        ax.set_xlabel("Environment steps (millions)")
        ax.grid(alpha=0.3, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    titles = ["(a) Episode return", "(b) Episode cost",
              "(c) Reward-value correlation", "(d) Cost-value correlation"]
    for ax, t in zip(axes, titles):
        ax.text(0.5, -0.42, t, style="italic", ha="center", transform=ax.transAxes)
    fig.legend(handles=[Line2D([], [], color=c, lw=2.2, label=n) for n, _, c in ALGOS],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}.pdf / .png  (block {args.block})")


if __name__ == "__main__":
    main()
