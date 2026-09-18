"""Figure 2: what the cost critic of Figure 1 gets right and wrong.

Sweep lsaldj7r, configuration gae_iters2_lr0.0001, 5 seeds, 102 probe states per
checkpoint, reference Gbar = mean of K=10 rollouts (simulation only, no critic value
folded into the tail).  Value groups on Gbar: low value < gamma^500 <= intermediate
value < gamma^5 <= high value.

Per algorithm, with the reference axis symlog and linear
(figures/fig2_cpo[_linear], figures/fig2_pidlag[_linear]):
  (a) critic vs reference at one descent-phase checkpoint, coloured by true value group
  (b) signed error V_hat - Gbar over all probes, across training
  (c) the same, grouped by true value
  (d) the critic as a binary classifier: AUROC for high value (Gbar >= gamma^5) against
      everything below it, across training, with the split-half ceiling dotted
Both algorithms together:
  figures/fig2_threshold -- AUROC against the threshold v defining a high-value state,
      before and after the cost reaches the limit
  figures/fig2_groups -- how the probe states split into the three value groups over
      training, one stacked panel per algorithm
Lines are the mean over 5 seeds, bands +-1 standard error; a probe-weighting check
(strata reweighted to a uniform distribution over episode steps) is printed.

Usage:
    uv run --python 3.12 --with "numpy<2" --with torch --with matplotlib \
        python MICE/plots/scripts/fig2_plot.py
"""

from __future__ import annotations

import argparse
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

from common.metrics import GAMMA, SPE, auroc, beta_crossing
from common.paths import FIG_DIR
from common.sweep_data import load_sweep

CELL = "gae_iters2_lr0.0001"               # the configuration plotted in Figure 1
ALGOS = [("CPO", "eval_data_lsaldj7r_cpo", "#4c72b0", "cpo"),
         ("PIDLag", "eval_data_lsaldj7r_pidlag", "#dd8452", "pidlag")]
T_LOW, T_HIGH = GAMMA ** 500, GAMMA ** 5
# not the algorithm colours: those are blue and orange in every figure of the paper
GROUPS = [("low value", "#bdbdbd"), ("intermediate value", "#9e9ac8"), ("high value", "#c44e52")]
DEFS = {"low value": r"$\tilde{V}_c < \gamma^{500}$",
        "intermediate value": r"$\gamma^{500} \leq \tilde{V}_c < \gamma^{5}$",
        "high value": r"$\tilde{V}_c \geq \gamma^{5}$"}
SCATTER_EPOCH = 10                         # 0.22M steps, before either algorithm reaches the limit
# high value = Gbar >= v, from a value far above gamma^0 = 1 down to gamma^500
THRESH_V = [4.0, 2.0, 1.0] + [GAMMA ** t for t in (5, 10, 25, 50, 100, 200, 500)]
THRESH_LAB = ["4", "2", r"$1{=}\gamma^0$", r"$\gamma^5$", r"$\gamma^{10}$", r"$\gamma^{25}$",
              r"$\gamma^{50}$", r"$\gamma^{100}$", r"$\gamma^{200}$", r"$\gamma^{500}$"]
TICKS, TICKLABELS = [3e4, 1e5, 3e5, 1e6, 3e6, 1e7], ["30k", "100k", "300k", "1M", "3M", "10M"]
# strata t = 0, 100, 300, 500, 700, 900 stand for episode steps [0,50), [50,200), [200,400), ...
STRATUM_W = np.array([50, 150, 200, 200, 200, 200]) / 1000
LINTHRESH = 0.01     # symlog: below this the scatter axes are linear, so zero is drawable


def group(v: np.ndarray) -> list[np.ndarray]:
    return [v < T_LOW, (v >= T_LOW) & (v < T_HIGH), v >= T_HIGH]


def wauroc(score, label, w):
    """AUROC with per-probe weights, for the step-weighting check."""
    pos, neg = np.where(label)[0], np.where(~label)[0]
    if not len(pos) or not len(neg):
        return np.nan
    d = score[pos][:, None] - score[neg][None, :]
    W = w[pos][:, None] * w[neg][None, :]
    return float((((d > 0) + 0.5 * (d == 0)).astype(float) * W).sum() / W.sum())


def point(pred, G, weighted=False):
    """Every per-checkpoint statistic the figure draws or prints."""
    gbar = G.mean(1)
    n, K = G.shape
    w = np.repeat(STRATUM_W / (n // 6), n // 6) if weighted else np.ones(n) / n
    e = pred - gbar
    out = {"auroc": wauroc(pred, gbar >= T_HIGH, w) if weighted else auroc(pred, gbar >= T_HIGH),
           "signed_all": float((w * e).sum() / w.sum()),
           "neg": float((w * (pred < 0)).sum() / w.sum())}
    h1, h2 = G[:, :K // 2].mean(1), G[:, K // 2:].mean(1)
    out["auroc_ceil"] = np.nanmean([auroc(h1, h2 >= T_HIGH), auroc(h2, h1 >= T_HIGH)])
    for tag, masks in (("true", group(gbar)), ("pred", group(pred))):
        for (g, _), m in zip(GROUPS, masks):
            out[f"{tag}_{g}"] = float((w[m] * e[m]).sum() / w[m].sum()) if m.any() else np.nan
    for (g, _), m in zip(GROUPS, group(gbar)):
        out[f"share_{g}"] = float(w[m].sum() / w.sum())
    for i, v in enumerate(THRESH_V):
        lab = gbar >= v
        out[f"auroc_v{i}"] = wauroc(pred, lab, w) if weighted else auroc(pred, lab)
        out[f"prev_v{i}"] = float((w * lab).sum() / w.sum())
    var_v = gbar.var(ddof=1) - G.var(1, ddof=1).mean() / K
    cov = np.cov(pred, gbar)[0, 1]
    out["s"] = float(np.sqrt(pred.var(ddof=1) / var_v)) if var_v > 0 else np.nan
    out["beta"] = float(cov / pred.var(ddof=1)) if pred.var() > 0 else np.nan
    return out


def logx(ax):
    ax.set_xscale("log"); ax.set_xlim(1.5e4, 1.03e7)
    ax.xaxis.set_major_locator(FixedLocator(TICKS)); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(TICKLABELS); ax.set_xlabel("Total env steps")


def band(ax, x, M, col, lw=1.6, ls="-", label=None):
    """Seed mean with a +-1 standard-error band.

    The full range over 5 seeds, which this used to draw, is dominated by whichever
    seed is most extreme at each checkpoint and buries the curves it surrounds.
    """
    m = np.nanmean(M, 0)
    se = np.nanstd(M, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(M), 0))
    ax.plot(x, m, color=col, lw=lw, ls=ls, label=label)
    ax.fill_between(x, m - se, m + se, color=col, alpha=0.18, lw=0)


def algo_figure(name, d, col, out, linear=False):
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), constrained_layout=True)
    x, P = d["x"], d["P"]              # every checkpoint, epoch 0 (20k steps) included

    ax = axes[0]
    pts = d["scatter"]
    pred = np.concatenate([p.pred["c"] for p in pts]); gbar = np.concatenate([p.G["c"].mean(1) for p in pts])
    for (g, gc), m in reversed(list(zip(GROUPS, group(gbar)))):
        ax.scatter(gbar[m], pred[m], s=12, color=gc, alpha=0.7, edgecolors="none", label=f"{g}: {DEFS[g]}")
    px = 0.05 * np.ptp(gbar)
    xl = (gbar.min() - px, gbar.max() + px)
    xs = np.linspace(*xl, 800)          # y = x sampled densely: symlog y bends the identity line
    ax.plot(xs, xs, color="0.3", ls="--", lw=1, label="perfect prediction")
    if linear:
        py = 0.05 * np.ptp(pred)
        ax.set_xlim(*xl)
        ax.set_ylim(pred.min() - py, pred.max() + py)
    else:
        # symlog on both axes: the critic predicts negative values, which no log axis
        # takes, and the reference is zero on ~40% of the probes
        ax.set_xscale("symlog", linthresh=LINTHRESH)
        ax.set_yscale("symlog", linthresh=LINTHRESH)
        # a simulation-only reference is non-negative, so don't waste a decade below zero
        ax.set_xlim(min(gbar.min() * 1.3, -0.02) if gbar.min() < 0 else -0.003, gbar.max() * 1.4)
        ax.set_ylim(pred.min() * 1.3, pred.max() * 1.4)
    ax.set_xlabel(r"Reference $\tilde V_c$ (10 rollouts)" + ("" if linear else ", symlog"))
    ax.set_ylabel(r"Cost critic prediction $\hat V_c$" + ("" if linear else ", symlog"))
    ax.set_title(f"Critic vs reference, {(SCATTER_EPOCH + 1) * SPE / 1e6:.2f}M steps")
    h_, l_ = ax.get_legend_handles_labels()
    order = [l_.index(f"{g}: {DEFS[g]}") for g, _ in GROUPS] + [l_.index("perfect prediction")]
    ax.legend([h_[i] for i in order], [l_[i] for i in order], fontsize=7.5, frameon=False,
              loc="lower right")

    ax = axes[1]
    band(ax, x, np.array([[p["signed_all"] for p in s_] for s_ in P]), col)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_ylabel(r"$\hat V_c - \tilde{V}_c^{\,c}$, all probes")
    ax.set_title("Signed error, all probe states")

    ax = axes[2]
    for g, gc in GROUPS:
        band(ax, x, np.array([[p[f"true_{g}"] for p in s_] for s_ in P]), gc, label=g)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_ylabel(r"$\hat V_c - \tilde{V}_c^{\,c}$")
    ax.set_title("Signed error by true value group")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = axes[3]
    band(ax, x, np.array([[p["auroc"] for p in s_] for s_ in P]), col)
    ax.plot(x, np.nanmean([[p["auroc_ceil"] for p in s_] for s_ in P], 0), color=col, lw=1.1, ls=":")
    ax.axhline(0.5, color="0.3", lw=1); ax.set_ylim(0.3, 1.02)
    ax.set_ylabel(r"AUROC (Mann-Whitney), high value $= \tilde{V}_c \geq \gamma^5$")
    ax.set_title("The critic as a binary classifier")
    ax.legend(handles=[Line2D([], [], color=col, lw=1.6, label="critic"),
                       Line2D([], [], color="0.3", lw=1.1, ls=":", label="split-half ceiling")],
              fontsize=8, frameon=False, loc="lower right")

    for ax in axes[1:]:
        logx(ax)
        ax.axvline(d["x0"], color="0.4", ls="-.", lw=0.9)
    for ax in axes:
        ax.grid(alpha=0.3, lw=0.6)
    fig.suptitle(f"{name} on PointGoal1: cost critic quality on its own probe states "
                 f"(5 seeds; mean $\\pm$ 1 s.e.; dash-dot: cost reaches the limit)", fontsize=10.5, color=col)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def threshold_figure(D, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    for name, _, col, _ in ALGOS:
        d = D[name]
        for test, ls in ((lambda x: x < d["x0"], "-"), (lambda x: x >= d["x0"], "--")):
            sel = [(s_, t) for s_ in range(len(d["P"])) for t in range(len(d["x"])) if test(d["x"][t])]
            V = np.array([[d["P"][s_][t][f"auroc_v{i}"] for i in range(len(THRESH_V))] for s_, t in sel])
            ax.plot(THRESH_V, np.nanmedian(V, 0), color=col, ls=ls, lw=1.6, marker="o", ms=3)
            ax.fill_between(THRESH_V, np.nanpercentile(V, 25, 0), np.nanpercentile(V, 75, 0),
                            color=col, alpha=0.12, lw=0)
    ax.axhline(0.5, color="0.3", lw=1); ax.set_ylim(0.3, 1.02)
    shown = [0, 1, 2, 6, 8, 9]
    ax.set_xscale("log"); ax.set_xticks([THRESH_V[i] for i in shown])
    ax.set_xticklabels([THRESH_LAB[i] for i in shown], fontsize=8)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"threshold $v$: high value $= \tilde{V}_c \geq v$")
    ax.set_ylabel("AUROC (median, IQR over seeds and checkpoints)")
    ax.set_title("The critic as a binary classifier: threshold sensitivity", fontsize=10.5)
    ax.grid(alpha=0.3, lw=0.6)
    ax.legend(handles=[Line2D([], [], color=c, lw=1.6, label=n) for n, _, c, _ in ALGOS] +
              [Line2D([], [], color="0.3", lw=1.6, ls="-", label="before cost reaches limit"),
               Line2D([], [], color="0.3", lw=1.6, ls="--", label="after")],
              fontsize=8, frameon=False, loc="lower left")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def composition_figure(D, out):
    """How the probe states split into value groups as training proceeds."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True, sharey=True)
    for ax, (name, _, _, _) in zip(axes, ALGOS):
        d = D[name]
        S = np.array([[[p[f"share_{g}"] for g, _ in GROUPS] for p in s_] for s_ in d["P"]])
        ax.stackplot(d["x"], S.mean(0).T, colors=[c for _, c in GROUPS],
                     labels=[f"{g}: {DEFS[g]}" for g, _ in GROUPS], edgecolor="none")
        ax.axvline(d["x0"], color="0.2", ls="-.", lw=1.1)
        ax.set_xscale("log")
        ax.set_xlim(d["x"][0], 1.03e7)
        ax.xaxis.set_major_locator(FixedLocator(TICKS)); ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(TICKLABELS)
        ax.set_xlabel("Total env steps")
        ax.set_ylim(0, 1)
        ax.set_title(name)
    axes[0].set_ylabel("Share of probe states")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=8.5, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("Value groups of the probe states over training "
                 "(mean over 5 seeds; dash-dot: cost reaches the limit)", fontsize=10.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(FIG_DIR))
    args = ap.parse_args()

    D = {}
    for name, root, col, slug in ALGOS:
        runs = [pts for _, pts in sorted(load_sweep(root, "pooled")[CELL].seeds.items())]
        D[name] = {"x": np.array([(p.epoch + 1) * SPE for p in runs[0]], float),
                   "x0": beta_crossing(root, CELL), "col": col, "slug": slug,
                   "P": [[point(p.pred["c"], p.G["c"]) for p in pts] for pts in runs],
                   "PW": [[point(p.pred["c"], p.G["c"], weighted=True) for p in pts] for pts in runs],
                   "scatter": [next(p for p in pts if p.epoch == SCATTER_EPOCH) for pts in runs]}
        print(f"{name}: cost reaches the limit at {D[name]['x0'] / 1e6:.2f}M steps; "
              f"scatter at {(SCATTER_EPOCH + 1) * SPE / 1e6:.2f}M")

    os.makedirs(args.outdir, exist_ok=True)
    for name, _, col, slug in ALGOS:
        algo_out = os.path.join(args.outdir, f"fig2_{slug}")
        algo_figure(name, D[name], col, algo_out)
        algo_figure(name, D[name], col, f"{algo_out}_linear", linear=True)
    threshold_figure(D, os.path.join(args.outdir, "fig2_threshold"))
    composition_figure(D, os.path.join(args.outdir, "fig2_groups"))

    keys = (["auroc", "signed_all", "neg", "s", "beta"]
            + [f"{t}_{g}" for t in ("true", "pred") for g, _ in GROUPS])
    for name, d in D.items():
        for phase, test in (("descent", lambda x: x < d["x0"]),
                            ("after", lambda x: x >= d["x0"])):
            m = [t for t in range(len(d["x"])) if test(d["x"][t])]
            for lab, P in (("unweighted", d["P"]), ("step-weighted", d["PW"])):
                vals = {k: np.nanmedian([P[s][t][k] for s in range(len(P)) for t in m]) for k in keys}
                print(f"{name:>6} {phase:>7} {lab:>13}: "
                      + "  ".join(f"{k} {v:+.2f}" for k, v in vals.items()))
            print(f"{name:>6} {phase:>7} AUROC by threshold (share high value): " + "  ".join(
                f"v={v:.3g}: {np.nanmedian([d['P'][s][t][f'auroc_v{i}'] for s in range(len(d['P'])) for t in m]):.2f} "
                f"({np.median([d['P'][s][t][f'prev_v{i}'] for s in range(len(d['P'])) for t in m]):.2f})"
                for i, v in enumerate(THRESH_V)))
    print(f"\nwrote {args.outdir}/fig2_{{cpo,pidlag}}[_linear], fig2_threshold, fig2_groups "
          "(.pdf/.png)")


if __name__ == "__main__":
    main()
