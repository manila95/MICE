"""Five seeds at 3M steps, in progress: per-seed curves and the across-seed mean.

SACLag on SafetyPointGoal1-v0: cost_limit 10, warmup_epochs 0, lambda_lr 7.5e-6,
lagrangian_upper_bound 0.5. No warmup phase, no cost-limit annealing.

Both views are drawn because neither alone is honest here. The across-seed distribution
is bimodal - two seeds track the target band and three sit near zero - so the mean lies
in a gap no run occupies, while the per-seed panel hides how the aggregate is moving.
The dashed line marks where the earlier 1M runs stopped.
"""
from __future__ import annotations

import csv
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = '/home/kaustubh/projects/calibration_rl/MICE/runs/seeds_nw3M'
SMOOTH, LIMIT, CEILING, STOP_1M = 41, 10.0, 20.0, 500
COLOURS = {0: '#1f77b4', 1: '#2ca02c', 2: '#d62728', 3: '#9467bd', 4: '#ff7f0e'}
SERIES = [('Metrics/EpRet', 'episodic return'),
          ('Metrics/EpCost', 'episodic cost'),
          ('Metrics/LagrangeMultiplier', r'Lagrange multiplier $\lambda$')]


def load(seed):
    hits = glob.glob(f'{RUNS}/nw7e6_s{seed}/*/seed-*/progress.csv')
    if not hits:
        return None
    rows = list(csv.DictReader(open(hits[0])))
    return None if len(rows) < SMOOTH else {k: np.array([float(r[k]) for r in rows]) for k, _ in SERIES}


def smooth(y, w=SMOOTH):
    w = min(w, len(y) if len(y) % 2 else len(y) - 1)
    pad = w // 2
    padded = np.concatenate([np.full(pad, y[:pad].mean()), y, np.full(pad, y[-pad:].mean())])
    return np.convolve(padded, np.ones(w) / w, mode='valid')[:len(y)]


data = {s: d for s in range(5) if (d := load(s)) is not None}
n = len(data)
length = min(len(d['Metrics/EpRet']) for d in data.values())

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for col, (key, ylabel) in enumerate(SERIES):
    stack = np.vstack([smooth(d[key])[:length] for d in data.values()])
    x = np.arange(length)

    top = axes[0, col]
    for s, d in data.items():
        ys = smooth(d[key])
        top.plot(np.arange(len(ys)), ys, color=COLOURS[s], lw=1.8,
                 label=f'seed {s}' if col == 0 else None)
    top.set_ylabel(ylabel + '\n(per seed)')

    bot = axes[1, col]
    mean = stack.mean(axis=0)
    sem = stack.std(axis=0, ddof=1) / np.sqrt(n)
    bot.fill_between(x, mean - sem, mean + sem, color='#1f77b4', alpha=0.28, lw=0,
                     label=f'mean $\\pm$ s.e.m. (n={n})' if col == 0 else None)
    bot.plot(x, mean, color='#1f77b4', lw=3.0, label='mean' if col == 0 else None)
    bot.set_ylabel(ylabel + '\n(across seeds)')

    for ax in (top, bot):
        ax.axvline(STOP_1M, color='#777777', ls='--', lw=1.2)
        ax.set_xlabel('epoch (2000 steps each)')
        ax.grid(alpha=0.25)
        ax.set_xlim(0, max(length, 600))
        if key == 'Metrics/EpRet':
            ax.axhspan(3, 10, color='#2ca02c', alpha=0.08)
            ax.axhline(CEILING, color='#d62728', ls=':', lw=1.4)
            ax.set_ylim(-8, 24)
        elif key == 'Metrics/EpCost':
            ax.axhspan(10, 30, color='#2ca02c', alpha=0.08)
            ax.axhline(LIMIT, color='k', ls=':', lw=1.3)
            ax.set_ylim(0, 110)
        else:
            ax.axhline(0.5, color='#d62728', ls='--', lw=1.3)
            ax.set_ylim(-0.02, 0.57)

axes[0, 0].text(STOP_1M - 10, 21.5, 'earlier 1M runs stopped here', ha='right',
                fontsize=8.5, color='#777777')
axes[0, 2].text(20, 0.06, 'bistable: pinned at the cap, or collapsed to ~0',
                fontsize=8.5, color='#555555')
handles = [h for ax in (axes[0, 0], axes[1, 0]) for h in ax.get_legend_handles_labels()[0]]
labels = [l for ax in (axes[0, 0], axes[1, 0]) for l in ax.get_legend_handles_labels()[1]]
fig.legend(handles, labels, loc='lower center', ncol=7, frameon=False,
           bbox_to_anchor=(0.5, -0.04), fontsize=9.5)
fig.suptitle(f'SACLag, no warmup, capped $\\lambda$ — 5 seeds at epoch {length} of 1500 '
             f'({SMOOTH}-epoch moving average)', fontsize=13, y=1.0)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'/home/kaustubh/projects/calibration_rl/MICE/figs/offpolicy_seeds_3m.{ext}',
                dpi=150, bbox_inches='tight')
print(f'wrote figs/offpolicy_seeds_3m.{{png,pdf}}  (n={n}, epoch {length} of 1500)\n')
for key, label in SERIES:
    stack = np.vstack([smooth(d[key])[:length] for d in data.values()])
    m, s = stack[:, -1].mean(), stack[:, -1].std(ddof=1) / np.sqrt(n)
    print(f'  {label:<30} {m:>7.2f} +/- {s:.2f}')
