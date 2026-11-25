"""Mean +/- standard error across seeds, SACLag on SafetyPointGoal1-v0.

Config: cost_limit 10, warmup_epochs 0, lambda_lr 7.5e-6, lagrangian_upper_bound 0.5.
No warmup phase and no cost-limit annealing; lambda is live from the first step.

Each seed is smoothed with a 51-epoch moving average (~100 episodes) before aggregation,
then the band is mean +/- s.e.m. = sd / sqrt(n) over the n seeds at each epoch. Individual
seeds stay visible underneath, because the across-seed distribution here is bimodal: the
mean sits in a gap no single run occupies.
"""
from __future__ import annotations

import csv
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = '/home/kaustubh/projects/calibration_rl/MICE/runs/seeds_nw'
SMOOTH, LIMIT, CEILING = 51, 10.0, 20.0
ACCENT, GREY = '#1f77b4', '#9a9a9a'
SERIES = [('Metrics/EpRet', 'episodic return'),
          ('Metrics/EpCost', 'episodic cost'),
          ('Metrics/LagrangeMultiplier', r'Lagrange multiplier $\lambda$')]


def load(seed):
    hits = glob.glob(f'{RUNS}/nw7e6_s{seed}/*/seed-*/progress.csv')
    if not hits:
        return None
    rows = list(csv.DictReader(open(hits[0])))
    return {k: np.array([float(r[k]) for r in rows]) for k, _ in SERIES}


def smooth(y, w=SMOOTH):
    """Moving average that keeps the series length, so seeds stay aligned by epoch."""
    if len(y) < 3:
        return y
    w = min(w, len(y) if len(y) % 2 else len(y) - 1)
    pad = w // 2
    padded = np.concatenate([np.full(pad, y[:pad].mean()), y, np.full(pad, y[-pad:].mean())])
    return np.convolve(padded, np.ones(w) / w, mode='valid')[:len(y)]


data = {s: d for s in range(5) if (d := load(s)) is not None}
n = len(data)
length = min(len(d['Metrics/EpRet']) for d in data.values())

fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
for ax, (key, ylabel) in zip(axes, SERIES):
    stack = np.vstack([smooth(d[key])[:length] for d in data.values()])
    x = np.arange(length)
    mean = stack.mean(axis=0)
    sem = stack.std(axis=0, ddof=1) / np.sqrt(n)
    for row in stack:
        ax.plot(x, row, color=GREY, lw=1.0, alpha=0.55)
    ax.fill_between(x, mean - sem, mean + sem, color=ACCENT, alpha=0.30, lw=0,
                    label=f'mean $\\pm$ s.e.m. (n={n})' if key == 'Metrics/EpRet' else None)
    ax.plot(x, mean, color=ACCENT, lw=3.0,
            label='mean' if key == 'Metrics/EpRet' else None)
    ax.plot([], [], color=GREY, lw=1.0,
            label='individual seeds' if key == 'Metrics/EpRet' else None)
    ax.set_xlabel('epoch (2000 steps each)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, length)

ax = axes[0]
ax.axhspan(3, 10, color='#2ca02c', alpha=0.08)
ax.axhline(CEILING, color='#d62728', ls=':', lw=1.5)
ax.text(length * 0.99, CEILING + 0.4, 'ceiling 20', ha='right', fontsize=9, color='#d62728')
ax.text(length * 0.99, 6.4, 'target 3-10', ha='right', fontsize=9, color='#2ca02c')
ax.set_ylim(-6, 23)

ax = axes[1]
ax.axhspan(10, 30, color='#2ca02c', alpha=0.08)
ax.axhline(LIMIT, color='k', ls=':', lw=1.4)
ax.text(length * 0.99, 32, 'target 10-30', ha='right', fontsize=9, color='#2ca02c')
ax.set_ylim(0, 105)

ax = axes[2]
ax.axhline(0.5, color='#d62728', ls='--', lw=1.4)
ax.text(length * 0.99, 0.512, 'cap 0.5', ha='right', fontsize=9, color='#d62728')
ax.set_ylim(-0.02, 0.57)

fig.legend(loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=10)
fig.suptitle(f'SACLag, no warmup, capped $\\lambda$: mean $\\pm$ s.e.m. over {n} seeds '
             f'({SMOOTH}-epoch moving average)', fontsize=13, y=1.02)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'/home/kaustubh/projects/calibration_rl/MICE/figs/offpolicy_seeds_mean_sem.{ext}',
                dpi=150, bbox_inches='tight')
print(f'wrote figs/offpolicy_seeds_mean_sem.{{png,pdf}}  (n={n}, {length} epochs)\n')
for key, label in SERIES:
    stack = np.vstack([smooth(d[key])[:length] for d in data.values()])
    m, s = stack[:, -1].mean(), stack[:, -1].std(ddof=1) / np.sqrt(n)
    print(f'  final {label:<28} {m:>7.2f} +/- {s:.2f} (s.e.m.)')
