"""Two λ schedules on SafetyPointGoal1-v0, SACLag, cost_limit 10.

Phase 1 starts at λ=0 and lets λ rise: reward forms first, the constraint is violated for
most of the run, and reward decays as λ climbs. Phase 2 starts at λ=init and lets λ decay:
the constraint holds early, but reward never forms. The question the entropy arms test is
whether that second failure is caused by exploration collapse under a dominant cost term.
"""
from __future__ import annotations

import csv
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = '/home/kaustubh/projects/calibration_rl/MICE/runs/offpolicy_3M'
CAP, WINDOW, LIMIT = 1000, 25, 10.0

ARMS = [
    # label, dir, colour, linestyle, linewidth
    ('phase 1: $\\lambda_0$=0, rising  (reward first)', 'saclag_cl10', '#1f77b4', '-', 2.6),
    ('$\\lambda_0$=2, lr 2e-6', 'init2_lr2e6', '#d62728', '--', 1.6),
    ('$\\lambda_0$=5, lr 5e-6', 'init5_lr5e6', '#ff7f0e', '--', 1.6),
    ('$\\lambda_0$=5, lr 5e-7 (static)', 'init5_lr5e7', '#8c564b', '--', 1.6),
    ('$\\lambda_0$=10, lr 1e-5', 'init10_lr1e5', '#e377c2', '--', 1.6),
    ('$\\lambda_0$=5 + alpha 0.2', 'init5_alpha02', '#2ca02c', '-', 2.2),
    ('$\\lambda_0$=5 + auto alpha', 'init5_autoalph', '#17becf', '-', 2.2),
    ('$\\lambda_0$=5 + alpha .05, warm 250', 'init5_a005_w250', '#9467bd', '-', 2.2),
]
SERIES = [('Metrics/EpRet', 'episodic return'),
          ('Metrics/EpCost', 'episodic cost'),
          ('Metrics/LagrangeMultiplier', r'Lagrange multiplier $\lambda$')]


def load(tag):
    hits = glob.glob(f'{RUNS}/{tag}/*/seed-*/progress.csv')
    if not hits:
        return None
    rows = list(csv.DictReader(open(hits[0])))[:CAP]
    if len(rows) < 5:
        return None
    return {k: np.array([float(r[k]) for r in rows]) for k, _ in SERIES}


def smooth(y, w=WINDOW):
    return y if len(y) < w else np.convolve(y, np.ones(w) / w, mode='valid')


fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
for ax, (key, ylabel) in zip(axes, SERIES):
    for label, tag, colour, ls, lw in ARMS:
        d = load(tag)
        if d is None:
            continue
        y = d[key]
        ax.plot(np.arange(len(y)), y, color=colour, alpha=0.10, lw=0.7)
        ys = smooth(y)
        ax.plot(np.arange(len(y) - len(ys), len(y)), ys, color=colour, ls=ls, lw=lw,
                label=label if key == 'Metrics/EpRet' else None)
    ax.set_xlabel('epoch (2000 steps each)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, CAP)

axes[0].axhline(0, color='k', lw=0.8, ls=':')
axes[0].set_ylim(-25, 32)
axes[1].axhline(LIMIT, color='k', ls=':', lw=1.5)
axes[1].text(CAP * 0.99, LIMIT + 1.5, 'cost_limit 10', ha='right', fontsize=9)
axes[1].set_ylim(0, 85)
fig.legend(loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.21), fontsize=8.8)
fig.suptitle('Starting $\\lambda$ high satisfies the constraint early but reward never forms',
             fontsize=13, y=1.02)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'/home/kaustubh/projects/calibration_rl/MICE/figs/offpolicy_lambda_init.{ext}',
                dpi=150, bbox_inches='tight')
print('wrote figs/offpolicy_lambda_init.{png,pdf}\n')

print(f"{'arm':<40}{'ep':>5}{'EpRet':>8}{'EpCost':>9}{'lam':>8}{'viol%':>7}")
for label, tag, *_ in ARMS:
    d = load(tag)
    if d is None:
        print(f'{label:<40}{"(too early)":>29}')
        continue
    n = len(d['Metrics/EpRet'])
    m = lambda k: d[k][-50:].mean()
    viol = 100 * (d['Metrics/EpCost'] > LIMIT).mean()
    print(f'{label:<40}{n:>5}{m("Metrics/EpRet"):>8.2f}{m("Metrics/EpCost"):>9.1f}'
          f'{m("Metrics/LagrangeMultiplier"):>8.2f}{viol:>7.0f}')
