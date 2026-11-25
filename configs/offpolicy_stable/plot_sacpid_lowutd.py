"""SACPID against SACLag at the same low update-to-data ratio, SafetyPointGoal1-v0.

Both run 5 parallel environments with update_cycle 2 — one gradient step per ten
environment steps — at cost_limit 10 with the multiplier live from the first step.

The question is whether PID survives here. Its original failure on this task was lambda
reaching 629: the integral term accumulates once per _update(), and update_cycle=1 calls
that 2000 times an epoch. At UTD 0.10 it accumulates 200 times instead. The scaled arm
raises the gains tenfold so the multiplier's trajectory against environment steps matches
the per-step schedule; the stock arm leaves them, so its dual also moves ten times slower.

Trailing 41-epoch means, the same reduction used throughout this work: a centred average
has to invent values past the end, which badly distorts a series still swinging.
"""
from __future__ import annotations

import csv
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = '/home/kaustubh/projects/calibration_rl/MICE/runs'
SMOOTH, LIMIT = 41, 10.0
ARMS = [
    ('SACLag — 5 envs, uc 2 (reference)', f'{RUNS}/update_cycle/v5uc2', '#2ca02c', '-', 2.8),
    ('SACPID — $\\lambda$ matched to SACLag', f'{RUNS}/sacpid_lowutd/pid_match', '#1f77b4', '-', 2.4),
    ('SACPID — $\\lambda$ matched + real P/D', f'{RUNS}/sacpid_lowutd/pid_match_p', '#17becf', '-', 2.4),
    ('SACPID — gains x10 (slow $\\lambda$)', f'{RUNS}/sacpid_lowutd/pid_v5uc2', '#d62728', '--', 1.8),
    ('SACPID — stock gains (slowest)', f'{RUNS}/sacpid_lowutd/pid_v5uc2_g', '#ff7f0e', '--', 1.8),
    ('SACPID — the original failure, uc 1', f'{RUNS}/sacpid_5M/seed2', '#9467bd', ':', 1.6),
]
SERIES = [('Metrics/EpRet', 'episodic return'),
          ('Metrics/EpCost', 'episodic cost'),
          ('Metrics/LagrangeMultiplier', r'Lagrange multiplier $\lambda$')]


def load(base):
    hits = glob.glob(f'{base}/*/seed-*/progress.csv') or glob.glob(f'{base}/*/*/seed-*/progress.csv')
    if not hits:
        return None
    rows = list(csv.DictReader(open(hits[0])))
    if len(rows) < 10:
        return None
    out = {}
    for k, _ in SERIES:
        v = np.array([float(r[k]) for r in rows])
        if np.isnan(v).any():          # a 5-env run can log NaN before any episode ends
            idx = np.where(~np.isnan(v))[0]
            v = np.interp(np.arange(len(v)), idx, v[idx]) if len(idx) else np.zeros_like(v)
        out[k] = v
    return out


def smooth(y, w=SMOOTH):
    return np.array([y[max(0, i - w + 1):i + 1].mean() for i in range(len(y))])


data = {label: load(base) for label, base, *_ in ARMS}
longest = max((len(d['Metrics/EpRet']) for d in data.values() if d), default=1500)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
for ax, (key, ylabel) in zip(axes, SERIES):
    for label, base, colour, ls, lw in ARMS:
        d = data.get(label)
        if d is None:
            continue
        ys = smooth(d[key])
        ax.plot(np.arange(len(ys)), ys, color=colour, ls=ls, lw=lw,
                label=label if key == 'Metrics/EpRet' else None)
    ax.set_xlabel('epoch (2000 steps each)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, longest)

axes[0].axhline(0, color='k', lw=0.8, ls=':')
axes[0].set_ylim(-10, 30)
axes[1].axhline(LIMIT, color='k', ls=':', lw=1.4)
axes[1].text(longest * 0.99, LIMIT + 3, 'cost_limit 10', ha='right', fontsize=9)
axes[1].set_ylim(0, 110)
axes[2].set_yscale('symlog', linthresh=1)
axes[2].axhline(0.5, color='#555555', ls='--', lw=1.2)
axes[2].text(longest * 0.02, 120, 'log scale: the uc 1 run reached 629; dashed line is the 0.5 cap',
             fontsize=8.5, color='#9467bd')

fig.legend(loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13), fontsize=9)
fig.suptitle('SACPID at SACLag\'s update rate, with and without SACLag\'s multiplier schedule', fontsize=13, y=1.02)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'/home/kaustubh/projects/calibration_rl/MICE/figs/sacpid_lowutd.{ext}',
                dpi=150, bbox_inches='tight')
print('wrote figs/sacpid_lowutd.{png,pdf}\n')
print(f"{'arm':<42}{'ep':>6}{'peak':>8}{'EpRet':>8}{'EpCost':>9}{'lambda':>9}")
for label, base, *_ in ARMS:
    d = data.get(label)
    if d is None:
        print(f'{label:<42}{"(no data)":>40}'); continue
    q, c, l = (smooth(d[k]) for k, _ in SERIES)
    print(f'{label:<42}{len(q):>6}{q.max():>8.1f}{q[-1]:>8.1f}{c[-1]:>9.1f}{l[-1]:>9.2f}')
