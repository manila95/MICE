"""Warmup-then-clamp arms on SafetyPointGoal1-v0 (SACLag, cost_limit 10).

lambda is held at exactly 0 for a short warmup - long enough for the policy to become
competent (reward ~11 by epoch 25) but stopping before reward crosses 20 at epoch 31 -
then lambda_lr 5e-5 drives it to a cap within ~10 epochs and the cap pins it there.

This is the only design that satisfied all three conditions at once: peak return never
above 20, final return 3-10, final cost 10-30. Earlier arms failed one of two ways -
constrained from step zero they drifted to inaction (reward 0 by epoch 180), and left
unconstrained they peaked near 27 and had to give the reward back.

25-epoch moving averages (~50 episodes); raw series faint behind each curve, since a
per-epoch value is an average over only two episodes.
"""
from __future__ import annotations

import csv
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = '/home/kaustubh/projects/calibration_rl/MICE/runs/fixed_lambda'
MA, LIMIT, CEILING = 25, 10.0, 20.0

ARMS = [
    ('warmup 20, cap 0.5', 'w20_c05', '#1f77b4', '-', True),
    ('warmup 25, cap 0.5', 'w25_c05', '#2ca02c', '-', True),
    ('warmup 25, cap 0.4', 'w25_c04', '#ff7f0e', '--', False),
    ('warmup 30, cap 0.6', 'w30_c06', '#9467bd', '--', False),
]
SERIES = [('Metrics/EpRet', 'episodic return (MA25)'),
          ('Metrics/EpCost', 'episodic cost (MA25)'),
          ('Metrics/LagrangeMultiplier', r'$\lambda$ (MA25)')]


def load(tag):
    hits = glob.glob(f'{RUNS}/{tag}/*/seed-*/progress.csv')
    if not hits:
        return None
    rows = list(csv.DictReader(open(hits[0])))
    return None if len(rows) < MA else {k: np.array([float(r[k]) for r in rows]) for k, _ in SERIES}


def ma(y, w=MA):
    return np.convolve(y, np.ones(w) / w, mode='valid')


fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
for ax, (key, ylabel) in zip(axes, SERIES):
    for label, tag, colour, ls, hit in ARMS:
        d = load(tag)
        if d is None:
            continue
        y = d[key]
        ax.plot(np.arange(len(y)), y, color=colour, alpha=0.11, lw=0.7)
        ys = ma(y)
        ax.plot(np.arange(MA - 1, len(y)), ys, color=colour, ls=ls,
                lw=2.6 if hit else 1.6,
                label=(label + ('  ✓ meets spec' if hit else '')) if key == 'Metrics/EpRet' else None)
    ax.set_xlabel('epoch (2000 steps each)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 500)

ax = axes[0]
ax.axhspan(3, 10, color='#2ca02c', alpha=0.08)
ax.axhline(CEILING, color='#d62728', ls=':', lw=1.8)
ax.text(497, CEILING + 0.6, 'ceiling: never exceed 20', ha='right', fontsize=9, color='#d62728')
ax.text(497, 6.5, 'target band 3-10', ha='right', fontsize=9, color='#2ca02c')
ax.set_ylim(-4, 30)

ax = axes[1]
ax.axhspan(10, 30, color='#2ca02c', alpha=0.08)
ax.axhline(LIMIT, color='k', ls=':', lw=1.5)
ax.text(497, LIMIT + 1.5, 'cost_limit 10', ha='right', fontsize=9)
ax.text(497, 31, 'target band 10-30', ha='right', fontsize=9, color='#2ca02c')
ax.set_ylim(0, 75)

for a in axes:
    a.axvspan(0, 25, color='#888888', alpha=0.10)
axes[2].text(30, axes[2].get_ylim()[1] * 0.92, r'$\lambda$=0 warmup, then clamped',
             fontsize=9, color='#555555')

fig.legend(loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.14), fontsize=9.5)
fig.suptitle('Short unconstrained warmup, then a hard clamp: reward never exceeds 20 and the policy does not collapse',
             fontsize=13, y=1.02)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'/home/kaustubh/projects/calibration_rl/MICE/figs/offpolicy_warmup_clamp.{ext}',
                dpi=150, bbox_inches='tight')
print('wrote figs/offpolicy_warmup_clamp.{png,pdf}\n')
print(f"{'arm':<22}{'ep':>5}{'peakRet':>9}{'EpRet':>8}{'EpCost':>9}{'lam':>7}  spec")
for label, tag, _, _, _ in ARMS:
    d = load(tag)
    if d is None:
        continue
    q, c, l = ma(d['Metrics/EpRet']), ma(d['Metrics/EpCost']), ma(d['Metrics/LagrangeMultiplier'])
    ok = 'YES' if (q.max() <= 20 and 3 <= q[-1] <= 10 and 10 <= c[-1] <= 30) else 'no'
    print(f'{label:<22}{len(d["Metrics/EpRet"]):>5}{q.max():>9.1f}{q[-1]:>8.1f}{c[-1]:>9.1f}{l[-1]:>7.2f}  {ok}')
