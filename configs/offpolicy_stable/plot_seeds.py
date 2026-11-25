"""Five seeds of the no-warmup capped-lambda config on SafetyPointGoal1-v0.

SACLag, cost_limit 10, warmup_epochs 0, lambda_lr 7.5e-6, lagrangian_upper_bound 0.5.
Lambda is live from the first step, starts at 0, and reaches its cap near epoch 33.

The point of the figure is the spread, not the mean: one seed meets the target spec, one
misses only by holding cost just under the band, and three fail in two distinct ways -
never learning, or collapsing with cost running away. The median and interquartile band
are drawn behind the seeds because a mean over a bimodal set describes no run that
happened.

25-epoch moving averages (~50 episodes); a per-epoch value averages only two episodes.
"""
from __future__ import annotations

import csv
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = '/home/kaustubh/projects/calibration_rl/MICE/runs/seeds_nw'
MA, LIMIT, CEILING = 25, 10.0, 20.0
# seed -> (colour, how it ended)
SEEDS = {
    0: ('#1f77b4', 'near-miss: cost under band'),
    1: ('#2ca02c', 'meets spec'),
    2: ('#d62728', 'never learned'),
    3: ('#9467bd', 'collapsed, cost ran away'),
    4: ('#ff7f0e', 'never learned'),
}
SERIES = [('Metrics/EpRet', 'episodic return (MA25)'),
          ('Metrics/EpCost', 'episodic cost (MA25)'),
          ('Metrics/LagrangeMultiplier', r'$\lambda$ (MA25)')]


def load(seed):
    hits = glob.glob(f'{RUNS}/nw7e6_s{seed}/*/seed-*/progress.csv')
    if not hits:
        return None
    rows = list(csv.DictReader(open(hits[0])))
    return {k: np.array([float(r[k]) for r in rows]) for k, _ in SERIES}


def ma(y, w=MA):
    return np.convolve(y, np.ones(w) / w, mode='valid')


data = {s: load(s) for s in SEEDS}
data = {s: d for s, d in data.items() if d is not None}

fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
for ax, (key, ylabel) in zip(axes, SERIES):
    curves = [ma(d[key]) for d in data.values()]
    n = min(len(c) for c in curves)
    stack = np.vstack([c[:n] for c in curves])
    x = np.arange(MA - 1, MA - 1 + n)
    lo, med, hi = np.percentile(stack, [25, 50, 75], axis=0)
    ax.fill_between(x, lo, hi, color='#999999', alpha=0.22, lw=0)
    ax.plot(x, med, color='#444444', lw=2.6, zorder=5,
            label='median of 5 seeds' if key == 'Metrics/EpRet' else None)
    for s, d in data.items():
        colour, note = SEEDS[s]
        ys = ma(d[key])
        ax.plot(np.arange(MA - 1, MA - 1 + len(ys)), ys, color=colour, lw=1.5, alpha=0.95,
                label=f'seed {s} — {note}' if key == 'Metrics/EpRet' else None)
    ax.set_xlabel('epoch (2000 steps each)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 500)

ax = axes[0]
ax.axhspan(3, 10, color='#2ca02c', alpha=0.08)
ax.axhline(CEILING, color='#d62728', ls=':', lw=1.6)
ax.text(497, CEILING + 0.5, 'ceiling 20', ha='right', fontsize=9, color='#d62728')
ax.text(497, 6.4, 'target 3-10', ha='right', fontsize=9, color='#2ca02c')
ax.set_ylim(-6, 24)

ax = axes[1]
ax.axhspan(10, 30, color='#2ca02c', alpha=0.08)
ax.axhline(LIMIT, color='k', ls=':', lw=1.4)
ax.text(497, 31, 'target 10-30', ha='right', fontsize=9, color='#2ca02c')
ax.set_ylim(0, 110)

ax = axes[2]
ax.axhline(0.5, color='#d62728', ls='--', lw=1.4)
ax.text(497, 0.515, 'cap 0.5', ha='right', fontsize=9, color='#d62728')
ax.set_ylim(-0.02, 0.58)
ax.text(20, 0.06, 'bistable: sticks at the cap, or falls to ~0', fontsize=9, color='#555555')

fig.legend(loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.15), fontsize=9)
fig.suptitle('Five seeds, one configuration: 1 of 5 meets the target — the spread is the result',
             fontsize=13, y=1.02)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'/home/kaustubh/projects/calibration_rl/MICE/figs/offpolicy_seeds_nowarmup.{ext}',
                dpi=150, bbox_inches='tight')
print('wrote figs/offpolicy_seeds_nowarmup.{png,pdf}\n')
for s, d in data.items():
    q, c, l = ma(d['Metrics/EpRet']), ma(d['Metrics/EpCost']), ma(d['Metrics/LagrangeMultiplier'])
    print(f'  seed {s}: peak {q.max():5.1f}  final {q[-1]:5.1f} / cost {c[-1]:5.1f} / lam {l[-1]:.2f}  '
          f'({SEEDS[s][1]})')
