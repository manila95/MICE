"""The lambda-initialisation sweep on its own: SACLag, SafetyPointGoal1-v0, cost_limit 10.

Solid lines vary only lambda_0 (and the decay rate needed to unwind it over the run),
shaded light to dark with lambda_0. Dashed lines hold lambda_0=5 and vary the entropy
coefficient instead, testing whether exploration collapse is what kills reward.
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
BLUES = plt.get_cmap('Blues')

INIT = [   # lambda_0 sweep: one hue, darker with larger lambda_0
    (r'$\lambda_0$=2, lr 2e-6', 'init2_lr2e6', BLUES(0.42)),
    (r'$\lambda_0$=5, lr 5e-6', 'init5_lr5e6', BLUES(0.62)),
    (r'$\lambda_0$=5, lr 5e-7 (near-static)', 'init5_lr5e7', BLUES(0.80)),
    (r'$\lambda_0$=10, lr 1e-5', 'init10_lr1e5', BLUES(0.97)),
]
ENTROPY = [   # lambda_0 = 5 throughout; only the entropy coefficient changes
    (r'$\lambda_0$=5, alpha 0.2', 'init5_alpha02', '#2ca02c'),
    (r'$\lambda_0$=5, auto alpha', 'init5_autoalph', '#17becf'),
    (r'$\lambda_0$=5, alpha 0.05, warm 250', 'init5_a005_w250', '#9467bd'),
]
SERIES = [('Metrics/EpRet', 'episodic return'),
          ('Metrics/EpCost', 'episodic cost'),
          ('Metrics/LagrangeMultiplier', r'Lagrange multiplier $\lambda$')]


def load(tag):
    hits = glob.glob(f'{RUNS}/{tag}/*/seed-*/progress.csv')
    if not hits:
        return None
    rows = list(csv.DictReader(open(hits[0])))[:CAP]
    return None if len(rows) < 5 else {k: np.array([float(r[k]) for r in rows]) for k, _ in SERIES}


def smooth(y, w=WINDOW):
    return y if len(y) < w else np.convolve(y, np.ones(w) / w, mode='valid')


fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
for ax, (key, ylabel) in zip(axes, SERIES):
    for group, ls, lw in ((INIT, '-', 2.3), (ENTROPY, '--', 1.8)):
        for label, tag, colour in group:
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

ax = axes[0]
ax.axhline(0, color='k', lw=0.9, ls=':')
ax.set_ylim(-22, 30)
ax.annotate('reward the same schedule reaches\nwith $\\lambda_0$=0 (peak 27)', xy=(210, 27),
            xytext=(330, 20), fontsize=8.5, color='#999999',
            arrowprops=dict(arrowstyle='->', color='#999999', lw=1.1))
ax.plot([0, 1000], [27, 27], color='#999999', lw=1.1, ls=':')
axes[1].axhline(LIMIT, color='k', ls=':', lw=1.5)
axes[1].text(CAP * 0.99, LIMIT + 1.8, 'cost_limit 10', ha='right', fontsize=9)
axes[1].set_ylim(0, 70)
fig.legend(loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.17), fontsize=8.8)
fig.suptitle(r'Initialising $\lambda$ high: every $\lambda_0$ holds the constraint, none lets reward form',
             fontsize=13, y=1.02)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'/home/kaustubh/projects/calibration_rl/MICE/figs/offpolicy_init_sweep.{ext}',
                dpi=150, bbox_inches='tight')
print('wrote figs/offpolicy_init_sweep.{png,pdf}\n')
print(f"{'arm':<38}{'ep':>5}{'EpRet':>8}{'EpCost':>9}{'lam now':>9}{'viol%':>7}")
for label, tag, _ in INIT + ENTROPY:
    d = load(tag)
    if d is None:
        print(f'{label:<38}{"too early":>29}'); continue
    m = lambda k: d[k][-50:].mean()
    print(f'{label:<38}{len(d["Metrics/EpRet"]):>5}{m("Metrics/EpRet"):>8.2f}'
          f'{m("Metrics/EpCost"):>9.1f}{m("Metrics/LagrangeMultiplier"):>9.2f}'
          f'{100*(d["Metrics/EpCost"]>LIMIT).mean():>7.0f}')
