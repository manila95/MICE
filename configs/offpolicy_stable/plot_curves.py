"""Training curves for the off-policy safe-RL arms on SafetyPointGoal1-v0.

Reward, cost and the Lagrange multiplier against epoch, capped at 1000 epochs (2M steps).
Per-epoch values are an average over two episodes, so each curve is smoothed with a
rolling mean before plotting; the raw series is drawn faintly behind it.
"""
from __future__ import annotations

import csv
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = '/home/kaustubh/projects/calibration_rl/MICE/runs/offpolicy_3M'
CAP, WINDOW = 1000, 25

# label -> (directory, colour, is-a-winner)
ARMS = [
    ('SACLag, cost_limit 10  (shipped schedule)', 'saclag_cl10', '#1f77b4', True),
    ('SACLag, cost_limit 25  (shipped schedule)', 'saclag_ref', '#2ca02c', True),
    ('TD3Lag, cost_limit 10', 'td3lag_cl10', '#9467bd', False),
    ('SACLag, lambda_lr 1e-5 (20x)', 'saclag_cl10_lr20x', '#d62728', False),
    ('SACLag, lambda_lr 1e-5, warmup 250', 'saclag_cl10_warm250', '#ff7f0e', False),
    ('SACLag, lambda_lr 2e-6 (4x)', 'saclag_cl10_lr4x', '#8c564b', False),
]
SERIES = [
    ('Metrics/EpRet', 'episodic return', None),
    ('Metrics/EpCost', 'episodic cost', 10.0),
    ('Metrics/LagrangeMultiplier', r'Lagrange multiplier $\lambda$', None),
]


def load(tag):
    hits = glob.glob(f'{RUNS}/{tag}/*/seed-*/progress.csv')
    if not hits:
        return None
    rows = list(csv.DictReader(open(hits[0])))[:CAP]
    return {k: np.array([float(r[k]) for r in rows]) for k, _, _ in SERIES} | {
        'epoch': np.arange(len(rows))
    }


def smooth(y, w=WINDOW):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode='valid')


fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, (key, ylabel, ref) in zip(axes, SERIES):
    for label, tag, colour, winner in ARMS:
        d = load(tag)
        if d is None:
            continue
        y, x = d[key], d['epoch']
        ax.plot(x, y, color=colour, alpha=0.12, lw=0.8)
        ys = smooth(y)
        ax.plot(x[len(x) - len(ys):], ys, color=colour, lw=2.2 if winner else 1.3,
                ls='-' if winner else '--', label=label if key == 'Metrics/EpRet' else None)
    if ref is not None:
        ax.axhline(ref, color='k', ls=':', lw=1.4)
        ax.text(CAP * 0.99, ref, f'  cost_limit {ref:.0f}', va='bottom', ha='right', fontsize=9)
    ax.set_xlabel('epoch (2000 steps each)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, CAP)
    if key == 'Metrics/EpRet':
        # the winner's peak: reward is still falling at the cap, so the endpoint is a
        # point on a declining curve, not a converged value
        d = load('saclag_cl10')
        if d is not None:
            ys = smooth(d[key]); xs = d['epoch'][len(d['epoch']) - len(ys):]
            i = int(np.argmax(ys))
            ax.plot(xs[i], ys[i], 'o', color='#1f77b4', ms=7, mfc='none', mew=2)
            ax.annotate(f'peak {ys[i]:.0f} @ epoch {xs[i]}\ncost ~46 there: limit violated',
                        (xs[i], ys[i]), xytext=(430, -14), fontsize=8.5, color='#1f77b4',
                        arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2,
                                        connectionstyle='arc3,rad=-0.25'))
axes[1].set_ylim(0, 85)
fig.legend(loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.17), fontsize=9)
fig.suptitle('Off-policy safe RL on SafetyPointGoal1-v0: the shipped schedule wins, but reward is still falling at 1000 epochs',
             fontsize=13, y=1.02)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'/home/kaustubh/projects/calibration_rl/MICE/figs/offpolicy_stable_curves.{ext}',
                dpi=150, bbox_inches='tight')
print('wrote figs/offpolicy_stable_curves.{png,pdf}')

# final-window summary, the numbers the curves should be read against
print(f"\n{'arm':<38}{'epochs':>7}{'EpRet':>8}{'EpCost':>9}{'lambda':>8}")
for label, tag, _, _ in ARMS:
    d = load(tag)
    if d is None:
        continue
    n = len(d['epoch'])
    m = lambda k: d[k][-50:].mean()
    print(f'{label:<38}{n:>7}{m("Metrics/EpRet"):>8.2f}{m("Metrics/EpCost"):>9.1f}'
          f'{m("Metrics/LagrangeMultiplier"):>8.3f}')
