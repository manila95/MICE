"""Training curves for the lambda-schedule arms on SafetyPointGoal1-v0 (SACLag, limit 10).

Everything is a 25-epoch moving average (~50 episodes). A per-epoch value is an average
over two episodes on a 1000-step task, which is far too noisy to read a trend from; the
raw series is drawn faintly behind each curve so the smoothing is visible.

The arms differ in when cost pressure starts and how the multiplier is driven:
  * warmup 100 (stock): no pressure for 200k steps, then lambda ramps
  * warmup 0:           lambda rises from the first step
  * jc_ema:             lambda follows a ~10-episode moving average of cost, not the
                        two episodes that happened to finish last
  * update_every:       lambda integrates once per epoch instead of 2000 times
"""
from __future__ import annotations

import csv
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = '/home/kaustubh/projects/calibration_rl/MICE/runs/offpolicy_3M'
MA, LIMIT = 25, 10.0

ARMS = [
    ('stock: warmup 100, $\\lambda$ ramps', 'saclag_cl10_long', '#1f77b4', '-'),
    ('stock, $\\lambda$ rate /2', 'lam_slow_2x', '#17becf', '-'),
    ('warmup 0, stock rate', 'grad_w0_slow', '#2ca02c', '-'),
    ('warmup 0, rate 2e-7', 'grad_w0_vslow', '#98df8a', '-'),
    ('warmup 0, $\\lambda_0$=0.2', 'grad_w0_i02', '#d62728', ':'),
    ('$\\lambda_0$=0.5, lr 1e-6', 'init0p5_lr1e6', '#8c564b', ':'),
    ('$\\lambda$ from cost MA (ema)', 'lam_ema', '#ff7f0e', '--'),
    ('$\\lambda$ once per epoch', 'lam_perepoch', '#9467bd', '--'),
    ('warmup 0 + cost MA', 'grad_w0_ema', '#e377c2', '-.'),
    ('warmup 0 + MA + per-epoch', 'grad_w0_ema_pe', '#7f7f7f', '-.'),
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
    for label, tag, colour, ls in ARMS:
        d = load(tag)
        if d is None:
            continue
        y = d[key]
        ax.plot(np.arange(len(y)), y, color=colour, alpha=0.10, lw=0.7)
        ys = ma(y)
        ax.plot(np.arange(MA - 1, len(y)), ys, color=colour, ls=ls, lw=2.0,
                label=label if key == 'Metrics/EpRet' else None)
    ax.set_xlabel('epoch (2000 steps each)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
axes[1].axhline(LIMIT, color='k', ls=':', lw=1.5)
axes[1].text(0.99, LIMIT, ' cost_limit 10', transform=axes[1].get_yaxis_transform(),
             ha='right', va='bottom', fontsize=9)
axes[0].axhline(0, color='k', lw=0.8, ls=':')
fig.legend(loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.16), fontsize=8.6)
fig.suptitle('Where cost pressure starts, and what drives $\\lambda$ '
             '(25-epoch moving averages)', fontsize=13, y=1.02)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'/home/kaustubh/projects/calibration_rl/MICE/figs/offpolicy_lambda_schedule.{ext}',
                dpi=150, bbox_inches='tight')
print('wrote figs/offpolicy_lambda_schedule.{png,pdf}\n')
print(f"{'arm':<32}{'ep':>5}{'EpRet':>8}{'EpCost':>9}{'lam':>8}")
for label, tag, *_ in ARMS:
    d = load(tag)
    if d is None:
        print(f'{label:<32}{"(too early)":>30}'); continue
    f = lambda k: ma(d[k])[-1]
    print(f'{label:<32}{len(d["Metrics/EpRet"]):>5}{f("Metrics/EpRet"):>8.1f}'
          f'{f("Metrics/EpCost"):>9.1f}{f("Metrics/LagrangeMultiplier"):>8.3f}')
