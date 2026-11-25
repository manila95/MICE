"""Update-to-data ratio and parallel rollouts on SafetyPointGoal1-v0 (SACLag, limit 10).

SAC takes one gradient step per environment step. update_cycle divides that rate, and
vector_env_nums divides it again while also cutting the per-step rollout overhead that
update_cycle cannot touch. lambda_lr is scaled by env-steps-per-update in every arm, so
the multiplier's trajectory against environment steps is unchanged and the only variable
is how often the critic and policy are updated.

Each arm is one seed. The seed-noise band on this configuration spans about 10 reward,
so only the largest gap here is outside it.
"""
from __future__ import annotations

import csv
import glob
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = '/home/kaustubh/projects/calibration_rl/MICE/runs'
SMOOTH, LIMIT = 41, 10.0
ARMS = [
    ('stock: 1 env, uc 1  (UTD 1.00)', f'{RUNS}/seeds_nw3M/nw7e6_s0', '#555555', '-', 2.6, '8.0h'),
    ('1 env, uc 10  (UTD 0.10)', f'{RUNS}/update_cycle/uc10', '#1f77b4', '--', 1.9, '3.4h'),
    ('1 env, uc 100  (UTD 0.01)', f'{RUNS}/update_cycle/uc100', '#9467bd', '--', 1.9, '2.8h'),
    ('5 envs, uc 1  (UTD 0.20)', f'{RUNS}/update_cycle/v5uc1', '#ff7f0e', '-', 1.9, '2.6h'),
    ('5 envs, uc 2  (UTD 0.10)', f'{RUNS}/update_cycle/v5uc2', '#2ca02c', '-', 2.8, '1.8h'),
]
SERIES = [('Metrics/EpRet', 'episodic return'),
          ('Metrics/EpCost', 'episodic cost'),
          ('Metrics/LagrangeMultiplier', r'Lagrange multiplier $\lambda$')]


def y200(d):
    return float(np.mean(d['Metrics/EpRet'][-200:]))


def load(base):
    hits = glob.glob(f'{base}/*/seed-*/progress.csv')
    if not hits:
        return None
    rows = list(csv.DictReader(open(hits[0])))
    if len(rows) < SMOOTH:
        return None
    out = {}
    for k, _ in SERIES:
        v = np.array([float(r[k]) for r in rows])
        # a 5-env run can log a NaN before any episode has finished; carry the last value
        if np.isnan(v).any():
            idx = np.where(~np.isnan(v))[0]
            v = np.interp(np.arange(len(v)), idx, v[idx]) if len(idx) else np.zeros_like(v)
        out[k] = v
    return out


def smooth(y, w=SMOOTH):
    """Trailing mean over the last w epochs, the same reduction used everywhere else here.

    A centred average has to invent values past the end, and on a series that is still
    swinging that padding drags the final point badly: it put the stock arm at 13.7 where
    the trailing mean, and the plain mean of the last 50 epochs, both say about 6-7.
    """
    return np.array([y[max(0, i - w + 1):i + 1].mean() for i in range(len(y))])


fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
for ax, (key, ylabel) in zip(axes, SERIES):
    for label, base, colour, ls, lw, wall in ARMS:
        d = load(base)
        if d is None:
            continue
        ys = smooth(d[key])
        ax.plot(np.arange(len(ys)), ys, color=colour, ls=ls, lw=lw,
                label=f'{label} — {wall}' if key == 'Metrics/EpRet' else None)
    ax.set_xlabel('epoch (2000 steps each)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1500)

axes[0].axhline(0, color='k', lw=0.8, ls=':')
axes[0].set_ylim(-8, 30)
axes[1].axhline(LIMIT, color='k', ls=':', lw=1.4)
axes[1].text(1490, LIMIT + 2, 'cost_limit 10', ha='right', fontsize=9)
axes[1].set_ylim(0, 90)
axes[2].axhline(0.5, color='#d62728', ls='--', lw=1.3)
axes[2].text(1490, 0.515, 'cap 0.5', ha='right', fontsize=9, color='#d62728')
axes[2].set_ylim(-0.02, 0.58)
fig.legend(loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.13), fontsize=9)
fig.suptitle('Cutting gradient steps 10-100x costs no reward, and parallel envs make it 4.3x faster',
             fontsize=13, y=1.02)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'/home/kaustubh/projects/calibration_rl/MICE/figs/offpolicy_update_cycle.{ext}',
                dpi=150, bbox_inches='tight')
print('wrote figs/offpolicy_update_cycle.{png,pdf}\n')
print(f"{'arm':<34}{'ep':>6}{'peak':>8}{'EpRet':>8}{'EpCost':>9}{'wall':>7}")
for label, base, _, _, _, wall in ARMS:
    d = load(base)
    if d is None:
        continue
    q, c = smooth(d['Metrics/EpRet']), smooth(d['Metrics/EpCost'])
    print(f'{label:<34}{len(q):>6}{q.max():>8.1f}{q[-1]:>8.1f}{c[-1]:>9.1f}{wall:>7}'
          f'   last200 {y200(d):>5.1f}')
