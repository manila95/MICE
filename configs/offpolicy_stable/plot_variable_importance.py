"""Which variables actually moved SACLag on SafetyPointGoal1-v0.

Every bar is a measured pair of runs from this investigation, not an estimate. The bar
spans the final episodic return at the two settings compared; the label gives the setting
that produced each end.

The grey band is the noise floor: five seeds of one fixed configuration spanned reward
-0.8 to 9.3 (sd 3.9). Any effect narrower than that band is not distinguishable from
seed luck on the single-seed comparisons most of these bars come from, which is the main
thing this chart is for.

Variables split cleanly in two. A few decide whether the task is learned at all - peak
return is either ~26 or ~1, with nothing in between. The rest leave peak return at ~26
and only move where the run ends up along the reward/cost frontier.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# (label, low value, high value, low setting, high setting, tier)
# tier: 0 = decides whether learning happens, 1 = moves along the frontier, 2 = no gain / harmful
BARS = [
    ('Tuned env block\n(alpha, actor lr)', 0.2, 26.4, 'base-lr override', 'stock block', 0),
    ('$\\lambda_0$ (multiplier init)', -0.3, 26.4, '$\\lambda_0$=2', '$\\lambda_0$=0', 0),
    ('SR (td_ridge)', -2.8, 12.0, 'SR on', 'SR off', 2),
    ('Training horizon', 0.5, 12.0, '300k steps', '2M steps', 0),
    ('cost-limit annealing', 6.2, 12.0, '50$\\to$10', 'fixed', 2),
    ('$\\lambda$ ramp rate', 7.3, 12.0, '20$\\times$ (1e-5)', 'stock (5e-7)', 1),
    ('warmup_epochs', 8.0, 12.0, '250', '100', 1),
    ('Entropy $\\alpha$\n(under high $\\lambda$)', -1.0, -1.6, '0.2', '1e-5', 2),
    ('Algorithm', 11.1, 12.0, 'TD3Lag', 'SACLag', 1),
    ('cost_limit', 11.7, 12.0, '25', '10', 1),
    ('$\\lambda$ upper bound', 12.0, 12.0, 'capped', 'uncapped', 2),
]
SEED_LO, SEED_HI, SEED_SD = -0.8, 9.3, 3.88
COLOUR = {0: '#1f77b4', 1: '#9a9a9a', 2: '#c0603a'}
TIER = {0: 'decides whether the task is learned', 1: 'moves along the frontier',
        2: 'no gain, or harmful'}

BARS = sorted(BARS, key=lambda b: abs(b[2] - b[1]))
fig, ax = plt.subplots(figsize=(12.5, 7.2))
y = np.arange(len(BARS))

ax.axvspan(SEED_LO, SEED_HI, color='#555555', alpha=0.13, zorder=0)
ax.text((SEED_LO + SEED_HI) / 2, len(BARS) - 0.3,
        f'seed noise floor\n5 seeds, one config: {SEED_LO} to {SEED_HI} (sd {SEED_SD})',
        ha='center', va='top', fontsize=9, color='#444444', style='italic')

for i, (label, lo, hi, lo_s, hi_s, tier) in enumerate(BARS):
    a, b = min(lo, hi), max(lo, hi)
    width = max(b - a, 0.18)          # keep a zero-width effect visible
    ax.barh(i, width, left=a, height=0.52, color=COLOUR[tier], alpha=0.9, zorder=3)
    ax.text(a - 0.5, i, lo_s, ha='right', va='center', fontsize=8.5, color='#333333')
    ax.text(b + 0.5, i, hi_s, ha='left', va='center', fontsize=8.5, color='#333333')
    ax.text(b + 0.5, i - 0.33, f'$\\Delta$ {b - a:.1f}', ha='left', va='center',
            fontsize=8, color=COLOUR[tier], weight='bold')

ax.set_yticks(y)
ax.set_yticklabels([b[0] for b in BARS], fontsize=10)
ax.set_xlabel('final episodic return at each setting (MA41)')
ax.set_xlim(-9, 34)
ax.axvline(0, color='k', lw=0.8, ls=':')
ax.grid(axis='x', alpha=0.25)
handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR[t]) for t in (0, 1, 2)]
ax.legend(handles, [TIER[t] for t in (0, 1, 2)], loc='lower right', fontsize=9.5,
          framealpha=0.95)
ax.set_title('What actually moved SACLag on SafetyPointGoal1-v0\n'
             'measured pairs from 67 runs; effects inside the grey band are within seed noise',
             fontsize=12.5)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'/home/kaustubh/projects/calibration_rl/MICE/figs/saclag_variable_importance.{ext}',
                dpi=150)
print('wrote figs/saclag_variable_importance.{png,pdf}')
for label, lo, hi, lo_s, hi_s, tier in sorted(BARS, key=lambda b: -abs(b[2] - b[1])):
    flag = '' if abs(hi - lo) > (SEED_HI - SEED_LO) else '   <- within seed noise'
    print(f'  {label.replace(chr(10)," "):<34}{abs(hi-lo):>7.1f}{flag}')
