"""The achievable reward-cost frontier on SafetyPointGoal1-v0 (SACLag and friends).

Every 100-epoch window of every run on disk is one achievable (cost, reward) pair. Pooling
them across 40 runs - which between them vary the multiplier's initial value, its learning
rate, its update rule, the entropy coefficient, when cost pressure starts, and whether the
cost limit itself is annealed - maps out what this task actually permits, as opposed to
what any one schedule happens to reach.

The point of the pooled view: if a cost band's best reward is low across every mechanism
tried, that band is a property of the task, not of the configuration.
"""
from __future__ import annotations

import csv
import glob
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOTS = ['/home/kaustubh/projects/calibration_rl/MICE/runs/offpolicy_3M',
         '/home/kaustubh/projects/calibration_rl/MICE/runs/offpolicy_screen']
W = 100
BANDS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 45), (45, 70), (70, 200)]

pts = []
for root in ROOTS:
    for f in glob.glob(f'{root}/*/*/seed-*/progress.csv'):
        tag = f[len(root) + 1:].split('/')[0]
        rows = list(csv.DictReader(open(f)))
        if len(rows) < 150:
            continue
        ret = [float(x['Metrics/EpRet']) for x in rows]
        cost = [float(x['Metrics/EpCost']) for x in rows]
        for a in range(W, len(rows), W):
            pts.append((st.mean(cost[a - W:a]), st.mean(ret[a - W:a]), tag, a))

fig, ax = plt.subplots(figsize=(9.5, 5.6))
early = [p for p in pts if p[3] <= 300]
late = [p for p in pts if p[3] > 300]
ax.scatter([p[0] for p in early], [p[1] for p in early], s=26, c='#c7c7c7',
           label='epochs 100-300 (policy still forming)', zorder=2)
ax.scatter([p[0] for p in late], [p[1] for p in late], s=30, c='#1f77b4', alpha=0.75,
           label='epochs 300+ (settled)', zorder=3)

fx, fy = [], []
for lo, hi in BANDS:
    band = [p for p in pts if lo <= p[0] < hi]
    if band:
        fx.append((lo + min(hi, 100)) / 2)
        fy.append(max(p[1] for p in band))
ax.plot(fx, fy, color='#d62728', lw=2.4, marker='o', ms=7, zorder=4,
        label='best reward achieved in each cost band')

ax.axvline(10, color='k', ls=':', lw=1.6)
ax.text(10.6, ax.get_ylim()[0] + 1, 'cost_limit 10', fontsize=9, rotation=90, va='bottom')
ax.axvspan(20, 30, color='#2ca02c', alpha=0.08, zorder=1)
ax.annotate('the knee: cost 20-30 buys reward ~20,\nvs ~10 at cost 10',
            xy=(25, 19.9), xytext=(46, 6), fontsize=9.5, color='#2ca02c',
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.4))
ax.set_xscale('symlog', linthresh=20)
ax.set_xlabel('episodic cost (100-epoch mean)')
ax.set_ylabel('episodic return (100-epoch mean)')
ax.set_title(f'What SafetyPointGoal1-v0 permits: {len(pts)} windows from '
             f'{len(set(p[2] for p in pts))} runs', fontsize=12)
ax.grid(alpha=0.25)
ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'/home/kaustubh/projects/calibration_rl/MICE/figs/offpolicy_frontier.{ext}', dpi=150)
print(f'wrote figs/offpolicy_frontier.{{png,pdf}}  ({len(pts)} windows, '
      f'{len(set(p[2] for p in pts))} runs)')
