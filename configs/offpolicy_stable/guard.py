"""Kill arms that are already decided, so a sweep costs minutes instead of hours.

Derived from 32 completed runs on SafetyPointGoal1-v0. Episodic return at epoch 60
separates learners from failures with no overlap: learners sit at 22-25, failures at
-2.5 to 1.2, and no run below 5 at epoch 60 ever finished above 5.

Two checks, because one is not enough: `sacpid_init5` read 25.5 at epoch 60 and still
finished at -16.5, since its multiplier only engaged when warmup ended. So an arm is
checked again once lambda has had time to act.
"""
from __future__ import annotations

import csv
import glob
import os
import subprocess
import sys
import time

RUNS = sys.argv[1] if len(sys.argv) > 1 else '/home/kaustubh/projects/calibration_rl/MICE/runs/offpolicy_3M'
LEARN_BY, LEARN_MIN = 60, 5.0        # "is it learning at all"
HOLD_AFTER, HOLD_MIN = 180, 5.0      # "did lambda destroy it" (warmup 100 + 80 to act)


def window(path, lo, hi):
    rows = list(csv.DictReader(open(path)))
    if len(rows) < hi:
        return None, len(rows)
    vals = [float(r['Metrics/EpRet']) for r in rows[lo:hi]]
    return sum(vals) / len(vals), len(rows)


def verdict(tag):
    hits = glob.glob(f'{RUNS}/{tag}/*/seed-*/progress.csv')
    if not hits:
        return None
    for lo, hi, floor, why in ((LEARN_BY - 20, LEARN_BY, LEARN_MIN, 'never learned'),
                               (HOLD_AFTER - 20, HOLD_AFTER, HOLD_MIN, 'collapsed after lambda engaged')):
        val, n = window(hits[0], lo, hi)
        if val is not None and val < floor:
            return f'{why} (EpRet={val:.1f} at epoch {hi})'
    return None


def main():
    killed = set()
    while True:
        alive = 0
        for line in open(f'{RUNS}/grid.txt'):
            tag = line.split()[0] if line.split() else None
            if not tag or tag in killed:
                continue
            if subprocess.run(['pgrep', '-f', f'run.py {tag} '], capture_output=True).returncode:
                continue
            alive += 1
            bad = verdict(tag)
            if bad:
                subprocess.run(['pkill', '-f', f'run.py {tag} '])
                killed.add(tag)
                print(f'{time.strftime("%H:%M")}  KILLED {tag}: {bad}', flush=True)
        if not alive:
            print('no arms left', flush=True)
            return
        time.sleep(90)


if __name__ == '__main__':
    main()
