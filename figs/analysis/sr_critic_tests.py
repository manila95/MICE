"""Does the SR critic differ from the base critic on the probe diagnostics?

Same four configurations as analysis/sr_cost_budget.py (plain advantage, best
EpRet/TotalCost):

    CPO     base  hyfcu0dx, update_iters 2, critic lr 3e-4     SR  260zwq3s cfg3
    PIDLag  base  hyfcu0dx, update_iters 10, critic lr 1e-4    SR  84vmrbam best

All metrics are computed on the pooled probe block at the pre-update evaluation point,
per seed and checkpoint, then averaged over the checkpoints of a phase (before or after
the episodic cost first reaches the limit) to give one number per seed. SR and base are
compared with the exact 5 vs 5 permutation test on those per-seed numbers.

Metrics: correlation with the reference, AUROC for high value (Gbar >= gamma^5) and its
split-half ceiling, signed error over all probes and by true value group, share of
negative predictions, and rho_max(K) -- reported because the two arms were run under
different evaluation protocols.

A second table measures how conservative the resulting updates are, over the same
windows and with the same test: for CPO, the share of epochs whose update falls back to
a pure cost-recovery step (OptimCase 0, the infeasible branch); for TRPO-PIDLag, the
average Lagrange multiplier. Both are read from the W&B run histories, which needs
network access and the run ids below; pass --no-conservatism to skip that table.

Usage (from the RL_CAL root):
    PYTHONPATH=analysis uv run --python 3.12 --with "numpy<2" --with torch \
        python analysis/sr_critic_tests.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import os
from collections import defaultdict

import numpy as np

from metrics import GAMMA, auroc
from sweep_data import _load_dir, load_sweep

SPE, BETA = 20000, 10.0
T_LOW, T_HIGH = GAMMA ** 500, GAMMA ** 5
GROUPS = ["low", "intermediate", "high"]
ARMS = [("CPO", "base", "eval_data_hyfcu0dx", "plain_iters2_lr0.0003",
         "hist_sr_base", "plain_iters2_lr0.0003"),
        ("CPO", "SR", "eval_data_260zwq3s_front/cfg3", None,
         "eval_data_260zwq3s_front/cfg3", None),
        ("PIDLag", "base", "eval_data_hyfcu0dx", "plain_iters10_lr0.0001",
         "hist_sr_base", "plain_iters10_lr0.0001"),
        ("PIDLag", "SR", "eval_data_84vmrbam_best", None, "hist_sr_pidlag", None)]
KEYS = ["corr", "auroc", "ceil", "signed"] + GROUPS + ["neg", "rho_max"]
PROJECT = "liam-paull/calibration_rl"
# run ids of the same four configurations, for the history-based conservatism table
RUN_IDS = {("CPO", "base"): "cc0mi669,uwan2h99,tm8oq6lz,9qx3rf9v,8ip4rznu",
           ("CPO", "SR"): "832astfc,sriyczlb,yoacs0tv,g0oqw2fs,05m2oui9",
           ("PIDLag", "base"): "13ha0olu,a88z4l5u,2ep2akwv,hxh6zo7s,07yi1jlf",
           ("PIDLag", "SR"): "xbxn4c75,chdf5fda,fbdm4uep,n2lotret,x94fks2o"}
# CPO: OptimCase 0 is the infeasible branch, a pure cost-recovery step, so its share is
# the fraction of updates that abandon the reward objective. PIDLag: the multiplier is
# how much the cost term weighs in the objective. Lower means less conservative.
CONSERVATISM = {"CPO": "Misc/OptimCase", "PIDLag": "Metrics/LagrangeMultiplier"}
WINDOWS = (("<= 0.5M", 0.0, 5e5), ("0.5-2M", 5e5, 2e6), ("2-10M", 2e6, 1.1e7),
           ("all", 0.0, 1.1e7))


def point(p) -> dict[str, float]:
    pred, G = p.pred["c"], p.G["c"]
    gbar, K = G.mean(1), G.shape[1]
    lab, e = gbar >= T_HIGH, pred - gbar
    h1, h2 = G[:, :K // 2].mean(1), G[:, K // 2:].mean(1)
    s2b = gbar.var(ddof=1)
    var_v = max(s2b - G.var(1, ddof=1).mean() / K, 0.0)
    out = {"corr": float(np.corrcoef(pred, gbar)[0, 1]) if pred.std() > 0 else np.nan,
           "auroc": auroc(pred, lab),
           "ceil": np.nanmean([auroc(h1, h2 >= T_HIGH), auroc(h2, h1 >= T_HIGH)]),
           "signed": float(e.mean()), "neg": float(np.mean(pred < 0)),
           "rho_max": float(np.sqrt(var_v / s2b)) if s2b > 0 else np.nan}
    for g, m in zip(GROUPS, (gbar < T_LOW, (gbar >= T_LOW) & (gbar < T_HIGH), lab)):
        out[g] = float(e[m].mean()) if m.any() else np.nan
    return out


def seeds_of(root: str, cell: str | None) -> list[list]:
    if cell is not None:
        return [pts for _, pts in sorted(load_sweep(root, "pooled")[cell].seeds.items())]
    return [_load_dir(d, "pooled") for d in sorted(glob.glob(os.path.join(root, "*", "seed*")))]


def beta_crossing(root: str, cell: str | None) -> float:
    per = defaultdict(list)
    with open(os.path.join(root, "history.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if (cell is None or r["cell"] == cell) and r["Metrics/EpCost"] not in ("", "None"):
                per[r["run_id"]].append((float(r["TotalEnvSteps"]), float(r["Metrics/EpCost"])))
    a = [np.array(sorted(v)) for v in per.values()]
    n = min(len(x) for x in a)
    mean_cost = np.mean([x[:n, 1] for x in a], 0)
    below = np.where(mean_cost <= BETA)[0]
    return float(a[0][below[0], 0]) if len(below) else float(a[0][n - 1, 0])


def perm_p(x: np.ndarray, y: np.ndarray) -> float:
    r"""Exact two-sided permutation test on the difference of means.

    One value per seed (here, a metric averaged over the checkpoints of a window), so
    ``x`` and ``y`` hold one number per run and the test needs independence across
    seeds only -- not across the correlated checkpoints those numbers average over.

    Under the null the arm label is exchangeable: the pooled values come from one
    distribution and the base/SR assignment carries no information. The statistic is
    :math:`|\bar x - \bar y|`, and the reference distribution is obtained by
    enumerating *all* :math:`\binom{|x| + |y|}{|x|}` assignments of the pooled values
    to two groups of the original sizes -- exhaustively, so there is no sampling error,
    and no normality or equal-variance assumption. The returned $p$ is the fraction of
    assignments whose statistic is at least the observed one, the observed assignment
    included (each partition appears twice in the enumeration, once as itself and once
    as its complement; the statistic is symmetric, so this does not bias the count).

    With five seeds per arm there are $\binom{10}{5} = 252$ assignments, so the
    smallest attainable value is $2/252 \approx 0.008$: a printed $p = 0.008$ means the
    observed split is the most extreme of all, typically because the two groups of
    seeds do not overlap, and not that the evidence is stronger than that.

    No multiplicity correction is applied. The tables here run many metrics over
    several windows, so single cells at $p < 0.05$ should be read with that in mind;
    the differences worth trusting are those that repeat across windows or algorithms.

    Returns NaN if either input contains a non-finite value (e.g. a metric undefined
    at a checkpoint where the critic was constant).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return np.nan
    pool, obs, n = np.concatenate([x, y]), abs(x.mean() - y.mean()), len(x)
    d = [abs(pool[list(c)].mean() - pool[[i for i in range(len(pool)) if i not in c]].mean())
         for c in itertools.combinations(range(len(pool)), n)]
    return float(np.mean(np.array(d) >= obs - 1e-12))


def conservatism(arm: tuple[str, str]) -> list[np.ndarray]:
    """Per-seed (steps, metric) from the run history, for the arm's conservatism proxy."""
    import wandb
    api = wandb.Api(timeout=120)
    key = CONSERVATISM[arm[0]]
    out = []
    for rid in RUN_IDS[arm].split(","):
        rows = [(rec["TotalEnvSteps"], rec[key]) for rec in
                api.run(f"{PROJECT}/{rid}").scan_history(keys=["TotalEnvSteps", key],
                                                         page_size=10000)
                if rec.get(key) is not None and rec.get("TotalEnvSteps") is not None]
        out.append(np.array(sorted(rows), float))
    return out


def conservatism_table() -> None:
    C = {arm: conservatism(arm) for arm in RUN_IDS}
    print("\n===== conservatism of the update =====")
    print("CPO: share of epochs in OptimCase 0 (pure cost-recovery step);  "
          "PIDLag: mean Lagrange multiplier")
    print(f"{'steps':>9} | {'CPO base':>9} {'CPO SR':>9} {'p':>6} | "
          f"{'PID base':>9} {'PID SR':>9} {'p':>6}")
    for lab, lo, hi in WINDOWS:
        row = []
        for algo in ("CPO", "PIDLag"):
            agg = ((lambda a: float((a[:, 1] == 0).mean())) if algo == "CPO"
                   else (lambda a: float(a[:, 1].mean())))
            v = {k: np.array([agg(a[(a[:, 0] >= lo) & (a[:, 0] < hi)]) for a in C[(algo, k)]])
                 for k in ("base", "SR")}
            row.append(f"{v['base'].mean():9.3f} {v['SR'].mean():9.3f} "
                       f"{perm_p(v['SR'], v['base']):6.3f}")
        print(f"{lab:>9} | " + " | ".join(row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-conservatism", action="store_true",
                    help="skip the history-based table (no network access needed)")
    args = ap.parse_args()

    D = {}
    for algo, kind, root, cell, hroot, hcell in ARMS:
        runs = seeds_of(root, cell)
        x0 = beta_crossing(hroot, hcell)
        x = np.array([(p.epoch + 1) * SPE for p in runs[0]], float)
        P = [[point(p) for p in r] for r in runs]
        D[(algo, kind)] = {"x": x, "P": P, "x0": x0}
        print(f"{algo:7s} {kind:4s}: {len(runs)} seeds x {len(x)} checkpoints, "
              f"limit reached at {x0 / 1e6:.2f}M")

    # Matched step windows: the arms reach the limit at very different times, so a
    # phase split defined per arm would compare different stages of training.
    for phase, lo, hi in WINDOWS:
        if phase == "all":
            continue
        print(f"\n===== steps {phase} =====")
        print(f"{'metric':>13} | {'CPO base':>9} {'CPO SR':>9} {'p':>6} | "
              f"{'PID base':>9} {'PID SR':>9} {'p':>6}")
        for k in KEYS:
            cells = {}
            for (algo, kind), d in D.items():
                m = [t for t in range(len(d["x"])) if lo <= d["x"][t] < hi]
                cells[(algo, kind)] = np.array(
                    [np.nanmean([d["P"][s][t][k] for t in m]) for s in range(len(d["P"]))])
            row = []
            for algo in ("CPO", "PIDLag"):
                b, s = cells[(algo, "base")], cells[(algo, "SR")]
                row.append(f"{b.mean():9.3f} {s.mean():9.3f} {perm_p(s, b):6.3f}")
            print(f"{k:>13} | " + " | ".join(row))

    if not args.no_conservatism:
        conservatism_table()


if __name__ == "__main__":
    main()
