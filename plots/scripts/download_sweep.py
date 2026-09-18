"""Download every eval_data artifact from a wandb sweep into a per-cell/per-seed tree.

Layout (what analysis/ expects):
    eval_data_<sweep>/<cell>/seed<N>/epoch_XXXXX.pkl
    eval_data_<sweep>/manifest.csv

`cell` encodes the sweep's design point, e.g. ``gae_iters10_lr0.0003``.  Sweep
6m8yuigd is 2 (adv estimator) x 2 (update_iters) x 2 (critic lr) x 5 seeds = 40
runs, 37 eval epochs each.

Usage:
    python download_sweep.py                       # eval_data, sweep 6m8yuigd
    python download_sweep.py --type scatter_data
    python download_sweep.py --sweep <id> --workers 8

Sweeps that mix algorithms need one tree per algorithm: `cell` encodes the estimator,
update_iters and critic lr but NOT the algorithm, so CPO and TRPOPID runs of the same
design point would collide.  Sweep lsaldj7r (2 algos x 2 estimators x 2 update_iters x
2 critic lrs x 5 seeds) was downloaded by passing each algorithm's run ids separately,
with `--history` for the training curves and a second pass for the eval_data of the
cell the figures plot:

    S=liam-paull/calibration_rl/lsaldj7r
    python download_sweep.py --sweep $S --out eval_data_lsaldj7r_cpo --runs <cpo ids> --history
    python download_sweep.py --sweep $S --out eval_data_lsaldj7r_cpo --runs <cpo ids of the cell>

and likewise into eval_data_lsaldj7r_pidlag.  Run ids per algorithm come from
`run.config["algo"]` over `wandb.Api().sweep(S).runs`.  The trees are written relative
to the working directory, which is where the figure scripts expect to find them.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import wandb

DEFAULT_SWEEP = "liam-paull/calibration_rl/6m8yuigd"


def cell_name(run) -> str:
    a = run.config["algo_cfgs"]
    lr = run.config["model_cfgs"]["critic"]["lr"]
    adv = a["adv_estimation_method"]
    cadv = a["cost_adv_estimation_method"]
    # The sweep co-varies these; keep both in the name if that ever stops holding.
    tag = adv if adv == cadv else f"{adv}-{cadv}"
    return f"{tag}_iters{a['update_iters']}_lr{lr:g}"


def fetch_run(entity_project: str, run_id: str, dest_root: str, art_type: str) -> tuple[str, int, int]:
    """Download one run's artifacts of `art_type`.  Returns (run_id, got, skipped)."""
    api = wandb.Api(timeout=120)          # one Api per thread
    run = api.run(f"{entity_project}/{run_id}")
    dest = os.path.join(dest_root, cell_name(run), f"seed{run.config['seed']}")
    os.makedirs(dest, exist_ok=True)
    got = skipped = 0
    for art in run.logged_artifacts():
        if art.type != art_type:
            continue
        for f in art.files():
            if not f.name.lower().endswith(".pkl"):
                continue
            if os.path.exists(os.path.join(dest, f.name)):
                skipped += 1
                continue
            f.download(root=dest, replace=False)
            got += 1
    return run_id, got, skipped


HISTORY_KEYS = [
    "Train/Epoch", "TotalEnvSteps", "Metrics/EpRet", "Metrics/EpCost", "Metrics/EpLen",
    # In-sample (training-batch) critic diagnostics, logged at eval epochs only.
    # *CriticCorr  = Corr(V_hat, y^kappa) on the training batch
    # *PredTrueCorr = Corr(V_hat, discounted return of the training trajectory)
    # AfterUpdate is the post-fit critic, i.e. the same one the probes see.
    "Value/Train/AfterUpdate/CostCriticCorr",
    "Value/Train/AfterUpdate/RewardCriticCorr",
    "Value/Train/BeforeUpdate/CostCriticCorr",
    "Value/Train/BeforeUpdate/RewardCriticCorr",
    "Value/Train/AfterUpdate/CostPredTrueCorr",
    "Value/Train/AfterUpdate/RewardPredTrueCorr",
    "Value/Train/CostTargetTrueCorr",
    "Value/Train/RewardTargetTrueCorr",
]


def fetch_history(runs, out: str) -> None:
    """Training curves (episodic return/cost) -- these live in the run history,
    not in the eval_data artifacts."""
    path = os.path.join(out, "history.csv")
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "cell", "seed"] + HISTORY_KEYS)
        for i, r in enumerate(runs, 1):
            cell, seed = cell_name(r), r.config["seed"]
            rows = 0
            for rec in r.scan_history(keys=HISTORY_KEYS, page_size=10000):
                w.writerow([r.id, cell, seed] + [rec.get(k) for k in HISTORY_KEYS])
                rows += 1
            n += rows
            print(f"  [{i}/{len(runs)}] {r.id} {cell} seed{seed}: {rows} epochs")
    print(f"wrote {path} ({n} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=DEFAULT_SWEEP)
    ap.add_argument("--type", default="eval_data", help="artifact type to pull")
    ap.add_argument("--out", default=None, help="default: eval_data_<sweep id>")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--runs", default=None,
                    help="comma-separated run ids to restrict the download to")
    ap.add_argument("--history", action="store_true",
                    help="fetch Metrics/EpRet + Metrics/EpCost into history.csv "
                         "instead of downloading artifacts")
    args = ap.parse_args()

    entity_project, sweep_id = args.sweep.rsplit("/", 1)
    out = args.out or f"eval_data_{sweep_id}"
    if args.type != "eval_data":
        out = f"{args.type}_{sweep_id}"
    os.makedirs(out, exist_ok=True)

    api = wandb.Api(timeout=120)
    runs = list(api.sweep(args.sweep).runs)
    if args.runs:
        keep = set(args.runs.split(","))
        runs = [r for r in runs if r.id in keep]
    print(f"{args.sweep}: {len(runs)} runs -> {out}/")

    if args.history:  # before the manifest, so both modes can run concurrently
        fetch_history(runs, out)
        return

    with open(os.path.join(out, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "run_name", "state", "cell", "seed", "adv", "cost_adv",
                    "update_iters", "critic_lr", "env_id", "cost_limit", "steps_per_epoch"])
        for r in runs:
            a = r.config["algo_cfgs"]
            w.writerow([r.id, r.name, r.state, cell_name(r), r.config["seed"],
                        a["adv_estimation_method"], a["cost_adv_estimation_method"],
                        a["update_iters"], r.config["model_cfgs"]["critic"]["lr"],
                        r.config.get("env_id"), r.config.get("cost-limit"),
                        a.get("steps_per_epoch")])

    ids = [r.id for r in runs]
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_run, entity_project, rid, out, args.type): rid for rid in ids}
        for fut in as_completed(futs):
            rid = futs[fut]
            try:
                _, got, skipped = fut.result()
            except Exception as exc:                       # keep going; report at the end
                print(f"  !! {rid}: {type(exc).__name__}: {exc}")
                continue
            done += 1
            print(f"  [{done}/{len(ids)}] {rid}: +{got} new, {skipped} already present")
    print(f"done -> {out}/")


if __name__ == "__main__":
    main()
