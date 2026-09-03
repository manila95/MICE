"""
Reusable, parallelized wandb puller for multi-seed comparison plots.

Pulls, for each of a list of run IDs, (1) scalar history (EpRet/EpCost/etc.) and
(2) every `eval_data`-type artifact logged by that run, downloaded and reduced to a
per-epoch pooled correlation (pred vs. MC-true, reward + cost, pooled across every
evaluated state) -- exactly the data shape the "1x4 plot, mean +/- seed band"
comparisons in this project use.

Why parallel: each run's pull is a sequence of small, independent network calls
(wandb API + artifact downloads) -- I/O-bound, not CPU-bound, so a thread pool
(not a process pool) is the right tool; the GIL is released during network I/O,
so threads genuinely run concurrently here despite Python's GIL. For N runs this
turns "N sequential run-pulls" into roughly "1 run-pull's wall-clock", the same
win vectorization/wave-batching gets elsewhere in this codebase for CPU-bound work
via subprocess parallelism instead.

Usage:
    from pull_wandb_runs_parallel import pull_runs_parallel
    data = pull_runs_parallel(
        {"gae": ["553r7i1o", "0xtlo1ih", ...], "plain": ["q55g6673", ...]},
        project="liam-paull/calibration_rl",
        steps_per_epoch=20000,
    )
    # data["gae"] -> list of {"run_id", "hist" (DataFrame), "corr_df" (DataFrame)},
    # one entry per seed, ready to be aggregated into a mean +/- SEM band per group.
"""
from __future__ import annotations

import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import wandb


def _pull_one_run(run_id: str, project: str, steps_per_epoch: int, cache_dir: str) -> dict:
    """Pull scalar history + pooled per-epoch correlation for a single run. Runs in a worker
    thread -- must not touch shared mutable state other than the filesystem cache dir (each
    run gets its own subdirectory, so concurrent writers never collide)."""
    api = wandb.Api()  # each thread gets its own Api object -- not documented as thread-safe
    run = api.run(f"{project}/{run_id}")

    hist = run.history(keys=["TotalEnvSteps", "Metrics/EpRet", "Metrics/EpCost"], pandas=True)
    for c in ["TotalEnvSteps", "Metrics/EpRet", "Metrics/EpCost"]:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    hist = hist.sort_values("TotalEnvSteps")

    arts = sorted((a for a in run.logged_artifacts() if a.type == "eval_data"), key=lambda a: a.name)
    corr_rows = []
    for art in arts:
        local_dir = art.download(root=os.path.join(cache_dir, run_id, art.name))
        fname = os.listdir(local_dir)[0]
        with open(os.path.join(local_dir, fname), "rb") as f:
            bundle = pickle.load(f)
        if "pooled" not in bundle:
            continue
        raw = bundle["pooled"]["raw"]
        epoch = bundle["epoch"]
        pred_r, true_r = np.array(raw["r"]["pred"]), np.array(raw["r"]["mc_mean"])
        pred_c, true_c = np.array(raw["c"]["pred"]), np.array(raw["c"]["mc_mean"])
        corr_r = np.corrcoef(pred_r, true_r)[0, 1] if len(pred_r) > 1 else np.nan
        corr_c = np.corrcoef(pred_c, true_c)[0, 1] if len(pred_c) > 1 else np.nan
        corr_rows.append(dict(epoch=epoch, total_steps=(epoch + 1) * steps_per_epoch,
                               corr_r=corr_r, corr_c=corr_c))
    corr_df = pd.DataFrame(corr_rows).sort_values("total_steps") if corr_rows else pd.DataFrame()
    return dict(run_id=run_id, hist=hist, corr_df=corr_df)


def pull_runs_parallel(
    groups: dict[str, list[str]],
    project: str,
    steps_per_epoch: int,
    max_workers: int = 8,
    cache_dir: str = "/tmp/wandb_pull_cache",
) -> dict[str, list[dict]]:
    """Pull scalar history + pooled correlation for every run in every group, concurrently.

    Args:
        groups: {group_label: [run_id, ...]} -- e.g. {"gae": [...5 seeds...], "plain": [...]}.
        project: wandb "entity/project" string.
        steps_per_epoch: to convert epoch -> TotalEnvSteps for the correlation series
            (the eval-data artifact's own `epoch` field doesn't carry this).
        max_workers: thread pool size. I/O-bound, so this can comfortably exceed core count --
            8-16 is reasonable for wandb's API; going much higher risks rate-limiting.
        cache_dir: where artifact downloads land (one subdir per run_id, so reruns of this
            function reuse already-downloaded artifacts instead of re-fetching them).

    Returns:
        {group_label: [{"run_id", "hist", "corr_df"}, ...]} -- one entry per run, in the same
        order as `groups[group_label]`. Aggregate across the list (e.g. mean/SEM per
        total_steps bucket) to build a "mean +/- seed band" plot.
    """
    os.makedirs(cache_dir, exist_ok=True)
    all_run_ids = [(label, rid) for label, rids in groups.items() for rid in rids]

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_pull_one_run, rid, project, steps_per_epoch, cache_dir): (label, rid)
            for label, rid in all_run_ids
        }
        for fut in as_completed(futures):
            label, rid = futures[fut]
            results[(label, rid)] = fut.result()
            print(f"{label} {rid}: done ({len(results[(label, rid)]['corr_df'])} eval epochs)", flush=True)

    return {
        label: [results[(label, rid)] for rid in rids]
        for label, rids in groups.items()
    }


if __name__ == "__main__":
    # Example / smoke test -- mirrors the GAE-vs-Plain, 5-seeds-each comparison this was built for.
    import time

    groups = {
        "gae": ["553r7i1o", "0xtlo1ih", "qzlq7yzv", "q0q88eze", "h528otgw"],
        "plain": ["q55g6673", "v2qc3yxa", "16wfcies", "mkncndw6", "6qjo235x"],
    }
    t0 = time.time()
    data = pull_runs_parallel(groups, project="liam-paull/calibration_rl", steps_per_epoch=20000)
    print(f"pulled {sum(len(v) for v in groups.values())} runs in {time.time()-t0:.1f}s (parallel)")
