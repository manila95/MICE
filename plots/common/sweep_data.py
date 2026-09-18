"""Load a wandb-sweep eval tree (cell/seed/epoch_*.pkl) and aggregate across seeds.

Sweep 6m8yuigd is 2 (adv estimator: gae|plain) x 2 (update_iters) x 2 (critic lr)
x 5 seeds, 37 eval epochs to 500, K=10 rollouts per probe.

Aggregation is per-seed-then-across-seeds: rho_obs and rho_max are computed
separately for each run (each seed has its own critic and its own probe values),
and only then averaged.  Pooling states across seeds would mix different critics
and different value scales into a single correlation, which is not the same
quantity.  The band across seeds is therefore the headline uncertainty; the
bootstrap over states is a secondary, per-run quantity (off by default -- it is
40x37x2 fits, so pass want_boot only when you need it).
"""

from __future__ import annotations

import csv
import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np

from common.ceiling import (EvalPoint, _blocks, leverage_share, rho_max, rho_obs,
                         rho_spearman)

CELL_RE = re.compile(r"^(?P<adv>[a-z-]+)_iters(?P<iters>\d+)_lr(?P<lr>[0-9.e-]+)$")


@dataclass
class Cell:
    """One design point of the sweep, holding its seeds."""
    name: str
    adv: str
    update_iters: int
    critic_lr: float
    seeds: dict[int, list[EvalPoint]] = field(default_factory=dict)

    def label(self) -> str:
        return f"iters={self.update_iters}, lr={self.critic_lr:g}"


def _load_dir(directory: str, block: str) -> list[EvalPoint]:
    import pickle
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "epoch_*.pkl"))):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        sub = dict(_blocks(d)).get(block)
        if sub is None:
            raise KeyError(f"{path}: no block {block!r}")
        raw = sub["raw"]
        G = {h: np.asarray(raw[h]["returns"], dtype=float) for h in ("r", "c")}
        pred = {h: np.asarray(raw[h]["pred"], dtype=float) for h in ("r", "c")}
        ep = EvalPoint(algo=directory, epoch=int(d["epoch"]), block=block,
                       K=G["c"].shape[1], pred=pred, G=G)
        # The training-style regression target y^kappa, computed by the run's OWN
        # estimator -- the kappa-return for gae cells, the plug-in tail estimate
        # for plain cells.  'target' is its mean over the K repeats;
        # 'target_repeats' are the K single-sample versions, which is the form the
        # critic actually regresses on during training.
        ep.target = {h: np.asarray(raw[h]["target"], dtype=float) for h in ("r", "c")}
        ep.target_repeats = {h: np.asarray(raw[h]["target_repeats"], dtype=float)
                             for h in ("r", "c")}
        out.append(ep)
    return sorted(out, key=lambda p: p.epoch)


def load_sweep(root: str, block: str = "pooled") -> dict[str, Cell]:
    """Read root/<cell>/seed<N>/epoch_*.pkl into Cell objects."""
    cells: dict[str, Cell] = {}
    for entry in sorted(os.listdir(root)):
        m = CELL_RE.match(entry)
        if not m or not os.path.isdir(os.path.join(root, entry)):
            continue
        cell = Cell(name=entry, adv=m["adv"], update_iters=int(m["iters"]),
                    critic_lr=float(m["lr"]))
        for sd in sorted(os.listdir(os.path.join(root, entry))):
            sm = re.match(r"^seed(\d+)$", sd)
            if not sm:
                continue
            pts = _load_dir(os.path.join(root, entry, sd), block)
            if pts:
                cell.seeds[int(sm[1])] = pts
        if cell.seeds:
            cells[entry] = cell
    return cells


# --------------------------------------------------------------------------- #
# vectorised bootstrap over states (optional; the seed band is the default one)
# --------------------------------------------------------------------------- #

def _corr(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def boot_over_states(pred: np.ndarray, G: np.ndarray, n_boot: int = 1000,
                     seed: int = 0) -> tuple[float, float, float, float]:
    """95% CI for rho_max and for rho_obs/rho_max, all replicates at once.

    Same estimator as ceiling.rho_max_boot / ceiling.disattenuated, but computed
    on a (n_boot, n, K) gather instead of a Python loop -- the sweep needs
    ~3k of these, which the looping version is too slow for.
    """
    G = np.asarray(G, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n, K = G.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    Gi = G[idx]                                     # (B, n, K)
    within = Gi.var(axis=2, ddof=1).mean(axis=1)    # (B,)
    means = Gi.mean(axis=2)                         # (B, n)
    between = means.var(axis=1, ddof=1)             # (B,)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_v = np.clip(between - within / K, 0.0, None)
        rm = np.sqrt(np.where(between > 0, var_v / between, 0.0))

        p = pred[idx]                               # (B, n)
        pc = p - p.mean(axis=1, keepdims=True)
        mc = means - means.mean(axis=1, keepdims=True)
        denom = np.sqrt((pc ** 2).sum(1) * (mc ** 2).sum(1))
        ro = np.where(denom > 0, (pc * mc).sum(1) / denom, np.nan)
        dis = np.where(rm > 0, np.minimum(ro / rm, 1.0), np.nan)

    ok = np.isfinite(rm)
    rlo, rhi = np.percentile(rm[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan)
    ok2 = np.isfinite(dis)
    dlo, dhi = np.percentile(dis[ok2], [2.5, 97.5]) if ok2.any() else (np.nan, np.nan)
    ok3 = np.isfinite(ro)
    olo, ohi = np.percentile(ro[ok3], [2.5, 97.5]) if ok3.any() else (np.nan, np.nan)
    return float(rlo), float(rhi), float(dlo), float(dhi), float(olo), float(ohi)


def splithalf_fast(G: np.ndarray, n_rep: int = 200, seed: int = 0) -> float:
    """Vectorised ceiling.splithalf_ceiling -- same estimator, all reps at once.

    Splits each state's K rollouts into disjoint halves, correlates the two
    half-means across states, and inverts
    Corr(Gbar^(K1), Gbar^(K2)) = rho_max(K1) * rho_max(K2) exactly.
    """
    from ceiling import _lam_from_splithalf
    G = np.asarray(G, dtype=float)
    n, K = G.shape
    if n < 3 or K < 2:
        return np.nan
    K1, K2 = K // 2, K - K // 2
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((n_rep, n, K)), axis=2)
    Gp = np.take_along_axis(np.broadcast_to(G, (n_rep, n, K)), order, axis=2)
    A, B = Gp[:, :, :K1].mean(2), Gp[:, :, K1:].mean(2)
    Ac = A - A.mean(1, keepdims=True)
    Bc = B - B.mean(1, keepdims=True)
    den = np.sqrt((Ac ** 2).sum(1) * (Bc ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, (Ac * Bc).sum(1) / den, np.nan)
    out = []
    for ri in r[np.isfinite(r)]:
        lam = _lam_from_splithalf(float(ri), K1, K2)
        out.append(1.0 if lam == np.inf else
                   (np.sqrt(lam * K / (lam * K + 1)) if lam > 0 else 0.0))
    return float(np.mean(out)) if out else np.nan


# --------------------------------------------------------------------------- #
# per-run curves, then across seeds
# --------------------------------------------------------------------------- #

FIELDS = ("rho_obs", "rho_max", "rho_dis", "rho_spearman", "leverage_top3",
          "sigma2", "var_v", "frac_zero_returns",
          # target-agreement diagnostics (see cell_curves docstring)
          "rho_target", "rho_target_1s", "self_gap", "self_gap_1s",
          "rho_target_ref")


def run_curves(points: list[EvalPoint], head: str, want_boot: bool = False,
               n_boot: int = 1000, seed: int = 0) -> dict[str, np.ndarray]:
    """Point estimates (and optionally state-bootstrap CIs) for one run."""
    rows: dict[str, list] = {k: [] for k in ("epoch", "K") + FIELDS}
    if want_boot:
        rows.update({k: [] for k in ("rho_max_lo", "rho_max_hi",
                                     "rho_dis_lo", "rho_dis_hi",
                                     "rho_obs_lo", "rho_obs_hi",
                                     "rho_max_splithalf")})
    for p in points:
        G, pred = p.G[head], p.pred[head]
        rm = rho_max(G)
        ro = rho_obs(pred, G)
        sigma2 = G.var(axis=1, ddof=1).mean()
        s2b = G.mean(axis=1).var(ddof=1)
        rows["epoch"].append(p.epoch)
        rows["K"].append(p.K)
        rows["rho_obs"].append(ro)
        rows["rho_max"].append(rm)
        rows["rho_dis"].append(min(ro / rm, 1.0) if rm and rm > 0 else np.nan)
        rows["rho_spearman"].append(rho_spearman(pred, G))
        rows["leverage_top3"].append(leverage_share(G, top=3))
        rows["sigma2"].append(sigma2)
        rows["var_v"].append(max(s2b - sigma2 / p.K, 0.0))
        rows["frac_zero_returns"].append(float((G == 0).mean()))

        # How much of the critic's apparent accuracy is agreement with its own
        # regression target rather than with the truth.  For kappa<1 (gae) the
        # target is anchored to the frozen critic at every interior step, so
        # rho_target is inflated; for kappa=1 (plain) the target reduces to the
        # plug-in tail estimate and rho_target == rho_obs by construction.
        tgt = getattr(p, "target", {}).get(head)
        tr = getattr(p, "target_repeats", {}).get(head)
        rt = _corr(pred, tgt) if tgt is not None else np.nan
        # Single-sample form: correlate against each repeat separately, then
        # average -- this is the noise level the critic actually trains at.
        rt1 = (float(np.nanmean([_corr(pred, tr[:, k]) for k in range(tr.shape[1])]))
               if tr is not None else np.nan)
        rows["rho_target"].append(rt)
        rows["rho_target_1s"].append(rt1)
        rows["self_gap"].append(rt - ro)
        rows["self_gap_1s"].append(rt1 - ro)
        rows["rho_target_ref"].append(_corr(tgt, G.mean(axis=1))
                                      if tgt is not None else np.nan)
        if want_boot:
            rlo, rhi, dlo, dhi, olo, ohi = boot_over_states(
                pred, G, n_boot=n_boot, seed=seed)
            rows["rho_max_lo"].append(rlo)
            rows["rho_max_hi"].append(rhi)
            rows["rho_dis_lo"].append(dlo)
            rows["rho_dis_hi"].append(dhi)
            rows["rho_obs_lo"].append(olo)
            rows["rho_obs_hi"].append(ohi)
            rows["rho_max_splithalf"].append(splithalf_fast(G, seed=seed))
    return {k: np.asarray(v) for k, v in rows.items()}


def cell_curves(cell: Cell, head: str, **kw) -> dict[str, np.ndarray]:
    """Per-seed curves aligned on the common epoch grid, plus seed mean/sd.

    Returns 'epoch', per-seed matrices '<field>_seeds' of shape (n_seeds,
    n_epochs), and '<field>' / '<field>_sd' for the across-seed mean and
    (ddof=1) standard deviation.
    """
    per = {sd: run_curves(pts, head, **kw) for sd, pts in sorted(cell.seeds.items())}
    grids = [set(c["epoch"].tolist()) for c in per.values()]
    epochs = np.array(sorted(set.intersection(*grids))) if grids else np.array([])

    extra = tuple(k for k in ("rho_max_lo", "rho_max_hi", "rho_dis_lo", "rho_dis_hi",
                              "rho_obs_lo", "rho_obs_hi", "rho_max_splithalf")
                  if per and k in next(iter(per.values())))
    out: dict[str, np.ndarray] = {"epoch": epochs,
                                  "seeds": np.array(sorted(per)),
                                  "n_seeds": np.array(len(per))}
    for f in FIELDS + extra:
        M = np.vstack([
            np.interp(epochs, c["epoch"], c[f], left=np.nan, right=np.nan)
            if not np.array_equal(c["epoch"], epochs)
            else c[f]
            for c in per.values()
        ]) if per else np.zeros((0, len(epochs)))
        out[f + "_seeds"] = M
        out[f] = np.nanmean(M, axis=0)
        out[f + "_sd"] = np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.zeros(len(epochs))
    out["K"] = np.array([c["K"][0] for c in per.values()])
    return out


def read_manifest(root: str) -> list[dict]:
    path = os.path.join(root, "manifest.csv")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def load_history(root: str) -> dict[str, dict[str, np.ndarray]]:
    """Episodic return/cost curves per cell, averaged across seeds.

    These come from the run history (download_sweep.py --history), not from the
    eval_data artifacts, and are logged every epoch rather than only at eval
    epochs.
    """
    path = os.path.join(root, "history.csv")
    if not os.path.exists(path):
        return {}

    dense = ["Metrics/EpRet", "Metrics/EpCost"]          # logged every epoch
    sparse = ["Value/Train/AfterUpdate/CostCriticCorr",  # logged at eval epochs only
              "Value/Train/AfterUpdate/RewardCriticCorr",
              "Value/Train/BeforeUpdate/CostCriticCorr",
              "Value/Train/BeforeUpdate/RewardCriticCorr",
              "Value/Train/AfterUpdate/CostPredTrueCorr",
              "Value/Train/AfterUpdate/RewardPredTrueCorr"]
    alias = {"Metrics/EpRet": "EpRet", "Metrics/EpCost": "EpCost",
             "Value/Train/AfterUpdate/CostCriticCorr": "TrainCriticCorr_c",
             "Value/Train/AfterUpdate/RewardCriticCorr": "TrainCriticCorr_r",
             "Value/Train/BeforeUpdate/CostCriticCorr": "TrainCriticCorrPre_c",
             "Value/Train/BeforeUpdate/RewardCriticCorr": "TrainCriticCorrPre_r",
             "Value/Train/AfterUpdate/CostPredTrueCorr": "TrainPredTrue_c",
             "Value/Train/AfterUpdate/RewardPredTrueCorr": "TrainPredTrue_r"}

    per: dict[str, dict[int, list[dict]]] = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                r["_step"] = float(r["TotalEnvSteps"])
            except (TypeError, ValueError):
                continue
            per.setdefault(r["cell"], {}).setdefault(int(r["seed"]), []).append(r)

    def series(rows, key):
        """(steps, values) for one metric, dropping rows where it is absent."""
        xs, ys = [], []
        for r in rows:
            v = r.get(key)
            if v in (None, "", "NaN"):
                continue
            try:
                y = float(v)
            except ValueError:
                continue
            if np.isfinite(y):
                xs.append(r["_step"])
                ys.append(y)
        o = np.argsort(xs)
        return np.asarray(xs)[o], np.asarray(ys)[o]

    out: dict[str, dict[str, np.ndarray]] = {}
    for cell, seeds in per.items():
        d: dict[str, np.ndarray] = {"n_seeds": np.array(len(seeds))}
        for key in dense + sparse:
            per_seed = [series(rows, key) for rows in seeds.values()]
            per_seed = [(x, y) for x, y in per_seed if len(x)]
            if not per_seed:
                continue
            grid = min((x for x, _ in per_seed), key=len)
            M = np.vstack([np.interp(grid, x, y) for x, y in per_seed])
            name = alias[key]
            d[name + "_steps"] = grid
            d[name] = M.mean(0)
            d[name + "_sd"] = M.std(0, ddof=1) if len(M) > 1 else np.zeros(len(grid))
        d["steps"] = d.get("EpRet_steps", np.array([]))
        out[cell] = d
    return out
