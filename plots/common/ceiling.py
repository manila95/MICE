"""Reliability ceiling for critic-vs-Monte-Carlo correlation.

The evaluation logs, for each probe state i, a critic prediction V_hat_i and K
independent Monte-Carlo returns G_ik.  Panels (c)/(d) plot

    rho_obs = Corr(V_hat, Gbar),      Gbar_i = mean_k G_ik

against training progress.  Gbar is a *noisy* estimate of V^pi, so even a
perfect critic cannot reach rho_obs = 1.  Writing Gbar_i = V_i + eps_i with
Var[eps_i] = sigma^2 / K,

    Corr(V, Gbar) = Var[V] / (sd[V] * sd[Gbar]) = sd[V] / sd[Gbar]

so the attainable ceiling is

    rho_max = sqrt(Var[V] / Var[Gbar]),   Var[V] = Var[Gbar] - sigma^2 / K

estimated by one-way ANOVA over states: the between-state variance of the Gbar_i
estimates Var[Gbar]; the pooled within-state variance estimates sigma^2.

rho_max depends on the current policy (through Var[V^pi] and E[sigma^2]) and so
is recomputed at every evaluation point from the same states x K rollouts that
are already collected.  No extra data is required.

Attenuation is exact -- Corr(V_hat, Gbar) = Corr(V_hat, V) * rho_max(K), the same
factor for every critic -- provided V_hat is s-measurable and the rollouts are
fresh.  Two caveats on the current data, both from CLAUDE.md:

  * Full-state resets: the reference sees the full simulator state while the
    critic sees only the observation, so rho_max is slightly GENEROUS.  The
    correction favours the critic, i.e. errs in the safe direction.

  * NOT a plug-in tail.  Verified against the generating code (MICE branch
    value_function_estimation_study_intermediate_states,
    omnisafe/utils/value_eval.py): both estimate_true_value_same_state_mc and
    estimate_value_from_snapshots build raw[...]['returns'] as a pure sampled
    discounted sum (`g_r += disc_r * r_np`).  The critic's boot_r/boot_c go only
    into v_seq[-1], which feeds _rollout_target -> 'target'/'target_repeats'.
    So Gbar is plug-in-free and independent of V_hat, and the attenuation
    identity applies as stated.  ('target' is NOT plug-in-free -- it is the
    training-style target and carries the critic, which is why its within-state
    variance is ~50x smaller and its apparent ceiling ~0.999.  Do not feed
    'target' to these estimators.)

  * Heterogeneous horizons ACROSS strata -- the real defect, per CLAUDE.md.
    Probes are rolled out for remaining_horizon = max_episode_steps - t, so a
    t=0 probe's reference is a 1000-step return and a t=900 probe's is a
    100-step one, while V_hat predicts full-horizon V^pi for both.  The pooled
    block therefore mixes six different estimands (mean Gbar_r falls to 0.22 of
    its t=0 value by t=900).  Prefer block='mc_study' (t=0, full horizon,
    gamma^1000 ~ 4e-5) for the cleanest read until references run a fixed
    horizon H in every stratum.
"""

from __future__ import annotations

import glob
import os
import pickle
import re
from dataclasses import dataclass

import numpy as np

# Below this, rho_max is too small to divide by: the disattenuated correlation
# rho_obs / rho_max is marked unreliable rather than plotted.
RHO_MAX_FLOOR = 0.10


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #

def rho_max(G: np.ndarray) -> float:
    """Ceiling on Corr(V_hat, Gbar) for a perfect critic.

    G : (n_states, K) Monte-Carlo returns.  Uses ddof=1 for both the within- and
    between-state variances, so both are unbiased and their difference is an
    unbiased estimate of Var[V] before the max(., 0) truncation.
    """
    G = np.asarray(G, dtype=float)
    n, K = G.shape
    if n < 2 or K < 2:
        return np.nan
    sigma2 = G.var(axis=1, ddof=1).mean()      # pooled within-state
    s2_between = G.mean(axis=1).var(ddof=1)    # between-state (of the means)
    if s2_between <= 0:
        return 0.0
    var_v = max(s2_between - sigma2 / K, 0.0)
    return float(np.sqrt(var_v / s2_between))


def rho_max_boot(G: np.ndarray, n_boot: int = 1000, seed: int = 0,
                 q: tuple[float, float] = (2.5, 97.5)) -> tuple[float, float]:
    """Percentile bootstrap CI for rho_max, resampling states with replacement.

    The max(., 0) truncation in rho_max makes the sampling distribution pile up
    at zero when the signal is weak, so the band is asymmetric by construction.
    """
    G = np.asarray(G, dtype=float)
    n = len(G)
    if n < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = np.array([rho_max(G[i]) for i in idx])
    lo, hi = np.percentile(draws, q)
    return float(lo), float(hi)


def rho_obs(pred: np.ndarray, G: np.ndarray) -> float:
    """Observed Corr(V_hat, Gbar).  Reproduces the logged Correlation_{r,c}."""
    pred = np.asarray(pred, dtype=float)
    gbar = np.asarray(G, dtype=float).mean(axis=1)
    if pred.std() == 0 or gbar.std() == 0:
        return np.nan
    return float(np.corrcoef(pred, gbar)[0, 1])


def rho_spearman(pred: np.ndarray, G: np.ndarray) -> float:
    """Rank correlation between V_hat and Gbar.

    Pearson on the cost head is dominated by a handful of high-|cost| states; the
    gap between this and rho_obs measures how much of the Pearson value rides on
    those states.
    """
    pred = np.asarray(pred, dtype=float)
    gbar = np.asarray(G, dtype=float).mean(axis=1)
    rp = np.argsort(np.argsort(pred)).astype(float)
    rg = np.argsort(np.argsort(gbar)).astype(float)
    if rp.std() == 0 or rg.std() == 0:
        return np.nan
    return float(np.corrcoef(rp, rg)[0, 1])


def leverage_share(G: np.ndarray, top: int = 3) -> float:
    """Fraction of the between-state variance carried by the `top` states.

    Near 1 means the correlation in that panel is determined by a few states, so
    both rho_obs and rho_max have a far smaller effective sample size than n.
    """
    gbar = np.asarray(G, dtype=float).mean(axis=1)
    dev = (gbar - gbar.mean()) ** 2
    total = dev.sum()
    if total <= 0:
        return np.nan
    return float(np.sort(dev)[::-1][:top].sum() / total)


def _lam_from_splithalf(r: float, K1: int, K2: int) -> float:
    """Invert Corr(Gbar^(K1), Gbar^(K2)) = rho_max(K1) * rho_max(K2) for
    lam = Var[V] / sigma^2.

    With rho_max(K)^2 = lam*K / (lam*K + 1), the identity is a quadratic in lam:
        lam^2 K1 K2 (1 - r^2) - lam r^2 (K1 + K2) - r^2 = 0.
    For the equal-split case K1 == K2 this reduces to Spearman-Brown.
    """
    if r <= 0:
        return 0.0
    r2 = r * r
    if r2 >= 1:
        return np.inf
    a = K1 * K2 * (1 - r2)
    b = -r2 * (K1 + K2)
    c = -r2
    return float((-b + np.sqrt(b * b - 4 * a * c)) / (2 * a))


def splithalf_ceiling(G: np.ndarray, n_rep: int = 200, seed: int = 0) -> float:
    """Assumption-light cross-check on rho_max (appendix).

    Splits the K rollouts of each state into disjoint halves of size K1 and K2,
    correlates the two half-means across states, and inverts the identity
    Corr(Gbar^(K1), Gbar^(K2)) = rho_max(K1) * rho_max(K2) exactly for the
    full-K ceiling -- the same quantity as rho_max, without the ANOVA model.

    Solving exactly rather than Spearman-Brown-correcting matters because K=5
    splits unevenly (K1=2, K2=3); against simulated data with known Var[V] and
    sigma^2 the exact inversion is closer to truth than the K1==K2 approximation.
    """
    G = np.asarray(G, dtype=float)
    n, K = G.shape
    if n < 3 or K < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    K1 = K // 2
    K2 = K - K1
    out = []
    for _ in range(n_rep):
        a = np.empty(n)
        b = np.empty(n)
        for i in range(n):
            perm = rng.permutation(K)
            a[i] = G[i, perm[:K1]].mean()
            b[i] = G[i, perm[K1:]].mean()
        if a.std() == 0 or b.std() == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        lam = _lam_from_splithalf(r, K1, K2)
        out.append(1.0 if lam == np.inf else
                   (np.sqrt(lam * K / (lam * K + 1)) if lam > 0 else 0.0))
    if not out:
        return np.nan
    return float(np.nanmean(out))


def disattenuated(pred: np.ndarray, G: np.ndarray, n_boot: int = 1000,
                  seed: int = 0) -> tuple[float, float, float, bool]:
    """rho_obs / rho_max with a bootstrap CI, clipped to 1.

    Both numerator and denominator are recomputed on each bootstrap replicate so
    the CI carries their correlation.  Returns (point, lo, hi, reliable).
    """
    pred = np.asarray(pred, dtype=float)
    G = np.asarray(G, dtype=float)
    n = len(G)
    rm = rho_max(G)
    reliable = bool(np.isfinite(rm) and rm >= RHO_MAX_FLOOR)
    if not np.isfinite(rm) or rm <= 0:
        return np.nan, np.nan, np.nan, False

    point = min(rho_obs(pred, G) / rm, 1.0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = []
    for i in idx:
        rm_b = rho_max(G[i])
        if rm_b <= 0:
            continue
        draws.append(min(rho_obs(pred[i], G[i]) / rm_b, 1.0))
    if len(draws) < 0.5 * n_boot:
        return point, np.nan, np.nan, reliable
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), reliable


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #

@dataclass
class EvalPoint:
    """One evaluation checkpoint for one algorithm."""
    algo: str
    epoch: int
    block: str          # 'pooled' | 'mc_study' | 'intermediate:<step>'
    K: int
    pred: dict[str, np.ndarray]     # 'r' / 'c' -> (n_states,)
    G: dict[str, np.ndarray]        # 'r' / 'c' -> (n_states, K)

    @property
    def n_states(self) -> int:
        return len(self.G["c"])


def _blocks(d: dict):
    yield "pooled", d["pooled"]
    yield "mc_study", d["mc_study"]
    for step, sub in sorted(d["intermediate_study"].items()):
        yield f"intermediate:{step}", sub


def load_algo(directory: str, algo: str, block: str = "pooled") -> list[EvalPoint]:
    """Load one algorithm's epoch_*.pkl files from `directory`."""
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
        out.append(EvalPoint(algo=algo, epoch=int(d["epoch"]), block=block,
                             K=G["c"].shape[1], pred=pred, G=G))
    return sorted(out, key=lambda p: p.epoch)


def discover(root: str = "eval_data", block: str = "pooled") -> dict[str, list[EvalPoint]]:
    """Find algorithms under `root`.

    Either `root` itself holds epoch_*.pkl (a single unnamed run), or each
    subdirectory is one algorithm (e.g. eval_data/cpo, eval_data/pidlag).
    """
    if glob.glob(os.path.join(root, "epoch_*.pkl")):
        return {os.path.basename(os.path.normpath(root)): load_algo(root, "run", block)}
    algos = {}
    for entry in sorted(os.listdir(root)):
        sub = os.path.join(root, entry)
        if os.path.isdir(sub) and glob.glob(os.path.join(sub, "epoch_*.pkl")):
            algos[entry] = load_algo(sub, entry, block)
    return algos


# --------------------------------------------------------------------------- #
# per-algorithm curves
# --------------------------------------------------------------------------- #

def curves(points: list[EvalPoint], head: str, n_boot: int = 1000,
           seed: int = 0) -> dict[str, np.ndarray]:
    """Assemble every quantity panel (c)/(d)/(e) needs, for one head."""
    rows = {k: [] for k in ("epoch", "K", "n_states", "rho_obs", "rho_spearman",
                            "rho_max", "rho_max_lo", "rho_max_hi",
                            "rho_max_splithalf", "rho_dis", "rho_dis_lo",
                            "rho_dis_hi", "reliable", "sigma2", "var_v",
                            "frac_zero_returns", "leverage_top3")}
    for p in points:
        G, pred = p.G[head], p.pred[head]
        rm = rho_max(G)
        lo, hi = rho_max_boot(G, n_boot=n_boot, seed=seed)
        dis, dlo, dhi, ok = disattenuated(pred, G, n_boot=n_boot, seed=seed)
        sigma2 = G.var(axis=1, ddof=1).mean()
        s2b = G.mean(axis=1).var(ddof=1)
        rows["epoch"].append(p.epoch)
        rows["K"].append(p.K)
        rows["n_states"].append(p.n_states)
        rows["rho_obs"].append(rho_obs(pred, G))
        rows["rho_spearman"].append(rho_spearman(pred, G))
        rows["rho_max"].append(rm)
        rows["rho_max_splithalf"].append(splithalf_ceiling(G, seed=seed))
        rows["leverage_top3"].append(leverage_share(G, top=3))
        rows["rho_max_lo"].append(lo)
        rows["rho_max_hi"].append(hi)
        rows["rho_dis"].append(dis)
        rows["rho_dis_lo"].append(dlo)
        rows["rho_dis_hi"].append(dhi)
        rows["reliable"].append(ok)
        rows["sigma2"].append(sigma2)
        rows["var_v"].append(max(s2b - sigma2 / p.K, 0.0))
        rows["frac_zero_returns"].append(float((G == 0).mean()))
    return {k: np.asarray(v) for k, v in rows.items()}
