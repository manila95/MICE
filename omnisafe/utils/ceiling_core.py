"""
Correlation-ceiling estimator for value-function evaluation under noisy MC targets.

The question: a critic learns V*(s) = E[Y | s] (expected discounted cost-to-go).
Your evaluation correlates the critic's prediction against a SINGLE realized
return Y drawn from a state. If the within-state return variance Var(Y | s) is
large relative to the between-state variance Var_s(E[Y | s]), then even a PERFECT
critic (V = E[Y|s]) cannot correlate strongly with single samples. The ceiling is

    corr_max = sqrt( B / (B + W) ),   B = Var_s(E[Y|s]),  W = E_s[Var(Y|s)].

We estimate B and W by drawing K independent rollouts (branches) from each of N
anchor states, giving a matrix Y[N, K]. Per-anchor mean m_i estimates E[Y|s_i];
per-anchor variance v_i estimates Var(Y|s_i).

Bias correction (important at small K): Var_i(m_i) = B + W/K in expectation, so
    B_hat = Var_i(m_i) - mean_i(v_i) / K.

This module is numpy-only. Feed it Y[N,K] and your critic predictions V[N].
"""

from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# rank helper (Spearman without scipy)
# --------------------------------------------------------------------------- #
def _rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum - 1) / 2.0
    return avg[inv]


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    return _pearson(_rankdata(x), _rankdata(y))


def _explained_variance(pred, target):
    """1 - Var(target - pred)/Var(target). Standard PPO critic diagnostic.
    NOTE: uses the VARIANCE of the residual, so it is INVARIANT to a constant
    offset -- it penalizes scale/structure errors but NOT pure bias. Use
    _r2_score (below) or stratified_bias to detect a constant/level bias such
    as the downward flattening of a truncated-bootstrap TD(lambda) cost critic."""
    target = np.asarray(target, float); pred = np.asarray(pred, float)
    vt = np.var(target)
    if vt < 1e-12:
        return np.nan
    return float(1.0 - np.var(target - pred) / vt)


def _r2_score(pred, target):
    """1 - MSE/Var(target) = coefficient of determination. Unlike explained
    variance, the MSE numerator includes the squared mean error, so this DOES
    penalize a constant bias. The gap (explained_variance - r2) isolates how
    much of the error is pure level bias vs. structural."""
    target = np.asarray(target, float); pred = np.asarray(pred, float)
    vt = np.var(target)
    if vt < 1e-12:
        return np.nan
    mse = np.mean((target - pred) ** 2)
    return float(1.0 - mse / vt)


# --------------------------------------------------------------------------- #
# variance components + ceiling
# --------------------------------------------------------------------------- #
def variance_components(Y):
    """Y: [N, K] array of returns, K independent branches per anchor state.
    Returns per-anchor means/vars and the (bias-corrected) ceiling."""
    Y = np.asarray(Y, float)
    N, K = Y.shape
    if K < 2:
        raise ValueError("need K >= 2 branches per anchor to estimate within-var")
    m = Y.mean(axis=1)                       # E[Y|s_i] estimate
    v = Y.var(axis=1, ddof=1)                # Var(Y|s_i) estimate
    W = float(v.mean())                      # within-state variance
    B_naive = float(np.var(m, ddof=1))       # inflated by W/K
    B_corr = max(B_naive - W / K, 0.0)       # bias-corrected between-variance

    def ceil(B):
        denom = B + W
        return float(np.sqrt(B / denom)) if denom > 0 else np.nan

    return {
        "N": N, "K": K,
        "m": m, "v": v,
        "within_W": W,
        "between_naive": B_naive,
        "between_corrected": B_corr,
        "ceiling_naive": ceil(B_naive),
        "ceiling_corrected": ceil(B_corr),
        "noise_to_signal": (W / B_corr) if B_corr > 0 else np.inf,
    }


def bootstrap_ceiling(Y, n_boot=1000, seed=0):
    """Bootstrap over anchors -> CI on the corrected ceiling."""
    Y = np.asarray(Y, float)
    N = Y.shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        vals.append(variance_components(Y[idx])["ceiling_corrected"])
    vals = np.array([x for x in vals if np.isfinite(x)])
    return {
        "ceiling_mean": float(vals.mean()),
        "ceiling_lo": float(np.percentile(vals, 2.5)),
        "ceiling_hi": float(np.percentile(vals, 97.5)),
    }


# --------------------------------------------------------------------------- #
# full report: is the observed correlation AT the ceiling, or below it?
# --------------------------------------------------------------------------- #
def full_report(V, Y, y_single_col=0, n_boot=1000, seed=0):
    """
    V: [N] critic predictions at the anchor states.
    Y: [N, K] branched returns from those states.

    Compares four things:
      - corr(V, Y_single): what you currently measure (uses one branch as the
        single realized return, mimicking on-policy MC targets).
      - corr(V, m):        critic vs the DENOISED target E[Y|s]. If this is high
        while corr(V, Y_single) is low, the critic is good and the metric was
        the problem.
      - ceiling:           best achievable corr(perfect, Y_single).
      - ratio:             corr(V, Y_single) / ceiling. Near 1 => metric-limited.
    """
    V = np.asarray(V, float)
    Y = np.asarray(Y, float)
    vc = variance_components(Y)
    m = vc["m"]
    y_single = Y[:, y_single_col]

    corr_single = _pearson(V, y_single)
    corr_mean = _pearson(V, m)
    ceiling = vc["ceiling_corrected"]
    boot = bootstrap_ceiling(Y, n_boot=n_boot, seed=seed)

    rep = {
        **vc,
        "corr_V_Ysingle": corr_single,
        "corr_V_mean": corr_mean,
        "spearman_V_Ysingle": _spearman(V, y_single),
        "spearman_V_mean": _spearman(V, m),
        "explained_var_V_mean": _explained_variance(V, m),   # blind to level bias
        "explained_var_V_Ysingle": _explained_variance(V, y_single),
        "r2_V_mean": _r2_score(V, m),                        # penalizes level bias
        "r2_V_Ysingle": _r2_score(V, y_single),
        "level_bias_V_mean": float(np.mean(V - m)),          # signed mean error
        "ceiling": ceiling,
        "ceiling_ci": (boot["ceiling_lo"], boot["ceiling_hi"]),
        "ratio_to_ceiling": (corr_single / ceiling) if ceiling and np.isfinite(ceiling) and ceiling > 0 else np.nan,
        "V_mean": float(V.mean()), "V_std": float(V.std()),
        "V_min": float(V.min()), "V_max": float(V.max()),
    }
    return rep


def stratified_bias(V, target, mask):
    """Signed bias mean(V - target) inside vs outside a boolean mask (e.g.
    near-hazard states). This is where a truncated-bootstrap TD(lambda) critic
    should show a downward bias that MC/SR critics remove."""
    mask = np.asarray(mask, bool)
    d = np.asarray(V, float) - np.asarray(target, float)
    return {
        "bias_in_mask": float(d[mask].mean()) if mask.any() else np.nan,
        "bias_out_mask": float(d[~mask].mean()) if (~mask).any() else np.nan,
        "n_in": int(mask.sum()), "n_out": int((~mask).sum()),
    }


# --------------------------------------------------------------------------- #
# offline fallback: kNN-in-observation-space (no env reset needed).
# Approximates E[Y|s] by averaging Y over nearest neighbors in obs space.
# BIASED: neighbors are not the same state, so between-state variation leaks
# into the "within" estimate -> W overestimated -> ceiling UNDERESTIMATED.
# Treat the result as a conservative lower bound on the ceiling.
# --------------------------------------------------------------------------- #
def knn_ceiling(obs, Y_single, k=16, standardize=True, seed=0, max_anchors=4000):
    """obs: [M, d] observations. Y_single: [M] single realized returns.
    Returns a lower-bound ceiling estimate from local neighborhoods."""
    obs = np.asarray(obs, float)
    Y_single = np.asarray(Y_single, float)
    M = obs.shape[0]
    if standardize:
        obs = (obs - obs.mean(0)) / (obs.std(0) + 1e-8)
    rng = np.random.default_rng(seed)
    anchors = np.arange(M) if M <= max_anchors else rng.choice(M, max_anchors, replace=False)

    local_means, local_vars = [], []
    for i in anchors:
        d2 = np.sum((obs - obs[i]) ** 2, axis=1)
        nn = np.argpartition(d2, min(k, M - 1))[:k]
        yv = Y_single[nn]
        local_means.append(yv.mean())
        local_vars.append(yv.var(ddof=1))
    local_means = np.array(local_means); local_vars = np.array(local_vars)
    W = float(local_vars.mean())
    B = max(float(np.var(local_means, ddof=1)) - W / k, 0.0)
    ceiling = float(np.sqrt(B / (B + W))) if (B + W) > 0 else np.nan
    return {
        "ceiling_lowerbound": ceiling, "within_W": W, "between_corrected": B,
        "note": "kNN neighbors are not identical states; ceiling is a lower bound.",
    }