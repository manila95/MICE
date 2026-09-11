# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Calibration losses for value-function critics.

A regression critic (reward or cost) is fit with a task loss -- MSE or Huber, see
:meth:`~omnisafe.algorithms.on_policy.base.policy_gradient.PolicyGradient._critic_loss` -- that
scores *average* squared error but says nothing about whether the prediction is systematically
too high in one part of its range and too low in another. That's the failure mode this module
targets: "calibration" here means the classic regression-reliability sense already used elsewhere
in this repo's diagnostics (see ``Value/*/PredTrueCorr`` in
:mod:`omnisafe.algorithms.on_policy.base.policy_gradient` and the pred-vs-true scatter plots in
:mod:`omnisafe.utils.value_eval`) -- if you bucket transitions by predicted value, the average
prediction in each bucket should match the average realized/target value in that bucket. A critic
can have low MSE while being badly miscalibrated in this sense (e.g. systematically overestimating
high-cost states and underestimating low-cost ones, which is exactly the failure mode of interest
for a cost critic feeding a Lagrangian/CPO safety constraint: a biased-high cost estimate in the
states that matter most makes the agent falsely believe it is safe).

Both losses are pure functions of ``(pred, target)`` so they don't need any actor-critic-specific
wiring; the per-stream config resolution (which type, how many bins, and the reward/cost blend
coefficient) lives in
:meth:`~omnisafe.algorithms.on_policy.base.policy_gradient.PolicyGradient._calibration_loss`,
which calls these.
"""

from __future__ import annotations

import torch


def moment_calibration_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Cheap, binning-free calibration loss: batch-level bias and slope.

    A critic is calibrated in the reliability sense (see module docstring) to first order iff
    regressing ``target`` on ``pred`` over the batch recovers the identity line -- intercept 0,
    slope 1. This computes exactly that pair of terms:

    .. math::

        L = (\bar{y} - \bar{t})^2 + \left(\frac{\mathrm{Cov}(y, t)}{\mathrm{Var}(y)} - 1\right)^2

    where :math:`y` is ``pred`` and :math:`t` is ``target``. The first term is the usual bias
    (systematic over/under-prediction); the second penalizes a fitted slope other than 1, which
    catches range-dependent miscalibration (e.g. compressed or inverted spread) that a pure bias
    term misses even though it's binning-free and so needs no minimum batch size the way
    :func:`binned_calibration_loss` does.

    ``Var(y)`` is detached in the denominator: differentiating through it would let the loss
    shrink by collapsing prediction variance (trivially driving the ratio toward whatever the
    numerator needs) rather than by actually moving the covariance with the target, which is the
    only change that reflects real calibration improvement. Both terms use ``target.detach()``
    since the target (a GAE/GAE-RTG buffer value) carries no useful gradient here and must not be
    pulled toward the prediction.

    Args:
        pred: Critic predictions for the batch, shape ``(B,)``.
        target: Regression targets for the batch, shape ``(B,)``.

    Returns:
        A scalar tensor, differentiable w.r.t. ``pred``.
    """
    target = target.detach()
    bias = pred.mean() - target.mean()

    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    var_pred = (pred_centered * pred_centered).mean()
    cov = (pred_centered * target_centered).mean()

    eps = 1e-8
    slope = cov / (var_pred.detach() + eps)
    slope_term = (slope - 1.0) ** 2

    return bias * bias + slope_term


def binned_calibration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 10,
) -> torch.Tensor:
    r"""Population-weighted binned calibration loss (regression-ECE).

    The regression analogue of Expected Calibration Error: partition the batch into ``n_bins``
    equal-population buckets by predicted value, and for each non-empty bucket compare the mean
    prediction against the mean target. The loss is the population-weighted mean squared gap:

    .. math::

        L = \sum_{k=1}^{K} \frac{|B_k|}{N} \left(\bar{y}_{B_k} - \bar{t}_{B_k}\right)^2

    Unlike :func:`moment_calibration_loss`, this can catch calibration errors that are non-linear
    in the predicted value (e.g. well-calibrated in the bulk of the range but biased high only for
    the top decile of predicted cost) -- exactly the shape of failure a global bias/slope check can
    average away.

    Bin *edges* are quantiles of ``pred.detach()`` -- gradient is deliberately cut there. Letting
    the boundary itself move in response to the loss would hand the optimizer an escape hatch that
    has nothing to do with calibration: a prediction sitting at a bin edge with a large local error
    could reduce the loss by nudging the *edge* out of the way rather than by correcting the
    prediction. Gradient still flows into the loss through each bin's mean prediction
    (``pred[mask].mean()``), which is what actually gets penalized toward the bin's mean target --
    the standard trick behind differentiable ECE-style calibration objectives (e.g. Kumar et al.
    2018's MMCE): fix bin *membership*, keep gradient on bin *content*.

    ``n_bins`` is clamped to ``batch_size // 2`` (and the whole loss short-circuits to 0 for
    batches smaller than that) so a minibatch too small to support the requested resolution
    degrades gracefully instead of producing single-sample, all-noise "bins".

    Args:
        pred: Critic predictions for the batch, shape ``(B,)``.
        target: Regression targets for the batch, shape ``(B,)``.
        n_bins: Requested number of equal-population bins. Defaults to ``10``.

    Returns:
        A scalar tensor, differentiable w.r.t. ``pred``. Exactly ``0`` (no gradient) if the batch
        is too small (fewer than 2 usable bins) to bin at all.
    """
    assert pred.shape == target.shape
    batch_size = pred.shape[0]

    n_bins = min(int(n_bins), batch_size // 2)
    if n_bins < 1:
        return pred.new_zeros(())

    target = target.detach()
    pred_detached = pred.detach()
    with torch.no_grad():
        quantiles = torch.linspace(0.0, 1.0, n_bins + 1, device=pred.device, dtype=pred.dtype)
        edges = torch.quantile(pred_detached, quantiles)

    total_sq_err = pred.new_zeros(())
    total_count = 0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (pred_detached >= lo) & (pred_detached < hi)
        else:
            # Last bin is closed on both ends so the single largest prediction (== edges[-1],
            # the 1.0-quantile) isn't left out of every bin by the half-open test above.
            mask = (pred_detached >= lo) & (pred_detached <= hi)
        count = int(mask.sum().item())
        if count == 0:
            continue
        bin_pred_mean = pred[mask].mean()
        bin_target_mean = target[mask].mean()
        total_sq_err = total_sq_err + count * (bin_pred_mean - bin_target_mean) ** 2
        total_count += count

    if total_count == 0:
        return pred.new_zeros(())
    return total_sq_err / total_count
