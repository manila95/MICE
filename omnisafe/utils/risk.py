"""Risk estimation for SRPL (risk-aware safe policy learning).

Ported from the reference implementation at
https://github.com/montrealrobotics/risk_aware_exploration (``working_safepo`` branch,
``src/models/risk_models.py`` and ``src/utils.py``). The SafePO integration the research branch
carries (``manila95/Safe-Policy-Optimization``, ``risk_aware_exploration``) references these
classes but does not define them; this module is the port of the originals.

The method has three parts:

- **A risk label.** :func:`compute_fear` turns an episode's cost sequence into a per-step
  distance to the nearest cost event, looking both forward and backward. A step adjacent to a
  violation scores 0; one far from any violation scores ``max_dist``. Those distances are
  bucketed into ``quantile_num`` bins of ``quantile_size`` steps, giving a one-hot label.
- **A risk estimator.** :class:`BayesRiskEst` is a classifier from observation to a distribution
  over those bins -- "how close to a violation is this state". It is trained online, on a replay
  buffer of states labelled once their episode finishes (the label is only knowable in
  hindsight), with the loss weighted towards the nearest-violation bin because that class is rare.
- **A risk-conditioned policy.** The estimator's output is fed to the actor and critics alongside
  the observation; see :class:`omnisafe.models.risk_net.RiskNet`.

The estimator is only ever a *predictor* here -- no gradient flows from the policy loss into it
(the policy consumes its output under ``no_grad``), so it cannot be bent to make the policy's
objective easier.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def get_activation(name: str) -> nn.Module:
    """Activation lookup, matching the reference implementation's set."""
    return {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'softmax': nn.Softmax(dim=-1),
        'logsoftmax': nn.LogSoftmax(dim=-1),
    }[name]


def compute_fear(costs: torch.Tensor, max_dist: float = 1000.0) -> torch.Tensor:
    """Per-step distance to the nearest cost event in an episode, forward or backward.

    A step where a cost fires scores 0; otherwise the score is how many steps away the nearest
    cost is, in either direction, capped at ``max_dist`` when the episode has none on that side.
    Both directions matter: a state is risky whether it is about to cause a violation or has just
    come out of one.

    Verbatim in behaviour from the reference ``compute_fear``, rewritten as two vectorized passes
    instead of a Python loop over timesteps -- the loop ran once per step per environment per
    episode, which is the hot path here.

    Args:
        costs: ``(T,)`` cost sequence for one episode. Non-zero is treated as a violation.
        max_dist: Score for a step with no cost event on that side.

    Returns:
        ``(T,)`` tensor of distances.
    """
    costs = costs.reshape(-1)
    horizon = costs.numel()
    fear = torch.full((horizon,), float(max_dist), dtype=torch.float32)
    if horizon == 0:
        return fear
    hit = costs != 0
    if not bool(hit.any()):
        return fear

    idx = torch.arange(horizon, dtype=torch.float32)
    # Forward pass: distance since the most recent cost. Running maximum of the indices where a
    # cost fired gives the latest one at or before each step; -inf before the first.
    last = torch.where(hit, idx, torch.full_like(idx, float('-inf')))
    last = torch.cummax(last, dim=0).values
    fear_fwd = torch.where(torch.isfinite(last), idx - last, torch.full_like(idx, float(max_dist)))
    # Backward pass: the mirror image, distance until the next cost.
    nxt = torch.where(hit, idx, torch.full_like(idx, float('inf')))
    nxt = torch.cummin(nxt.flip(0), dim=0).values.flip(0)
    fear_bwd = torch.where(torch.isfinite(nxt), nxt - idx, torch.full_like(idx, float(max_dist)))
    return torch.minimum(torch.minimum(fear_fwd, fear_bwd), fear.new_full((horizon,), max_dist))


def quantile_bins(quantile_num: int, quantile_size: int) -> np.ndarray:
    """Bin edges for bucketing :func:`compute_fear` distances into a one-hot risk label."""
    return np.array([i * quantile_size for i in range(quantile_num + 1)])


def fear_to_quantile(fear: torch.Tensor, bins: np.ndarray) -> torch.Tensor:
    """One-hot encode fear distances into the quantile bins.

    A distance past the last edge falls in no bin and its row is all zeros, exactly as the
    reference's ``np.histogram``-based encoding leaves it: those states are far from any
    violation and carry no risk signal to learn from.

    Args:
        fear: ``(N,)`` distances from :func:`compute_fear`.
        bins: Edges from :func:`quantile_bins`.

    Returns:
        ``(N, len(bins) - 1)`` one-hot tensor.
    """
    values = fear.reshape(-1).cpu().numpy()
    counts = np.stack([np.histogram(v, bins=bins)[0] for v in values]) if values.size else (
        np.zeros((0, len(bins) - 1))
    )
    return torch.as_tensor(counts, dtype=torch.float32)


def fear_to_binary(fear: torch.Tensor, fear_radius: int) -> torch.Tensor:
    """One-hot encode fear distances as "within ``fear_radius`` of a violation, or not"."""
    near = (fear.reshape(-1) <= fear_radius).long()
    return torch.nn.functional.one_hot(near, num_classes=2).float()


class RiskReplayBuffer:
    """States and their hindsight risk labels, for training the estimator online.

    A ring buffer rather than the reference's unbounded list: this is fed every finished episode
    of every parallel environment for the whole run, and the labels are non-stationary anyway
    (they describe where *this* policy gets into trouble), so old entries are the ones worth
    dropping.
    """

    def __init__(self, capacity: int, obs_dim: int, risk_size: int) -> None:
        """Initialize an instance of :class:`RiskReplayBuffer`."""
        self._obs = torch.zeros(capacity, obs_dim, dtype=torch.float32)
        self._risks = torch.zeros(capacity, risk_size, dtype=torch.float32)
        self._capacity = capacity
        self._pos = 0
        self._full = False

    def __len__(self) -> int:
        """Number of entries currently stored."""
        return self._capacity if self._full else self._pos

    def add(self, obs: torch.Tensor, risks: torch.Tensor) -> None:
        """Append a batch of states and their one-hot risk labels, wrapping when full."""
        obs = obs.detach().reshape(-1, self._obs.shape[1]).cpu().float()
        risks = risks.detach().reshape(-1, self._risks.shape[1]).cpu().float()
        for start in range(0, obs.shape[0], self._capacity):
            chunk_obs = obs[start : start + self._capacity]
            chunk_risks = risks[start : start + self._capacity]
            n = chunk_obs.shape[0]
            end = self._pos + n
            if end <= self._capacity:
                self._obs[self._pos : end] = chunk_obs
                self._risks[self._pos : end] = chunk_risks
            else:
                split = self._capacity - self._pos
                self._obs[self._pos :] = chunk_obs[:split]
                self._risks[self._pos :] = chunk_risks[:split]
                self._obs[: n - split] = chunk_obs[split:]
                self._risks[: n - split] = chunk_risks[split:]
                self._full = True
            self._pos = end % self._capacity
            if end >= self._capacity:
                self._full = True

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a uniform batch of ``(obs, risk_label)``."""
        size = len(self)
        idx = torch.randint(0, size, (min(batch_size, size),))
        return self._obs[idx], self._risks[idx]


class BayesRiskEst(nn.Module):
    """Classifier from observation to a distribution over risk bins.

    Ported from the reference ``BayesRiskEst``, keeping its layer sizes, batch-norm placement,
    dropout and log-softmax output -- the output is log-probabilities, so a consumer wanting
    probabilities exponentiates (as the reference's rollout does).

    Batch norm means the module's behaviour depends on train/eval mode, and on a batch of one it
    cannot normalize at all. It is therefore kept in ``eval()`` except during its own update; see
    :meth:`omnisafe.adapter.onpolicy_adapter.OnPolicyAdapter._update_risk_model`.
    """

    def __init__(
        self,
        obs_size: int,
        out_size: int = 2,
        fc1_size: int = 128,
        fc2_size: int = 128,
        fc3_size: int = 128,
        fc4_size: int = 128,
        batch_norm: bool = True,
        activation: str = 'relu',
    ) -> None:
        """Initialize an instance of :class:`BayesRiskEst`."""
        super().__init__()
        self.obs_size = obs_size
        self.batch_norm = batch_norm
        self.fc1 = nn.Linear(obs_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        self.out = nn.Linear(fc4_size, out_size)
        self.bnorm1 = nn.BatchNorm1d(fc1_size)
        self.bnorm2 = nn.BatchNorm1d(fc2_size)
        self.bnorm3 = nn.BatchNorm1d(fc3_size)
        self.bnorm4 = nn.BatchNorm1d(fc4_size)
        self.activation = get_activation(activation)
        self.logsoftmax = get_activation('logsoftmax')
        self.dropout = nn.Dropout(0.2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Log-probabilities over the risk bins for each observation."""
        # Trailing features (an augmented observation, say) are dropped rather than fed in, same
        # as the reference: the estimator is defined over the raw observation it was sized for.
        x = obs[..., : self.obs_size]
        if self.batch_norm:
            x = self.bnorm1(self.activation(self.fc1(x)))
            x = self.bnorm2(self.activation(self.fc2(x)))
            x = self.bnorm3(self.activation(self.dropout(self.fc3(x))))
            x = self.bnorm4(self.activation(self.dropout(self.fc4(x))))
        else:
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.activation(self.dropout(self.fc3(x)))
            x = self.activation(self.dropout(self.fc4(x)))
        return self.logsoftmax(self.out(x))
