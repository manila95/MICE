"""Constants and scoring helpers shared by the paper's figures.

The value groups referred to throughout are defined on the Monte Carlo reference
Gbar: low value (Gbar < gamma^500), intermediate value (gamma^500 <= Gbar <
gamma^5) and high value (Gbar >= gamma^5).  A "high value" state is one whose
discounted cost-to-go is large; it is a statement about the value function, not a
claim about whether the state is safe to be in.
"""

from __future__ import annotations

import numpy as np

from common.sweep_data import load_history

GAMMA = 0.99         # cost_gamma in every run
SPE = 20000          # steps per epoch
COST_LIMIT = 10.0    # beta


def avg_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks of x, ties sharing their average rank."""
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inv]


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Probability that a labelled state outranks an unlabelled one (Mann-Whitney).

    0.5 for any constant score, 1 for a perfect ranking of the labelled states.
    """
    n1 = int(label.sum())
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((avg_ranks(score)[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def beta_crossing(root: str, cell: str) -> float:
    """Steps at which the seed-mean episodic cost first reaches the limit.

    Everything to the left is the constraint-reduction phase -- the safe
    exploration the paper is about -- and it is where the critic diagnostics
    must be read, not at convergence.
    """
    h = load_history(root)[cell]
    below = np.where(h["EpCost"] <= COST_LIMIT)[0]
    return float(h["steps"][below[0]]) if len(below) else float(h["steps"][-1])
