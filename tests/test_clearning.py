"""Correctness tests for the C-learning objective backing ``sr_cfgs.psi_objective: 'contrastive'``.

Run either way -- ``pytest tests/test_clearning.py`` or ``python tests/test_clearning.py``.

The test that matters is :func:`test_fits_the_successor_measure_of_a_tabular_mdp`. It fits the
bilinear critic on a tabular chain whose successor measure ``sum_t gamma^t P^t`` is known *exactly*
by linear solve, so a sign error, a missing ``1 / (1 - gamma)``, or positives drawn from the wrong
distribution show up as a failed correlation against ground truth rather than as a
plausible-looking training curve. :func:`test_value_readout_matches_the_true_value_function` then
closes the loop the mode actually exists for: that the ridge read-out ``psi . w`` recovers the true
discounted value, which is the claim that justifies fitting the ratio in its linear scale.
"""

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from omnisafe.utils import clearning  # noqa: E402


SEED = 0
N_STATES = 8
SR_DIM = 8
GAMMA = 0.8


def _transition_matrix() -> np.ndarray:
    """A lazy random walk on a path graph -- symmetric, so its stationary distribution is uniform."""
    mat = np.zeros((N_STATES, N_STATES))
    for s in range(N_STATES):
        for step in (-1, 1):
            nxt = s + step
            mat[s, nxt if 0 <= nxt < N_STATES else s] += 0.5
    return mat


def _true_successor_measure() -> np.ndarray:
    """``M = sum_t gamma^t P^t = (I - gamma P)^-1``, the quantity the critic factorizes."""
    return np.linalg.inv(np.eye(N_STATES) - GAMMA * _transition_matrix())


def _walk(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    walk = np.empty(n_steps, dtype=np.int64)
    state = int(rng.integers(N_STATES))
    for i in range(n_steps):
        walk[i] = state
        nxt = state + (1 if rng.random() < 0.5 else -1)
        state = nxt if 0 <= nxt < N_STATES else state
    return walk


def _fit(n_steps: int = 3000) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Fit tabular ``psi`` and ``phi`` jointly; return both tables and the last stats."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    # Episodes long relative to the effective horizon 1 / (1 - gamma) = 5, so the truncation bias
    # sample_geometric_futures documents affects only a small tail of rows.
    lengths = [60] * 400
    walk = np.concatenate([_walk(length, rng) for length in lengths])
    states = torch.as_tensor(walk)
    one_hot = torch.eye(N_STATES)[states]

    # Bias-free linear maps on one-hot inputs are free tables, so nothing about network capacity
    # can be mistaken for a property of the objective.
    psi = torch.nn.Linear(N_STATES, SR_DIM, bias=False)
    phi = torch.nn.Linear(N_STATES, SR_DIM, bias=False)
    for table in (psi, phi):
        torch.nn.init.normal_(table.weight, std=0.3)
    optimizer = torch.optim.Adam([*psi.parameters(), *phi.parameters()], lr=0.01)

    anchors, futures = clearning.sample_geometric_futures(lengths, gamma=GAMMA)
    n_rows, n_pairs = one_hot.shape[0], anchors.shape[0]
    batch = 512
    stats: dict[str, float] = {}
    for _ in range(n_steps):
        pairs = torch.randperm(n_pairs)[:batch]
        negatives = torch.randperm(n_rows)[:batch]
        loss, stats = clearning.density_ratio_loss(
            psi_anchor=psi(one_hot[anchors[pairs]]),
            phi_future=phi(one_hot[futures[pairs]]),
            phi_random=phi(one_hot[negatives]),
            gamma=GAMMA,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        eye = torch.eye(N_STATES)
        return psi(eye), phi(eye), stats


def test_fits_the_successor_measure_of_a_tabular_mdp() -> None:
    """``psi(s) . phi(g)`` recovers ``M(s, g) / rho(g)`` for the exactly-solvable chain."""
    psi_table, phi_table, stats = _fit()
    ratio = (psi_table @ phi_table.T).numpy()
    # The fit is against the ratio to rho, and rho is uniform here, so the true target is N * M.
    truth = _true_successor_measure() * N_STATES

    corr = np.corrcoef(ratio.ravel(), truth.ravel())[0, 1]
    assert corr > 0.98, f'|corr| with the true successor measure is only {corr:.3f}'
    rel = np.abs(ratio - truth).max() / np.abs(truth).max()
    assert rel < 0.2, f'max relative error {rel:.3f}\nfitted:\n{ratio.round(2)}\ntrue:\n{truth.round(2)}'

    # The ratio must integrate to the total discounted mass under rho. This catches a wrong or
    # missing 1 / (1 - gamma) that a correlation check would sail straight past.
    assert abs(stats['RatioMass'] - stats['RatioMassTarget']) < 0.15 * stats['RatioMassTarget'], stats


def test_value_readout_matches_the_true_value_function() -> None:
    """``psi . w`` with ``w = E_rho[phi r]`` recovers the true discounted value of a reward."""
    psi_table, phi_table, _ = _fit()
    rng = np.random.default_rng(SEED)
    reward = torch.as_tensor(rng.normal(size=N_STATES), dtype=torch.float32)

    # w as the identity in the module docstring gives it: an expectation under rho (uniform here).
    w = (phi_table * reward[:, None]).mean(0)
    predicted = (psi_table @ w).numpy()
    truth = _true_successor_measure() @ reward.numpy()

    corr = np.corrcoef(predicted, truth)[0, 1]
    assert corr > 0.98, f'value read-out correlates only {corr:.3f} with the true V'
    rel = np.abs(predicted - truth).max() / np.abs(truth).max()
    assert rel < 0.25, f'max relative value error {rel:.3f}\npred {predicted.round(3)}\ntrue {truth.round(3)}'


def test_geometric_offsets_have_the_right_distribution() -> None:
    """Offsets are forward, include zero, and are geometric on ``{0, 1, ...}``."""
    gamma = 0.9
    # One long segment, so truncation touches almost nothing and the empirical law is the clean one.
    anchors, futures = clearning.sample_geometric_futures([20000], gamma=gamma)
    offsets = (futures - anchors).float()
    assert (offsets >= 0).all(), 'offsets must be forward'

    # E[k] of a geometric on {0, 1, ...} with P(k) proportional to gamma^k is gamma / (1 - gamma).
    assert abs(offsets.mean().item() - gamma / (1.0 - gamma)) < 0.5, offsets.mean().item()
    # P(k = 0) = 1 - gamma. This is the term whose omission biases the fit to gamma P (I - gamma P)^-1
    # rather than the successor measure itself, so it is worth pinning down numerically.
    assert abs((offsets == 0).float().mean().item() - (1.0 - gamma)) < 0.02

    # No row is dropped: every row can draw k = 0, including the last of each segment.
    anchors, futures = clearning.sample_geometric_futures([4, 4], gamma=gamma)
    assert anchors.tolist() == list(range(8))
    assert (futures[[3, 7]] == anchors[[3, 7]]).all(), 'a segment-final row can only stay put'
    assert clearning.sample_geometric_futures([1, 1], gamma=gamma)[0].numel() == 2


def test_offsets_never_leave_their_segment() -> None:
    """A future index always lands inside the anchor's own episode, at every gamma."""
    for gamma in (0.0, 0.5, 0.99, 1.0 - 1e-9):
        lengths = [3, 7, 2, 11]
        anchors, futures = clearning.sample_geometric_futures(lengths, gamma=gamma)
        bounds = torch.tensor(np.cumsum([0, *lengths]))
        for a, f in zip(anchors.tolist(), futures.tolist()):
            seg = int(torch.searchsorted(bounds, torch.tensor(a), right=True)) - 1
            assert bounds[seg] <= f < bounds[seg + 1], (gamma, a, f, seg)
            assert f >= a
    # gamma = 0 is the degenerate case: all the discounted mass sits on t = 0.
    anchors, futures = clearning.sample_geometric_futures([50], gamma=0.0)
    assert torch.equal(futures, anchors)


def test_density_ratio_loss_is_minimized_at_the_true_ratio() -> None:
    """The loss is a proper scoring rule: its argmin is the density ratio, scale factor included.

    Built on explicit distributions rather than a fitted critic, so the optimum is known in closed
    form. With ``psi = 1`` and a one-dimensional ``phi``, the critic's value at ``g`` is just
    ``phi(g)``, and the loss reduces to a quadratic per state whose stationary point is
    ``p(g) / q(g) / (1 - gamma)`` -- the target including the factor that makes it the
    *unnormalized* successor measure. Scaling the solution away from 1.0 in either direction must
    raise the loss; getting the ``1 / (1 - gamma)`` wrong would move the argmin off 1.0.
    """
    gamma = 0.5
    n = 2000
    # q(g) = [1/2, 1/2] for the negatives, p(g) = [4/5, 1/5] for the futures.
    negatives = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    futures = torch.cat([torch.zeros(4 * n // 5), torch.ones(n // 5)])
    true_ratio = torch.tensor([0.8 / 0.5, 0.2 / 0.5]) / (1.0 - gamma)

    def loss_at(scale: float) -> float:
        phi_of = (true_ratio * scale).unsqueeze(-1)  # phi(g) for g in {0, 1}
        value, _ = clearning.density_ratio_loss(
            psi_anchor=torch.ones(n, 1),
            phi_future=phi_of[futures.long()],
            phi_random=phi_of[negatives.long()],
            gamma=gamma,
        )
        return value.item()

    base = loss_at(1.0)
    for scale in (0.5, 0.8, 0.95, 1.05, 1.25, 2.0):
        assert loss_at(scale) > base - 1e-6, (scale, base, loss_at(scale))


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('all clearning tests passed')
