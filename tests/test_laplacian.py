"""Correctness tests for the ALLO objective backing ``sr_cfgs.phi_source: 'laplacian'``.

Run either way -- ``pytest tests/test_laplacian.py`` or ``python tests/test_laplacian.py``. The
``__main__`` block is not decoration: the other modules in this directory have none, so running
them as scripts exits 0 having executed nothing, which looks exactly like a pass.

The test that matters is :func:`test_allo_recovers_laplacian_eigenvectors`. It fits a *tabular*
phi on a random walk over a path graph, whose Laplacian eigenvectors are known in closed form and
-- unlike a cycle's, which come in degenerate cos/sin pairs -- are non-degenerate, so the
recovered features can be checked coordinate by coordinate rather than only as a subspace. That
distinction is the whole point of ALLO over the penalty methods it replaces: those recover the
right subspace but an arbitrary rotation inside it, and a subspace-only test would pass for both.
"""

from __future__ import annotations

import os
import sys


# Resolve ``omnisafe`` to *this* fork rather than whichever checkout happens to be pip-installed
# (the sibling calibration_rl/omnisafe is installed editable in the shared env).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from omnisafe.utils import laplacian  # noqa: E402
from omnisafe.utils.contrastive import sample_temporal_pairs  # noqa: E402


SEED = 0
N_STATES = 16
SR_DIM = 4


def _path_graph_walk(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a random walk on a path graph with lazy (reflecting) endpoints.

    The transition matrix is symmetric, hence doubly stochastic, hence has a uniform stationary
    distribution -- which is what lets the test compare against the *unweighted* Laplacian
    eigenvectors without having to reweight anything.
    """
    walk = np.empty(n_steps, dtype=np.int64)
    state = int(rng.integers(N_STATES))
    for i in range(n_steps):
        walk[i] = state
        step = 1 if rng.random() < 0.5 else -1
        nxt = state + step
        state = nxt if 0 <= nxt < N_STATES else state  # laziness at the two endpoints
    return walk


def _true_transition_matrix() -> np.ndarray:
    """The transition matrix ``_path_graph_walk`` samples from, built explicitly."""
    mat = np.zeros((N_STATES, N_STATES))
    for s in range(N_STATES):
        for step in (-1, 1):
            nxt = s + step
            mat[s, nxt if 0 <= nxt < N_STATES else s] += 0.5
    return mat


def _bottom_eigenvectors(k: int) -> np.ndarray:
    """The ``k`` eigenvectors of ``I - P`` with the smallest eigenvalues, ``rho``-normalized.

    Scaled so that ``(1 / N) * V^T V = I``, i.e. to satisfy exactly the constraint ALLO enforces,
    so the learned table is directly comparable without any renormalization step that could
    paper over a scale error.
    """
    delta = np.eye(N_STATES) - _true_transition_matrix()
    eigvals, eigvecs = np.linalg.eigh(delta)
    order = np.argsort(eigvals)[:k]
    return eigvecs[:, order] * np.sqrt(N_STATES)


def _fit_tabular_phi(n_steps: int = 4000) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Fit a free ``(N_STATES, SR_DIM)`` table by ALLO on sampled walks.

    Returns the table, the empirical state distribution it was fitted under, and the last step's
    stats. Many short segments rather than a few long ones, and not for realism: a lazy random
    walk on a 16-state path mixes in ~N^2 = 256 steps, so eight 500-step walks carry only ~15
    independent draws of the state and their empirical distribution sits visibly off uniform --
    against which the closed-form eigenvectors are not the right answer to compare to. Each
    segment starts uniformly and the transition matrix is doubly stochastic, so every marginal is
    already uniform in expectation; it is only the variance that needs the segment count.
    """
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    lengths = [40] * 500
    walk = np.concatenate([_path_graph_walk(length, rng) for length in lengths])
    states = torch.as_tensor(walk)
    one_hot = torch.eye(N_STATES)[states]

    # A bias-free linear map on one-hot inputs *is* a free table, so nothing about the network's
    # capacity or nonlinearity can be confused for a property of the objective.
    table = torch.nn.Linear(N_STATES, SR_DIM, bias=False)
    torch.nn.init.normal_(table.weight, std=0.1)
    dual = torch.zeros(SR_DIM, SR_DIM)
    optimizer = torch.optim.Adam(table.parameters(), lr=0.01)

    src_pool, dst_pool = sample_temporal_pairs(lengths, horizon=1)
    n_rows, n_edges = one_hot.shape[0], src_pool.shape[0]
    batch = 512
    stats: dict[str, float] = {}
    for _ in range(n_steps):
        edges = torch.randperm(n_edges)[:batch]
        rows = torch.randperm(n_rows)[: 2 * batch]
        loss, violation, stats = laplacian.allo_loss(
            phi_s=table(one_hot[src_pool[edges]]),
            phi_s_next=table(one_hot[dst_pool[edges]]),
            phi_rho_a=table(one_hot[rows[:batch]]),
            phi_rho_b=table(one_hot[rows[batch:]]),
            dual=dual,
            barrier=2.0,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        laplacian.dual_ascent_(dual, violation, dual_lr=0.01)

    with torch.no_grad():
        learned = table(torch.eye(N_STATES))
    rho = torch.bincount(states, minlength=N_STATES).float() / states.shape[0]
    return learned, rho, stats


def test_allo_recovers_laplacian_eigenvectors() -> None:
    """ALLO recovers the bottom eigenvectors of the true Laplacian, in order and to scale."""
    learned, rho, stats = _fit_tabular_phi()
    truth = torch.as_tensor(_bottom_eigenvectors(SR_DIM), dtype=torch.float32)

    # 0. The comparison is only legitimate if the states really were visited near-uniformly --
    #    the closed-form eigenvectors above are those of the *uniformly* weighted Laplacian.
    rho_dev = ((rho - 1 / N_STATES).abs().max() * N_STATES).item()
    assert rho_dev < 0.3, f'visitation is {rho_dev:.2%} off uniform; lengthen the walk'

    # 1. The orthonormality constraint is satisfied: E_rho[phi phi^T] = I, weighted by the
    #    distribution it was actually enforced under. This is what fixes the scale (these features
    #    are deliberately unnormalized) and what hands the ridge solve an N * I design matrix.
    gram = (learned * rho[:, None]).T @ learned
    err = (gram - torch.eye(SR_DIM)).abs().max().item()
    assert err < 0.1, f'orthonormality violated by {err:.3f}\n{gram}'

    # 2. Each coordinate matches its own eigenvector, up to sign. An eigenvector is defined only
    #    up to sign, but *not* up to a rotation among coordinates -- checking per-coordinate is
    #    what distinguishes ALLO from the penalty methods, which recover only the joint span.
    #    Cosine, not correlation: coordinate 0 is the constant function, whose zero variance makes
    #    a correlation coefficient undefined.
    for i in range(SR_DIM):
        cos = (
            torch.dot(learned[:, i], truth[:, i]).abs()
            / (learned[:, i].norm() * truth[:, i].norm())
        ).item()
        assert cos > 0.95, (
            f'coordinate {i} does not match eigenvector {i} (|cos| = {cos:.3f}); '
            'the features may be a rotation of the right subspace rather than the eigenvectors.'
        )

    # 3. The first coordinate is the constant function (the eigenvalue-0 eigenvector). Called out
    #    separately because it is the one component whose identity is known without any solve.
    first = learned[:, 0]
    assert first.std().item() < 0.05 * first.abs().mean().item(), (
        f'coordinate 0 should be ~constant, got std {first.std().item():.4f} '
        f'against mean |value| {first.abs().mean().item():.4f}'
    )

    assert stats['OffDiagErr'] < 0.06, stats


def test_dirichlet_energy_matches_hand_computation() -> None:
    """The graph-drawing term is half the mean squared feature displacement along an edge."""
    phi_s = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    phi_next = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    # Rows contribute ||(1, 0)||^2 = 1 and ||(0, 2)||^2 = 4; mean 2.5, halved is 1.25.
    assert abs(laplacian.dirichlet_energy(phi_s, phi_next).item() - 1.25) < 1e-6
    # Identical endpoints (a self-loop) must cost nothing.
    assert laplacian.dirichlet_energy(phi_s, phi_s).item() == 0.0


def test_orthogonality_error_is_lower_triangular_and_asymmetric() -> None:
    """Only the lower triangle carries a residual, and only its row index carries gradient."""
    phi = torch.randn(64, 3, requires_grad=True)
    err = laplacian.orthogonality_error(phi, asymmetric=True)
    assert torch.equal(err, torch.tril(err)), 'upper triangle must be exactly zero'

    # The constraint <u_2, u_0> = 0 must move u_2 and leave u_0 alone: that asymmetry is what
    # imposes the deflation order and breaks the rotation symmetry.
    phi.grad = None
    err[2, 0].backward()
    per_coord = phi.grad.abs().sum(0)
    assert per_coord[2] > 0, 'the later coordinate must receive gradient'
    assert per_coord[0].item() == 0.0, 'the earlier coordinate must be stop-gradiented'

    # Without the asymmetry both sides move, which is the plain (symmetric) penalty formulation.
    phi.grad = None
    laplacian.orthogonality_error(phi, asymmetric=False)[2, 0].backward()
    assert phi.grad.abs().sum(0)[0] > 0


def test_dual_ascent_moves_multipliers_up_the_violation() -> None:
    """Ascent, in place: a persistently violated constraint grows its multiplier without bound."""
    dual = torch.zeros(2, 2)
    violation = torch.tensor([[0.5, 0.0], [-0.25, 0.5]])
    for _ in range(4):
        laplacian.dual_ascent_(dual, violation, dual_lr=0.1)
    assert torch.allclose(dual, 0.4 * violation, atol=1e-6), dual


def test_sample_temporal_pairs_honors_the_requested_device() -> None:
    """Indices come back on the caller's device, so a CUDA batch can index a CUDA minibatch.

    Guards the failure mode directly: PyTorch permits CPU indices into a CUDA tensor but not the
    reverse, so a pool left on the CPU raises the moment a caller draws a minibatch of it with a
    CUDA index tensor -- which is exactly what both trained-phi update loops do every step.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    anchor, positive = sample_temporal_pairs([6, 6], horizon=1, device=device)
    assert anchor.device.type == torch.device(device).type
    idx = torch.randint(anchor.shape[0], (4,), device=device)
    assert anchor[idx].shape == (4,)  # the call that used to raise
    assert torch.equal((anchor - positive).abs(), torch.ones_like(anchor)), 'horizon=1 edges only'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('all laplacian tests passed')
