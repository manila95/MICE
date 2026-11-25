"""Correctness tests for the ``sr_mode: 'td_ridge'`` successor-representation Q-critic.

Run either way -- ``pytest tests/test_sr_td_ridge.py`` or ``python tests/test_sr_td_ridge.py``.
The ``__main__`` block is not decoration: a module with no such block exits 0 having executed
nothing when run as a script, which looks exactly like a pass.

The SR critic factorizes ``Q(s,a) = psi(s,a) . w``, where ``psi`` is the discounted sum of
future features and ``w`` regresses the immediate reward onto the immediate feature ``phi``.
Each half can be wrong independently and still produce a plausible-looking training curve, so
each is pinned against a closed form rather than against itself:

* the ridge solve, against ``(X^T X + kappa I)^-1 X^T y`` computed in numpy, and -- the
  sharper version -- against the exact ``w_true`` when the reward really is ``phi . w_true``,
  which is only identifiable when the design matrix has more rows than ``sr_dim``;
* ``psi``, against ``(I - gamma P)^-1 Phi`` on a finite MDP with a known transition matrix;
* the composition, against the exact discounted return ``(I - gamma P)^-1 r``. When the reward
  is linear in the features the factorization is *exact*, so this is a real equality and not a
  tolerance dressed up as one.

``psi``'s fixed point is reached by repeated bootstrapping, and each target-network sync
propagates roughly one more step of lookahead. A test that syncs too rarely therefore measures
a uniformly *shrunk* ``psi`` -- at gamma 0.9, eight syncs give ``0.9^8 ~ 0.43`` of the true
value -- and that shortfall is indistinguishable from a real bug by eye. ``SYNC_EVERY`` is set
so the sync count comfortably exceeds the horizon; do not raise it to make the test faster.
"""

from __future__ import annotations

import os
import sys

# Run as a script, sys.path[0] is tests/, so `import omnisafe` finds the *installed* release
# package instead of this checkout -- and the SR attributes simply would not be there. Same
# guard as tests/test_laplacian.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config


N_STATES = 40          # > SR_DIM, or w_true is not identifiable from the design matrix
SR_DIM = 16
GAMMA = 0.9
TRAIN_STEPS = 12000
BATCH = 256
SYNC_EVERY = 50        # ~240 syncs, against an effective horizon of 1/(1-gamma) = 10


def _sr_cfgs() -> dict:
    """``sr_cfgs`` with ``phi`` a frozen map of the raw input, so it is trunk-independent."""
    return {
        'sr_mode': 'td_ridge', 'cost_only': False, 'sr_dim': SR_DIM, 'sr_update_every': 1,
        'cond_every': 1, 'hidden_sizes': [64, 64], 'activation': 'tanh', 'lr': 3e-4,
        'lam_sr': 0.95, 'gamma_sr': None, 'ridge_kappa': 1e-6, 'ridge_kappa_cost': None,
        'cost_value_clip': None, 'ema_tau': 1.0, 'dropout': 0.0, 'use_layer_norm': False,
        'use_spectral_norm': False, 'weight_decay': 0.0, 'readout': 'ridge', 'w_source': 'phi',
        'ridge_data': 'epoch', 'ridge_buffer_samples': None, 'cost_replay_buffer_size': 1000,
        'w_lr': None, 'readout_buffer_size': 1000, 'readout_grad_steps': 50,
        'w_weight_decay': 0.0, 'w_weight_decay_cost': None, 'readout_batch_size': None,
        'phi_source': 'random', 'phi_hidden_sizes': [64, 64], 'phi_orthogonal_init': False,
        'phi_rff_bandwidth': 1.0, 'phi_ensemble_sources': None,
        'phi_contrastive_temperature': 0.1, 'phi_contrastive_horizon': 5,
        'phi_contrastive_pretrain_steps': 500, 'phi_contrastive_steps_per_epoch': 50,
        'phi_contrastive_batch_size': None, 'phi_lr': None, 'phi_laplacian_barrier': 2.0,
        'phi_laplacian_dual_lr': 0.01, 'phi_laplacian_batch_size': 1024,
        'phi_laplacian_lr': 3e-3,
    }


def _build_actor_critic() -> ConstraintActorQCritic:
    """A real :class:`ConstraintActorQCritic` in ``td_ridge`` mode over one-hot observations."""
    model_cfgs = Config(**{
        'use_successor_representation': True,
        'sr_cfgs': Config(**_sr_cfgs()),
        'actor_type': 'gaussian_sac',
        'linear_lr_decay': False,
        'exploration_noise_anneal': False,
        'std_range': [0.5, 0.1],
        'actor': Config(**{'hidden_sizes': [64, 64], 'activation': 'relu', 'lr': 3e-4,
                           'output_activation': 'identity'}),
        'critic': Config(**{'hidden_sizes': [64, 64], 'activation': 'relu', 'lr': 1e-3,
                            'num_critics': 2, 'num_critics_cost': 1}),
        'weight_initialization_mode': 'kaiming_uniform',
    })
    return ConstraintActorQCritic(
        obs_space=spaces.Box(low=-np.inf, high=np.inf, shape=(N_STATES,), dtype=np.float32),
        act_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        model_cfgs=model_cfgs,
        epochs=1,
    )


class _FixedPolicy(nn.Module):
    """Deterministic ``a = pi(s)`` over one-hot states, so the MDP stays a finite chain."""

    def __init__(self, actions: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer('actions', actions)

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.actions[obs.argmax(dim=-1)]


def _mdp(seed: int = 0):
    """A random row-stochastic transition matrix and one action per state."""
    rng = np.random.RandomState(seed)
    transitions = np.stack([rng.dirichlet(np.ones(N_STATES)) for _ in range(N_STATES)])
    obs = np.eye(N_STATES, dtype=np.float32)
    actions = rng.uniform(-1, 1, size=(N_STATES, 1)).astype(np.float32)
    return rng, transitions, obs, actions


def test_ridge_solve_matches_closed_form() -> None:
    """``w_r`` is the ridge solution of ``phi -> reward``, to float64 round-off."""
    torch.manual_seed(0)
    rng, _, obs, actions = _mdp()
    ac = _build_actor_critic()
    phi, _ = ac.sr_features(torch.as_tensor(obs), torch.as_tensor(actions))
    reward = torch.as_tensor(rng.uniform(-1, 1, size=N_STATES).astype(np.float32))
    cost = torch.as_tensor(rng.uniform(0, 1, size=N_STATES).astype(np.float32))

    ac.sr_trunk.ridge_update(phi, reward, cost, ridge_kappa=1e-3, ema_tau=1.0)

    design = phi.double().numpy()
    gram = design.T @ design
    scale = np.diag(gram).mean()
    expected = np.linalg.solve(gram + 1e-3 * scale * np.eye(SR_DIM), design.T @ reward.double().numpy())
    assert np.abs(ac.sr_trunk.w_r.numpy() - expected).max() < 1e-5


def test_ridge_recovers_exact_weights_when_reward_is_linear() -> None:
    """With ``r = phi . w_true`` and negligible shrinkage, the solve returns ``w_true`` itself."""
    torch.manual_seed(0)
    rng, _, obs, actions = _mdp()
    ac = _build_actor_critic()
    phi, _ = ac.sr_features(torch.as_tensor(obs), torch.as_tensor(actions))
    w_true = rng.uniform(-1, 1, size=SR_DIM)
    reward = torch.as_tensor((phi.double().numpy() @ w_true).astype(np.float32))
    cost = torch.zeros(N_STATES)

    ac.sr_trunk.ridge_update(phi, reward, cost, ridge_kappa=1e-9, ema_tau=1.0)

    assert np.abs(ac.sr_trunk.w_r.double().numpy() - w_true).max() < 1e-3


def _train_psi(ac: ConstraintActorQCritic, rng, transitions, obs, actions):
    """Drive the real :meth:`DDPG._update_successor_features` to the SR fixed point."""
    stub = type('_Stub', (), {})()
    stub._cfgs = Config(**{
        'model_cfgs': Config(**{'sr_cfgs': Config(**_sr_cfgs())}),
        'algo_cfgs': Config(**{'gamma': GAMMA, 'use_critic_norm': False,
                               'critic_norm_coeff': 0.0, 'max_grad_norm': 0.0}),
    })
    stub._actor_critic = ac
    stub._logger = type('_L', (), {'store': lambda self, *a, **k: None})()
    for step in range(TRAIN_STEPS):
        idx = rng.randint(0, N_STATES, size=BATCH)
        nxt = np.array([rng.choice(N_STATES, p=transitions[i]) for i in idx])
        DDPG._update_successor_features(
            stub,
            torch.as_tensor(obs[idx]),
            torch.as_tensor(actions[idx]),
            torch.zeros(BATCH),
            torch.as_tensor(obs[nxt]),
        )
        if step % SYNC_EVERY == 0:
            ac.target_sr_trunk.load_state_dict(ac.sr_trunk.state_dict())
    ac.target_sr_trunk.load_state_dict(ac.sr_trunk.state_dict())


def test_psi_converges_to_exact_successor_representation() -> None:
    """``psi`` reaches ``(I - gamma P)^-1 Phi``, the SR's closed form on a finite MDP."""
    torch.manual_seed(0)
    rng, transitions, obs, actions = _mdp()
    ac = _build_actor_critic()
    ac.actor = _FixedPolicy(torch.as_tensor(actions))
    obs_t, act_t = torch.as_tensor(obs), torch.as_tensor(actions)
    phi, _ = ac.sr_features(obs_t, act_t)

    _train_psi(ac, rng, transitions, obs, actions)

    expected = np.linalg.solve(np.eye(N_STATES) - GAMMA * transitions, phi.double().numpy())
    with torch.no_grad():
        learned = ac.sr_trunk.psi(obs_t, act_t)[0].double().numpy()
    assert np.abs(learned - expected).max() / np.abs(expected).max() < 0.05


def test_q_readout_equals_exact_discounted_return() -> None:
    """``psi . w`` is the true value function when the reward is linear in ``phi``."""
    torch.manual_seed(0)
    rng, transitions, obs, actions = _mdp()
    ac = _build_actor_critic()
    ac.actor = _FixedPolicy(torch.as_tensor(actions))
    obs_t, act_t = torch.as_tensor(obs), torch.as_tensor(actions)
    phi, _ = ac.sr_features(obs_t, act_t)
    w_true = rng.uniform(-1, 1, size=SR_DIM)
    reward_np = phi.double().numpy() @ w_true

    _train_psi(ac, rng, transitions, obs, actions)
    ac.sr_trunk.ridge_update(
        phi, torch.as_tensor(reward_np.astype(np.float32)), torch.zeros(N_STATES),
        ridge_kappa=1e-9, ema_tau=1.0,
    )
    ac.sync_sr_readout_weights()

    expected = np.linalg.solve(np.eye(N_STATES) - GAMMA * transitions, reward_np)
    with torch.no_grad():
        q = ac.reward_critic(obs_t, act_t)[0].double().numpy()
    assert np.abs(q - expected).max() / np.abs(expected).max() < 0.05


def test_sync_copies_readout_weights_onto_target_trunk() -> None:
    """``w`` is a buffer, so ``polyak_update`` cannot reach it and the sync must."""
    torch.manual_seed(0)
    ac = _build_actor_critic()
    ac.sr_trunk.w_r.add_(1.234)
    ac.sr_trunk.w_c.add_(-0.567)

    ac.sync_sr_readout_weights()

    assert torch.allclose(ac.target_sr_trunk.w_r, ac.sr_trunk.w_r)
    assert torch.allclose(ac.target_sr_trunk.w_c, ac.sr_trunk.w_c)


def test_critic_gradient_reaches_psi_but_not_readout_weights() -> None:
    """The critic loss trains ``psi``; ``w`` is detached and belongs to the ridge solve.

    ``w`` is zero-initialized, and ``Q = (psi * w).sum(-1)`` is identically zero until the
    first ridge solve -- so is its gradient with respect to ``psi``. That is benign, because
    ``psi`` is trained by the SR feature loss rather than by the critic loss, but it means a
    gradient check run on a freshly built trunk measures the zero initialization and not the
    wiring. Solve ``w`` first, then check.
    """
    torch.manual_seed(0)
    rng, _, obs, actions = _mdp()
    ac = _build_actor_critic()
    obs_t, act_t = torch.as_tensor(obs), torch.as_tensor(actions)

    assert not ac.sr_trunk.w_r.requires_grad, 'w_r must be a buffer, not a trained parameter'
    ac.reward_critic(obs_t, act_t)[0].sum().backward()
    grad_at_init = ac.sr_trunk.psi_heads[0][0].weight.grad
    assert grad_at_init is None or grad_at_init.abs().sum().item() == 0

    phi, _ = ac.sr_features(obs_t, act_t)
    reward = torch.as_tensor(rng.uniform(-1, 1, size=N_STATES).astype(np.float32))
    ac.sr_trunk.ridge_update(phi, reward, torch.zeros(N_STATES), ridge_kappa=1e-3, ema_tau=1.0)
    ac.zero_grad(set_to_none=True)
    ac.reward_critic(obs_t, act_t)[0].sum().backward()

    grad = ac.sr_trunk.psi_heads[0][0].weight.grad
    assert grad is not None and grad.abs().sum().item() > 0
    assert ac.sr_trunk.w_r.grad is None


if __name__ == '__main__':
    for name, fn in sorted(
        (n, f) for n, f in globals().items() if n.startswith('test_') and callable(f)
    ):
        fn()
        print(f'PASS  {name}')
    print('all td_ridge SR tests passed')
