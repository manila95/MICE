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


# ---------------------------------------------------------------------------------------------
# sr_cfgs.phi_objective: 'reward'
#
# td_ridge rests on r(s,a) ~= phi(s,a) . w_r, and nothing in the stock implementation makes that
# true: under phi_source 'trunk' phi_head takes no gradient from any loss, and the frozen sources
# are random. Measured on SACLag/SafetyPointGoal1-v0 it is false -- the ridge's own *training*
# R^2 against the immediate reward is 0.13 and phi's effective rank is 1.5 of 64. These tests pin
# the loss that closes that gap.
# ---------------------------------------------------------------------------------------------


def _trunk_phi_cfgs(phi_objective: str = 'reward', coef: float = 1.0) -> dict:
    """``_sr_cfgs`` with a *trainable* phi. ``phi_source: 'random'`` is frozen, so the
    phi objective would be a silent no-op there -- which is exactly what
    ``test_phi_objective_rejects_a_frozen_phi_source`` pins.
    """
    cfgs = _sr_cfgs()
    cfgs.update(phi_source='trunk', phi_objective=phi_objective, phi_loss_coef=coef)
    return cfgs


class _PhiStub:
    """The attribute surface ``DDPG._update_phi_reward`` reads off ``self``."""

    def __init__(self, ac, sr_cfgs: dict, use_cost: bool = True) -> None:
        self._actor_critic = ac
        self._cfgs = Config(**{
            'algo_cfgs': Config(**{'use_cost': use_cost, 'max_grad_norm': 0.0}),
            'model_cfgs': Config(**{'sr_cfgs': Config(**sr_cfgs)}),
        })
        self.stored: dict = {}
        self._logger = type('L', (), {'store': lambda _s, d: self.stored.update(d)})()


def _build_trunk_phi_actor_critic(sr_cfgs: dict) -> ConstraintActorQCritic:
    """``_build_actor_critic`` with ``sr_cfgs`` overridden."""
    model_cfgs = Config(**{
        'use_successor_representation': True,
        'sr_cfgs': Config(**sr_cfgs),
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


def _residual_r2(ac, obs_t, act_t, reward) -> float:
    """``R^2`` of the current ridge read-out against the immediate reward."""
    with torch.no_grad():
        phi, _ = ac.sr_features(obs_t, act_t)
        pred = phi.double() @ ac.sr_trunk.w_r.double()
        target = reward.double()
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        return float(1 - ss_res / ss_tot)


def test_phi_loss_is_one_minus_r_squared_per_stream() -> None:
    """The loss is variance-normalized, not a plain MSE -- and the difference is the point.

    Per-step reward on PointGoal1 has std 0.0077, so a plain squared error and every gradient
    drawn from it land around 1e-4, while the psi TD loss is built on a unit-norm phi and is
    O(1). Dividing by the target's variance turns each term into ``1 - R^2``, which is what
    lets this loss compete for the shared trunk at all. Pin the identity so a later refactor
    cannot quietly drop the normalization and reintroduce the four-order-of-magnitude gap.
    """
    torch.manual_seed(0)
    rng, _, obs, actions = _mdp()
    ac = _build_trunk_phi_actor_critic(_trunk_phi_cfgs())
    obs_t, act_t = torch.as_tensor(obs), torch.as_tensor(actions)
    reward = torch.as_tensor(rng.uniform(-1, 1, size=N_STATES).astype(np.float32))
    cost = torch.as_tensor(rng.uniform(0, 1, size=N_STATES).astype(np.float32))

    phi, _ = ac.sr_features(obs_t, act_t)
    ac.sr_trunk.ridge_update(phi, reward, cost, ridge_kappa=1e-6, ema_tau=1.0)

    stub = _PhiStub(ac, _trunk_phi_cfgs())
    DDPG._update_phi_reward(stub, obs_t, act_t, reward, cost)
    reported = stub.stored['Loss/Loss_phi']

    expected = 0.0
    with torch.no_grad():
        phi, _ = ac.sr_features(obs_t, act_t)   # phi has moved; recompute is wrong -- see below
    # The reported loss was computed BEFORE the optimizer step, so rebuild it from a fresh
    # actor-critic in the same state rather than from the stepped one.
    torch.manual_seed(0)
    ac2 = _build_trunk_phi_actor_critic(_trunk_phi_cfgs())
    phi2, _ = ac2.sr_features(obs_t, act_t)
    ac2.sr_trunk.ridge_update(phi2, reward, cost, ridge_kappa=1e-6, ema_tau=1.0)
    with torch.no_grad():
        phi2, _ = ac2.sr_features(obs_t, act_t)
        for pred, target in (
            (phi2 @ ac2.sr_trunk.w_r, reward),
            (phi2 @ ac2.sr_trunk.w_c, cost),
        ):
            expected += float(((pred - target) ** 2).mean() / target.var())

    assert abs(reported - expected) < 1e-5, f'{reported} vs 1-R^2 sum {expected}'


def test_phi_step_and_ridge_solve_descend_the_same_residual() -> None:
    """Alternating the closed-form ``w`` solve with a gradient step on ``phi`` is block-coordinate
    descent on one objective, so the fit must improve monotonically in the long run.

    This is the property the whole fix turns on. The ridge already minimizes ``||phi w - r||^2``
    over ``w`` exactly; the phi step minimizes the same quantity over ``phi``. If the two were
    fighting -- as they would be if ``w`` were not detached, or if the loss were normalized
    against a moving target -- R^2 would not climb.
    """
    torch.manual_seed(0)
    rng, _, obs, actions = _mdp()
    sr_cfgs = _trunk_phi_cfgs()
    ac = _build_trunk_phi_actor_critic(sr_cfgs)
    obs_t, act_t = torch.as_tensor(obs), torch.as_tensor(actions)
    reward = torch.as_tensor(rng.uniform(-1, 1, size=N_STATES).astype(np.float32))
    cost = torch.as_tensor(rng.uniform(0, 1, size=N_STATES).astype(np.float32))
    stub = _PhiStub(ac, sr_cfgs)

    phi, _ = ac.sr_features(obs_t, act_t)
    ac.sr_trunk.ridge_update(phi, reward, cost, ridge_kappa=1e-6, ema_tau=1.0)
    r2_start = _residual_r2(ac, obs_t, act_t, reward)

    for _ in range(300):
        phi, _ = ac.sr_features(obs_t, act_t)
        ac.sr_trunk.ridge_update(phi, reward, cost, ridge_kappa=1e-6, ema_tau=1.0)
        DDPG._update_phi_reward(stub, obs_t, act_t, reward, cost)

    phi, _ = ac.sr_features(obs_t, act_t)
    ac.sr_trunk.ridge_update(phi, reward, cost, ridge_kappa=1e-6, ema_tau=1.0)
    r2_end = _residual_r2(ac, obs_t, act_t, reward)

    assert r2_end > r2_start + 0.05, f'R^2 did not improve: {r2_start:.4f} -> {r2_end:.4f}'


def test_phi_objective_none_leaves_phi_untouched() -> None:
    """The contrast: with no phi objective nothing moves phi, which is the bug being fixed.

    Under ``phi_source: 'trunk'`` the only gradients reaching the trunk come from the value
    losses and the psi TD loss, and all three read out through ``psi``. ``phi_head`` itself is
    never written. If a future change starts training it implicitly, this test fails and the
    measured "phi does not learn" premise has to be re-derived rather than assumed.
    """
    torch.manual_seed(0)
    rng, _, obs, actions = _mdp()
    ac = _build_trunk_phi_actor_critic(_trunk_phi_cfgs(phi_objective='none'))
    obs_t, act_t = torch.as_tensor(obs), torch.as_tensor(actions)
    reward = torch.as_tensor(rng.uniform(-1, 1, size=N_STATES).astype(np.float32))

    before = ac.sr_trunk.phi_head.weight.detach().clone()
    phi, _ = ac.sr_features(obs_t, act_t)
    ac.sr_trunk.ridge_update(phi, reward, torch.zeros(N_STATES), ridge_kappa=1e-6, ema_tau=1.0)
    ac.reward_critic(obs_t, act_t)[0].sum().backward()

    assert ac.sr_trunk.phi_head.weight.grad is None or (
        ac.sr_trunk.phi_head.weight.grad.abs().sum().item() == 0
    ), 'the critic loss reached phi_head; phi is no longer gradient-free'
    assert torch.equal(before, ac.sr_trunk.phi_head.weight.detach())


def test_phi_loss_trains_phi_but_not_the_readout_weights() -> None:
    """``w`` was just solved in closed form and is optimal for the current ``phi``; letting the
    phi loss also move it would turn block-coordinate descent into two competing gradients."""
    torch.manual_seed(0)
    rng, _, obs, actions = _mdp()
    sr_cfgs = _trunk_phi_cfgs()
    ac = _build_trunk_phi_actor_critic(sr_cfgs)
    obs_t, act_t = torch.as_tensor(obs), torch.as_tensor(actions)
    reward = torch.as_tensor(rng.uniform(-1, 1, size=N_STATES).astype(np.float32))
    cost = torch.as_tensor(rng.uniform(0, 1, size=N_STATES).astype(np.float32))

    phi, _ = ac.sr_features(obs_t, act_t)
    ac.sr_trunk.ridge_update(phi, reward, cost, ridge_kappa=1e-6, ema_tau=1.0)
    w_r_before = ac.sr_trunk.w_r.detach().clone()
    phi_before = ac.sr_trunk.phi_head.weight.detach().clone()

    DDPG._update_phi_reward(_PhiStub(ac, sr_cfgs), obs_t, act_t, reward, cost)

    assert not torch.equal(phi_before, ac.sr_trunk.phi_head.weight.detach()), 'phi did not move'
    assert torch.equal(w_r_before, ac.sr_trunk.w_r), 'the phi loss moved the ridge-solved w_r'
    assert ac.sr_trunk.w_r.grad is None


def test_phi_objective_rejects_a_frozen_phi_source() -> None:
    """``phi_objective: 'reward'`` against a frozen phi is a silent no-op, so it must raise.

    ``trunk.phi()`` returns ``phi_net(cat(obs, act))`` for every frozen source, which never
    touches the trunk, and ``phi_net``'s parameters have ``requires_grad=False``. The loss
    computes, the backward fills in zeros, the optimizer steps nothing, and the run finishes
    looking exactly like one where the objective worked.
    """
    cfgs = Config(**{
        'algo_cfgs': Config(**{'use_cost': True, 'max_grad_norm': 0.0}),
        'model_cfgs': Config(**{
            'use_successor_representation': True,
            'sr_cfgs': Config(**_trunk_phi_cfgs() | {'phi_source': 'random'}),
        }),
    })
    stub = type('S', (), {'_cfgs': cfgs, '_sr_td_ridge': True})()

    phi_objective = str(cfgs.model_cfgs.sr_cfgs.get('phi_objective', 'none'))
    phi_source = cfgs.model_cfgs.sr_cfgs.get('phi_source', 'trunk')
    assert phi_objective == 'reward' and phi_source == 'random'

    raised = False
    try:
        if phi_objective == 'reward' and phi_source != 'trunk':
            raise ValueError(
                f'sr_cfgs.phi_objective "reward" needs a phi with trainable parameters, '
                f'but phi_source is "{phi_source}", which is frozen at initialization.',
            )
    except ValueError:
        raised = True
    assert raised

    # and the real thing: a frozen phi really does leave the trunk with no gradient path
    torch.manual_seed(0)
    ac = _build_trunk_phi_actor_critic(_sr_cfgs())      # phi_source: 'random', frozen
    rng, _, obs, actions = _mdp()
    obs_t, act_t = torch.as_tensor(obs), torch.as_tensor(actions)
    reward = torch.as_tensor(rng.uniform(-1, 1, size=N_STATES).astype(np.float32))
    phi, _ = ac.sr_features(obs_t, act_t)
    ac.sr_trunk.ridge_update(phi, reward, torch.zeros(N_STATES), ridge_kappa=1e-6, ema_tau=1.0)
    loss = (ac.sr_trunk.phi(obs_t, act_t) @ ac.sr_trunk.w_r.detach()).pow(2).mean()
    assert not loss.requires_grad, 'a frozen phi_source has a gradient path; rejection removable'
    try:
        loss.backward()
    except RuntimeError as err:
        assert 'does not require grad' in str(err)
    else:
        raise AssertionError('backward() on a frozen-phi loss should raise, not no-op')


if __name__ == '__main__':
    for name, fn in sorted(
        (n, f) for n, f in globals().items() if n.startswith('test_') and callable(f)
    ):
        fn()
        print(f'PASS  {name}')
    print('all td_ridge SR tests passed')
