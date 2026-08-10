"""Utilities for the critic-calibration / sample-complexity experiment.

The experiment this module supports (see ``experiments/eval_critic_calibration.py``) asks a
narrow question: *given a policy frozen after k epochs of training, how many on-policy samples M
does a critic trained from scratch need to recover a well-calibrated V^pi?* That requires three
pieces the normal training loop doesn't expose on their own:

* :func:`reinit_critics` -- throw away the current reward/cost critics and build fresh ones with
  the same architecture, isolating "how good is a critic *this* capacity can get with M samples"
  from whatever the critic learned during the k-epoch pretraining phase.
* :func:`collect_fixed_policy_rollout` -- collect exactly M on-policy transitions under a frozen
  policy, but -- unlike :meth:`omnisafe.adapter.onpolicy_adapter.OnPolicyAdapter.rollout`, which
  bootstraps any trajectory still open at the epoch boundary -- extends collection to let every
  in-flight episode finish naturally, so ``discounted_ret`` (the Monte-Carlo return) is unbiased
  for every collected transition rather than truncated for whichever episodes happened to be
  mid-flight at the M-th sample.
* :func:`train_critic` -- fit the (freshly reinitialized) critics on exactly that batch of M
  transitions, targeting either the algorithm's normal bootstrapped target (``target_value_r/c``,
  e.g. GAE) or the pure Monte-Carlo return (``discounted_ret`` / ``discounted_cost_ret``), with a
  choice of a fixed number of epochs or held-out-episode early stopping.

Each function operates on a live :class:`~omnisafe.models.actor_critic.constraint_actor_critic.ConstraintActorCritic`
and the algorithm's :class:`~omnisafe.utils.config.Config`, mirroring the tensors and update
mechanics already used by :class:`~omnisafe.algorithms.on_policy.base.policy_gradient.PolicyGradient`
(:meth:`_update_reward_critic` / :meth:`_update_cost_critic`) so that switching between "how CPO
already trains its critic" and this isolated evaluation involves changing as little as possible.

.. note::
    Only the plain per-stream V-critic case is supported (``model_cfgs.use_successor_representation
    = False``). The successor-representation critic factors ``V`` through a shared trunk that this
    module does not know how to reinitialize/retrain in isolation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import Config, ModelConfig


def assert_plain_critic(model_cfgs: ModelConfig) -> None:
    """Fail fast if the model uses the successor-representation critic.

    Args:
        model_cfgs (ModelConfig): The model configuration to check.

    Raises:
        NotImplementedError: If ``use_successor_representation`` is set.
    """
    if bool(model_cfgs.get('use_successor_representation', False)):
        raise NotImplementedError(
            'eval_critic_calibration only supports the plain per-stream V-critic '
            '(model_cfgs.use_successor_representation must be False). The successor-'
            'representation critic factors V through a shared trunk that reinit_critics/'
            'train_critic do not know how to reinitialize or retrain in isolation.',
        )


def reinit_critics(
    actor_critic: torch.nn.Module,
    obs_space: OmnisafeSpace,
    act_space: OmnisafeSpace,
    model_cfgs: ModelConfig,
    device: torch.device,
) -> None:
    """Replace ``actor_critic``'s reward/cost critics with freshly initialized ones, in place.

    Builds new critic networks with the exact architecture ``model_cfgs.critic`` specifies (same
    code path :class:`~omnisafe.models.actor_critic.actor_critic.ActorCritic` uses at
    construction time), and fresh Adam optimizers to match -- so there is no leftover Adam
    momentum/second-moment state from whatever the critic learned before this call. The actor is
    left untouched.

    Args:
        actor_critic (torch.nn.Module): The live ``ConstraintActorCritic`` to modify in place.
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configuration (``model_cfgs.critic.*`` is read).
        device (torch.device): Device the new critics are moved to.
    """
    assert_plain_critic(model_cfgs)

    def _build() -> torch.nn.Module:
        return CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        ).build_critic('v').to(device)

    actor_critic.reward_critic = _build()
    actor_critic.cost_critic = _build()

    if model_cfgs.critic.lr is not None:
        actor_critic.reward_critic_optimizer = optim.Adam(
            actor_critic.reward_critic.parameters(),
            lr=model_cfgs.critic.lr,
        )
        actor_critic.cost_critic_optimizer = optim.Adam(
            actor_critic.cost_critic.parameters(),
            lr=model_cfgs.critic.lr,
        )


@torch.no_grad()
def collect_fixed_policy_rollout(
    adapter: Any,
    actor_critic: torch.nn.Module,
    cfgs: Config,
    target_m: int,
    max_episode_len: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[tuple[int, int]], int]:
    r"""Collect on-policy transitions under a frozen policy, without truncating any episode.

    This is deliberately not :meth:`OnPolicyAdapter.rollout`: that method bootstraps whatever
    trajectory is still open once the step budget is hit, which biases ``discounted_ret`` low for
    the last few states of the trajectory that happened to be mid-flight. Here, once at least
    ``target_m`` transitions have been stored, each environment is allowed to keep running until
    *its own* current episode ends naturally, and only then stops contributing further
    transitions -- so every stored transition's ``discounted_ret`` is a true, untruncated
    Monte-Carlo return.

    Args:
        adapter (OnPolicyAdapter): The (already-wrapped) online adapter to step. Its
            ``actor_critic`` should already carry the frozen policy and whatever critic is
            currently installed (typically freshly reinitialized by :func:`reinit_critics`).
        actor_critic (torch.nn.Module): The ``ConstraintActorCritic`` used to act and to record
            per-step value estimates (only used as diagnostics / bootstrap values here, not
            trained).
        cfgs (Config): The algorithm configuration (``algo_cfgs.gamma``, ``lam``, etc. -- the same
            fields :class:`VectorOnPolicyBuffer` would use inside normal training).
        target_m (int): Minimum number of transitions to collect.
        max_episode_len (int): Safety bound on how long any single episode may run while we wait
            for it to finish; also sizes the buffer's per-environment overshoot margin.
        device (torch.device): Device the buffer's tensors live on.

    Returns:
        data: The buffer's ``get()`` dict (``obs``, ``act``, ``target_value_r/c``,
            ``discounted_ret``, ``discounted_cost_ret``, ...), holding every collected transition
            (at least ``target_m``, typically a little more from finishing in-flight episodes).
        episode_slices: ``(start, end)`` index pairs into ``data``, one per complete episode.
        n_collected: The exact number of transitions collected (``== data['obs'].shape[0]``).
    """
    num_envs = adapter._env.num_envs  # pylint: disable=protected-access
    per_env_capacity = math.ceil(target_m / num_envs) + max_episode_len + 1

    buf = VectorOnPolicyBuffer(
        obs_space=adapter.observation_space,
        act_space=adapter.action_space,
        size=per_env_capacity,
        gamma=cfgs.algo_cfgs.gamma,
        lam=cfgs.algo_cfgs.lam,
        lam_c=cfgs.algo_cfgs.lam_c,
        advantage_estimator=cfgs.algo_cfgs.adv_estimation_method,
        penalty_coefficient=cfgs.algo_cfgs.penalty_coef,
        standardized_adv_r=False,
        standardized_adv_c=False,
        num_envs=num_envs,
        device=device,
        cost_gamma=getattr(cfgs.algo_cfgs, 'cost_gamma', None),
        cost_advantage_estimator=getattr(cfgs.algo_cfgs, 'cost_adv_estimation_method', None),
    )

    obs, _ = adapter.reset()
    active = torch.ones(num_envs, dtype=torch.bool)
    stored = 0
    past_threshold = False
    # Hard fallback: if some environment's episode still hasn't ended once its per-env buffer is
    # about to overflow, force-finish it (bootstrapped, like a normal epoch cutoff) rather than
    # crash. Only bites for pathologically long episodes relative to max_episode_len.
    for _ in range(per_env_capacity - 1):
        if not active.any():
            break
        act, value_r, value_c, logp = actor_critic.step(obs)
        next_obs, reward, cost, terminated, truncated, info = adapter.step(act)

        for idx in range(num_envs):
            if not active[idx]:
                continue
            buf.buffers[idx].store(
                obs=obs[idx],
                act=act[idx],
                reward=reward[idx],
                cost=cost[idx],
                value_r=value_r[idx],
                value_c=value_c[idx],
                logp=logp[idx],
            )
            stored += 1
            if stored >= target_m:
                past_threshold = True

            done = bool(terminated[idx]) or bool(truncated[idx])
            if done:
                if bool(truncated[idx]) and not bool(terminated[idx]):
                    # Time-limit cutoff, not a true terminal state: bootstrap from the value of
                    # the state the episode was cut off at.
                    _, last_value_r, last_value_c, _ = actor_critic.step(
                        info['final_observation'][idx],
                    )
                    last_value_r, last_value_c = last_value_r.unsqueeze(0), last_value_c.unsqueeze(0)
                else:
                    last_value_r = last_value_c = None
                buf.buffers[idx].finish_path(last_value_r, last_value_c)
                if past_threshold:
                    active[idx] = False

        obs = next_obs
    else:
        # Loop exhausted per_env_capacity without every env finishing -- force-finish whatever is
        # left so buf.get() below sees a consistent state, at the cost of a truncation bias for
        # those specific trailing transitions only.
        if active.any():
            act, value_r, value_c, _ = actor_critic.step(obs)
            for idx in active.nonzero(as_tuple=False).flatten().tolist():
                buf.buffers[idx].finish_path(value_r[idx].unsqueeze(0), value_c[idx].unsqueeze(0))

    # ``VectorOnPolicyBuffer.get()`` returns each sub-buffer's *entire* preallocated tensor
    # (shape ``(per_env_capacity, ...)``), not just the ``ptr`` transitions actually stored --
    # correct for normal training, where the buffer size is chosen to be filled exactly, but
    # here ``per_env_capacity`` deliberately overshoots so in-flight episodes have room to
    # finish. Capture each env's final ``ptr`` before ``get()`` resets it, then compact the
    # concatenated tensors down to only the transitions actually written, remapping the (already
    # per-env-local) episode slices onto the compacted layout.
    final_ptrs = [b.ptr for b in buf.buffers]
    raw_data = buf.get()

    data = {
        key: torch.cat(
            [tensor[i * per_env_capacity : i * per_env_capacity + final_ptrs[i]] for i in range(num_envs)],
            dim=0,
        )
        for key, tensor in raw_data.items()
    }

    episode_slices: list[tuple[int, int]] = []
    compacted_offset = 0
    for i, sub_buf in enumerate(buf.buffers):
        for s, e in sub_buf._last_episode_slices:  # pylint: disable=protected-access
            episode_slices.append((compacted_offset + s, compacted_offset + e))
        compacted_offset += final_ptrs[i]

    return data, episode_slices, sum(final_ptrs)


def _critic_grad_step(
    critic: torch.nn.Module,
    optimizer: optim.Optimizer,
    obs: torch.Tensor,
    target: torch.Tensor,
    cfgs: Config,
) -> float:
    """One MSE gradient step, mirroring ``PolicyGradient._update_reward_critic`` /
    ``_update_cost_critic`` exactly (critic-norm regularizer, grad clipping) but returning the
    loss instead of only logging it, so callers can track train/val curves locally.
    """
    optimizer.zero_grad()
    loss = F.mse_loss(critic(obs)[0], target)
    if cfgs.algo_cfgs.use_critic_norm:
        for param in critic.parameters():
            if param.requires_grad:
                loss = loss + param.pow(2).sum() * cfgs.algo_cfgs.critic_norm_coef
    loss.backward()
    if cfgs.algo_cfgs.use_max_grad_norm:
        clip_grad_norm_(critic.parameters(), cfgs.algo_cfgs.max_grad_norm)
    optimizer.step()
    return loss.item()


def train_critic(  # pylint: disable=too-many-arguments,too-many-locals
    actor_critic: torch.nn.Module,
    cfgs: Config,
    data: dict[str, torch.Tensor],
    episode_slices: list[tuple[int, int]],
    target_mode: str,
    stopping: str,
    batch_size: int,
    fixed_epochs: int = 200,
    val_frac: float = 0.1,
    patience: int = 20,
    min_delta: float = 1e-4,
    max_epochs: int = 2000,
) -> dict[str, float]:
    """Train ``actor_critic``'s reward/cost critics on a fixed batch of ``data``.

    Args:
        actor_critic (torch.nn.Module): The ``ConstraintActorCritic`` whose ``reward_critic`` /
            ``cost_critic`` (and their optimizers) are trained in place -- typically freshly
            reinitialized by :func:`reinit_critics` just before this call.
        cfgs (Config): The algorithm configuration.
        data (dict): One buffer's worth of transitions, as returned by
            :func:`collect_fixed_policy_rollout` (or ``VectorOnPolicyBuffer.get()``).
        episode_slices (list): ``(start, end)`` pairs into ``data``, used to carve out a
            held-out validation set by whole episodes (``stopping='early_stop'``) so adjacent,
            highly-correlated in-episode states never leak across the split.
        target_mode ({'bootstrap', 'mc'}): What the critics regress toward. ``'bootstrap'`` uses
            ``data['target_value_r'/'target_value_c']`` -- the same GAE/TD target
            ``PolicyGradient._update`` trains on, computed once from the rollout-time critic
            values. ``'mc'`` uses ``data['discounted_ret'/'discounted_cost_ret']`` -- the plain,
            un-bootstrapped Monte-Carlo return -- which is unbiased for V^pi precisely because
            :func:`collect_fixed_policy_rollout` guarantees every episode ran to completion.
        stopping ({'fixed', 'early_stop'}): ``'fixed'`` runs exactly ``fixed_epochs`` passes over
            all of ``data``. ``'early_stop'`` holds out ``val_frac`` of the episodes, and stops
            once the held-out MSE hasn't improved by ``min_delta`` for ``patience`` consecutive
            epochs (capped at ``max_epochs`); the critics are rolled back to their best-val-loss
            state before returning.
        batch_size (int): Minibatch size for the gradient steps.
        fixed_epochs (int): Epoch count for ``stopping='fixed'``.
        val_frac (float): Held-out episode fraction for ``stopping='early_stop'``.
        patience (int): Epochs without improvement before stopping, for ``stopping='early_stop'``.
        min_delta (float): Minimum held-out MSE improvement to reset patience.
        max_epochs (int): Hard cap on epochs for ``stopping='early_stop'``.

    Returns:
        A dict of summary stats: ``epochs_run``, ``final_train_loss_r/c``,
        ``final_val_loss_r/c`` (``nan`` under ``'fixed'``), ``best_val_loss`` (``nan`` unless
        early stopping actually improved on its first epoch's value).
    """
    assert target_mode in ('bootstrap', 'mc'), f'Unknown target_mode {target_mode!r}'
    assert stopping in ('fixed', 'early_stop'), f'Unknown stopping {stopping!r}'

    obs = data['obs']
    if target_mode == 'mc':
        target_r_all = data['discounted_ret']
        target_c_all = data['discounted_cost_ret']
    else:
        target_r_all = data['target_value_r']
        target_c_all = data['target_value_c']

    use_cost = cfgs.algo_cfgs.use_cost
    n = obs.shape[0]
    device = obs.device

    val_idx = torch.tensor([], dtype=torch.long, device=device)
    train_idx = torch.arange(n, device=device)
    if stopping == 'early_stop':
        n_eps = len(episode_slices)
        n_val_eps = max(1, round(n_eps * val_frac)) if n_eps > 1 else 0
        if n_val_eps > 0:
            val_ep_idx = set(torch.randperm(n_eps)[:n_val_eps].tolist())
            train_mask = torch.ones(n, dtype=torch.bool, device=device)
            val_mask = torch.zeros(n, dtype=torch.bool, device=device)
            for i, (s, e) in enumerate(episode_slices):
                if i in val_ep_idx:
                    train_mask[s:e] = False
                    val_mask[s:e] = True
            train_idx = train_mask.nonzero(as_tuple=True)[0]
            val_idx = val_mask.nonzero(as_tuple=True)[0]

    epochs_cap = fixed_epochs if stopping == 'fixed' else max_epochs
    best_val = float('inf')
    best_state: dict[str, Any] | None = None
    patience_ctr = 0
    train_loss_r = train_loss_c = float('nan')
    val_loss_r = val_loss_c = float('nan')
    epochs_run = 0

    for epoch in range(epochs_cap):
        epochs_run = epoch + 1
        perm = train_idx[torch.randperm(train_idx.shape[0], device=device)]
        losses_r, losses_c = [], []
        for start in range(0, perm.shape[0], batch_size):
            batch = perm[start : start + batch_size]
            losses_r.append(
                _critic_grad_step(
                    actor_critic.reward_critic,
                    actor_critic.reward_critic_optimizer,
                    obs[batch],
                    target_r_all[batch],
                    cfgs,
                ),
            )
            if use_cost:
                losses_c.append(
                    _critic_grad_step(
                        actor_critic.cost_critic,
                        actor_critic.cost_critic_optimizer,
                        obs[batch],
                        target_c_all[batch],
                        cfgs,
                    ),
                )
        train_loss_r = sum(losses_r) / len(losses_r)
        train_loss_c = sum(losses_c) / len(losses_c) if losses_c else float('nan')

        if stopping == 'early_stop' and val_idx.numel() > 0:
            with torch.no_grad():
                val_loss_r = F.mse_loss(
                    actor_critic.reward_critic(obs[val_idx])[0],
                    target_r_all[val_idx],
                ).item()
                val_loss_c = (
                    F.mse_loss(
                        actor_critic.cost_critic(obs[val_idx])[0],
                        target_c_all[val_idx],
                    ).item()
                    if use_cost
                    else float('nan')
                )
            monitor = val_loss_r + (val_loss_c if use_cost else 0.0)
            if monitor < best_val - min_delta:
                best_val = monitor
                patience_ctr = 0
                best_state = {'reward_critic': {
                    k: v.clone() for k, v in actor_critic.reward_critic.state_dict().items()
                }}
                if use_cost:
                    best_state['cost_critic'] = {
                        k: v.clone() for k, v in actor_critic.cost_critic.state_dict().items()
                    }
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

    if stopping == 'early_stop' and best_state is not None:
        actor_critic.reward_critic.load_state_dict(best_state['reward_critic'])
        if use_cost:
            actor_critic.cost_critic.load_state_dict(best_state['cost_critic'])

    return {
        'epochs_run': epochs_run,
        'n_train': int(train_idx.numel()),
        'n_val': int(val_idx.numel()),
        'final_train_loss_r': train_loss_r,
        'final_train_loss_c': train_loss_c,
        'final_val_loss_r': val_loss_r if stopping == 'early_stop' else float('nan'),
        'final_val_loss_c': val_loss_c if stopping == 'early_stop' else float('nan'),
        'best_val_loss': best_val if best_state is not None else float('nan'),
    }
