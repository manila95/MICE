"""Utility for evaluating true vs. estimated value functions."""

from __future__ import annotations

import os

import numpy as np
import torch
from rich.progress import Progress

from omnisafe.utils.gae import calculate_adv_and_value_targets


def estimate_true_value(agent, env, cfgs, discount_r, discount_c, eval_episodes=100, epoch=None):
    """Estimate true V(s) vs. critic estimate by rolling out full episodes.

    Uses the provided (already-wrapped) environment directly so that observation
    and reward normalizers are identical to those seen during training.

    Two evaluation regimes:
    - Eval_s0: initial state of each episode — V(s_0) vs. G_0.
    - Eval_all: every visited state — V(s_t) vs. G_t (MC return from step t).
    """

    def _predict(obs):
        act, est_r, est_c, _ = agent.step(obs)
        return act, est_r, est_c

    return _rollout_and_report(
        predict=_predict,
        env=env,
        cfgs=cfgs,
        discount_r=discount_r,
        discount_c=discount_c,
        eval_episodes=eval_episodes,
        epoch=epoch,
        value_symbol='V',
    )


def estimate_true_q_value(
    agent,
    env,
    cfgs,
    discount_r,
    discount_c,
    eval_episodes=100,
    epoch=None,
    deterministic=False,
):
    """Off-policy counterpart of :func:`estimate_true_value` for Q-critics.

    Identical protocol -- roll the *current* policy out for whole episodes on the (already
    wrapped) training environment and compare the critic's prediction at each visited state
    against the Monte-Carlo return from that state -- except that the estimate is
    ``Q(s_t, a_t)`` for the action actually taken rather than ``V(s_t)``. That is the right
    pairing: the MC return is the return of the trajectory that follows ``a_t``, so it is an
    unbiased sample of ``Q^pi(s_t, a_t)``.

    Actions are sampled from the stochastic policy by default (``deterministic=False``),
    matching how transitions are generated during training; the MC return must come from the
    same policy the critic is being fit to.

    .. note::
        For SAC-style agents the reward critic is trained toward an *entropy-augmented*
        target, so ``Q_r`` estimates the soft return, whereas the MC return here is the plain
        discounted reward sum. The gap between the two is the discounted entropy term
        ``alpha * sum_t gamma^t H(pi(.|s_t))``, i.e. ``Eval_*/EstimationError_r`` carries a
        systematic offset of that size at nonzero ``alpha``. The correlations, and every cost
        statistic (the cost critic has no entropy bonus), are unaffected.
    """
    use_cost = cfgs.algo_cfgs.use_cost

    def _predict(obs):
        with torch.no_grad():
            act = agent.step(obs, deterministic=deterministic)
            q_r = list(agent.reward_critic(obs, act))
            # Twin critics: the min is what the actor loss and the TD target actually consume,
            # so it is the estimate worth calibrating.
            est_r = torch.min(torch.stack(q_r, dim=0), dim=0).values if len(q_r) > 1 else q_r[0]
            est_c = agent.cost_critic(obs, act)[0] if use_cost else torch.zeros_like(est_r)
        return act, est_r.reshape(-1), est_c.reshape(-1)

    return _rollout_and_report(
        predict=_predict,
        env=env,
        cfgs=cfgs,
        discount_r=discount_r,
        discount_c=discount_c,
        eval_episodes=eval_episodes,
        epoch=epoch,
        value_symbol='Q',
    )


def _rollout_and_report(  # pylint: disable=too-many-locals,too-many-statements
    predict,
    env,
    cfgs,
    discount_r,
    discount_c,
    eval_episodes,
    epoch,
    value_symbol,
):
    """Roll ``predict``'s policy out for ``eval_episodes`` episodes and log the calibration stats.

    ``predict(obs)`` returns ``(action, estimate_r, estimate_c)``, where the estimates are the
    critic's prediction for the state (V) or the state-action pair (Q) about to be stepped.
    """
    device = torch.device(cfgs.train_cfgs.device)
    num_envs = env.num_envs

    obs, _ = env.reset()
    act, cur_est_r, cur_est_c = predict(obs)

    # Per-env episode history: each entry is (est_r, est_c, reward, cost) at step t
    ep_history = [[] for _ in range(num_envs)]

    # s0 collected data
    s0_true_r, s0_true_c = [], []
    s0_est_r,  s0_est_c  = [], []
    # all-states collected data
    all_true_r, all_true_c = [], []
    all_est_r,  all_est_c  = [], []

    episodes_done = 0

    with Progress() as progress:
        task = progress.add_task('Evaluating value function...', total=eval_episodes)
        while episodes_done < eval_episodes:
            next_obs, r, c, terminated, truncated, _ = env.step(act)

            # ``.squeeze(-1)`` is unsafe here: when num_envs == 1 a (1,) or (1, 1) reward
            # collapses all the way to a 0-d scalar (there is no trailing size-1 feature dim to
            # remove, only the batch dim), breaking the per-env indexing below. ``.reshape``
            # instead pins the shape to exactly ``(num_envs,)`` regardless of whether the env
            # returns ``(num_envs,)`` or ``(num_envs, 1)``.
            r_sq = r.reshape(num_envs)
            c_sq = c.reshape(num_envs)

            # Record V(s_t), V_c(s_t), r_t, c_t BEFORE moving to next state
            for i in range(num_envs):
                ep_history[i].append((
                    cur_est_r[i].item(),
                    cur_est_c[i].item(),
                    r_sq[i].item(),
                    c_sq[i].item(),
                ))

            done = (terminated.bool() | truncated.bool()).reshape(num_envs)

            newly_done = 0
            for i in done.nonzero(as_tuple=False).flatten().tolist():
                if episodes_done < eval_episodes:
                    hist = ep_history[i]
                    T = len(hist)

                    # Backward scan: G_t = r_t + gamma * G_{t+1}
                    G_r, G_c = 0.0, 0.0
                    step_true_r = [0.0] * T
                    step_true_c = [0.0] * T
                    for t in range(T - 1, -1, -1):
                        G_r = hist[t][2] + discount_r * G_r
                        G_c = hist[t][3] + discount_c * G_c
                        step_true_r[t] = G_r
                        step_true_c[t] = G_c

                    # s0
                    s0_true_r.append(step_true_r[0])
                    s0_true_c.append(step_true_c[0])
                    s0_est_r.append(hist[0][0])
                    s0_est_c.append(hist[0][1])

                    # all states
                    for t in range(T):
                        all_true_r.append(step_true_r[t])
                        all_true_c.append(step_true_c[t])
                        all_est_r.append(hist[t][0])
                        all_est_c.append(hist[t][1])

                    episodes_done += 1
                    newly_done += 1

                ep_history[i] = []

            progress.update(task, advance=newly_done)

            obs = next_obs
            act, cur_est_r, cur_est_c = predict(obs)

    def _to_tensor(lst):
        return torch.tensor(lst, device=device, dtype=torch.float32)

    s0_true_r_t  = _to_tensor(s0_true_r)
    s0_true_c_t  = _to_tensor(s0_true_c)
    s0_est_r_t   = _to_tensor(s0_est_r)
    s0_est_c_t   = _to_tensor(s0_est_c)

    all_true_r_t = _to_tensor(all_true_r)
    all_true_c_t = _to_tensor(all_true_c)
    all_est_r_t  = _to_tensor(all_est_r)
    all_est_c_t  = _to_tensor(all_est_c)

    def _stats(true_t, est_t):
        error    = torch.mean(true_t - est_t)
        true_m   = torch.mean(true_t)
        est_m    = torch.mean(est_t)
        corr     = torch.corrcoef(torch.stack([true_t, est_t]))[0, 1]
        return error, true_m, est_m, corr

    s0_c_error,  s0_true_c_m,  s0_est_c_m,  s0_corr_c  = _stats(s0_true_c_t,  s0_est_c_t)
    s0_r_error,  s0_true_r_m,  s0_est_r_m,  s0_corr_r  = _stats(s0_true_r_t,  s0_est_r_t)
    all_c_error, all_true_c_m, all_est_c_m, all_corr_c = _stats(all_true_c_t, all_est_c_t)
    all_r_error, all_true_r_m, all_est_r_m, all_corr_r = _stats(all_true_r_t, all_est_r_t)

    # Opt-in local dump of the raw per-state (true, estimate) pairs behind the scatter plots
    # above, keyed by epoch. Independent of use_wandb -- offline analysis (e.g. reliability /
    # calibration diagrams binned across many epochs) needs the raw arrays, not just the wandb
    # scatter images and the aggregate corr/error scalars logged below.
    dump_dir = os.environ.get('MICE_VALUE_EVAL_DUMP_DIR')
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        np.savez(
            os.path.join(dump_dir, f'{value_symbol}_epoch_{(epoch or 0):04d}.npz'),
            s0_true_c=s0_true_c_t.cpu().numpy(),
            s0_est_c=s0_est_c_t.cpu().numpy(),
            s0_true_r=s0_true_r_t.cpu().numpy(),
            s0_est_r=s0_est_r_t.cpu().numpy(),
            all_true_c=all_true_c_t.cpu().numpy(),
            all_est_c=all_est_c_t.cpu().numpy(),
            all_true_r=all_true_r_t.cpu().numpy(),
            all_est_r=all_est_r_t.cpu().numpy(),
        )

    if cfgs.logger_cfgs.use_wandb:
        import matplotlib.pyplot as plt
        import wandb

        def _scatter_fig(true_vals, est_vals, label, color, title):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(true_vals, est_vals, alpha=0.3, s=8, color=color)
            lo = min(true_vals.min(), est_vals.min())
            hi = max(true_vals.max(), est_vals.max())
            ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, label='ideal')
            ax.set_xlabel(f'True {label}')
            ax.set_ylabel(f'Estimated {label}')
            ax.set_title(title)
            ax.legend()
            plt.tight_layout()
            return fig

        s0_c_np  = s0_true_c_t.cpu().numpy();  s0_ec_np  = s0_est_c_t.cpu().numpy()
        s0_r_np  = s0_true_r_t.cpu().numpy();  s0_er_np  = s0_est_r_t.cpu().numpy()
        all_c_np = all_true_c_t.cpu().numpy(); all_ec_np = all_est_c_t.cpu().numpy()
        all_r_np = all_true_r_t.cpu().numpy(); all_er_np = all_est_r_t.cpu().numpy()

        word = 'Values' if value_symbol == 'V' else 'Q-Values'
        fig_s0_c   = _scatter_fig(s0_c_np,  s0_ec_np,  'C', 'steelblue',  f'C-{word}: True vs Estimated (s0)')
        fig_s0_r   = _scatter_fig(s0_r_np,  s0_er_np,  'R', 'darkorange', f'R-{word}: True vs Estimated (s0)')
        fig_all_c  = _scatter_fig(all_c_np, all_ec_np, 'C', 'steelblue',  f'C-{word}: True vs Estimated (all states)')
        fig_all_r  = _scatter_fig(all_r_np, all_er_np, 'R', 'darkorange', f'R-{word}: True vs Estimated (all states)')

        wandb.log({
            # Scatter plots
            'scatter/s0_c_values':  wandb.Image(fig_s0_c),
            'scatter/s0_r_values':  wandb.Image(fig_s0_r),
            'scatter/all_c_values': wandb.Image(fig_all_c),
            'scatter/all_r_values': wandb.Image(fig_all_r),
            # Eval_s0 stats
            'Eval_s0/Correlation_c':      s0_corr_c.item(),
            'Eval_s0/Correlation_r':      s0_corr_r.item(),
            'Eval_s0/EstimationError_c':  s0_c_error.item(),
            'Eval_s0/true_value_c':       s0_true_c_m.item(),
            'Eval_s0/estimate_value_c':   s0_est_c_m.item(),
            'Eval_s0/EstimationError_r':  s0_r_error.item(),
            'Eval_s0/true_value_r':       s0_true_r_m.item(),
            'Eval_s0/estimate_value_r':   s0_est_r_m.item(),
            # Eval_all stats
            'Eval_all/Correlation_c':     all_corr_c.item(),
            'Eval_all/Correlation_r':     all_corr_r.item(),
            'Eval_all/EstimationError_c': all_c_error.item(),
            'Eval_all/true_value_c':      all_true_c_m.item(),
            'Eval_all/estimate_value_c':  all_est_c_m.item(),
            'Eval_all/EstimationError_r': all_r_error.item(),
            'Eval_all/true_value_r':      all_true_r_m.item(),
            'Eval_all/estimate_value_r':  all_est_r_m.item(),
        }, step=epoch)

        plt.close(fig_s0_c)
        plt.close(fig_s0_r)
        plt.close(fig_all_c)
        plt.close(fig_all_r)

    return (
        s0_c_error,  s0_true_c_m,  s0_est_c_m,  s0_corr_c,
        s0_r_error,  s0_true_r_m,  s0_est_r_m,  s0_corr_r,
        all_c_error, all_true_c_m, all_est_c_m, all_corr_c,
        all_r_error, all_true_r_m, all_est_r_m, all_corr_r,
    )


def _find_obs_normalizer(env):
    """Walk an env's wrapper chain (outermost first) for its ``ObsNormalize`` instance, if any.

    ``Wrapper.__getattr__`` only forwards non-underscore names (see ``omnisafe.envs.core.Wrapper``),
    so an outer wrapper (``ActionScale``, ``Unsqueeze``, ...) can't reach an inner
    ``ObsNormalize``'s ``_obs_normalizer`` through attribute delegation alone -- this walks the
    ``._env`` chain explicitly instead. Returns ``None`` if no ``ObsNormalize`` wrapper is present
    (``algo_cfgs.obs_normalize=False``), in which case there is nothing to sync.
    """
    # Local import: value_eval.py must not import omnisafe.envs.wrapper at module load time (it
    # would create a circular import, since that module's callers import from here too).
    from omnisafe.envs.wrapper import ObsNormalize  # noqa: PLC0415

    while True:
        if isinstance(env, ObsNormalize):
            return env._obs_normalizer  # noqa: SLF001
        inner = getattr(env, '_env', None)
        if inner is None:
            return None
        env = inner


def sync_obs_normalizer(target_env, source_env) -> None:
    """Copy ``source_env``'s ``ObsNormalize`` running statistics into ``target_env``'s.

    A no-op if either env has no ``ObsNormalize`` wrapper (``algo_cfgs.obs_normalize=False``).
    Used to snapshot a dedicated eval-only env's normalizer from the live training env's current
    statistics before probing with it -- see ``estimate_true_value_same_state_mc``'s
    ``sync_normalizer_from`` and the on-policy intermediate-state study's collection env, which
    both need the critic's inputs to be processed exactly as they would be during real training,
    without the dedicated env's own normalizer (typically built with ``update_stats=False``, see
    ``omnisafe.envs.wrapper.ObsNormalize``) needing to independently accumulate its own history to
    get there.
    """
    target_norm = _find_obs_normalizer(target_env)
    source_norm = _find_obs_normalizer(source_env)
    if target_norm is not None and source_norm is not None:
        target_norm.load_state_dict(source_norm.state_dict())


def _rollout_target(
    r_seq,
    v_seq_incl_boot,
    terminated,
    lam,
    gamma,
    advantage_estimator,
    logp_seq=None,
):
    r"""The training-style regression target for one already-completed rollout.

    Runs the *exact same* GAE/plain/vtrace/etc. formula (:func:`omnisafe.utils.gae.
    calculate_adv_and_value_targets`) the training buffer uses on its own on-policy trajectories
    -- applied here to a probe rollout's own reward/value sequence -- and returns the target at
    that rollout's own first timestep (index 0), which is the value the current policy/critic
    combination would actually have been trained to predict for that starting state, had this
    rollout been part of a training batch.

    Mirrors ``OnPolicyAdapter.rollout``'s bootstrap convention exactly (see
    ``omnisafe.utils.gae``'s docstring): 0 if the rollout ended in a true terminal, otherwise the
    critic's own value at the final observation, appended as both the last "value" *and* the last
    "reward" entry (the latter so rewards-to-go-style targets fold the bootstrap in the same way
    GAE's ``values[1:]`` does).

    Args:
        r_seq: ``(T,)`` per-step reward (or cost) sequence for this one rollout.
        v_seq_incl_boot: ``(T+1,)`` per-step critic values *including* the bootstrap value as the
            final entry (already zeroed by the caller for terminated rollouts).
        terminated: Whether this rollout ended in a true terminal (vs. a horizon truncation) --
            only used to decide whether the appended bootstrap "reward" should be 0 or the
            bootstrap value itself (it's always the same value already in ``v_seq_incl_boot[-1]``
            either way, this only controls the pseudo-reward, matching ``finish_path``'s
            ``rewards = torch.cat([..., last_value_r])`` regardless of terminated/truncated --
              the bootstrap value is 0 for a true terminal, so the appended pseudo-reward is 0
              too automatically; kept as an explicit arg for clarity at call sites, not because
              the formula branches on it).
        lam: GAE lambda for this stream.
        gamma: Discount factor for this stream.
        advantage_estimator: ``algo_cfgs.adv_estimation_method`` / ``cost_adv_estimation_method``.
        logp_seq: ``(T,)`` log-probabilities, only read under ``advantage_estimator == 'vtrace'``.

    Returns:
        The scalar target value at this rollout's first timestep.
    """
    del terminated  # see docstring -- already folded into v_seq_incl_boot[-1]
    rewards = torch.cat([r_seq, v_seq_incl_boot[-1:]])
    action_probs = logp_seq.exp() if advantage_estimator == 'vtrace' else None
    _, target = calculate_adv_and_value_targets(
        values=v_seq_incl_boot,
        rewards=rewards,
        lam=lam,
        gamma=gamma,
        advantage_estimator=advantage_estimator,
        action_probs=action_probs,
        behavior_action_probs=action_probs,
    )
    return target[0].item()


def estimate_true_value_same_state_mc(
    agent,
    env,
    cfgs,
    discount_r,
    discount_c,
    probe_seeds,
    mc_repeats=5,
    epoch=None,
    sync_normalizer_from=None,
    max_episode_steps=None,
    return_raw=False,
):
    r"""Compare the critic's V(s0) against a genuine same-layout Monte-Carlo estimate.

    ``estimate_true_value`` above scores the critic against one MC sample per visited state --
    each state is seen once, under whatever action the current policy happened to sample there,
    so its "true" return is a single noisy draw, not an estimate with a known variance. This
    function instead fixes a state (a Safety-Gymnasium *layout*, reproduced exactly by resetting
    with the same seed -- procedural generation is deterministic in the seed) and re-rolls the
    current *stochastic* policy out from it ``mc_repeats`` times, so the resulting sample mean is
    a genuine Monte-Carlo estimate of :math:`V^\pi(s_0)` / :math:`V_c^\pi(s_0)` with a directly
    measurable sample variance -- the closest thing to an oracle this codebase can produce without
    an analytic model of the environment.

    ``env`` may be vectorized (``num_envs = N > 1``): the same-layout requirement only constrains
    *one env instance* per probe (each repeat needs a clean, uninterrupted ``reset(seed=X)`` ->
    rollout on the *same* underlying env slot), not the whole call to run on a single env. With
    N > 1, the ``len(probe_seeds) * mc_repeats`` independent rollouts are packed into
    ``ceil(total / N)`` waves of N concurrent (subprocess-parallel, since
    ``safety_gymnasium.vector.make`` defaults to ``asynchronous=True``) rollouts instead of
    running one at a time -- close to an N-x speedup on what was the dominant cost of the whole
    training loop. Pass a dedicated eval env (see ``sync_normalizer_from``) built via
    ``omnisafe.envs.core.make(..., num_envs=N)`` with the same wrapper recipe as training
    (``TimeLimit``/``AutoReset``/``ObsNormalize``/``ActionScale``, plus ``Unsqueeze`` only when
    N == 1) -- training itself does not need to give up its own vectorization for this.

    Assumes every probe episode reaches ``done`` at exactly ``max_episode_steps`` (uniformly
    across the whole vectorized batch) -- true for Safety-Gymnasium's Goal/Push/etc. tasks, which
    only end via the time limit and never terminate early (verified empirically: ``Metrics/EpLen``
    is exactly ``max_episode_steps`` in every logged epoch of every run in this codebase). This
    lets the rollout loop below always run for exactly ``max_episode_steps`` steps rather than
    tracking a per-slot done mask, which keeps the vectorized loop simple; it is checked with an
    assertion after each wave, so a future environment that terminates early would fail loudly
    here instead of silently producing wrong (early-terminated-episode-truncated) returns -- such
    an environment would need a done mask added to freeze each slot's accumulation independently.

    Args:
        agent: The actor-critic; must expose ``step(obs) -> (action, value_r, value_c, log_prob)``.
        env: An already-wrapped environment (any ``num_envs >= 1``) to roll the probes out in.
            May be a dedicated eval env (see ``sync_normalizer_from``) or the training env itself.
        cfgs: The resolved algorithm config.
        discount_r (float): Reward discount factor.
        discount_c (float): Cost discount factor.
        probe_seeds (list of int): The fixed set of layouts (env reset seeds) to probe. Held
            constant across calls so the same states are re-evaluated at every eval epoch.
        mc_repeats (int): Number of independent same-layout rollouts averaged per probe seed.
            Defaults to 5 -- a deliberately modest budget: this routine is
            ``len(probe_seeds) * mc_repeats`` full episodes *per call*, so the cost scales
            directly with this number (before accounting for the ``env.num_envs``-way speedup).
        epoch (int or None): Current epoch, for logging only.
        sync_normalizer_from: Optional live env (e.g. the training env) whose ``ObsNormalize``
            running statistics are copied into ``env``'s ``ObsNormalize`` (via
            :func:`_find_obs_normalizer` and a ``state_dict`` copy) before probing. A no-op if
            either env has no ``ObsNormalize`` wrapper (``algo_cfgs.obs_normalize=False``).
            ``None`` (the default) skips syncing -- appropriate only if ``env`` already *is* the
            live training env, or normalization is off.
        max_episode_steps (int or None): Episode horizon to roll every probe out for. Required
            when ``env.num_envs > 1`` (a vectorized ``AsyncVectorEnv`` has no ``.spec`` to read
            this off directly -- the caller must supply it, e.g. from a disposable single-env
            instance). Optional when ``env.num_envs == 1``, where it defaults to
            ``env.max_episode_steps``.
        return_raw (bool): If True, also return the per-probe arrays the aggregate stats were
            computed from (predictions, MC-mean returns, per-repeat variances) -- e.g. for
            pickling alongside the aggregates for later offline analysis, or for scatter plots
            (each probe is one point). Defaults to False (the original, stats-dict-only return).

    Returns:
        A dict of aggregate statistics over the ``len(probe_seeds)`` probes: for each of
        ``r`` (reward) and ``c`` (cost), ``EstimationError_{r,c}`` (mean of MC estimate minus
        critic prediction), ``Correlation_{r,c}`` (Pearson correlation between the
        ``len(probe_seeds)`` critic predictions and MC-mean estimates), and ``MeanVar_{r,c}``
        (the sample variance of the ``mc_repeats`` returns at each probe, averaged over probes --
        how noisy the MC "oracle" itself is at this point in training). If ``return_raw``, a
        ``(stats, raw)`` tuple instead, where ``raw`` is a dict with ``probe_seeds`` and, for each
        of ``r``/``c``: ``pred`` (critic's V(s) at each probe), ``mc_mean`` (MC-estimated true
        value at each probe), ``mc_var`` (MC sample variance at each probe) -- all
        ``len(probe_seeds)``-length lists, in ``probe_seeds`` order.
    """
    if sync_normalizer_from is not None:
        sync_obs_normalizer(env, sync_normalizer_from)
    device = torch.device(cfgs.train_cfgs.device)

    n_envs = env.num_envs
    if max_episode_steps is None:
        assert n_envs == 1, (
            'max_episode_steps must be passed explicitly when env.num_envs > 1 (a vectorized '
            'env has no .spec to read it off).'
        )
        max_episode_steps = env.max_episode_steps
    assert max_episode_steps and max_episode_steps > 0

    # Which advantage estimator/lambda each stream's target should mirror -- the same config
    # training itself reads (cost null-falls-back to reward's, same convention as
    # cost_gamma/critic_norm_coef_cost/etc. elsewhere in this codebase).
    adv_estimator_r = getattr(cfgs.algo_cfgs, 'adv_estimation_method', 'gae')
    adv_estimator_c = getattr(cfgs.algo_cfgs, 'cost_adv_estimation_method', None) or adv_estimator_r
    lam_r = getattr(cfgs.algo_cfgs, 'lam', 0.95)
    lam_c = getattr(cfgs.algo_cfgs, 'lam_c', lam_r)
    penalty_coef = getattr(cfgs.algo_cfgs, 'penalty_coef', 0.0)

    # Flat task list, mc_repeats consecutive entries per probe seed -- waves below cut across
    # this list without regard to seed boundaries; the seed-grouping happens afterward, purely
    # by array-index arithmetic, so it doesn't matter which wave a given repeat lands in.
    tasks = [seed for seed in probe_seeds for _ in range(mc_repeats)]
    n_tasks = len(tasks)
    pred_r_of: list[float | None] = [None] * n_tasks
    pred_c_of: list[float | None] = [None] * n_tasks
    ret_r_of: list[float] = [0.0] * n_tasks
    ret_c_of: list[float] = [0.0] * n_tasks
    target_r_of: list[float] = [0.0] * n_tasks
    target_c_of: list[float] = [0.0] * n_tasks

    n_waves = (n_tasks + n_envs - 1) // n_envs
    for w in range(n_waves):
        start = w * n_envs
        wave_task_idxs = list(range(start, min(start + n_envs, n_tasks)))
        wave_size = len(wave_task_idxs)
        wave_seeds = [tasks[i] for i in wave_task_idxs]
        if wave_size < n_envs:
            # Last, partial wave: pad with a repeated seed so every env slot still gets a valid
            # reset -- the padding slots' results are simply never read back out below.
            wave_seeds += [wave_seeds[0]] * (n_envs - wave_size)

        obs, _ = env.reset(seed=wave_seeds if n_envs > 1 else wave_seeds[0])
        act, value_r, value_c, log_prob = agent.step(obs)
        # V(s0) is deterministic given s0 (no action taken yet), so it only needs computing once
        # per rollout -- cheap regardless (one batched NN forward pass vs. hundreds of env steps).
        pred_r_batch = value_r.reshape(-1).detach().cpu().numpy()
        pred_c_batch = value_c.reshape(-1).detach().cpu().numpy()

        # Per-step sequences, needed (in addition to the running discounted sum below, which is
        # all the *true* MC estimate needs) to compute the training-style target after the
        # rollout -- see _rollout_target. Row max_episode_steps is the bootstrap slot, filled in
        # after the loop.
        r_seq = np.zeros((max_episode_steps + 1, n_envs), dtype=np.float64)
        c_seq = np.zeros((max_episode_steps + 1, n_envs), dtype=np.float64)
        v_r_seq = np.zeros((max_episode_steps + 1, n_envs), dtype=np.float64)
        v_c_seq = np.zeros((max_episode_steps + 1, n_envs), dtype=np.float64)
        logp_seq = np.zeros((max_episode_steps, n_envs), dtype=np.float64)
        v_r_seq[0] = pred_r_batch
        v_c_seq[0] = pred_c_batch

        g_r = np.zeros(n_envs, dtype=np.float64)
        g_c = np.zeros(n_envs, dtype=np.float64)
        disc_r, disc_c = 1.0, 1.0
        terminated = truncated = None
        for t in range(max_episode_steps):
            obs, r, c, terminated, truncated, _ = env.step(act)
            r_np = r.reshape(-1).detach().cpu().numpy()
            c_np = c.reshape(-1).detach().cpu().numpy()
            r_seq[t] = r_np
            c_seq[t] = c_np
            g_r += disc_r * r_np
            g_c += disc_c * c_np
            disc_r *= discount_r
            disc_c *= discount_c
            if t < max_episode_steps - 1:
                act, value_r, value_c, log_prob = agent.step(obs)
                v_r_seq[t + 1] = value_r.reshape(-1).detach().cpu().numpy()
                v_c_seq[t + 1] = value_c.reshape(-1).detach().cpu().numpy()
                logp_seq[t] = log_prob.reshape(-1).detach().cpu().numpy()
        done_at_end = terminated.reshape(-1).bool() | truncated.reshape(-1).bool()
        if not bool(done_at_end.all()):
            raise RuntimeError(
                'estimate_true_value_same_state_mc: not every probe reached done at '
                f'max_episode_steps={max_episode_steps} (got done={done_at_end.tolist()}). This '
                'function assumes homogeneous, fixed-length episodes across the vectorized '
                "batch (see docstring) -- this environment doesn't satisfy that and needs a "
                'per-slot done mask added instead.',
            )

        # Bootstrap value at the final observation -- 0 for a true terminal, the critic's own
        # prediction there otherwise -- mirroring OnPolicyAdapter.rollout's last_value_r/c
        # construction exactly (see omnisafe.utils.gae's docstring). Queried for every slot
        # uniformly (cheap: one more batched forward pass) rather than branching per-env; the
        # terminated slots' values are simply zeroed out afterward.
        _, boot_value_r, boot_value_c, _ = agent.step(obs)
        is_terminated = terminated.reshape(-1).bool().detach().cpu().numpy()
        boot_r = np.where(is_terminated, 0.0, boot_value_r.reshape(-1).detach().cpu().numpy())
        boot_c = np.where(is_terminated, 0.0, boot_value_c.reshape(-1).detach().cpu().numpy())
        v_r_seq[max_episode_steps] = boot_r
        v_c_seq[max_episode_steps] = boot_c

        for local_i, task_idx in enumerate(wave_task_idxs):
            pred_r_of[task_idx] = float(pred_r_batch[local_i])
            pred_c_of[task_idx] = float(pred_c_batch[local_i])
            ret_r_of[task_idx] = float(g_r[local_i])
            ret_c_of[task_idx] = float(g_c[local_i])

            r_this = torch.from_numpy(r_seq[:max_episode_steps, local_i]).float()
            c_this = torch.from_numpy(c_seq[:max_episode_steps, local_i]).float()
            v_r_this = torch.from_numpy(v_r_seq[:, local_i]).float()
            v_c_this = torch.from_numpy(v_c_seq[:, local_i]).float()
            logp_this = torch.from_numpy(logp_seq[:, local_i]).float()
            # Mirrors finish_path's `rewards -= penalty_coefficient * costs` -- an intrinsic-cost
            # penalty folded into the reward stream's target only; a no-op at the (default) 0.0.
            r_this_penalized = r_this - penalty_coef * c_this
            target_r_of[task_idx] = _rollout_target(
                r_this_penalized, v_r_this, bool(is_terminated[local_i]), lam_r, discount_r,
                adv_estimator_r, logp_seq=logp_this,
            )
            target_c_of[task_idx] = _rollout_target(
                c_this, v_c_this, bool(is_terminated[local_i]), lam_c, discount_c,
                adv_estimator_c, logp_seq=logp_this,
            )

    pred_r_list: list[float] = []
    pred_c_list: list[float] = []
    mc_mean_r_list: list[float] = []
    mc_mean_c_list: list[float] = []
    mc_var_r_list: list[float] = []
    mc_var_c_list: list[float] = []
    target_r_list: list[float] = []
    target_c_list: list[float] = []
    idx = 0
    for _seed in probe_seeds:
        idxs = range(idx, idx + mc_repeats)
        idx += mc_repeats
        pred_r_list.append(pred_r_of[idxs[0]])
        pred_c_list.append(pred_c_of[idxs[0]])
        returns_r = [ret_r_of[i] for i in idxs]
        returns_c = [ret_c_of[i] for i in idxs]
        mc_mean_r_list.append(float(np.mean(returns_r)))
        mc_mean_c_list.append(float(np.mean(returns_c)))
        mc_var_r_list.append(float(np.var(returns_r)))
        mc_var_c_list.append(float(np.var(returns_c)))
        # Averaged across the mc_repeats independent rollouts, same as mc_mean -- the target
        # genuinely varies per repeat (it depends on the realized trajectory, unlike pred which
        # only depends on s0), so this is the same kind of reduction, not a different one.
        target_r_list.append(float(np.mean([target_r_of[i] for i in idxs])))
        target_c_list.append(float(np.mean([target_c_of[i] for i in idxs])))

    def _t(lst):
        return torch.tensor(lst, device=device, dtype=torch.float32)

    pred_r_t, pred_c_t = _t(pred_r_list), _t(pred_c_list)
    mc_mean_r_t, mc_mean_c_t = _t(mc_mean_r_list), _t(mc_mean_c_list)
    target_r_t, target_c_t = _t(target_r_list), _t(target_c_list)

    def _corr(a, b):
        if a.numel() < 2 or a.std() <= 0 or b.std() <= 0:
            return float('nan')
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    stats = {
        'MCStudy/EstimationError_r': (mc_mean_r_t - pred_r_t).mean().item(),
        'MCStudy/EstimationError_c': (mc_mean_c_t - pred_c_t).mean().item(),
        'MCStudy/Correlation_r': _corr(pred_r_t, mc_mean_r_t),
        'MCStudy/Correlation_c': _corr(pred_c_t, mc_mean_c_t),
        # Decomposes the pred-vs-true correlation above into its two independent failure modes
        # (mirrors Value/Train/*'s TargetTrueCorr/CriticCorr, applied here to the eval probes
        # instead of the training buffer's own trajectories -- see PolicyGradient.
        # _log_critic_diagnostics): is the *target construction itself* (whatever
        # adv_estimation_method computes) biased relative to the true MC value, independent of how
        # well the critic fits it (Correlation_target_true); and is the critic actually fitting
        # what it's trained to fit, independent of whether that target is itself a good proxy for
        # the truth (Correlation_pred_target).
        'MCStudy/EstimationError_target_true_r': (mc_mean_r_t - target_r_t).mean().item(),
        'MCStudy/EstimationError_target_true_c': (mc_mean_c_t - target_c_t).mean().item(),
        'MCStudy/Correlation_target_true_r': _corr(target_r_t, mc_mean_r_t),
        'MCStudy/Correlation_target_true_c': _corr(target_c_t, mc_mean_c_t),
        'MCStudy/Correlation_pred_target_r': _corr(pred_r_t, target_r_t),
        'MCStudy/Correlation_pred_target_c': _corr(pred_c_t, target_c_t),
        'MCStudy/MeanTarget_r': target_r_t.mean().item(),
        'MCStudy/MeanTarget_c': target_c_t.mean().item(),
        'MCStudy/MeanVar_r': float(np.mean(mc_var_r_list)),
        'MCStudy/MeanVar_c': float(np.mean(mc_var_c_list)),
        'MCStudy/MeanTrue_r': mc_mean_r_t.mean().item(),
        'MCStudy/MeanTrue_c': mc_mean_c_t.mean().item(),
        'MCStudy/MeanPred_r': pred_r_t.mean().item(),
        'MCStudy/MeanPred_c': pred_c_t.mean().item(),
        'MCStudy/NumProbes': float(len(probe_seeds)),
        'MCStudy/MCRepeats': float(mc_repeats),
    }
    del epoch  # accepted for call-site symmetry with estimate_true_value; not used here
    if not return_raw:
        return stats
    raw = {
        'probe_seeds': list(probe_seeds),
        'r': {'pred': pred_r_list, 'mc_mean': mc_mean_r_list, 'mc_var': mc_var_r_list, 'target': target_r_list},
        'c': {'pred': pred_c_list, 'mc_mean': mc_mean_c_list, 'mc_var': mc_var_c_list, 'target': target_c_list},
    }
    return stats, raw


def estimate_value_from_snapshots(
    agent,
    env,
    cfgs,
    discount_r,
    discount_c,
    snapshots,
    remaining_horizon,
    mc_repeats=5,
    epoch=None,
    return_raw=False,
):
    r"""Like :func:`estimate_true_value_same_state_mc`, but for arbitrary on-policy *intermediate*
    states captured via :mod:`omnisafe.utils.state_snapshot`, instead of states reachable by
    ``env.reset(seed=X)``. Where the s0 study asks "is the critic accurate at episode starts", this
    asks the question the critic's accuracy actually needs to answer for TD-learning/GAE/advantages
    to work: is it accurate at states the *current* policy actually visits mid-episode?

    Structurally almost identical to :func:`estimate_true_value_same_state_mc` -- same
    wave-batched rollout, same per-probe ``mc_repeats`` averaging, same aggregate stats -- with two
    differences: probes are restored from pre-captured snapshots
    (:func:`omnisafe.utils.state_snapshot.restore_and_get_obs`) instead of reset from seeds, and
    the rollout horizon is ``remaining_horizon`` (however many steps were left in the episode when
    the snapshot was captured), not the full episode length.

    Args:
        agent: The actor-critic; must expose ``step(obs) -> (action, value_r, value_c, log_prob)``.
        env: An already-wrapped, vectorized (``num_envs >= 1``) env to roll the probes out in --
            same recipe as ``estimate_true_value_same_state_mc``'s ``env`` argument, but there is
            no ``sync_normalizer_from`` here: normalizer syncing has to happen once, before
            *capturing* the snapshots (so the on-policy actions that produced them were sampled
            under realistic normalization), not per scoring call -- see the state-collection call
            site for where that sync happens.
        discount_r (float): Reward discount factor.
        discount_c (float): Cost discount factor.
        snapshots (list of dict): Pre-captured states (one entry per probe), all from the *same*
            within-episode step index, so they share a single ``remaining_horizon``.
        remaining_horizon (int): Steps left in the episode from this snapshot's step index (i.e.
            ``max_episode_steps - step_index_captured_at``). Every snapshot passed in one call
            must share this same horizon -- a captured-at-step-300 and a captured-at-step-700
            probe can't be scored in the same call, since they'd finish at different times.
        mc_repeats (int): Independent rollouts averaged per probe state.
        epoch (int or None): Current epoch, for logging only.
        return_raw (bool): See :func:`estimate_true_value_same_state_mc` -- same meaning, same
            per-probe ``raw`` dict shape (just no ``probe_seeds`` key, since these probes came
            from pre-captured snapshots rather than reset seeds).

    Returns:
        Same shape/semantics as :func:`estimate_true_value_same_state_mc`'s return dict, just
        without the ``MCStudy/`` key prefix -- callers should apply their own prefix (e.g. tagging
        which within-episode position these snapshots came from). If ``return_raw``, a
        ``(stats, raw)`` tuple -- see :func:`estimate_true_value_same_state_mc`'s docstring.
    """
    # Local import: mirrors state_snapshot.py's own local import of _find_obs_normalizer from
    # here -- avoids import-time coupling between the two modules in either direction.
    from omnisafe.utils.state_snapshot import restore_and_get_obs  # noqa: PLC0415

    device = torch.device(cfgs.train_cfgs.device)
    n_envs = env.num_envs
    assert remaining_horizon and remaining_horizon > 0

    adv_estimator_r = getattr(cfgs.algo_cfgs, 'adv_estimation_method', 'gae')
    adv_estimator_c = getattr(cfgs.algo_cfgs, 'cost_adv_estimation_method', None) or adv_estimator_r
    lam_r = getattr(cfgs.algo_cfgs, 'lam', 0.95)
    lam_c = getattr(cfgs.algo_cfgs, 'lam_c', lam_r)
    penalty_coef = getattr(cfgs.algo_cfgs, 'penalty_coef', 0.0)

    # Flat task list, mc_repeats consecutive entries per probe state -- same rationale as
    # estimate_true_value_same_state_mc's `tasks` list.
    tasks = [snap for snap in snapshots for _ in range(mc_repeats)]
    n_tasks = len(tasks)
    pred_r_of: list[float | None] = [None] * n_tasks
    pred_c_of: list[float | None] = [None] * n_tasks
    ret_r_of: list[float] = [0.0] * n_tasks
    ret_c_of: list[float] = [0.0] * n_tasks
    target_r_of: list[float] = [0.0] * n_tasks
    target_c_of: list[float] = [0.0] * n_tasks

    n_waves = (n_tasks + n_envs - 1) // n_envs
    for w in range(n_waves):
        start = w * n_envs
        wave_task_idxs = list(range(start, min(start + n_envs, n_tasks)))
        wave_size = len(wave_task_idxs)
        wave_snaps = [tasks[i] for i in wave_task_idxs]
        if wave_size < n_envs:
            # Last, partial wave: pad with a repeated snapshot so every env slot still gets a
            # valid restore -- the padding slots' results are simply never read back out below.
            wave_snaps += [wave_snaps[0]] * (n_envs - wave_size)

        obs = restore_and_get_obs(env, wave_snaps, device)
        act, value_r, value_c, log_prob = agent.step(obs)
        pred_r_batch = value_r.reshape(-1).detach().cpu().numpy()
        pred_c_batch = value_c.reshape(-1).detach().cpu().numpy()

        # Per-step sequences, needed to compute the training-style target after the rollout --
        # see estimate_true_value_same_state_mc's matching comment.
        r_seq = np.zeros((remaining_horizon + 1, n_envs), dtype=np.float64)
        c_seq = np.zeros((remaining_horizon + 1, n_envs), dtype=np.float64)
        v_r_seq = np.zeros((remaining_horizon + 1, n_envs), dtype=np.float64)
        v_c_seq = np.zeros((remaining_horizon + 1, n_envs), dtype=np.float64)
        logp_seq = np.zeros((remaining_horizon, n_envs), dtype=np.float64)
        v_r_seq[0] = pred_r_batch
        v_c_seq[0] = pred_c_batch

        g_r = np.zeros(n_envs, dtype=np.float64)
        g_c = np.zeros(n_envs, dtype=np.float64)
        disc_r, disc_c = 1.0, 1.0
        terminated = truncated = None
        for t in range(remaining_horizon):
            obs, r, c, terminated, truncated, _ = env.step(act)
            r_np = r.reshape(-1).detach().cpu().numpy()
            c_np = c.reshape(-1).detach().cpu().numpy()
            r_seq[t] = r_np
            c_seq[t] = c_np
            g_r += disc_r * r_np
            g_c += disc_c * c_np
            disc_r *= discount_r
            disc_c *= discount_c
            if t < remaining_horizon - 1:
                act, value_r, value_c, log_prob = agent.step(obs)
                v_r_seq[t + 1] = value_r.reshape(-1).detach().cpu().numpy()
                v_c_seq[t + 1] = value_c.reshape(-1).detach().cpu().numpy()
                logp_seq[t] = log_prob.reshape(-1).detach().cpu().numpy()
        done_at_end = terminated.reshape(-1).bool() | truncated.reshape(-1).bool()
        if not bool(done_at_end.all()):
            raise RuntimeError(
                'estimate_value_from_snapshots: not every probe reached done after '
                f'remaining_horizon={remaining_horizon} steps (got done={done_at_end.tolist()}). '
                'remaining_horizon must be exactly max_episode_steps minus the step index these '
                'snapshots were captured at.',
            )

        _, boot_value_r, boot_value_c, _ = agent.step(obs)
        is_terminated = terminated.reshape(-1).bool().detach().cpu().numpy()
        boot_r = np.where(is_terminated, 0.0, boot_value_r.reshape(-1).detach().cpu().numpy())
        boot_c = np.where(is_terminated, 0.0, boot_value_c.reshape(-1).detach().cpu().numpy())
        v_r_seq[remaining_horizon] = boot_r
        v_c_seq[remaining_horizon] = boot_c

        for local_i, task_idx in enumerate(wave_task_idxs):
            pred_r_of[task_idx] = float(pred_r_batch[local_i])
            pred_c_of[task_idx] = float(pred_c_batch[local_i])
            ret_r_of[task_idx] = float(g_r[local_i])
            ret_c_of[task_idx] = float(g_c[local_i])

            r_this = torch.from_numpy(r_seq[:remaining_horizon, local_i]).float()
            c_this = torch.from_numpy(c_seq[:remaining_horizon, local_i]).float()
            v_r_this = torch.from_numpy(v_r_seq[:, local_i]).float()
            v_c_this = torch.from_numpy(v_c_seq[:, local_i]).float()
            logp_this = torch.from_numpy(logp_seq[:, local_i]).float()
            r_this_penalized = r_this - penalty_coef * c_this
            target_r_of[task_idx] = _rollout_target(
                r_this_penalized, v_r_this, bool(is_terminated[local_i]), lam_r, discount_r,
                adv_estimator_r, logp_seq=logp_this,
            )
            target_c_of[task_idx] = _rollout_target(
                c_this, v_c_this, bool(is_terminated[local_i]), lam_c, discount_c,
                adv_estimator_c, logp_seq=logp_this,
            )

    pred_r_list: list[float] = []
    pred_c_list: list[float] = []
    mc_mean_r_list: list[float] = []
    mc_mean_c_list: list[float] = []
    mc_var_r_list: list[float] = []
    mc_var_c_list: list[float] = []
    target_r_list: list[float] = []
    target_c_list: list[float] = []
    idx = 0
    for _snap in snapshots:
        idxs = range(idx, idx + mc_repeats)
        idx += mc_repeats
        pred_r_list.append(pred_r_of[idxs[0]])
        pred_c_list.append(pred_c_of[idxs[0]])
        returns_r = [ret_r_of[i] for i in idxs]
        returns_c = [ret_c_of[i] for i in idxs]
        mc_mean_r_list.append(float(np.mean(returns_r)))
        mc_mean_c_list.append(float(np.mean(returns_c)))
        mc_var_r_list.append(float(np.var(returns_r)))
        mc_var_c_list.append(float(np.var(returns_c)))
        target_r_list.append(float(np.mean([target_r_of[i] for i in idxs])))
        target_c_list.append(float(np.mean([target_c_of[i] for i in idxs])))

    def _t(lst):
        return torch.tensor(lst, device=device, dtype=torch.float32)

    pred_r_t, pred_c_t = _t(pred_r_list), _t(pred_c_list)
    mc_mean_r_t, mc_mean_c_t = _t(mc_mean_r_list), _t(mc_mean_c_list)
    target_r_t, target_c_t = _t(target_r_list), _t(target_c_list)

    def _corr(a, b):
        if a.numel() < 2 or a.std() <= 0 or b.std() <= 0:
            return float('nan')
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    stats = {
        'EstimationError_r': (mc_mean_r_t - pred_r_t).mean().item(),
        'EstimationError_c': (mc_mean_c_t - pred_c_t).mean().item(),
        'Correlation_r': _corr(pred_r_t, mc_mean_r_t),
        'Correlation_c': _corr(pred_c_t, mc_mean_c_t),
        'EstimationError_target_true_r': (mc_mean_r_t - target_r_t).mean().item(),
        'EstimationError_target_true_c': (mc_mean_c_t - target_c_t).mean().item(),
        'Correlation_target_true_r': _corr(target_r_t, mc_mean_r_t),
        'Correlation_target_true_c': _corr(target_c_t, mc_mean_c_t),
        'Correlation_pred_target_r': _corr(pred_r_t, target_r_t),
        'Correlation_pred_target_c': _corr(pred_c_t, target_c_t),
        'MeanTarget_r': target_r_t.mean().item(),
        'MeanTarget_c': target_c_t.mean().item(),
        'MeanVar_r': float(np.mean(mc_var_r_list)),
        'MeanVar_c': float(np.mean(mc_var_c_list)),
        'MeanTrue_r': mc_mean_r_t.mean().item(),
        'MeanTrue_c': mc_mean_c_t.mean().item(),
        'MeanPred_r': pred_r_t.mean().item(),
        'MeanPred_c': pred_c_t.mean().item(),
        'NumProbes': float(len(snapshots)),
        'MCRepeats': float(mc_repeats),
    }
    del epoch  # accepted for call-site symmetry; not used here
    if not return_raw:
        return stats
    raw = {
        'r': {'pred': pred_r_list, 'mc_mean': mc_mean_r_list, 'mc_var': mc_var_r_list, 'target': target_r_list},
        'c': {'pred': pred_c_list, 'mc_mean': mc_mean_c_list, 'mc_var': mc_var_c_list, 'target': target_c_list},
    }
    return stats, raw


def pool_correlation_stats(raw_list: list[dict], prefix: str = '') -> tuple[dict, dict]:
    """Pool several ``return_raw=True`` dicts into one correlation, as if every probe from every
    source (e.g. s0 plus every on-policy intermediate position) had been scored together.

    The per-category ``Correlation_r``/``Correlation_c`` that ``estimate_true_value_same_state_mc``
    and ``estimate_value_from_snapshots`` report are each computed over only their own narrow slice
    of states (just resets, or just one fixed within-episode position) -- useful for spotting
    *where* the critic is worse, but neither one answers "how accurate is the critic across the
    actual diversity of states we evaluate on". This does: it concatenates the raw
    predicted/MC-true pairs across every source first, then computes one correlation over the
    pooled set, which is not the same as averaging the per-category correlations (pooling can
    surface a relationship -- or wash one out -- that no individual narrow slice shows on its own).

    Args:
        raw_list: The ``raw`` dicts to pool (each has ``'r'``/``'c'`` keys, each with
            ``'pred'``/``'mc_mean'`` lists of equal length, as returned by
            ``estimate_true_value_same_state_mc``/``estimate_value_from_snapshots`` with
            ``return_raw=True``).
        prefix: Optional key prefix for the returned stats dict (e.g. ``'PooledMC/'``).

    Returns:
        ``(stats, raw)`` -- ``stats`` has the same key shape as the per-category stats dicts
        (``Correlation_r/c``, ``EstimationError_r/c``, ``MeanTrue_r/c``, ``MeanPred_r/c``,
        ``NumProbes``), and ``raw`` is the merged ``{'r': {...}, 'c': {...}}`` dict (no
        ``mc_var``/``probe_seeds`` -- pooling those across heterogeneous sources isn't meaningful),
        suitable for passing straight into ``eval_data_dump.save_scatter_grid`` as one more series.
    """

    def _t(lst):
        return torch.tensor(lst, dtype=torch.float32)

    def _corr(a, b):
        if a.numel() < 2 or a.std() <= 0 or b.std() <= 0:
            return float('nan')
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    stats: dict = {}
    raw: dict = {}
    num_probes = None
    for stream in ('r', 'c'):
        pred_list: list[float] = []
        mc_mean_list: list[float] = []
        target_list: list[float] = []
        has_target = all('target' in src[stream] for src in raw_list)
        for src in raw_list:
            pred_list.extend(src[stream]['pred'])
            mc_mean_list.extend(src[stream]['mc_mean'])
            if has_target:
                target_list.extend(src[stream]['target'])
        pred_t, mc_mean_t = _t(pred_list), _t(mc_mean_list)
        stats[f'{prefix}EstimationError_{stream}'] = (mc_mean_t - pred_t).mean().item()
        stats[f'{prefix}Correlation_{stream}'] = _corr(pred_t, mc_mean_t)
        stats[f'{prefix}MeanTrue_{stream}'] = mc_mean_t.mean().item()
        stats[f'{prefix}MeanPred_{stream}'] = pred_t.mean().item()
        raw[stream] = {'pred': pred_list, 'mc_mean': mc_mean_list}
        if has_target:
            target_t = _t(target_list)
            stats[f'{prefix}EstimationError_target_true_{stream}'] = (mc_mean_t - target_t).mean().item()
            stats[f'{prefix}Correlation_target_true_{stream}'] = _corr(target_t, mc_mean_t)
            stats[f'{prefix}Correlation_pred_target_{stream}'] = _corr(pred_t, target_t)
            stats[f'{prefix}MeanTarget_{stream}'] = target_t.mean().item()
            raw[stream]['target'] = target_list
        num_probes = len(pred_list)
    stats[f'{prefix}NumProbes'] = float(num_probes or 0)
    return stats, raw
