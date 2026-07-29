"""Utility for evaluating true vs. estimated value functions.

Two estimators of the "true" :math:`V^\\pi(s)` are available:

``single-trajectory`` (``value_eval_mc_samples == 1``)
    The cheap, historical behaviour: roll the policy out once and compare
    :math:`V(s_t)` against the discounted cost-to-go :math:`G_t` of *that one*
    trajectory. :math:`G_t` is an unbiased but very high-variance sample of
    :math:`V^\\pi(s_t)`, so per-state scatter is dominated by return noise
    rather than by critic error.

``multi-sample Monte Carlo`` (``value_eval_mc_samples > 1``)
    Snapshot the full simulator state at the query state, then restore it and
    re-run the policy ``K`` times. Averaging the ``K`` returns cuts the
    sampling noise by :math:`\\sqrt{K}` and gives a genuine picture of
    :math:`V^\\pi(s)`. Requires an environment implementing
    ``snapshot_state`` / ``restore_state`` (see
    :mod:`omnisafe.envs.safety_gymnasium_state`).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import numpy as np
import torch
from rich.progress import Progress


@contextmanager
def _frozen_normalizers(env: Any) -> Generator[None, None, None]:
    """Freeze the running observation/reward/cost statistics for the duration of the block.

    Evaluation rollouts are not training data. Letting them push into the
    normalizers would shift the very statistics the critic was fitted against —
    a mild problem for a single-trajectory pass, a serious one once Monte-Carlo
    sampling multiplies the number of evaluation steps by ``K``.

    Args:
        env (Any): The wrapped environment whose normalizers should be frozen.

    Yields:
        None: With every normalizer in the wrapper chain switched to eval mode.
    """
    normalizers = []
    current = env
    while current is not None:
        for attr in ('_obs_normalizer', '_reward_normalizer', '_cost_normalizer'):
            norm = current.__dict__.get(attr)
            if norm is not None:
                normalizers.append((norm, norm.training))
        current = current.__dict__.get('_env')

    for norm, _ in normalizers:
        norm.eval()
    try:
        yield
    finally:
        for norm, was_training in normalizers:
            norm.train(was_training)


def _flat(tensor: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Reshape a per-environment step output to a flat ``(num_envs,)`` tensor.

    Args:
        tensor (torch.Tensor): A reward/cost/done tensor from ``env.step``.
        num_envs (int): Number of parallel environments.

    Returns:
        The tensor viewed as ``(num_envs,)``.
    """
    return tensor.reshape(num_envs)


def _collect_single_trajectory(  # pylint: disable=too-many-locals
    agent,
    env,
    discount_r: float,
    discount_c: float,
    eval_episodes: int,
) -> dict[str, list[float]]:
    """Compare V(s) against the one-sample discounted cost-to-go of an on-policy rollout.

    Args:
        agent: The actor-critic whose critics are being evaluated.
        env: The wrapped (already normalized) training environment.
        discount_r (float): Reward discount factor.
        discount_c (float): Cost discount factor.
        eval_episodes (int): Number of episodes to roll out.

    Returns:
        Per-state ``true``/``est`` values for the ``s0`` and ``all`` regimes.
    """
    num_envs = env.num_envs

    obs, _ = env.reset()
    act, cur_est_r, cur_est_c, _ = agent.step(obs)

    # Per-env episode history: each entry is (est_r, est_c, reward, cost) at step t
    ep_history: list[list[tuple[float, float, float, float]]] = [[] for _ in range(num_envs)]
    out: dict[str, list[float]] = {
        key: [] for key in ('s0_true_r', 's0_true_c', 's0_est_r', 's0_est_c')
    }
    out.update({key: [] for key in ('all_true_r', 'all_true_c', 'all_est_r', 'all_est_c')})

    episodes_done = 0

    with Progress() as progress:
        task = progress.add_task('Evaluating value function...', total=eval_episodes)
        while episodes_done < eval_episodes:
            next_obs, r, c, terminated, truncated, _ = env.step(act)

            r_flat = _flat(r, num_envs)
            c_flat = _flat(c, num_envs)

            # Record V(s_t), V_c(s_t), r_t, c_t BEFORE moving to next state
            for i in range(num_envs):
                ep_history[i].append(
                    (
                        cur_est_r[i].item(),
                        cur_est_c[i].item(),
                        r_flat[i].item(),
                        c_flat[i].item(),
                    ),
                )

            done = _flat(terminated.bool() | truncated.bool(), num_envs)

            newly_done = 0
            for i in done.nonzero(as_tuple=False).flatten().tolist():
                if episodes_done < eval_episodes:
                    hist = ep_history[i]
                    horizon = len(hist)

                    # Backward scan: G_t = r_t + gamma * G_{t+1}
                    ret_r, ret_c = 0.0, 0.0
                    step_true_r = [0.0] * horizon
                    step_true_c = [0.0] * horizon
                    for t in range(horizon - 1, -1, -1):
                        ret_r = hist[t][2] + discount_r * ret_r
                        ret_c = hist[t][3] + discount_c * ret_c
                        step_true_r[t] = ret_r
                        step_true_c[t] = ret_c

                    out['s0_true_r'].append(step_true_r[0])
                    out['s0_true_c'].append(step_true_c[0])
                    out['s0_est_r'].append(hist[0][0])
                    out['s0_est_c'].append(hist[0][1])

                    for t in range(horizon):
                        out['all_true_r'].append(step_true_r[t])
                        out['all_true_c'].append(step_true_c[t])
                        out['all_est_r'].append(hist[t][0])
                        out['all_est_c'].append(hist[t][1])

                    episodes_done += 1
                    newly_done += 1

                ep_history[i] = []

            progress.update(task, advance=newly_done)

            obs = next_obs
            act, cur_est_r, cur_est_c, _ = agent.step(obs)

    return out


def _collect_query_states(  # pylint: disable=too-many-locals
    agent,
    env,
    num_episodes: int,
    states_per_episode: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Roll the policy out and snapshot a sample of the states it visits.

    Every episode contributes its initial state (needed for the ``Eval_s0``
    regime) plus up to ``states_per_episode`` interior states drawn uniformly
    without replacement via reservoir sampling — episode length is not known in
    advance, and snapshotting every step would be wasteful.

    Args:
        agent: The actor-critic whose critics are being evaluated.
        env: The wrapped (already normalized) training environment.
        num_episodes (int): Number of episodes to draw query states from.
        states_per_episode (int): Interior states to keep per episode.
        rng (np.random.Generator): Source of randomness for reservoir sampling.

    Returns:
        One dict per query state with its snapshot, critic estimates and ``is_s0`` flag.
    """
    num_envs = env.num_envs

    obs, _ = env.reset()
    act, est_r, est_c, _ = agent.step(obs)

    queries: list[dict[str, Any]] = []
    reservoirs: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
    interior_seen = [0] * num_envs
    step_idx = [0] * num_envs

    episodes_started = 0
    episode_id = []
    for _ in range(num_envs):
        episode_id.append(episodes_started)
        episodes_started += 1

    def _query(env_idx: int, is_s0: bool) -> dict[str, Any]:
        return {
            'snapshot': env.snapshot_state(env_idx),
            'est_r': est_r[env_idx].item(),
            'est_c': est_c[env_idx].item(),
            'is_s0': is_s0,
        }

    with Progress() as progress:
        task = progress.add_task('Snapshotting query states...', total=num_episodes)
        while any(eid < num_episodes for eid in episode_id):
            for i in range(num_envs):
                if episode_id[i] >= num_episodes:
                    continue
                if step_idx[i] == 0:
                    queries.append(_query(i, is_s0=True))
                elif states_per_episode > 0:
                    # Reservoir sampling over the interior states of this episode.
                    interior_seen[i] += 1
                    if len(reservoirs[i]) < states_per_episode:
                        reservoirs[i].append(_query(i, is_s0=False))
                    else:
                        slot = int(rng.integers(interior_seen[i]))
                        if slot < states_per_episode:
                            reservoirs[i][slot] = _query(i, is_s0=False)

            next_obs, _, _, terminated, truncated, _ = env.step(act)
            done = _flat(terminated.bool() | truncated.bool(), num_envs)

            for i in range(num_envs):
                step_idx[i] += 1
                if not done[i]:
                    continue
                if episode_id[i] < num_episodes:
                    queries.extend(reservoirs[i])
                    progress.update(task, advance=1)
                reservoirs[i] = []
                interior_seen[i] = 0
                step_idx[i] = 0
                episode_id[i] = episodes_started
                episodes_started += 1

            obs = next_obs
            act, est_r, est_c, _ = agent.step(obs)

    return queries


def _rollout_monte_carlo(  # pylint: disable=too-many-locals,too-many-statements
    agent,
    env,
    queries: list[dict[str, Any]],
    discount_r: float,
    discount_c: float,
    mc_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Re-run the policy ``mc_samples`` times from each query state.

    All parallel environments are kept busy by treating them as worker slots: a
    slot is loaded with one ``(query, sample)`` job, stepped until its episode
    ends, then immediately reloaded with the next job.

    Auto-reset is switched off for the duration: rebuilding a Safety-Gymnasium
    world recompiles its MuJoCo model (~600 ms, versus ~0.2 ms per step), and
    every reset here would be thrown away by the restore that follows it.

    Args:
        agent: The actor-critic whose critics are being evaluated.
        env: The wrapped (already normalized) training environment.
        queries (list[dict[str, Any]]): Query states from :func:`_collect_query_states`.
        discount_r (float): Reward discount factor.
        discount_c (float): Cost discount factor.
        mc_samples (int): Rollouts per query state.
        rng (np.random.Generator): Source of per-rollout environment seeds.

    Returns:
        returns_r: ``(num_queries, mc_samples)`` discounted reward returns.
        returns_c: ``(num_queries, mc_samples)`` discounted cost returns.
        env_steps: Total number of environment steps consumed.
    """
    num_envs = env.num_envs
    num_queries = len(queries)

    jobs = [(q, s) for q in range(num_queries) for s in range(mc_samples)]
    returns_r = np.zeros((num_queries, mc_samples), dtype=np.float64)
    returns_c = np.zeros((num_queries, mc_samples), dtype=np.float64)

    slot_job: list[tuple[int, int] | None] = [None] * num_envs
    ret_r = np.zeros(num_envs)
    ret_c = np.zeros(num_envs)
    disc_r = np.ones(num_envs)
    disc_c = np.ones(num_envs)

    obs, _ = env.reset()
    obs = obs.clone()
    next_job = 0
    env_steps = 0

    def _load(env_idx: int) -> None:
        """Restore the next job's state into a slot, keeping the slot steppable."""
        nonlocal next_job
        if next_job < len(jobs):
            slot_job[env_idx] = jobs[next_job]
            next_job += 1
            snapshot = queries[slot_job[env_idx][0]]['snapshot']
        else:
            # No work left, but with auto-reset off a finished env cannot be
            # stepped — park the slot on an arbitrary state and discard its rollout.
            slot_job[env_idx] = None
            snapshot = queries[0]['snapshot']
        restored = env.restore_state(
            snapshot,
            env_idx,
            rng_seed=int(rng.integers(2**31 - 1)),
        )
        obs[env_idx] = restored.reshape(-1)
        ret_r[env_idx] = 0.0
        ret_c[env_idx] = 0.0
        disc_r[env_idx] = 1.0
        disc_c[env_idx] = 1.0

    env.set_auto_reset(False)
    try:
        for i in range(num_envs):
            _load(i)

        with Progress() as progress:
            task = progress.add_task('Monte-Carlo value rollouts...', total=len(jobs))
            while any(job is not None for job in slot_job):
                act, _, _, _ = agent.step(obs)
                next_obs, r, c, terminated, truncated, _ = env.step(act)
                env_steps += num_envs

                r_flat = _flat(r, num_envs).cpu().numpy()
                c_flat = _flat(c, num_envs).cpu().numpy()
                done = _flat(terminated.bool() | truncated.bool(), num_envs).cpu().numpy()

                active = np.array([job is not None for job in slot_job])
                ret_r[active] += disc_r[active] * r_flat[active]
                ret_c[active] += disc_c[active] * c_flat[active]
                disc_r[active] *= discount_r
                disc_c[active] *= discount_c

                obs = next_obs.clone()

                finished = 0
                for i in range(num_envs):
                    if not done[i]:
                        continue
                    if slot_job[i] is not None:
                        query_idx, sample_idx = slot_job[i]
                        returns_r[query_idx, sample_idx] = ret_r[i]
                        returns_c[query_idx, sample_idx] = ret_c[i]
                        finished += 1
                    _load(i)

                progress.update(task, advance=finished)
    finally:
        env.set_auto_reset(True)

    return returns_r, returns_c, env_steps


def _collect_monte_carlo(
    agent,
    env,
    discount_r: float,
    discount_c: float,
    num_episodes: int,
    states_per_episode: int,
    mc_samples: int,
    seed: int,
) -> dict[str, list[float]]:
    """Compare V(s) against a ``mc_samples``-sample Monte-Carlo estimate of V^pi(s).

    Args:
        agent: The actor-critic whose critics are being evaluated.
        env: The wrapped (already normalized) training environment.
        discount_r (float): Reward discount factor.
        discount_c (float): Cost discount factor.
        num_episodes (int): Number of episodes to draw query states from.
        states_per_episode (int): Interior query states per episode.
        mc_samples (int): Rollouts per query state.
        seed (int): Seed for query sampling and per-rollout environment seeds.

    Returns:
        Per-state ``true``/``est`` values plus the per-state Monte-Carlo standard
        error, for the ``s0`` and ``all`` regimes.
    """
    rng = np.random.default_rng(seed)

    queries = _collect_query_states(agent, env, num_episodes, states_per_episode, rng)
    returns_r, returns_c, env_steps = _rollout_monte_carlo(
        agent,
        env,
        queries,
        discount_r,
        discount_c,
        mc_samples,
        rng,
    )

    mean_r = returns_r.mean(axis=1)
    mean_c = returns_c.mean(axis=1)
    # Standard error of the mean: how much of the residual is sampling noise
    # rather than critic error.
    sem_r = returns_r.std(axis=1, ddof=1) / np.sqrt(mc_samples)
    sem_c = returns_c.std(axis=1, ddof=1) / np.sqrt(mc_samples)

    out: dict[str, list[float]] = {
        key: []
        for key in (
            's0_true_r',
            's0_true_c',
            's0_est_r',
            's0_est_c',
            's0_sem_r',
            's0_sem_c',
            'all_true_r',
            'all_true_c',
            'all_est_r',
            'all_est_c',
            'all_sem_r',
            'all_sem_c',
        )
    }
    out['env_steps'] = [float(env_steps)]

    for idx, query in enumerate(queries):
        for prefix in (['all'] + (['s0'] if query['is_s0'] else [])):
            out[f'{prefix}_true_r'].append(float(mean_r[idx]))
            out[f'{prefix}_true_c'].append(float(mean_c[idx]))
            out[f'{prefix}_est_r'].append(query['est_r'])
            out[f'{prefix}_est_c'].append(query['est_c'])
            out[f'{prefix}_sem_r'].append(float(sem_r[idx]))
            out[f'{prefix}_sem_c'].append(float(sem_c[idx]))

    return out


def estimate_true_value(  # pylint: disable=too-many-locals,too-many-statements
    agent,
    env,
    cfgs,
    discount_r,
    discount_c,
    eval_episodes=100,
    epoch=None,
):
    """Estimate true V(s) vs. critic estimate by rolling out full episodes.

    Uses the provided (already-wrapped) environment directly so that observation
    and reward normalizers are identical to those seen during training; the
    normalizers' running statistics are frozen for the duration of the
    evaluation so that these extra steps do not leak into training.

    Two evaluation regimes:
    - Eval_s0: initial state of each episode — V(s_0) vs. G_0.
    - Eval_all: every visited (or, under Monte Carlo, every sampled) state —
      V(s_t) vs. G_t.

    Set ``algo_cfgs.value_eval_mc_samples > 1`` to replace the one-sample
    cost-to-go with a multi-sample Monte-Carlo estimate. The environment is
    snapshotted at each query state and the policy re-run from it, which costs
    roughly ``value_eval_mc_episodes * (1 + value_eval_states_per_episode) *
    value_eval_mc_samples`` episodes' worth of environment steps.

    Args:
        agent: The actor-critic whose critics are being evaluated.
        env: The wrapped (already normalized) training environment.
        cfgs: The run configuration.
        discount_r (float): Reward discount factor.
        discount_c (float): Cost discount factor.
        eval_episodes (int, optional): Episodes for the single-trajectory
            estimator. Defaults to 100.
        epoch (int, optional): Epoch index, used as the wandb step. Defaults to None.

    Returns:
        Estimation error, true mean, estimated mean and correlation for cost and
        reward, in the ``s0`` and then the ``all`` regime.
    """
    device = torch.device(cfgs.train_cfgs.device)

    mc_samples = int(getattr(cfgs.algo_cfgs, 'value_eval_mc_samples', 1))
    mc_episodes = int(getattr(cfgs.algo_cfgs, 'value_eval_mc_episodes', 10))
    states_per_episode = int(getattr(cfgs.algo_cfgs, 'value_eval_states_per_episode', 5))
    seed = int(getattr(cfgs, 'seed', 0)) + (epoch or 0)

    use_mc = mc_samples > 1
    if use_mc and not env.supports_state_restore:
        print(
            'WARNING: value_eval_mc_samples > 1 but the environment cannot restore its '
            'state; falling back to the single-trajectory estimator.',
        )
        use_mc = False

    with _frozen_normalizers(env):
        if use_mc:
            data = _collect_monte_carlo(
                agent,
                env,
                discount_r,
                discount_c,
                num_episodes=mc_episodes,
                states_per_episode=states_per_episode,
                mc_samples=mc_samples,
                seed=seed,
            )
        else:
            data = _collect_single_trajectory(agent, env, discount_r, discount_c, eval_episodes)

    def _to_tensor(lst):
        return torch.tensor(lst, device=device, dtype=torch.float32)

    s0_true_r_t = _to_tensor(data['s0_true_r'])
    s0_true_c_t = _to_tensor(data['s0_true_c'])
    s0_est_r_t = _to_tensor(data['s0_est_r'])
    s0_est_c_t = _to_tensor(data['s0_est_c'])

    all_true_r_t = _to_tensor(data['all_true_r'])
    all_true_c_t = _to_tensor(data['all_true_c'])
    all_est_r_t = _to_tensor(data['all_est_r'])
    all_est_c_t = _to_tensor(data['all_est_c'])

    def _stats(true_t, est_t):
        error = torch.mean(true_t - est_t)
        true_m = torch.mean(true_t)
        est_m = torch.mean(est_t)
        corr = torch.corrcoef(torch.stack([true_t, est_t]))[0, 1]
        return error, true_m, est_m, corr

    s0_c_error, s0_true_c_m, s0_est_c_m, s0_corr_c = _stats(s0_true_c_t, s0_est_c_t)
    s0_r_error, s0_true_r_m, s0_est_r_m, s0_corr_r = _stats(s0_true_r_t, s0_est_r_t)
    all_c_error, all_true_c_m, all_est_c_m, all_corr_c = _stats(all_true_c_t, all_est_c_t)
    all_r_error, all_true_r_m, all_est_r_m, all_corr_r = _stats(all_true_r_t, all_est_r_t)

    if cfgs.logger_cfgs.use_wandb:
        import matplotlib.pyplot as plt
        import wandb

        def _scatter_fig(true_vals, est_vals, label, color, title, yerr=None):
            fig, ax = plt.subplots(figsize=(6, 5))
            if yerr is not None:
                ax.errorbar(
                    true_vals,
                    est_vals,
                    xerr=yerr,
                    fmt='none',
                    ecolor='gray',
                    alpha=0.25,
                    elinewidth=0.6,
                )
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

        s0_c_np = s0_true_c_t.cpu().numpy()
        s0_ec_np = s0_est_c_t.cpu().numpy()
        s0_r_np = s0_true_r_t.cpu().numpy()
        s0_er_np = s0_est_r_t.cpu().numpy()
        all_c_np = all_true_c_t.cpu().numpy()
        all_ec_np = all_est_c_t.cpu().numpy()
        all_r_np = all_true_r_t.cpu().numpy()
        all_er_np = all_est_r_t.cpu().numpy()

        suffix = f' [MC x{mc_samples}]' if use_mc else ''
        fig_s0_c = _scatter_fig(
            s0_c_np,
            s0_ec_np,
            'C',
            'steelblue',
            f'C-Values: True vs Estimated (s0){suffix}',
            np.asarray(data['s0_sem_c']) if use_mc else None,
        )
        fig_s0_r = _scatter_fig(
            s0_r_np,
            s0_er_np,
            'R',
            'darkorange',
            f'R-Values: True vs Estimated (s0){suffix}',
            np.asarray(data['s0_sem_r']) if use_mc else None,
        )
        fig_all_c = _scatter_fig(
            all_c_np,
            all_ec_np,
            'C',
            'steelblue',
            f'C-Values: True vs Estimated (all states){suffix}',
            np.asarray(data['all_sem_c']) if use_mc else None,
        )
        fig_all_r = _scatter_fig(
            all_r_np,
            all_er_np,
            'R',
            'darkorange',
            f'R-Values: True vs Estimated (all states){suffix}',
            np.asarray(data['all_sem_r']) if use_mc else None,
        )

        log_dict = {
            # Scatter plots
            'scatter/s0_c_values': wandb.Image(fig_s0_c),
            'scatter/s0_r_values': wandb.Image(fig_s0_r),
            'scatter/all_c_values': wandb.Image(fig_all_c),
            'scatter/all_r_values': wandb.Image(fig_all_r),
            # Eval_s0 stats
            'Eval_s0/Correlation_c': s0_corr_c.item(),
            'Eval_s0/Correlation_r': s0_corr_r.item(),
            'Eval_s0/EstimationError_c': s0_c_error.item(),
            'Eval_s0/true_value_c': s0_true_c_m.item(),
            'Eval_s0/estimate_value_c': s0_est_c_m.item(),
            'Eval_s0/EstimationError_r': s0_r_error.item(),
            'Eval_s0/true_value_r': s0_true_r_m.item(),
            'Eval_s0/estimate_value_r': s0_est_r_m.item(),
            # Eval_all stats
            'Eval_all/Correlation_c': all_corr_c.item(),
            'Eval_all/Correlation_r': all_corr_r.item(),
            'Eval_all/EstimationError_c': all_c_error.item(),
            'Eval_all/true_value_c': all_true_c_m.item(),
            'Eval_all/estimate_value_c': all_est_c_m.item(),
            'Eval_all/EstimationError_r': all_r_error.item(),
            'Eval_all/true_value_r': all_true_r_m.item(),
            'Eval_all/estimate_value_r': all_est_r_m.item(),
        }
        if use_mc:
            log_dict.update(
                {
                    'Eval_mc/samples_per_state': mc_samples,
                    'Eval_mc/num_query_states': len(data['all_true_r']),
                    'Eval_mc/env_steps': data['env_steps'][0],
                    'Eval_mc/s0_sem_c': float(np.mean(data['s0_sem_c'])),
                    'Eval_mc/s0_sem_r': float(np.mean(data['s0_sem_r'])),
                    'Eval_mc/all_sem_c': float(np.mean(data['all_sem_c'])),
                    'Eval_mc/all_sem_r': float(np.mean(data['all_sem_r'])),
                },
            )

        wandb.log(log_dict, step=epoch)

        plt.close(fig_s0_c)
        plt.close(fig_s0_r)
        plt.close(fig_all_c)
        plt.close(fig_all_r)

    return (
        s0_c_error,
        s0_true_c_m,
        s0_est_c_m,
        s0_corr_c,
        s0_r_error,
        s0_true_r_m,
        s0_est_r_m,
        s0_corr_r,
        all_c_error,
        all_true_c_m,
        all_est_c_m,
        all_corr_c,
        all_r_error,
        all_true_r_m,
        all_est_r_m,
        all_corr_r,
    )
