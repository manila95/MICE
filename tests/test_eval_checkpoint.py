"""Out-of-process evaluation must score the same thing the in-process path scores.

Run either way -- ``pytest tests/test_eval_checkpoint.py`` or
``python tests/test_eval_checkpoint.py``.

The failure this guards against is silent. If ``load_agent_from_checkpoint`` rebuilds the agent
even slightly wrong -- a critic left at its random initialisation, the observation normalizer not
applied, a hidden width read from the wrong place -- the value studies still return perfectly
plausible correlations and estimation errors. They are just measuring a different agent than the
one that trained, and nothing downstream can tell.

So these tests pin the two halves of the contract separately:

* structural -- every parameter of the rebuilt actor/critics equals the saved one, and the
  wrapper stack around the eval envs matches what ``_get_mc_value_study_env`` builds;
* behavioural -- with the RNG seeded identically, the *same study function* run on the live agent
  and on the rebuilt agent returns bit-identical statistics.

The behavioural test is the one that would catch a mistake the structural one misses, so it
deliberately does not mock the study: it calls ``estimate_true_value_same_state_mc`` for real,
with a deliberately tiny probe/repeat budget to stay fast.
"""

from __future__ import annotations

import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get('MUJOCO_GL', 'egl') is None,
    reason='needs a working safety_gymnasium',
)

ENV_ID = 'SafetyPointGoal1-v0'


def _cfgs():
    """A resolved config, exactly as a real run would carry it (CPO defaults + overrides)."""
    from omnisafe.utils.config import get_default_kwargs_yaml  # noqa: PLC0415

    cfgs = get_default_kwargs_yaml('CPO', ENV_ID, 'on-policy')
    cfgs.recurisve_update(
        {
            'env_id': ENV_ID,
            'train_cfgs': {'device': 'cpu', 'epochs': 2, 'vector_env_nums': 1},
            'algo_cfgs': {
                'obs_normalize': True,
                'mc_value_study_vector_envs': 2,
                'intermediate_state_study_probes': 2,
            },
            'model_cfgs': {'critic': {'hidden_sizes': [32, 32]}, 'actor': {'hidden_sizes': [32, 32]}},
        },
    )
    cfgs.env_cfgs = type(cfgs.train_cfgs)()  # empty Config: no extra env kwargs
    return cfgs


def _build_live_agent(cfgs):
    """An actor-critic with *non-default* weights, so a failure to load them is visible.

    A freshly constructed agent would compare equal to a freshly constructed agent even if
    ``load_state_dict`` were never called, which is exactly the bug this file exists to catch.
    """
    from omnisafe.envs.core import make as make_env  # noqa: PLC0415
    from omnisafe.models.actor_critic.constraint_actor_critic import (  # noqa: PLC0415
        ConstraintActorCritic,
    )

    env = make_env(ENV_ID, num_envs=1, device='cpu')
    agent = ConstraintActorCritic(
        obs_space=env.observation_space, act_space=env.action_space,
        model_cfgs=cfgs.model_cfgs, epochs=cfgs.train_cfgs.epochs,
    )
    env.close()
    torch.manual_seed(1234)
    with torch.no_grad():
        for p in agent.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    agent.eval()
    return agent


def _save_checkpoint(tmp_path, cfgs, agent, epoch: int, with_critics: bool = True):
    """Write a checkpoint in exactly the layout ``Logger.torch_save`` produces."""
    import json  # noqa: PLC0415

    run_dir = str(tmp_path)
    os.makedirs(os.path.join(run_dir, 'torch_save'), exist_ok=True)
    params = {'pi': agent.actor.state_dict()}
    if with_critics:
        params['reward_critic'] = agent.reward_critic.state_dict()
        params['cost_critic'] = agent.cost_critic.state_dict()
    torch.save(params, os.path.join(run_dir, 'torch_save', f'epoch-{epoch}.pt'))
    with open(os.path.join(run_dir, 'config.json'), encoding='utf-8', mode='w') as f:
        json.dump(cfgs.todict(), f)
    return run_dir


def test_rebuilt_agent_is_parameter_identical(tmp_path) -> None:
    """Structural: every parameter round-trips through the checkpoint unchanged."""
    from omnisafe.utils.eval_checkpoint import load_agent_from_checkpoint  # noqa: PLC0415

    cfgs = _cfgs()
    live = _build_live_agent(cfgs)
    run_dir = _save_checkpoint(tmp_path, cfgs, live, epoch=1)

    rebuilt, _ = load_agent_from_checkpoint(run_dir, 1, cfgs=cfgs)

    for name, module in (('actor', 'actor'), ('reward', 'reward_critic'), ('cost', 'cost_critic')):
        a = dict(getattr(live, module).named_parameters())
        b = dict(getattr(rebuilt, module).named_parameters())
        assert a.keys() == b.keys(), f'{name}: parameter names differ'
        for k in a:
            assert torch.equal(a[k], b[k]), f'{name}.{k} differs after reload'


def test_rebuilt_agent_is_in_eval_mode(tmp_path) -> None:
    """A critic left in train() mode with dropout on returns a different value each call."""
    from omnisafe.utils.eval_checkpoint import load_agent_from_checkpoint  # noqa: PLC0415

    cfgs = _cfgs()
    run_dir = _save_checkpoint(tmp_path, cfgs, _build_live_agent(cfgs), epoch=1)
    rebuilt, _ = load_agent_from_checkpoint(run_dir, 1, cfgs=cfgs)
    assert not rebuilt.training
    assert not rebuilt.reward_critic.training


def test_checkpoint_without_critics_raises(tmp_path) -> None:
    """Refusing beats silently scoring a randomly initialised critic as this epoch's calibration."""
    from omnisafe.utils.eval_checkpoint import load_agent_from_checkpoint  # noqa: PLC0415

    cfgs = _cfgs()
    run_dir = _save_checkpoint(tmp_path, cfgs, _build_live_agent(cfgs), epoch=1, with_critics=False)
    with pytest.raises(KeyError, match='reward_critic'):
        load_agent_from_checkpoint(run_dir, 1, cfgs=cfgs)


def test_checkpoint_epochs_listing(tmp_path) -> None:
    """Epochs come back ascending and numerically, not lexically (100 before 95 would be wrong)."""
    from omnisafe.utils.eval_checkpoint import checkpoint_epochs  # noqa: PLC0415

    cfgs = _cfgs()
    agent = _build_live_agent(cfgs)
    for e in (1, 5, 95, 100, 450):
        run_dir = _save_checkpoint(tmp_path, cfgs, agent, epoch=e)
    assert checkpoint_epochs(run_dir) == [1, 5, 95, 100, 450]


def test_eval_env_wrapper_stack_matches_training(tmp_path) -> None:
    """The evaluator's envs must be wrapped exactly as _get_mc_value_study_env wraps them.

    ``TimeLimit`` and ``AutoReset`` are applied *conditionally*, on the env's own
    ``need_*_wrapper`` flags, and :class:`SafetyGymnasiumEnv` only sets them in its
    ``num_envs == 1`` branch -- a vectorized env carries its own time limit. So at
    ``mc_value_study_vector_envs > 1`` the correct stack has no ``TimeLimit``, and asserting it
    unconditionally would pin the wrong behaviour. ``Unsqueeze`` likewise appears only at
    ``n_envs == 1``. What must hold in every case is that the evaluator's recipe agrees with
    training's on these same flags.
    """
    from omnisafe.envs.core import make as make_env  # noqa: PLC0415
    from omnisafe.envs.wrapper import ActionScale, ObsNormalize, TimeLimit, Unsqueeze  # noqa: PLC0415
    from omnisafe.utils.eval_checkpoint import build_eval_envs  # noqa: PLC0415

    cfgs = _cfgs()
    n_envs = int(cfgs.algo_cfgs.mc_value_study_vector_envs)
    raw = make_env(ENV_ID, num_envs=n_envs, device='cpu')
    needs_time_limit = raw.need_time_limit_wrapper
    raw.close()

    envs = build_eval_envs(cfgs, which='mc')
    try:
        stack, node = [], envs['mc_env']
        while node is not None:
            stack.append(type(node))
            node = getattr(node, '_env', None)
        assert ActionScale in stack, f'ActionScale missing: {stack}'
        assert ObsNormalize in stack, f'ObsNormalize missing (obs_normalize=True): {stack}'
        assert (TimeLimit in stack) == needs_time_limit, (
            f'TimeLimit presence ({TimeLimit in stack}) disagrees with the env\'s own '
            f'need_time_limit_wrapper ({needs_time_limit}): {stack}'
        )
        assert (Unsqueeze in stack) == (n_envs == 1), f'Unsqueeze at n_envs={n_envs}: {stack}'
        assert envs['max_episode_steps'] == 1000
    finally:
        envs['mc_env'].close()


def test_normalizer_state_is_applied(tmp_path) -> None:
    """apply_obs_normalizer must actually reach the ObsNormalize buried in the wrapper stack."""
    from omnisafe.utils.eval_checkpoint import apply_obs_normalizer, build_eval_envs  # noqa: PLC0415
    from omnisafe.envs.wrapper import ObsNormalize  # noqa: PLC0415

    cfgs = _cfgs()
    envs = build_eval_envs(cfgs, which='mc')
    try:
        node = envs['mc_env']
        while not isinstance(node, ObsNormalize):
            node = node._env
        sd = node._obs_normalizer.state_dict()
        marker = {k: torch.full_like(v, 3.5) if v.dtype.is_floating_point else v
                  for k, v in sd.items()}
        apply_obs_normalizer(envs['mc_env'], marker)
        after = node._obs_normalizer.state_dict()
        assert any(torch.allclose(after[k], torch.full_like(after[k], 3.5))
                   for k in after if after[k].dtype.is_floating_point), 'normalizer not applied'
    finally:
        envs['mc_env'].close()


@pytest.mark.slow
def test_study_gives_identical_numbers_for_live_and_rebuilt_agent(tmp_path) -> None:
    """Behavioural: same study, same seed, live agent vs checkpoint-rebuilt agent -> same stats.

    This is the test that would catch a reconstruction bug the parameter comparison misses (a
    normalizer applied to one path and not the other, say). The policy is stochastic, so the
    torch RNG is reseeded immediately before each call -- with the probe layouts already fixed by
    ``probe_seeds``, that makes the whole rollout deterministic and the two runs directly
    comparable rather than merely close.
    """
    from omnisafe.utils.eval_checkpoint import (  # noqa: PLC0415
        apply_obs_normalizer, build_eval_envs, load_agent_from_checkpoint,
    )
    from omnisafe.utils.value_eval import estimate_true_value_same_state_mc  # noqa: PLC0415

    cfgs = _cfgs()
    live = _build_live_agent(cfgs)
    run_dir = _save_checkpoint(tmp_path, cfgs, live, epoch=1)
    rebuilt, norm_state = load_agent_from_checkpoint(run_dir, 1, cfgs=cfgs)

    envs = build_eval_envs(cfgs, which='mc')
    try:
        def run(agent):
            apply_obs_normalizer(envs['mc_env'], norm_state)
            torch.manual_seed(20260919)
            return estimate_true_value_same_state_mc(
                agent=agent, env=envs['mc_env'], cfgs=cfgs,
                discount_r=cfgs.algo_cfgs.gamma, discount_c=cfgs.algo_cfgs.cost_gamma,
                probe_seeds=[100_000, 100_001], mc_repeats=2, epoch=1,
                max_episode_steps=envs['max_episode_steps'],
            )

        a, b = run(live), run(rebuilt)
    finally:
        envs['mc_env'].close()

    assert a.keys() == b.keys(), 'stat keys differ'
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=0, abs=0), (
            f'{k}: live={a[k]!r} rebuilt={b[k]!r} -- the rebuilt agent is not the saved one'
        )


if __name__ == '__main__':
    raise SystemExit(pytest.main([os.path.abspath(__file__), '-q']))


# ---------------------------------------------------------------------------------------------
# End-to-end: the worker's whole bundle must equal the in-process path's, value for value.
# ---------------------------------------------------------------------------------------------

def _deep_equal(x, y, path: str = '') -> list[str]:
    """Exact structural comparison, returning a list of human-readable differences.

    Exact, not approximate: with :class:`~omnisafe.utils.eval_checkpoint.eval_rng` pinning the
    seed on both sides the two paths roll out the *same* trajectories, so any tolerance here
    would mask precisely the bugs this is meant to catch. NaN compares equal to NaN, since the
    studies legitimately emit NaN for an undefined correlation.
    """
    import numpy as np  # noqa: PLC0415

    out: list[str] = []
    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        if x.shape != y.shape:
            out.append(f'{path}: shape {x.shape} vs {y.shape}')
        elif not np.array_equal(x, y, equal_nan=True):
            out.append(f'{path}: tensor differs')
    elif isinstance(x, dict):
        if set(x) != set(y):
            out.append(f'{path}: keys differ: {sorted(set(x) ^ set(y))}')
        else:
            for k in x:
                out += _deep_equal(x[k], y[k], f'{path}/{k}')
    elif isinstance(x, (list, tuple)):
        if len(x) != len(y):
            out.append(f'{path}: len {len(x)} vs {len(y)}')
        else:
            for i, (u, v) in enumerate(zip(x, y)):
                out += _deep_equal(u, v, f'{path}[{i}]')
    elif isinstance(x, np.ndarray):
        if x.shape != y.shape:
            out.append(f'{path}: shape {x.shape} vs {y.shape}')
        elif not np.array_equal(x, y, equal_nan=True):
            out.append(f'{path}: array differs')
    elif isinstance(x, float):
        import math  # noqa: PLC0415
        if not (x == y or (math.isnan(x) and math.isnan(y))):
            out.append(f'{path}: {x!r} vs {y!r}')
    elif x != y:
        out.append(f'{path}: {x!r} vs {y!r}')
    return out


@pytest.mark.slow
def test_worker_bundle_equals_in_process_bundle(tmp_path) -> None:
    """The out-of-process worker must reproduce the in-process bundle exactly.

    Trains briefly with both studies on (so the in-process path writes a real
    ``eval_data/epoch_00001.pkl``), then runs the worker's ``evaluate_epoch`` over the same
    checkpoint and compares *every* value: predictions, MC means and variances, regression
    targets, per-rollout returns, the probe observations and actions, and every estimation error
    and correlation -- for the s0 study, each intermediate position, and the pooled set.

    Comparing only the aggregate statistics is not enough, and was not enough in practice: the
    first version of the worker pooled the intermediate positions *without* s0 and called
    ``pool_correlation_stats`` without ``prefix='PooledMC/'``. Both produce entirely plausible
    pooled correlations. Only comparing the raw prediction-side arrays exposed them.
    """
    import pickle  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import sys  # noqa: PLC0415

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments'))
    import omnisafe  # noqa: PLC0415
    from eval_worker import evaluate_epoch  # noqa: PLC0415
    from omnisafe.utils.eval_checkpoint import build_eval_envs, load_run_config  # noqa: PLC0415

    log_dir = str(tmp_path / 'run')
    omnisafe.Agent('CPO', ENV_ID, seed=0, custom_cfgs={
        'seed': 0,
        'train_cfgs': {'device': 'cpu', 'torch_threads': 2, 'vector_env_nums': 1, 'total_steps': 4000},
        'algo_cfgs': {
            'steps_per_epoch': 2000, 'update_iters': 1, 'test_estimate': False,
            # Explicitly sync: this test's whole point is to compare the in-process bundle
            # against the worker's, which requires training to actually produce one.
            'async_eval': False,
            'mc_value_study': True, 'mc_value_study_probes': 3, 'mc_value_study_repeats': 2,
            'mc_value_study_vector_envs': 3,
            'intermediate_state_study': True, 'intermediate_state_study_positions': [100, 300],
            'intermediate_state_study_probes': 3, 'intermediate_state_study_repeats': 2,
            'early_eval_freq': 5, 'early_eval_epochs': 50,
        },
        'model_cfgs': {'critic': {'lr': 3e-4, 'hidden_sizes': [64, 64]}},
        'logger_cfgs': {'use_wandb': False, 'use_tensorboard': False, 'log_dir': log_dir},
    }).learn()

    run_dir = next(
        os.path.join(r, '') for r, _, fs in os.walk(log_dir) if 'config.json' in fs
    )
    in_process = os.path.join(str(tmp_path), 'in_process.pkl')
    shutil.copy(os.path.join(run_dir, 'eval_data', 'epoch_00001.pkl'), in_process)

    cfgs = load_run_config(run_dir)
    envs = build_eval_envs(cfgs, which='both')
    try:
        flat = evaluate_epoch(run_dir, 1, cfgs, envs)
    finally:
        for k in ('mc_env', 'interm_env'):
            envs[k].close()

    with open(in_process, 'rb') as f:
        expected = pickle.load(f)
    with open(os.path.join(run_dir, 'eval_data', 'epoch_00001.pkl'), 'rb') as f:
        actual = pickle.load(f)

    diffs = _deep_equal(expected, actual)
    assert not diffs, 'worker bundle differs from in-process bundle:\n  ' + '\n  '.join(diffs[:20])

    # The pooled gradient-alignment numbers only ever reach the logger in-process, so the bundle
    # comparison above cannot see them -- assert the worker emits them rather than silently
    # dropping two of the diagnostics from eval_progress.csv.
    for stream in ('r', 'c'):
        assert f'PooledMC/GradientAlignment_{stream}' in flat
    assert any(k.startswith('MCStudy/') for k in flat)
    assert any(k.startswith('IntermediateMC/pos100/') for k in flat)
    assert any(k.startswith('PooledMC/') for k in flat)


# ---------------------------------------------------------------------------------------------
# eval_rng: determinism from (seed, epoch), independence across seeds, no training side effect.
# ---------------------------------------------------------------------------------------------

def _rng_cfgs(seed: int):
    """Minimal stand-in for a resolved config: eval_rng only reads .seed and .algo_cfgs."""
    return type(
        'C', (), {'seed': seed, 'algo_cfgs': type('A', (), {'eval_rng_seed_base': 500_000})},
    )


def test_eval_rng_is_determined_by_seed_and_epoch() -> None:
    """Same (seed, epoch) -> same seed -> same draws, every time and in every process."""
    from omnisafe.utils.eval_checkpoint import eval_rng  # noqa: PLC0415

    def draw(seed, epoch):
        with eval_rng(_rng_cfgs(seed), epoch):
            return torch.randn(8)

    assert torch.equal(draw(0, 25), draw(0, 25)), 'same (seed, epoch) must reproduce'
    assert torch.equal(draw(3, 7), draw(3, 7))


def test_eval_rng_differs_across_run_seeds() -> None:
    """Different training seeds must draw independent eval rollouts.

    Without the run seed folded in, every seed of a study shares one eval-noise realisation and
    averaging across seeds does not average that noise away -- the error bars on a seed-averaged
    calibration curve come out silently too tight.
    """
    from omnisafe.utils.eval_checkpoint import eval_rng  # noqa: PLC0415

    def draw(seed, epoch):
        with eval_rng(_rng_cfgs(seed), epoch):
            return torch.randn(8)

    assert not torch.equal(draw(0, 25), draw(1, 25)), 'seeds 0 and 1 drew identical eval noise'


def test_eval_rng_seeds_never_collide() -> None:
    """(seed, epoch) -> seed must be injective over any realistic range."""
    from omnisafe.utils.eval_checkpoint import eval_rng  # noqa: PLC0415

    seen = {}
    for s in range(20):
        for e in (0, 1, 5, 500, 10_000):
            k = eval_rng(_rng_cfgs(s), e).seed
            assert k not in seen, f'collision: (seed={s}, epoch={e}) and {seen[k]} share {k}'
            seen[k] = (s, e)


def test_eval_rng_restores_caller_state() -> None:
    """Training's RNG stream must be exactly as it would have been with no evaluation at all."""
    from omnisafe.utils.eval_checkpoint import eval_rng  # noqa: PLC0415

    torch.manual_seed(999)
    expected = [torch.randn(4), torch.randn(4)]

    torch.manual_seed(999)
    first = torch.randn(4)
    with eval_rng(_rng_cfgs(0), 25):
        torch.randn(100)  # perturb hard inside the scope
    second = torch.randn(4)

    assert torch.equal(first, expected[0])
    assert torch.equal(second, expected[1]), 'eval seeding leaked into the training RNG stream'


# ---------------------------------------------------------------------------------------------
# async_eval: training must skip the studies but still leave the worker everything it needs.
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_async_eval_skips_studies_but_writes_usable_checkpoints(tmp_path) -> None:
    """With ``async_eval: True`` the run computes no studies, yet stays fully evaluable later.

    Two halves, and the second is the one that matters: skipping is easy to get right, but a run
    that skips the studies *and* writes a checkpoint the evaluator cannot consume would produce
    no calibration data at all -- and nothing in the training logs would say so. So this asserts
    the checkpoint actually round-trips through ``load_agent_from_checkpoint`` and that the worker
    then produces real statistics from it.
    """
    import sys  # noqa: PLC0415

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments'),
    )
    import omnisafe  # noqa: PLC0415
    from eval_worker import evaluate_epoch  # noqa: PLC0415
    from omnisafe.utils.eval_checkpoint import build_eval_envs, load_run_config  # noqa: PLC0415

    log_dir = str(tmp_path / 'run')
    omnisafe.Agent('CPO', ENV_ID, seed=0, custom_cfgs={
        'seed': 0,
        'train_cfgs': {'device': 'cpu', 'torch_threads': 2, 'vector_env_nums': 1, 'total_steps': 4000},
        'algo_cfgs': {
            'steps_per_epoch': 2000, 'update_iters': 1, 'test_estimate': False,
            'async_eval': True,
            'mc_value_study': True, 'mc_value_study_probes': 3, 'mc_value_study_repeats': 2,
            'mc_value_study_vector_envs': 3,
            'intermediate_state_study': False,
            'early_eval_freq': 5, 'early_eval_epochs': 50,
        },
        'model_cfgs': {'critic': {'lr': 3e-4, 'hidden_sizes': [64, 64]}},
        'logger_cfgs': {'use_wandb': False, 'use_tensorboard': False, 'log_dir': log_dir},
    }).learn()

    run_dir = next(os.path.join(r, '') for r, _, fs in os.walk(log_dir) if 'config.json' in fs)

    # Skipped in-process: no eval_data bundle should have been written by training.
    assert not os.path.exists(os.path.join(run_dir, 'eval_data', 'epoch_00001.pkl')), (
        'async_eval left the in-process studies running'
    )
    # But the checkpoint for that epoch must exist, for the worker to pick up.
    assert os.path.exists(os.path.join(run_dir, 'torch_save', 'epoch-1.pt'))

    # And it must actually be evaluable: full round-trip through the worker.
    cfgs = load_run_config(run_dir)
    assert cfgs.algo_cfgs.async_eval is True
    envs = build_eval_envs(cfgs, which='mc')
    try:
        flat = evaluate_epoch(run_dir, 1, cfgs, envs)
    finally:
        envs['mc_env'].close()

    assert flat['epoch'] == 1
    assert any(k.startswith('MCStudy/') for k in flat), f'worker produced no stats: {sorted(flat)}'
    assert os.path.exists(os.path.join(run_dir, 'eval_data', 'epoch_00001.pkl')), (
        'worker did not persist the eval bundle'
    )
