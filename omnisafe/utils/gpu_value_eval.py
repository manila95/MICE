"""Value-function studies on the GPU Safety-Gymnasium (``safety_gymnasium_gpu``).

Drop-in GPU counterparts of the CPU study functions, returning the same ``(stats, raw)`` pairs so
that logging, pooling, gradient alignment and the eval-data dump downstream need no changes:

- :meth:`GPUValueStudies.s0_study` <-> :func:`omnisafe.utils.value_eval.estimate_true_value_same_state_mc`
- :meth:`GPUValueStudies.collect` <-> :func:`omnisafe.utils.state_snapshot.collect_on_policy_snapshots`
- :meth:`GPUValueStudies.state_study` <-> :func:`omnisafe.utils.value_eval.estimate_value_from_snapshots`

Every (probe, repeat) pair is one GPU world; a probe state is captured once and broadcast into its
``repeats`` worlds. Rollouts mirror the CPU loops operation for operation (the same agent calls,
the same per-step sequences, the same ``_rollout_target`` for the training-style target).

Which value is estimated. A CPU probe reset with ``reset(seed=s)`` -- or a CPU snapshot, which
restores ``rng_state`` -- also fixes the sequence of *future goal positions* for every repeat. With
``rng_replay=True`` (default) the GPU worlds replay that same RNG stream, reproducing the CPU
estimand exactly; with ``False`` future goals are drawn independently per rollout, i.e. the value
is marginalized over them (the quantity an observation-based critic can represent).
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from omnisafe.utils.isolated_eval import _assemble
from omnisafe.utils.value_eval import _effective_rollout_horizon, _rollout_target


RNG_STREAM = 8192  # doubles of the CPU task RNG recorded per state (1 per step + 2 per goal draw)


class GPUValueStudies:
    """Owns the GPU eval envs and runs the value studies on them.

    Args:
        env_id: Safety-Gymnasium id.
        env_kwargs: ``safety_gymnasium_gpu.make`` kwargs (``config``, ``lidar_num_bins``, ...).
        max_worlds: Cap on parallel worlds per rollout wave (memory bound).
        sim_device: CUDA device for the simulation.
        rng_replay: See the module docstring.
    """

    def __init__(self, env_id: str, env_kwargs: dict, max_worlds: int = 8192, sim_device: str = 'cuda:0',
                 rng_replay: bool = True) -> None:
        self.env_id = env_id
        self.env_kwargs = dict(env_kwargs)
        self.max_worlds = int(max_worlds)
        self.sim_device = sim_device
        self.rng_replay = rng_replay
        self._envs: dict[int, Any] = {}
        self.timing: dict[str, float] = {}

    # ----------------------------------------------------------------- plumbing
    def _env(self, n: int):
        """A frozen-on-done GPU env with exactly ``n`` worlds (cached per size)."""
        if n not in self._envs:
            import safety_gymnasium_gpu as sgg  # noqa: PLC0415

            self._envs[n] = sgg.make(self.env_id, num_envs=n, device=self.sim_device, seed=0,
                                     autoreset=False, **self.env_kwargs)
        return self._envs[n]

    def _wave_size(self, total: int) -> int:
        n = min(total, self.max_worlds)
        return int(math.ceil(total / math.ceil(total / n)))  # balance waves

    def _states_from_seeds(self, env, seeds) -> dict:
        """CPU ``reset(seed)`` layouts captured as GPU states (cached: they do not depend on the policy)."""
        from safety_gymnasium_gpu.mc import stack_states  # noqa: PLC0415

        cache = self.__dict__.setdefault('_seed_cache', {})
        cpu = env.cpu_env
        out = []
        for s in seeds:
            s = int(s)
            if s not in cache:
                cpu.reset(seed=s)
                cache[s] = env.state_from_cpu(cpu.unwrapped, rng_stream=RNG_STREAM if self.rng_replay else 0)
            out.append(cache[s])
        if len(cache) > 4096:  # collection seeds change every epoch; keep the cache bounded
            for k in list(cache)[: len(cache) - 4096]:
                del cache[k]
        return stack_states(out)

    def run_all(self, agent, normalizer, cfgs, probe_seeds, s0_repeats, collected: dict, positions_repeats: int,
                discount_r, discount_c, bootstrap_threshold=None, tail_mode=None):
        """The s0 study and every position's state study in one batched rollout.

        Same per-probe computation as :meth:`s0_study` / :meth:`state_study` (each probe's rollouts
        are independent worlds, so batching them changes nothing but wall-clock); returns
        ``(s0 (stats, raw), {pos: (stats, raw)})``. Requires equal repeats for both studies.
        """
        assert s0_repeats == positions_repeats, 'run_all batches both studies with one repeat count'
        from safety_gymnasium_gpu.mc import stack_states  # noqa: PLC0415

        t0 = time.time()
        total = (len(probe_seeds) + sum(next(iter(s.values())).shape[0] for s in collected.values())) * s0_repeats
        env = self._env(self._wave_size(total))
        s0_states = self._states_from_seeds(env, probe_seeds)
        t1 = time.time()
        order = [('s0', len(probe_seeds))] + [(pos, next(iter(st.values())).shape[0]) for pos, st in collected.items()]
        keys = set(s0_states)
        for st in collected.values():
            keys &= set(st)
        allstates = stack_states([{k: s0_states[k] for k in keys}] + [{k: st[k].to(s0_states[k].device) for k in keys} for st in collected.values()])
        # s0 states are fresh resets (steps == 0); collected ones get their clock restarted, as the
        # CPU snapshot study does with reset_elapsed_steps=True -- so one reset_steps=True covers both.
        probes = self._rollout(agent, normalizer, cfgs, allstates, s0_repeats, env.num_steps, discount_r, discount_c,
                               bootstrap_threshold, tail_mode, reset_steps=True)
        self.timing['s0_capture'] = t1 - t0
        self.timing['rollout_all'] = time.time() - t1
        out, i = {}, 0
        for key, count in order:
            chunk = probes[i:i + count]
            i += count
            if key == 's0':
                for p, s in zip(chunk, probe_seeds):
                    p['seed'] = int(s)
                stats, raw = _assemble(chunk, s0_repeats, 'MCStudy/')
                raw['probe_seeds'] = list(probe_seeds)
                out['s0'] = (stats, raw)
            else:
                stats, raw = _assemble(chunk, s0_repeats, '')
                raw.pop('probe_seeds', None)
                out[key] = (stats, raw)
        return out.pop('s0'), out

    @staticmethod
    def _policy(agent, normalizer, env):
        dev = next(agent.parameters()).device
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=env.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=env.device)

        @torch.no_grad()
        def step(obs_raw):
            obs = obs_raw.to(dev)
            if normalizer is not None:
                obs = normalizer.normalize(obs, update=False)
            act, value_r, value_c, logp = agent.step(obs)
            env_act = low + (high - low) * (act.to(env.device) + 1) / 2  # ActionScale(-1, 1)
            return obs, act, value_r.reshape(-1), value_c.reshape(-1), logp.reshape(-1), env_act

        return step

    # ----------------------------------------------------------------- core rollout
    def _rollout(self, agent, normalizer, cfgs, states: dict, repeats: int, nominal: int,
                 discount_r: float, discount_c: float, bootstrap_threshold, tail_mode, reset_steps: bool,
                 seeds=None) -> list[dict]:
        """MC rollouts from each of P states, ``repeats`` each; returns per-probe entries."""
        horizon = _effective_rollout_horizon(nominal, discount_r, discount_c, bootstrap_threshold)
        if horizon < nominal and tail_mode not in ('bootstrap', 'drop'):
            raise ValueError("bootstrap_threshold truncates the rollout: set tail_mode='drop' or 'bootstrap'")
        bootstrap_tail = tail_mode == 'bootstrap'
        adv_r = getattr(cfgs.algo_cfgs, 'adv_estimation_method', 'gae')
        adv_c = getattr(cfgs.algo_cfgs, 'cost_adv_estimation_method', None) or adv_r
        lam_r = getattr(cfgs.algo_cfgs, 'lam', 0.95)
        lam_c = getattr(cfgs.algo_cfgs, 'lam_c', lam_r)
        penalty = getattr(cfgs.algo_cfgs, 'penalty_coef', 0.0)

        P = next(iter(states.values())).shape[0]
        total = P * repeats
        n = self._wave_size(total)
        env = self._env(n)
        policy = self._policy(agent, normalizer, env)
        task_state = torch.arange(total) // repeats
        per_task: list[dict] = [None] * total  # type: ignore[list-item]
        for start in range(0, total, n):
            ids = list(range(start, min(start + n, total)))
            m = len(ids)
            src = torch.cat([task_state[ids], task_state[ids[:1]].expand(n - m)]).to(env.device)
            obs_raw = env.set_state({k: v.to(env.device)[src] for k, v in states.items()}, reset_steps=reset_steps)
            obs, act, v_r, v_c, logp, env_act = policy(obs_raw)
            obs0, act0 = obs.cpu(), act.cpu()
            r_seq = torch.zeros(horizon, n, device=env.device)
            c_seq = torch.zeros(horizon, n, device=env.device)
            vr_seq = torch.zeros(horizon + 1, n, device=env.device)
            vc_seq = torch.zeros(horizon + 1, n, device=env.device)
            lp_seq = torch.zeros(horizon, n, device=env.device)
            vr_seq[0], vc_seq[0] = v_r.to(env.device), v_c.to(env.device)
            g_r = torch.zeros(n, dtype=torch.float64, device=env.device)
            g_c = torch.zeros(n, dtype=torch.float64, device=env.device)
            disc_r = disc_c = 1.0
            for t in range(horizon):
                obs_raw, r, c, _te, _tr, _ = env.step(env_act)
                r_seq[t], c_seq[t] = r, c
                g_r += disc_r * r.double()
                g_c += disc_c * c.double()
                disc_r *= discount_r
                disc_c *= discount_c
                if t < horizon - 1:
                    # Same bookkeeping as the CPU loop, including its logp alignment.
                    _, _, v_r, v_c, logp, env_act = policy(obs_raw)
                    vr_seq[t + 1], vc_seq[t + 1], lp_seq[t] = v_r.to(env.device), v_c.to(env.device), logp.to(env.device)
            if horizon == nominal:
                if not bool(env.done.all()):
                    raise RuntimeError('GPU value study: some world did not reach done at the horizon')
                is_term = env.terminated.clone()
            else:
                is_term = torch.zeros(n, dtype=torch.bool, device=env.device)
            # Bootstrap at the true final observation (the CPU path, behind AutoReset, reads the
            # next episode's first one; the weight of this slot in a 1000-step target is ~0).
            _, _, b_r, b_c, _, _ = policy(obs_raw)
            b_r = torch.where(is_term, 0.0, b_r.to(env.device))
            b_c = torch.where(is_term, 0.0, b_c.to(env.device))
            vr_seq[horizon], vc_seq[horizon] = b_r, b_c
            if horizon < nominal and bootstrap_tail:
                g_r += disc_r * b_r.double()
                g_c += disc_c * b_c.double()
            # training-style targets, per world, with the shared helper
            r_cpu, c_cpu = r_seq.cpu(), c_seq.cpu()
            vr_cpu, vc_cpu, lp_cpu = vr_seq.cpu(), vc_seq.cpu(), lp_seq.cpu()
            term_cpu = is_term.cpu()
            g_r_cpu, g_c_cpu = g_r.cpu(), g_c.cpu()
            pr, pc = vr_seq[0].cpu(), vc_seq[0].cpu()
            for local, task in enumerate(ids):
                tr_ = _rollout_target(r_cpu[:, local] - penalty * c_cpu[:, local], vr_cpu[:, local],
                                      bool(term_cpu[local]), lam_r, discount_r, adv_r, logp_seq=lp_cpu[:, local])
                tc_ = _rollout_target(c_cpu[:, local], vc_cpu[:, local], bool(term_cpu[local]), lam_c,
                                      discount_c, adv_c, logp_seq=lp_cpu[:, local])
                per_task[task] = {
                    'pred_r': float(pr[local]), 'pred_c': float(pc[local]),
                    'ret_r': float(g_r_cpu[local]), 'ret_c': float(g_c_cpu[local]),
                    'tgt_r': float(tr_), 'tgt_c': float(tc_),
                    'obs': obs0[local].clone(), 'act': act0[local].clone(),
                }
        probes = []
        for p in range(P):
            rows = per_task[p * repeats:(p + 1) * repeats]
            probes.append({
                'seed': None if seeds is None else int(seeds[p]),
                'pred_r': rows[0]['pred_r'], 'pred_c': rows[0]['pred_c'],
                'returns_r': [x['ret_r'] for x in rows], 'returns_c': [x['ret_c'] for x in rows],
                'targets_r': [x['tgt_r'] for x in rows], 'targets_c': [x['tgt_c'] for x in rows],
                'obs': [x['obs'] for x in rows], 'act': [x['act'] for x in rows],
            })
        return probes

    # ----------------------------------------------------------------- studies
    def s0_study(self, agent, normalizer, cfgs, probe_seeds, repeats, discount_r, discount_c,
                 bootstrap_threshold=None, tail_mode=None):
        t0 = time.time()
        env = self._env(self._wave_size(len(probe_seeds) * repeats))
        states = self._states_from_seeds(env, probe_seeds)
        t1 = time.time()
        probes = self._rollout(agent, normalizer, cfgs, states, repeats, env.num_steps, discount_r, discount_c,
                               bootstrap_threshold, tail_mode, reset_steps=False, seeds=probe_seeds)
        self.timing['s0_capture'] = t1 - t0
        self.timing['s0_rollout'] = time.time() - t1
        stats, raw = _assemble(probes, repeats, 'MCStudy/')
        raw['probe_seeds'] = list(probe_seeds)
        return stats, raw

    def collect(self, agent, normalizer, trigger_steps, base_seed: int, n_probes: int, transition_steps=None):
        """On-policy states at each step in ``trigger_steps`` from ``n_probes`` seeded episodes."""
        t0 = time.time()
        trigger_steps = set(trigger_steps)
        transition_steps = set(transition_steps or [])
        env = self._env(n_probes)
        policy = self._policy(agent, normalizer, env)
        states = self._states_from_seeds(env, range(base_seed, base_seed + n_probes))
        obs_raw = env.set_state({k: v.to(env.device) for k, v in states.items()})
        collected: dict[int, dict] = {}
        transitions: dict[int, dict] = {}
        for t in range(max(trigger_steps)):
            obs, act, _, _, _, env_act = policy(obs_raw)
            obs_raw, r, c, _, _, _ = env.step(env_act)
            if t in transition_steps:
                transitions[t] = {
                    'obs': [row.cpu().clone() for row in obs], 'action': [row.cpu().clone() for row in act],
                    'reward': r.cpu().tolist(), 'cost': c.cpu().tolist(),
                }
            if t + 1 in trigger_steps:
                if bool(env.done.any()):
                    raise RuntimeError(f'GPU collect: an episode ended before step {t + 1}')
                collected[t + 1] = env.get_state()
        self.timing['collect'] = self.timing.get('collect', 0.0) + time.time() - t0
        return collected, transitions

    def state_study(self, agent, normalizer, cfgs, states, repeats, discount_r, discount_c,
                    bootstrap_threshold=None, tail_mode=None):
        t0 = time.time()
        env = self._env(self._wave_size(next(iter(states.values())).shape[0] * repeats))
        probes = self._rollout(agent, normalizer, cfgs, states, repeats, env.num_steps, discount_r, discount_c,
                               bootstrap_threshold, tail_mode, reset_steps=True)
        self.timing['state_rollout'] = self.timing.get('state_rollout', 0.0) + time.time() - t0
        stats, raw = _assemble(probes, repeats, '')
        raw.pop('probe_seeds', None)
        return stats, raw


def gpu_env_kwargs(env_cfgs: dict) -> dict:
    """Translate omnisafe ``env_cfgs`` into ``safety_gymnasium_gpu.make`` kwargs."""
    kw = {k: v for k, v in dict(env_cfgs).items() if k not in ('class_name', 'sim_device', 'seed')}
    if kw.get('lidar_num_bins') is None:
        kw.pop('lidar_num_bins', None)
    hazards = kw.pop('hazards_num', None)
    if hazards is not None:
        config = dict(kw.pop('config', None) or {})
        config['hazards_num'] = int(hazards)
        kw['config'] = config
    return kw


__all__ = ['GPUValueStudies', 'gpu_env_kwargs', 'np']
