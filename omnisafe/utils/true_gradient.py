r"""Brute-force "true" policy gradients for the reward and cost surrogates (GPU env only).

Follows Ilyas et al., "A Closer Look at Deep Policy Gradients" (ICLR 2020): the true gradient is
the agent's own gradient estimator averaged over a sample far larger than a training batch, and
batch-size estimates are scored by their cosine with it.

:class:`TrueGradientProbe` owns a separate GPU env with ``num_worlds`` worlds (typically 50x the
training envs). Each epoch it is stepped ``frag_len`` steps -- exactly the per-env fragment the
training rollout uses -- with the current policy, *persistently* like training (``persistent_rollout``),
so its states come from the same distribution as the training batch: same episode phase, same
history of past policies. From those ``num_worlds * frag_len`` samples it computes, at the current
parameters:

* ``critic`` gradients: the training estimator exactly -- the same advantage method (GAE / TD(0) /
  ...), the same bootstrap rules (``V(final_obs)`` on time-out, 0 on termination, ``V(obs)`` at the
  fragment end) and the same standardization (reward: mean/std, cost: mean only). Its expectation
  is what the training batch gradient estimates, critic bias included.
* ``mc`` gradients (``mc=True``): critic-free. A copy of every world is continued with the same
  policy to the end of its episode, giving the full discounted return-to-go ``G_t`` for every
  sample, and the advantage is ``G_t - V(s_t)`` (the critic only as a baseline, which does not bias
  the gradient). Its expectation is the policy gradient of the episodic discounted objective.

Per-group gradients (groups of ``group_envs`` whole worlds, independent given the policy) are
computed in one vectorized pass with ``torch.func.vmap``; every larger-sample gradient is a mean of
group gradients (groups are equal-sized, so the mean is exact).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.func import functional_call, grad, vmap

from omnisafe.utils.gae import resolve_adv_and_target


_HALF_LOG_2PI = 0.9189385332046727


def _cos(u: torch.Tensor, v: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(u, v, dim=0).item()


def vector_advantage(
    rew: torch.Tensor,
    val: torch.Tensor,
    end: torch.Tensor,
    boot: torch.Tensor,
    gamma: float,
    lam: float,
    method: str,
) -> torch.Tensor:
    """Advantages for ``[T, N]`` rollouts, identical to per-path ``calculate_adv_and_value_targets``.

    ``end[t]`` marks the last step of a path (episode end or fragment end) and ``boot[t]`` the value
    appended after it (``V(final_obs)``, ``V(obs)`` or 0). Paths restart after every end.
    """
    t_len = rew.shape[0]
    nxt = torch.empty_like(val)
    nxt[:-1] = torch.where(end[:-1], boot[:-1], val[1:])
    nxt[-1] = boot[-1]
    delta = rew + gamma * nxt - val
    if method == 'td_zero':
        return delta
    out = torch.empty_like(val)
    acc = torch.zeros_like(val[0])
    cont = (~end).float()
    if method == 'gae':
        for t in range(t_len - 1, -1, -1):
            acc = delta[t] + gamma * lam * cont[t] * acc
            out[t] = acc
        return out
    if method == 'mc':  # discounted rewards-to-go, bootstrap folded in at each path end
        for t in range(t_len - 1, -1, -1):
            acc = rew[t] + gamma * torch.where(end[t], boot[t], acc)
            out[t] = acc
        return out
    raise NotImplementedError(f'advantage method {method!r} not supported by the true-gradient probe')


class TrueGradientProbe:
    """See the module docstring."""

    def __init__(
        self,
        make_env,
        num_worlds: int,
        frag_len: int,
        algo_cfgs: Any,
        device: torch.device,
        group_envs: int,
        mc: bool = True,
        seed: int = 12345,
        adv_repeats: int = 0,
        adv_horizon: int = 600,
        oracle_samples: int = 0,
    ) -> None:
        assert num_worlds % group_envs == 0, 'num_worlds must be a multiple of group_envs'
        # nested Monte-Carlo advantage reference (see :meth:`_nested_mc`): anchors * 2 * repeats <= worlds
        self.adv_repeats, self.adv_horizon = int(adv_repeats), int(adv_horizon)
        self.n_anchors = num_worlds // (2 * adv_repeats) if adv_repeats else 0
        # oracle-advantage gradient: this many (state, action) samples, each with ONE paired
        # common-random-number rollout (an unbiased A sample), see :func:`oracle_report`
        self.oracle_samples = int(oracle_samples)
        assert not (self.n_anchors or self.oracle_samples) or mc, 'nested rollouts need the continuation env (mc=True)'
        self.env = make_env(num_worlds)
        self.cont = make_env(num_worlds) if mc else None
        self.R, self.T, self.dev = num_worlds, frag_len, device
        self.group_envs = group_envs
        self.seed = seed
        self.obs: torch.Tensor | None = None
        a = algo_cfgs
        self.gamma, self.lam = a.gamma, a.lam
        self.cgamma = getattr(a, 'cost_gamma', None) or a.gamma
        self.lam_c = a.lam_c
        self.adv_r = resolve_adv_and_target(a.adv_estimation_method)[0]
        self.adv_c = resolve_adv_and_target(getattr(a, 'cost_adv_estimation_method', None) or a.adv_estimation_method)[0]
        self.std_r, self.std_c = a.standardized_rew_adv, a.standardized_cost_adv
        assert getattr(a, 'penalty_coef', 0.0) == 0.0, 'penalty_coef is not supported'

    # ------------------------------------------------------------------ rollout
    @staticmethod
    def _norm(env):
        from omnisafe.utils.value_eval import _find_obs_normalizer  # noqa: PLC0415

        return _find_obs_normalizer(env)

    @torch.no_grad()
    def collect(self, agent, train_env) -> dict[str, torch.Tensor]:
        from omnisafe.utils.value_eval import sync_obs_normalizer  # noqa: PLC0415

        if train_env is not None:  # offline use sets the normalizers itself
            sync_obs_normalizer(self.env, train_env)
            if self.cont is not None:
                sync_obs_normalizer(self.cont, train_env)
        if self.obs is None:
            self.obs, _ = self.env.reset(seed=self.seed)
        obs = self.obs
        T, R = self.T, self.R
        buf = {k: [] for k in ('obs', 'act', 'logp', 'rew', 'cost', 'vr', 'vc', 'end', 'bootr', 'bootc', 'term')}
        # anchors for the advantage reference: distinct random worlds, each at a random fragment step
        sets = {}
        if self.n_anchors:
            sets['anchor'] = (torch.randperm(R, device=self.dev)[:self.n_anchors],
                              torch.randint(0, T, (self.n_anchors,), device=self.dev), self.adv_repeats)
        if self.oracle_samples:
            flat = torch.randperm(T * R, device=self.dev)[:self.oracle_samples]  # distinct (t, world) samples
            sets['oracle'] = (flat % R, flat // R, 1)
        snaps = {k: [] for k in sets}
        for t in range(T):
            for k, (aw, at, _) in sets.items():
                if bool((at == t).any()):
                    sel = (at == t).nonzero().squeeze(1)
                    snaps[k].append((sel, self.env.gpu_env.get_state(aw[sel])))
            act, vr, vc, logp = agent.step(obs)
            nobs, rew, cost, term, trunc, info = self.env.step(act)
            term, trunc = term.bool(), trunc.bool()
            bootr = torch.zeros(R, device=self.dev)
            bootc = torch.zeros(R, device=self.dev)
            if bool(trunc.any()):
                _, fr, fc, _ = agent.step(info['final_observation'])
                m = trunc & ~term
                bootr = torch.where(m, fr, bootr)
                bootc = torch.where(m, fc, bootc)
            end = term | trunc
            if t == T - 1:
                _, lr, lc, _ = agent.step(nobs)
                bootr = torch.where(end, bootr, lr)
                bootc = torch.where(end, bootc, lc)
                end = torch.ones_like(end)
            for k, v in (('obs', obs), ('act', act), ('logp', logp), ('rew', rew), ('cost', cost), ('vr', vr),
                         ('vc', vc), ('end', end), ('bootr', bootr), ('bootc', bootc), ('term', term | trunc)):
                buf[k].append(v.to(self.dev))
            obs = nobs
        self.obs = obs
        d = {k: torch.stack(v) for k, v in buf.items()}  # [T, R, ...]
        d['adv_r'] = vector_advantage(d['rew'], d['vr'], d['end'], d['bootr'], self.gamma, self.lam, self.adv_r)
        d['adv_c'] = vector_advantage(d['cost'], d['vc'], d['end'], d['bootc'], self.cgamma, self.lam_c, self.adv_c)
        if self.cont is not None:
            fr, fc = self._continue(agent, ~d['term'][-1])
            # episode ends inside the fragment stop the return (true finite-horizon objective);
            # the fragment's last step continues with the rollout-to-episode-end return.
            ep_end = d['term']
            zero = torch.zeros_like(d['vr'])
            bootr = zero.clone()
            bootc = zero.clone()
            bootr[-1] = torch.where(ep_end[-1], 0.0, fr)
            bootc[-1] = torch.where(ep_end[-1], 0.0, fc)
            mc_end = ep_end.clone()
            mc_end[-1] = True
            # REINFORCE: raw discounted reward-to-go / cost-to-go to the episode end
            d['rf_r'] = vector_advantage(d['rew'], d['vr'], mc_end, bootr, self.gamma, 1.0, 'mc')
            d['rf_c'] = vector_advantage(d['cost'], d['vc'], mc_end, bootc, self.cgamma, 1.0, 'mc')
            # the same with the critic as a baseline (same expectation, lower variance)
            d['mc_r'] = d['rf_r'] - d['vr']
            d['mc_c'] = d['rf_c'] - d['vc']
        for k, (aw, at, reps) in sets.items():
            d[k] = self._nested_mc(agent, d, aw, at, snaps[k], reps)
        return d

    @torch.no_grad()
    def _continue(self, agent, alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Discounted reward/cost from the fragment end to the episode end of every ``alive`` world."""
        src, dst = self.env.gpu_env, self.cont.gpu_env
        raw = dst.set_state(src.get_state())
        obs = self._norm(self.cont).normalize(raw.to(self.dev), update=False)
        R = self.R
        alive = alive.clone()
        fr = torch.zeros(R, device=self.dev)
        fc = torch.zeros(R, device=self.dev)
        dr, dc = 1.0, 1.0
        while bool(alive.any()):
            act, _, _, _ = agent.step(obs)
            obs, rew, cost, term, trunc, _ = self.cont.step(act)
            fr += alive.float() * dr * rew.to(self.dev)
            fc += alive.float() * dc * cost.to(self.dev)
            dr *= self.gamma
            dc *= self.cgamma
            alive &= ~(term.bool() | trunc.bool()).to(self.dev)
        return fr, fc

    @torch.no_grad()
    def _nested_mc(self, agent, d: dict, aw, at, snaps, K: int) -> dict[str, torch.Tensor]:
        """Monte-Carlo Q(s,a) and V(s) at the anchor samples, by re-simulating from their exact state.

        Every anchor state is loaded into ``2 * K`` worlds (in waves of at most ``num_worlds``): rollout ``i`` of
        the first half takes the anchor's recorded action at the first step, rollout ``i`` of the
        second half samples a fresh one; all then follow the policy for ``adv_horizon`` steps.
        Common random numbers: rollouts ``i`` of the two halves share their policy noise and their
        future goal draws (a per-pair replayed goal RNG stream), so ``Q_i - V_i`` cancels most of
        the return noise while each half stays an unbiased sample. The episode clock is restarted so
        no rollout is cut by the time limit: the estimand is the discounted, time-unaware value the
        critic targets (gamma^600 ~ 0.002 of the tail is dropped).
        """
        M, H = aw.numel(), self.adv_horizon
        order = torch.cat([s for s, _ in snaps])
        state_all = {k: torch.cat([st[k] for _, st in snaps]) for k in snaps[0][1]}
        inv = torch.empty_like(order)
        inv[order] = torch.arange(M, device=order.device)
        state_all = {k: v[inv.to(v.device)] for k, v in state_all.items()}  # anchor order
        a_all = d['act'][at, aw]  # [M, act_dim]
        per_wave = self.R // (2 * K)
        gr_all, gc_all = [], []
        dst = self.cont.gpu_env
        actor = agent.actor
        for w0 in range(0, M, per_wave):
            m = min(per_wave, M - w0)
            n = m * 2 * K
            state = {k: v[w0:w0 + m].repeat_interleave(2 * K, dim=0) for k, v in state_all.items()}  # [m, 2, K]
            stream = torch.rand(m, 1, K, H + 2048, dtype=torch.float64, device=dst.device).expand(m, 2, K, -1)
            state['rng_stream'] = stream.reshape(n, -1)
            state['rng_cursor'] = torch.zeros(n, dtype=torch.long, device=dst.device)
            raw = dst.set_state(state, idx=torch.arange(n, device=dst.device), reset_steps=True)
            del state, stream
            obs = self._norm(self.cont).normalize(raw.to(self.dev), update=False)
            alive = torch.zeros(self.R, dtype=torch.bool, device=self.dev)
            alive[:n] = True
            gr = torch.zeros(self.R, device=self.dev)
            gc = torch.zeros(self.R, device=self.dev)
            dr, dc = 1.0, 1.0
            for h in range(H):
                mean = actor.mean(obs)
                std = torch.exp(actor.log_std)
                eps = torch.randn(m, 1, K, mean.shape[-1], device=self.dev).expand(m, 2, K, -1).reshape(n, -1)
                eps = torch.cat([eps, torch.randn(self.R - n, mean.shape[-1], device=self.dev)])
                act = mean + std * eps
                if h == 0:
                    act.view(-1)[: n * act.shape[-1]].view(m, 2, K, -1)[:, 0] = a_all[w0:w0 + m, None]
                obs, rew, cost, term, trunc, _ = self.cont.step(act)
                gr += alive.float() * dr * rew.to(self.dev)
                gc += alive.float() * dc * cost.to(self.dev)
                dr *= self.gamma
                dc *= self.cgamma
                alive &= ~(term.bool() | trunc.bool()).to(self.dev)
                if not bool(alive.any()):
                    break
            gr_all.append(gr[:n].view(m, 2, K))
            gc_all.append(gc[:n].view(m, 2, K))
        out = {'world': aw, 't': at, 'K': K}
        for lab, g in (('r', torch.cat(gr_all)), ('c', torch.cat(gc_all))):
            out[f'q_{lab}'], out[f'v_{lab}'] = g[:, 0].mean(1), g[:, 1].mean(1)
            if K > 1:
                out[f'v_var_{lab}'] = g[:, 1].var(1)
                out[f'a_var_{lab}'] = (g[:, 0] - g[:, 1]).var(1)  # paired: noise of A_MC is a_var / K
        return out

    # ------------------------------------------------------------------ gradients
    def group_grads(self, actor, d: dict, keys: tuple[str, ...], chunk: int = 64) -> dict[str, torch.Tensor]:
        """``[G, P]`` flat surrogate gradients per group of ``group_envs`` worlds, per advantage key.

        Each advantage is standardized over the whole collected batch first, as the buffer does.
        """
        names = [n for n, _ in actor.named_parameters()]
        params = {n: p.detach() for n, p in actor.named_parameters()}
        assert set(names) == {n for n in names if n.startswith('mean.')} | {'log_std'}, 'Gaussian actor expected'

        def surrogate(p, obs, act, logp, adv):
            mean = functional_call(actor.mean, {n[5:]: v for n, v in p.items() if n.startswith('mean.')}, (obs,))
            log_std = p['log_std']
            lp = (-0.5 * ((act - mean) / torch.exp(log_std)) ** 2 - log_std - _HALF_LOG_2PI).sum(-1)
            return (torch.exp(lp - logp) * adv).mean()

        gfn = vmap(grad(surrogate), in_dims=(None, 0, 0, 0, 0))
        T, R = self.T, self.R
        G = R // self.group_envs

        def grouped(x):  # [T, R, ...] -> [G, group_envs * T, ...], env-major like the buffer
            x = x.transpose(0, 1).reshape(R * T, *x.shape[2:])
            return x.reshape(G, self.group_envs * T, *x.shape[1:])

        obs, act, logp = grouped(d['obs']), grouped(d['act']), grouped(d['logp'])
        out = {}
        for key in keys:
            adv = d[key].flatten()
            if key in ('adv_r', 'mc_r') and self.std_r:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            elif key in ('adv_c', 'mc_c') and self.std_c:
                adv = adv - adv.mean()
            adv = grouped(adv.reshape(T, R))
            parts = []
            for s in range(0, G, chunk):
                gd = gfn(params, obs[s:s + chunk], act[s:s + chunk], logp[s:s + chunk], adv[s:s + chunk])
                parts.append(torch.cat([gd[n].reshape(gd[n].shape[0], -1) for n in names], dim=1))
            out[key] = torch.cat(parts)
        return out


def gradient_report(
    gg: dict[str, torch.Tensor],
    hinv,
    train_g: torch.Tensor,
    train_b: torch.Tensor,
    group_rows: int,
    sizes: tuple[int, ...] = (2_000, 5_000, 20_000, 100_000, 500_000),
    draws: int = 8,
) -> dict[str, float]:
    """Alignment and estimate-quality numbers from per-group gradients (see the module docstring).

    ``train_g``/``train_b`` are the training batch's own (ascent) gradients; the probe's samples are
    independent of them.
    """
    out: dict[str, float] = {}
    G = next(iter(gg.values())).shape[0]
    half = G // 2

    def fisher_cos(u, v):
        hv = hinv(v)
        hu = hinv(u)
        return float(u.dot(hv) / torch.sqrt(u.dot(hu) * v.dot(hv)).clamp_min(1e-30))

    pairs = [('adv_r', 'adv_c', '')]
    if 'mc_r' in gg:
        pairs.append(('mc_r', 'mc_c', 'MC'))
    full = {k: v.mean(0) for k, v in gg.items()}
    for kr, kc, tag in pairs:
        g, b = full[kr], full[kc]
        out[f'TrueGrad/Cos{tag}'] = _cos(g, b)
        out[f'TrueGrad/FisherCos{tag}'] = fisher_cos(g, b)
        # convergence of the reference itself: two independent halves
        out[f'TrueGrad/HalfAgreeR{tag}'] = _cos(gg[kr][:half].mean(0), gg[kr][half:].mean(0))
        out[f'TrueGrad/HalfAgreeC{tag}'] = _cos(gg[kc][:half].mean(0), gg[kc][half:].mean(0))
    # the training batch's estimates against the truth
    out['TrueGrad/TrainVsTrueR'] = _cos(train_g, full['adv_r'])
    out['TrueGrad/TrainVsTrueC'] = _cos(train_b, full['adv_c'])
    out['TrueGrad/TrainCos'] = _cos(train_g, train_b)
    if 'mc_r' in gg:
        out['TrueGrad/CriticVsMCR'] = _cos(full['adv_r'], full['mc_r'])
        out['TrueGrad/CriticVsMCC'] = _cos(full['adv_c'], full['mc_c'])
        out['TrueGrad/TrainVsMCR'] = _cos(train_g, full['mc_r'])
        out['TrueGrad/TrainVsMCC'] = _cos(train_b, full['mc_c'])
        # the same in the Fisher metric CPO steps in: natural gradients H^-1 g
        out['TrueGrad/NatTrainVsMCR'] = _cos(hinv(train_g), hinv(full['mc_r']))
        out['TrueGrad/NatTrainVsMCC'] = _cos(hinv(train_b), hinv(full['mc_c']))
        out['TrueGrad/NatCriticVsMCR'] = _cos(hinv(full['adv_r']), hinv(full['mc_r']))
        out['TrueGrad/NatCriticVsMCC'] = _cos(hinv(full['adv_c']), hinv(full['mc_c']))
    # batch-size curve: estimate from k groups vs the mean of all *other* groups
    gen = torch.Generator(device='cpu').manual_seed(0)
    for n in sizes:
        k = max(1, n // group_rows)
        if k > half:
            continue
        for key, lab in (('adv_r', 'R'), ('adv_c', 'C')):
            cs = []
            tot = gg[key].sum(0)
            for _ in range(draws):
                idx = torch.randperm(G, generator=gen)[:k].to(gg[key].device)
                est = gg[key][idx].sum(0)
                rest = (tot - est) / (G - k)
                cs.append(_cos(est / k, rest))
            out[f'TrueGrad/BatchCos{lab}_{n}'] = sum(cs) / len(cs)
            if f'mc_{key[-1]}' in gg:
                # the training estimator on n samples vs the critic-free truth from the *other* groups
                ref = gg[f'mc_{key[-1]}']
                rtot = ref.sum(0)
                cs = []
                for _ in range(draws):
                    idx = torch.randperm(G, generator=gen)[:k].to(gg[key].device)
                    cs.append(_cos(gg[key][idx].mean(0), (rtot - ref[idx].sum(0)) / (G - k)))
                out[f'TrueGrad/BatchVsMC{lab}_{n}'] = sum(cs) / len(cs)
    return out


def _flat_grad(actor, obs, act, logp, adv) -> torch.Tensor:
    """Ascent gradient of ``mean(ratio * adv)`` at the current parameters (ratio = 1)."""
    with torch.enable_grad():
        actor.zero_grad()
        mean = actor.mean(obs)
        lp = (-0.5 * ((act - mean) / torch.exp(actor.log_std)) ** 2 - actor.log_std - _HALF_LOG_2PI).sum(-1)
        (torch.exp(lp - logp) * adv).mean().backward()
        out = torch.cat([p.grad.reshape(-1) for _, p in actor.named_parameters()])
        actor.zero_grad()
    return out


ORACLE_KEYS = [f'Oracle/{m}{X}' for X in ('R', 'C') for m in ('Oracle', 'Learned', 'Rollout', 'LearnedVsOracle')]


def oracle_report(d: dict, actor, gg: dict[str, torch.Tensor], std_r: bool, std_c: bool) -> dict[str, float]:
    """Gradients on ONE batch of (state, action) samples that differ only in the advantage used.

    * ``Oracle``: one paired common-random-number rollout per sample, ``A = Q_1 - V_1`` -- unbiased
      for the true advantage, so this is what the batch would give with a perfect critic (plus the
      oracle's own rollout noise, which only adds variance).
    * ``Learned``: the training estimator's advantage (critic-based GAE / TD(0) / ...).
    * ``Rollout``: REINFORCE with the critic as baseline, ``G - V`` from the single trajectory.

    Each is normalized over the batch the way the buffer does (reward mean/std, cost mean only)
    and scored by its cosine with the critic-free truth (all probe groups).
    """
    o = d['oracle']
    t, w = o['t'], o['world']
    obs, act, logp = d['obs'][t, w], d['act'][t, w], d['logp'][t, w]
    out = {}
    for lab, X, std in (('r', 'R', std_r), ('c', 'C', std_c)):
        truth = gg[f'mc_{lab}'].mean(0)

        def norm(a):
            a = a - a.mean()
            return a / (a.std() + 1e-8) if (std and lab == 'r') else a

        advs = {
            'Oracle': o[f'q_{lab}'] - o[f'v_{lab}'],
            'Learned': d[f'adv_{lab}'][t, w],
            'Rollout': d[f'rf_{lab}'][t, w] - d[f'v{lab}'][t, w],
        }
        grads = {k: _flat_grad(actor, obs, act, logp, norm(a)) for k, a in advs.items()}
        for k, g in grads.items():
            out[f'Oracle/{k}{X}'] = _cos(g, truth)
        out[f'Oracle/LearnedVsOracle{X}'] = _cos(grads['Learned'], grads['Oracle'])
    return out


COMBINED_KEYS = ['TrueGrad/CombTrainVsMC', 'TrueGrad/CombNatTrainVsMC', 'TrueGrad/CombCriticVsMC',
                 'TrueGrad/CombBatchVsMC_20000', 'TrueGrad/CombTruthAcc', 'TrueGrad/CombVsRewardMC']


def combined_report(gg: dict[str, torch.Tensor], hinv, train_g, train_b, lam: float, n_batch: int = 20_000,
                    group_rows: int | None = None, draws: int = 8) -> dict[str, float]:
    """The direction a Lagrangian method actually follows, ``(g - lam * b) / (1 + lam)``, vs its truth.

    ``CombTruthAcc`` = estimated cos(truth estimate, exact truth) from its own split half:
    with c = cos(half A, half B), cos(full, exact) = 1 / sqrt(1 + (1/c - 1) / 2).
    """
    def comb(r, c):
        return (r - lam * c) / (1 + lam)

    G = gg['mc_r'].shape[0]
    truth_g = comb(gg['mc_r'], gg['mc_c'])  # [G, P] per-group combined truth
    est_g = comb(gg['adv_r'], gg['adv_c'])
    truth = truth_g.mean(0)
    out = {
        'TrueGrad/CombTrainVsMC': _cos(comb(train_g, train_b), truth),
        'TrueGrad/CombNatTrainVsMC': _cos(hinv(comb(train_g, train_b)), hinv(truth)),
        'TrueGrad/CombCriticVsMC': _cos(est_g.mean(0), truth),
        'TrueGrad/CombVsRewardMC': _cos(truth, gg['mc_r'].mean(0)),
    }
    half = G // 2
    c = _cos(truth_g[:half].mean(0), truth_g[half:].mean(0))
    out['TrueGrad/CombTruthAcc'] = (1 + (1 / c - 1) / 2) ** -0.5 if c > 0 else float('nan')
    if group_rows:
        k = max(1, n_batch // group_rows)
        gen = torch.Generator(device='cpu').manual_seed(2)
        tot = truth_g.sum(0)
        cs = []
        for _ in range(draws):
            idx = torch.randperm(G, generator=gen)[:k].to(truth_g.device)
            cs.append(_cos(est_g[idx].mean(0), (tot - truth_g[idx].sum(0)) / (G - k)))
        out['TrueGrad/CombBatchVsMC_20000'] = sum(cs) / len(cs)
    return out


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-30))


def advantage_report(d: dict) -> dict[str, float]:
    """Learned value/advantage vs the nested Monte-Carlo reference at the anchor samples.

    ``A_MC = Q_MC - V_MC`` carries rollout noise of variance ``var(Q_i - V_i) / K`` per anchor (paired);
    it is independent of the probe trajectory's own estimate, so it only *attenuates* correlations.
    ``Ceil`` is the best correlation any estimator could reach against the noisy reference, and
    ``Corr / Ceil`` (``CorrC``) is the noise-corrected correlation with the true advantage.
    """
    a = d['anchor']
    out: dict[str, float] = {}
    t, w = a['t'], a['world']
    for lab, X in (('r', 'R'), ('c', 'C')):
        q, v = a[f'q_{lab}'], a[f'v_{lab}']
        k = float(a['K'])
        na, nv = a[f'a_var_{lab}'] / k, a[f'v_var_{lab}'] / k
        A = q - v
        est = d[f'adv_{lab}'][t, w]
        V = d[f'v{lab}'][t, w]
        G1 = d[f'rf_{lab}'][t, w] - V  # single-rollout REINFORCE advantage with the critic baseline
        # value
        tv = float(v.var() - nv.mean())
        out[f'AdvQ/VCorr{X}'] = _pearson(V, v) / max(tv / float(v.var()), 1e-12) ** 0.5 if tv > 0 else float('nan')
        out[f'AdvQ/VEV{X}'] = 1.0 - float(((V - v) ** 2).mean() - nv.mean()) / tv if tv > 0 else float('nan')
        out[f'AdvQ/VBias{X}'] = float((V - v).mean()) / tv ** 0.5 if tv > 0 else float('nan')
        out[f'AdvQ/VMean{X}'] = float(v.mean())
        # advantage
        ta = float(A.var() - na.mean())
        ceil = (ta / float(A.var())) ** 0.5 if ta > 0 else 0.0
        out[f'AdvQ/Ceil{X}'] = ceil
        out[f'AdvQ/Corr{X}'] = _pearson(est, A)
        out[f'AdvQ/CorrC{X}'] = _pearson(est, A) / ceil if ceil > 0.05 else float('nan')
        out[f'AdvQ/CorrG{X}'] = _pearson(G1, A) / ceil if ceil > 0.05 else float('nan')
        cov = float(((est - est.mean()) * (A - A.mean())).mean())
        out[f'AdvQ/Slope{X}'] = cov / float(est.var())  # E[A | A_hat] = slope * A_hat (+c): 1 = right scale
        out[f'AdvQ/StdRatio{X}'] = float(est.std()) / ta ** 0.5 if ta > 0 else float('nan')
        # bias of the estimate, per sample: E[A_hat - A] after removing the batch mean (constant offsets do not matter)
        out[f'AdvQ/ErrEV{X}'] = 1.0 - float((((est - est.mean()) - (A - A.mean())) ** 2).mean() - na.mean()) / ta if ta > 0 else float('nan')
        out[f'AdvQ/TrueStd{X}'] = ta ** 0.5 if ta > 0 else float('nan')
    return out


ADV_KEYS = [f'AdvQ/{m}{X}' for X in ('R', 'C') for m in
            ('VCorr', 'VEV', 'VBias', 'VMean', 'Ceil', 'Corr', 'CorrC', 'CorrG', 'Slope', 'StdRatio', 'ErrEV', 'TrueStd')]


#: Sample counts (rows) for the REINFORCE split-half curve; sizes above half the probe are skipped.
SPLIT_SIZES = (2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000,
               5_000_000, 10_000_000)


def split_half_curve(gg: dict[str, torch.Tensor], group_rows: int, draws: int = 8) -> dict[str, float]:
    """cos between two gradient estimates from *disjoint* sets of N samples, for every N.

    Both estimates are unbiased for the same gradient, so their cosine reaches 1 exactly when N is
    large enough for the estimate to be the true gradient. Keys: ``RF`` = REINFORCE (raw
    reward-/cost-to-go), ``RFb`` = the same with the critic as baseline.
    """
    out: dict[str, float] = {}
    gen = torch.Generator(device='cpu').manual_seed(1)
    for key, lab in (('rf_r', 'RFR'), ('rf_c', 'RFC'), ('mc_r', 'RFbR'), ('mc_c', 'RFbC')):
        if key not in gg:
            continue
        x = gg[key]
        G = x.shape[0]
        for n in SPLIT_SIZES:
            k = max(1, n // group_rows)
            if 2 * k > G:
                continue
            cs = []
            for _ in range(draws):
                idx = torch.randperm(G, generator=gen).to(x.device)
                cs.append(_cos(x[idx[:k]].mean(0), x[idx[k:2 * k]].mean(0)))
            out[f'TrueGrad/Half{lab}_{n}'] = sum(cs) / len(cs)
        out[f'TrueGrad/Norm{lab}'] = float(x.mean(0).norm())
    return out


def report_keys(mc: bool, group_rows: int, sizes=(2_000, 5_000, 20_000, 100_000, 500_000), num_rows: int = 0,
                adv: bool = False, oracle: bool = False):
    keys = ['TrueGrad/Cos', 'TrueGrad/FisherCos', 'TrueGrad/HalfAgreeR', 'TrueGrad/HalfAgreeC',
            'TrueGrad/TrainVsTrueR', 'TrueGrad/TrainVsTrueC', 'TrueGrad/TrainCos', 'TrueGrad/Seconds']
    if mc:
        keys += ['TrueGrad/CosMC', 'TrueGrad/FisherCosMC', 'TrueGrad/HalfAgreeRMC', 'TrueGrad/HalfAgreeCMC',
                 'TrueGrad/CriticVsMCR', 'TrueGrad/CriticVsMCC', 'TrueGrad/TrainVsMCR', 'TrueGrad/TrainVsMCC',
                 'TrueGrad/NatTrainVsMCR', 'TrueGrad/NatTrainVsMCC', 'TrueGrad/NatCriticVsMCR', 'TrueGrad/NatCriticVsMCC']
    if adv:
        keys += ADV_KEYS
    if oracle:
        keys += ORACLE_KEYS
    G = num_rows // group_rows
    if mc:
        for lab in ('RFR', 'RFC', 'RFbR', 'RFbC'):
            keys.append(f'TrueGrad/Norm{lab}')
            keys += [f'TrueGrad/Half{lab}_{n}' for n in SPLIT_SIZES if 2 * max(1, n // group_rows) <= G]
    for n in sizes:
        if max(1, n // group_rows) <= G // 2:
            keys += [f'TrueGrad/BatchCosR_{n}', f'TrueGrad/BatchCosC_{n}']
            if mc:
                keys += [f'TrueGrad/BatchVsMCR_{n}', f'TrueGrad/BatchVsMCC_{n}']
    return keys
