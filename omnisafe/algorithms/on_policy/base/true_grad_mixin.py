"""Brute-force reference gradients/advantages for trust-region algorithms (GPU env only).

Shared by CPO and TRPOPID: :class:`TrueGradMixin` owns the :class:`TrueGradientProbe` (a separate
GPU env stepped in lockstep with training) and logs the ``TrueGrad/*`` and ``AdvQ/*`` keys of
:mod:`omnisafe.utils.true_gradient` given the training batch's own reward (ascent) gradient ``g``
and cost gradient ``b``. Enabled by ``algo_cfgs.true_grad_worlds > 0``.
"""

from __future__ import annotations

import time

import torch

from omnisafe.utils import true_gradient
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import get_flat_gradients_from


class TrueGradMixin:
    """Mix into a TRPO-family algorithm (needs ``self._fvp`` and ``self._actor_critic``)."""

    #: extra keys this algorithm logs alongside (e.g. the combined Lagrangian direction)
    _TRUE_GRAD_EXTRA_KEYS: tuple[str, ...] = ()

    def _true_grad_enabled(self) -> bool:
        return bool(getattr(self._cfgs.algo_cfgs, 'true_grad_worlds', 0))

    def _register_true_grad_keys(self) -> None:
        a = self._cfgs.algo_cfgs
        if not self._true_grad_enabled():
            return
        n = int(a.true_grad_worlds)
        frag = a.steps_per_epoch // self._cfgs.train_cfgs.vector_env_nums
        mc = bool(getattr(a, 'true_grad_mc', True))
        for key in true_gradient.report_keys(
            mc, int(a.true_grad_group_envs) * frag, num_rows=n * frag,
            adv=bool(getattr(a, 'true_grad_adv_repeats', 0)),
            oracle=bool(getattr(a, 'true_grad_oracle_samples', 0)),
        ):
            self._logger.register_key(key)
        if mc:
            for key in self._TRUE_GRAD_EXTRA_KEYS:
                self._logger.register_key(key)

    def _true_grad_probe(self) -> true_gradient.TrueGradientProbe:
        probe = getattr(self, '_tg_probe', None)
        if probe is None:
            from omnisafe.envs.core import make as make_env  # noqa: PLC0415
            from omnisafe.envs.wrapper import ActionScale, ObsNormalize  # noqa: PLC0415

            a = self._cfgs.algo_cfgs
            env_cfgs = self._cfgs.env_cfgs.todict()
            # A seed of its own: the reference worlds must be independent of the training worlds.
            env_cfgs['seed'] = 1_000_003 + int(self._seed)

            def mk(n):
                env = make_env(self._env_id, num_envs=n, device=self._device, **env_cfgs)
                if a.obs_normalize:
                    env = ObsNormalize(env, device=self._device, update_stats=False)
                return ActionScale(env, low=-1.0, high=1.0, device=self._device)

            probe = true_gradient.TrueGradientProbe(
                mk,
                num_worlds=int(a.true_grad_worlds),
                frag_len=a.steps_per_epoch // self._cfgs.train_cfgs.vector_env_nums,
                algo_cfgs=a,
                device=self._device,
                group_envs=int(a.true_grad_group_envs),
                mc=bool(getattr(a, 'true_grad_mc', True)),
                seed=1_000_003 + int(self._seed),
                adv_repeats=int(getattr(a, 'true_grad_adv_repeats', 0)),
                oracle_samples=int(getattr(a, 'true_grad_oracle_samples', 0)),
            )
            self._tg_probe = probe
        return probe

    def _log_true_gradient(self, grads: torch.Tensor, b_grads: torch.Tensor, penalty: float | None = None) -> None:
        """Reference gradients at the current parameters; call at theta_old, before the step.

        ``grads``: reward surrogate gradient (ascent); ``b_grads``: cost surrogate gradient.
        ``penalty``: Lagrange multiplier, to also score the combined direction ``(g - l b)/(1 + l)``.
        """
        t0 = time.time()
        probe = self._true_grad_probe()
        devices = [self._device] if self._device.type == 'cuda' else []
        # Fork the RNG: the probe's action sampling must not shift the training run's random stream.
        with torch.random.fork_rng(devices=devices):
            data = probe.collect(self._actor_critic, self._env._env)  # noqa: SLF001
            keys = ('adv_r', 'adv_c') + (('mc_r', 'mc_c', 'rf_r', 'rf_c') if probe.cont is not None else ())
            gg = probe.group_grads(self._actor_critic.actor, data, keys)
        hinv = lambda v: conjugate_gradients(self._fvp, v, self._cfgs.algo_cfgs.cg_iters)  # noqa: E731
        stats = true_gradient.gradient_report(
            gg, hinv=hinv, train_g=grads, train_b=b_grads, group_rows=probe.group_envs * probe.T,
        )
        if probe.cont is not None:
            stats.update(true_gradient.split_half_curve(gg, probe.group_envs * probe.T))
            if penalty is not None:
                stats.update(true_gradient.combined_report(
                    gg, hinv, grads, b_grads, float(penalty), group_rows=probe.group_envs * probe.T))
        if 'anchor' in data:
            stats.update(true_gradient.advantage_report(data))
        if 'oracle' in data:
            stats.update(true_gradient.oracle_report(data, self._actor_critic.actor, gg, probe.std_r, probe.std_c))
        stats['TrueGrad/Seconds'] = time.time() - t0
        self._logger.store(stats)

    def _flat_surrogate_grad(self, obs, act, logp, adv) -> torch.Tensor:
        """Flat gradient of the surrogate ``E[ratio * adv]`` (ascent direction), without logging."""
        self._actor_critic.actor.zero_grad()
        self._actor_critic.actor(obs)
        ratio = torch.exp(self._actor_critic.actor.log_prob(act) - logp)
        (ratio * adv).mean().backward()
        out = get_flat_gradients_from(self._actor_critic.actor)
        self._actor_critic.actor.zero_grad()
        return out
