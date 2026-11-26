"""Safety-Gymnasium on the GPU (``safety_gymnasium_gpu``) as an omnisafe CMDP.

Selected per run with ``env_cfgs.class_name = SafetyGymnasiumGPUEnv``; the env ids are the same as
:class:`~omnisafe.envs.safety_gymnasium_env.SafetyGymnasiumEnv`'s, and since that class is
registered first, every run that does not ask for this one is routed exactly as before.

``num_envs`` worlds are stepped in lockstep by mujoco_warp; the env resets finished worlds itself
and handles the time limit (``need_auto_reset_wrapper`` / ``need_time_limit_wrapper`` are False,
as for omnisafe's Isaac Gym env), reporting the observation that ended an episode in
``info['final_observation']`` with ``info['_final_observation']`` as the per-world mask -- the
shape omnisafe's ``ObsNormalize`` and on-policy adapter expect from a vector env.

Requires the ``safety_gpu`` python environment (MuJoCo 3.x + mujoco_warp); see
``envs/safety_gymnasium_gpu/README.md``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from omnisafe.envs.core import CMDP, env_register
from omnisafe.envs.safety_gymnasium_env import SafetyGymnasiumEnv
from omnisafe.typing import DEVICE_CPU


@env_register
class SafetyGymnasiumGPUEnv(CMDP):
    """Batched GPU Safety-Gymnasium.

    Keyword Args:
        sim_device (str): CUDA device for the simulation; defaults to ``device`` if that is a
            CUDA device, else ``'cuda:0'``. Outputs are moved to ``device``.
        hazards_num, lidar_num_bins: Same meaning as for :class:`SafetyGymnasiumEnv`.
        Anything else is passed to ``safety_gymnasium.make`` (e.g. ``config``).
    """

    need_auto_reset_wrapper: bool = False
    need_time_limit_wrapper: bool = False
    need_evaluation: bool = False
    _support_envs: ClassVar[list[str]] = list(SafetyGymnasiumEnv._support_envs)  # noqa: SLF001

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_id)
        import safety_gymnasium_gpu as sgg  # noqa: PLC0415  (only importable in the GPU env)

        self._num_envs = int(num_envs)
        self._device = torch.device(device)
        sim_device = kwargs.pop('sim_device', None)
        if sim_device is None:
            sim_device = str(self._device) if self._device.type == 'cuda' else 'cuda:0'
        hazards_num = kwargs.pop('hazards_num', None)
        if hazards_num is not None:
            config = dict(kwargs.pop('config', None) or {})
            config['hazards_num'] = int(hazards_num)
            kwargs['config'] = config
        seed = int(kwargs.pop('seed', 0))
        self._env = sgg.make(env_id, num_envs=self._num_envs, device=sim_device, seed=seed, autoreset=True, **kwargs)
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space
        self._metadata = {}

    def _out(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self._device, dtype=torch.float32)
        return x[0] if self._num_envs == 1 else x  # Unsqueeze re-adds the batch dim for 1 env

    def step(self, action: torch.Tensor):
        act = torch.as_tensor(action, dtype=torch.float32).detach().to(self._env.device)
        obs, reward, cost, terminated, truncated, info = self._env.step(act.reshape(self._num_envs, -1))
        out_info: dict[str, Any] = {}
        if 'final_obs' in info:
            mask = info['final_mask'].to(self._device)
            out_info['final_observation'] = self._out(info['final_obs'])
            out_info['_final_observation'] = mask[0] if self._num_envs == 1 else mask
        return (
            self._out(obs), self._out(reward), self._out(cost),
            self._out(terminated), self._out(truncated), out_info,
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._env.gen.manual_seed(int(seed))
        obs, info = self._env.reset()
        return self._out(obs), info

    @property
    def max_episode_steps(self) -> int:
        return int(self._env.num_steps)

    @property
    def gpu_env(self):
        """The underlying ``safety_gymnasium_gpu`` env (batched torch API)."""
        return self._env

    def set_seed(self, seed: int) -> None:
        self.reset(seed=seed)

    def render(self) -> Any:
        raise NotImplementedError('use envs/safety_gymnasium_gpu/examples/render_gif.py')

    def close(self) -> None:
        pass
