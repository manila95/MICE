# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the Deep Deterministic Policy Gradient algorithm."""

from __future__ import annotations

import time
from typing import Any

from tqdm import tqdm

import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.adapter import OffPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.envs.core import make as make_env
from omnisafe.envs.wrapper import ActionScale, AutoReset, ObsNormalize, TimeLimit, Unsqueeze
from omnisafe.utils.critic_ensemble import aggregate as aggregate_critic_ensemble
from omnisafe.utils.eval_data_dump import log_eval_data_to_wandb, save_eval_data
from omnisafe.utils.value_eval import estimate_true_q_value_same_state_mc
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class DDPG(BaseAlgo):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        - Title: Continuous control with deep reinforcement learning
        - Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess,
            Tom Erez, Yuval Tassa, David Silver, Daan Wierstra.
        - URL: `DDPG <https://arxiv.org/abs/1509.02971>`_
    """

    _epoch: int

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OffPolicyAdapter` to adapt the environment to this
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
            AssertionError: If the total number of steps is not divisible by the number of steps per
                epoch.
        """
        self._env: OffPolicyAdapter = OffPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (
            self._cfgs.algo_cfgs.steps_per_epoch % self._cfgs.train_cfgs.vector_env_nums == 0
        ), 'The number of steps per epoch is not divisible by the number of environments.'

        assert (
            int(self._cfgs.train_cfgs.total_steps) % self._cfgs.algo_cfgs.steps_per_epoch == 0
        ), 'The total number of steps is not divisible by the number of steps per epoch.'
        self._epochs: int = int(
            self._cfgs.train_cfgs.total_steps // self._cfgs.algo_cfgs.steps_per_epoch,
        )
        self._epoch: int = 0
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch // self._cfgs.train_cfgs.vector_env_nums
        )

        self._update_cycle: int = self._cfgs.algo_cfgs.update_cycle
        assert (
            self._steps_per_epoch % self._update_cycle == 0
        ), 'The number of steps per epoch is not divisible by the number of steps per sample.'
        self._samples_per_epoch: int = self._steps_per_epoch // self._update_cycle
        self._update_count: int = 0

    def _init_model(self) -> None:
        """Initialize the model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_q_critic.ConstraintActorQCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorQCritic()
        """
        self._cfgs.model_cfgs.critic['num_critics'] = 1
        self._actor_critic: ConstraintActorQCritic = ConstraintActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        self._buf: VectorOffPolicyBuffer = VectorOffPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._cfgs.algo_cfgs.size,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            penalty_coefficient=self._cfgs.algo_cfgs.get('penalty_coefficient', 0.0),
            device=self._device,
        )

        use_sr = bool(self._cfgs.model_cfgs.get('use_successor_representation', False))
        self._sr_td_ridge: bool = (
            use_sr and self._cfgs.model_cfgs.sr_cfgs.get('sr_mode', 'shared_trunk') == 'td_ridge'
        )
        # How the td_ridge read-out weights w_r / w_c are learned: 'ridge' (closed-form each
        # batch) or 'sgd' (gradient descent over the replay buffer).
        self._sr_readout: str = (
            self._cfgs.model_cfgs.sr_cfgs.get('readout', 'ridge') if self._sr_td_ridge else 'ridge'
        )

    def _init_log(self) -> None:
        """Log info about epoch.

        +-------------------------+----------------------------------------------------------------------+
        | Things to log           | Description                                                          |
        +=========================+======================================================================+
        | Train/Epoch             | Current epoch.                                                       |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpCost          | Average cost of the epoch.                                           |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpRet           | Average return of the epoch.                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpLen           | Average length of the epoch.                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpCost      | Average cost of the evaluate epoch.                                  |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpRet       | Average return of the evaluate epoch.                                |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpLen       | Average length of the evaluate epoch.                                |
        +-------------------------+----------------------------------------------------------------------+
        | Value/reward_critic     | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-------------------------+----------------------------------------------------------------------+
        | Values/cost_critic      | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi            | Loss of the policy network.                                          |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_reward_critic | Loss of the reward critic.                                           |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic   | Loss of the cost critic network.                                     |
        +-------------------------+----------------------------------------------------------------------+
        | Train/LR                | Learning rate of the policy network.                                 |
        +-------------------------+----------------------------------------------------------------------+
        | Misc/Seed               | Seed of the experiment.                                              |
        +-------------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps      | Total steps of the experiment.                                       |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Total              | Total time.                                                          |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Rollout            | Rollout time.                                                        |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Update             | Update time.                                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Evaluate           | Evaluate time.                                                       |
        +-------------------------+----------------------------------------------------------------------+
        | FPS                     | Frames per second of the epoch.                                      |
        +-------------------------+----------------------------------------------------------------------+
        """
        self._logger: Logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        what_to_save['pi'] = self._actor_critic.actor
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer

        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key(
            'Metrics/EpRet',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/EpCost',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/EpLen',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        # Running totals over the whole run. No window_length: these are monotone, so a
        # moving average of them would lag the true total rather than smooth anything.
        self._logger.register_key('Metrics/TotalCost')
        self._logger.register_key('Metrics/TotalViolations')
        self._logger.register_key('Metrics/ViolationRate')

        if getattr(self._cfgs.algo_cfgs, 'test_estimate', False):
            # Registered here, not when the first evaluation produces them: the progress.csv
            # header is written at the first dump_tabular, so a column added later is lost.
            if getattr(self._cfgs.algo_cfgs, 'mc_value_study', True):
                for key in self._mc_study_keys():
                    self._logger.register_key(key)
            else:
                for key in self._value_eval_keys() + self._value_eval_extra_keys():
                    self._logger.register_key(key)

        if self._cfgs.train_cfgs.eval_episodes > 0:
            self._logger.register_key(
                'Metrics/TestEpRet',
                window_length=self._cfgs.logger_cfgs.window_lens,
            )
            self._logger.register_key(
                'Metrics/TestEpCost',
                window_length=self._cfgs.logger_cfgs.window_lens,
            )
            self._logger.register_key(
                'Metrics/TestEpLen',
                window_length=self._cfgs.logger_cfgs.window_lens,
            )

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/LR')

        self._logger.register_key('TotalEnvSteps')

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward_critic')

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost_critic')

        if self._sr_td_ridge:
            # log information about the td_ridge successor-representation critic
            self._logger.register_key('Loss/Loss_sr', delta=True)
            self._logger.register_key('Misc/RidgeResidualReward')
            self._logger.register_key('Misc/RidgeResidualCost')
            self._logger.register_key('Misc/WrNorm')
            self._logger.register_key('Misc/WcNorm')
            self._logger.register_key('Misc/GramCond')
            self._logger.register_key('Misc/GramCondCost')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Evaluate')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')
        # register environment specific keys
        for env_spec_key in self._env.env_spec_keys:
            self.logger.register_key(env_spec_key)

    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: average episode return in final epoch.
            ep_cost: average episode cost in final epoch.
            ep_len: average episode length in final epoch.
        """
        self._logger.log('INFO: Start training')
        start_time = time.time()
        step = 0
        for epoch in range(self._epochs):
            self._epoch = epoch
            rollout_time = 0.0
            update_time = 0.0
            epoch_time = time.time()

            pbar = tqdm(
                range(epoch * self._samples_per_epoch, (epoch + 1) * self._samples_per_epoch),
                desc=f'Epoch {epoch}',
                unit='step',
                leave=False,
            )
            for sample_step in pbar:
                step = sample_step * self._update_cycle * self._cfgs.train_cfgs.vector_env_nums

                rollout_start = time.time()
                # set noise for exploration
                if self._cfgs.algo_cfgs.use_exploration_noise:
                    self._actor_critic.actor.noise = self._cfgs.algo_cfgs.exploration_noise

                # collect data from environment
                self._env.rollout(
                    rollout_step=self._update_cycle,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                    use_rand_action=(step <= self._cfgs.algo_cfgs.start_learning_steps),
                )
                rollout_time += time.time() - rollout_start

                # update parameters
                update_start = time.time()
                if step > self._cfgs.algo_cfgs.start_learning_steps:
                    self._update()
                # if we haven't updated the network, log 0 for the loss
                else:
                    self._log_when_not_update()
                update_time += time.time() - update_start

                pbar.set_postfix(
                    rollout=f'{rollout_time:.1f}s',
                    update=f'{update_time:.1f}s',
                    phase='warmup' if step <= self._cfgs.algo_cfgs.start_learning_steps else 'train',
                )

            eval_start = time.time()
            self._env.eval_policy(
                episode=self._cfgs.train_cfgs.eval_episodes,
                agent=self._actor_critic,
                logger=self._logger,
            )
            self._eval_value_function(epoch)
            eval_time = time.time() - eval_start
            print(f'Epoch {epoch} done — rollout: {rollout_time:.1f}s  update: {update_time:.1f}s  eval: {eval_time:.1f}s')

            self._logger.store({'Time/Update': update_time})
            self._logger.store({'Time/Rollout': rollout_time})
            self._logger.store({'Time/Evaluate': eval_time})

            if (
                step > self._cfgs.algo_cfgs.start_learning_steps
                and self._cfgs.model_cfgs.linear_lr_decay
            ):
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': step + 1,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': self._actor_critic.actor_scheduler.get_last_lr()[0],
                    **self._env.safety_totals,
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len

    @staticmethod
    def _value_eval_keys() -> tuple[str, ...]:
        """Names for the flat tuple :func:`estimate_true_q_value` returns, in its order.

        It reports only to wandb, so without this the evaluation runs, pays for its rollouts
        and discards every number it computed.
        """
        return (
            'Eval_s0/EstimationError_c', 'Eval_s0/true_value_c', 'Eval_s0/estimate_value_c',
            'Eval_s0/Correlation_c',
            'Eval_s0/EstimationError_r', 'Eval_s0/true_value_r', 'Eval_s0/estimate_value_r',
            'Eval_s0/Correlation_r',
            'Eval_all/EstimationError_c', 'Eval_all/true_value_c', 'Eval_all/estimate_value_c',
            'Eval_all/Correlation_c',
            'Eval_all/EstimationError_r', 'Eval_all/true_value_r', 'Eval_all/estimate_value_r',
            'Eval_all/Correlation_r',
        )

    @staticmethod
    def _value_eval_extra_keys() -> tuple[str, ...]:
        """Rank correlations, computed beside the Pearson ones in :mod:`value_eval`."""
        return (
            'Eval_s0/Spearman_c', 'Eval_s0/Spearman_r',
            'Eval_all/Spearman_c', 'Eval_all/Spearman_r',
        )

    def _is_value_eval_epoch(self, epoch: int) -> bool:
        """Whether an evaluation falls inside the epoch that just finished.

        The schedule counts **environment steps**, not epochs, so it does not shift when
        ``steps_per_epoch`` or ``vector_env_nums`` change -- an epoch is 2000 env steps at one
        environment and still 2000 at five, but an epoch-based cadence would silently evaluate
        at different points in training if either were retuned.

        Dense early, sparse later: ``early_eval_step_freq`` up to ``early_eval_steps``, then
        ``value_eval_step_freq``. The critics move fastest at the start, which is where the
        calibration is most worth watching and least expensive to measure.

        An evaluation fires when the epoch just finished *crossed* a multiple of the interval,
        rather than when the step count divides it exactly, so an interval that is not a whole
        number of epochs still fires once per interval instead of never.

        Args:
            epoch (int): Index of the epoch that has just finished.

        Returns:
            True when this epoch crossed an evaluation boundary.
        """
        cfgs = self._cfgs.algo_cfgs
        # Epoch-based keys still win if a config sets them, so existing runs are unaffected.
        if getattr(cfgs, 'value_eval_freq', None) is not None and getattr(
            cfgs, 'use_step_eval_schedule', True
        ) is False:
            freq = int(cfgs.value_eval_freq)
            early = int(getattr(cfgs, 'early_eval_freq', 5))
            span = int(getattr(cfgs, 'early_eval_epochs', 100))
            return epoch % (early if epoch < span else freq) == 0

        early_steps = int(getattr(cfgs, 'early_eval_steps', 100_000))
        early_freq = int(getattr(cfgs, 'early_eval_step_freq', 10_000))
        late_freq = int(getattr(cfgs, 'value_eval_step_freq', 100_000))
        per_epoch = int(cfgs.steps_per_epoch)

        done = (epoch + 1) * per_epoch
        before = epoch * per_epoch
        freq = early_freq if done <= early_steps else late_freq
        if freq <= 0:
            return False
        return done // freq > before // freq

    _mc_eval_env = None
    _mc_probe_seeds: list[int] | None = None
    _mc_max_episode_steps: int | None = None

    @staticmethod
    def _mc_study_keys() -> tuple[str, ...]:
        """Aggregate statistics the same-layout MC study reports, in a fixed order.

        Registered before training starts rather than when the first evaluation produces
        them: progress.csv's header is written at the first dump_tabular, so a column added
        later never appears in the file.

        ``*_target_true_*`` and ``*_pred_target_*`` decompose the prediction-versus-truth gap
        into the two failure modes it can come from -- a target that does not match the real
        return, and a critic that does not match its own target. ``MeanVar_*`` is the
        variance of the repeats at each probe, which says how much of any remaining gap is
        simply the spread of the return rather than bias.
        """
        pairs = (
            'EstimationError', 'Correlation', 'MeanTrue', 'MeanPred', 'MeanTarget', 'MeanVar',
            'EstimationError_target_true', 'Correlation_target_true', 'Correlation_pred_target',
        )
        return tuple(
            f'MCStudy/{name}_{stream}' for name in pairs for stream in ('r', 'c')
        ) + ('MCStudy/NumProbes', 'MCStudy/MCRepeats')

    def _ensure_mc_eval_env(self):
        """A dedicated environment for the same-layout probe resets, built once.

        It cannot be the training environment: each probe needs a clean ``reset(seed=X)``, and
        doing that on the training env would destroy the episode in flight. It gets the same
        wrapper recipe as training (minus the reward/cost normalizers, which must not update
        from evaluation rollouts) so the policy sees identical observation and action
        processing to what it was trained under.
        """
        if self._mc_eval_env is not None:
            return self._mc_eval_env
        env_cfgs = {}
        if getattr(self._cfgs, 'env_cfgs', None) is not None:
            env_cfgs = self._cfgs.env_cfgs.todict()
        n = int(getattr(self._cfgs.algo_cfgs, 'mc_value_study_vector_envs', 1))
        env = make_env(self._env_id, num_envs=n, device=self._device, **env_cfgs)
        if env.need_time_limit_wrapper:
            env = TimeLimit(env, time_limit=env.max_episode_steps, device=self._device)
        if env.need_auto_reset_wrapper:
            env = AutoReset(env, device=self._device)
        if self._cfgs.algo_cfgs.obs_normalize:
            env = ObsNormalize(env, device=self._device, update_stats=False)
        env = ActionScale(env, low=-1.0, high=1.0, device=self._device)
        if n == 1:
            env = Unsqueeze(env, device=self._device)
        probe = make_env(self._env_id, num_envs=1, device=self._device, **env_cfgs)
        self._mc_max_episode_steps = probe.max_episode_steps
        self._mc_eval_env = env
        return env

    def _run_mc_value_study(self, epoch: int) -> None:
        """Score the Q critics against a repeated same-layout Monte-Carlo reference.

        The protocol the on-policy algorithms use. The earlier off-policy study compared the
        critic against *one* rollout per visited state, so each reference was a single draw
        from the return distribution rather than an estimate -- with nothing to say how much
        of a gap was bias and how much was the variance of one sample. Here a fixed set of
        layouts is reproduced exactly by seed and the stochastic policy is re-rolled from each
        ``mc_value_study_repeats`` times, so the reference is a sample mean whose variance is
        reported beside it.

        Args:
            epoch (int): Current epoch, used as the logging step and the dump's filename.
        """
        if self._mc_probe_seeds is None:
            n = int(getattr(self._cfgs.algo_cfgs, 'mc_value_study_probes', 17))
            off = int(getattr(self._cfgs.algo_cfgs, 'mc_value_study_seed_offset', 100_000))
            self._mc_probe_seeds = list(range(off, off + n))
        gamma = self._cfgs.algo_cfgs.gamma
        stats, raw = estimate_true_q_value_same_state_mc(
            agent=self._actor_critic,
            env=self._ensure_mc_eval_env(),
            cfgs=self._cfgs,
            discount_r=gamma,
            discount_c=getattr(self._cfgs.algo_cfgs, 'cost_gamma', gamma),
            probe_seeds=self._mc_probe_seeds,
            mc_repeats=int(getattr(self._cfgs.algo_cfgs, 'mc_value_study_repeats', 10)),
            epoch=epoch,
            sync_normalizer_from=self._env._env,  # noqa: SLF001
            max_episode_steps=self._mc_max_episode_steps,
            return_raw=True,
        )
        for key in self._mc_study_keys():
            value = stats.get(key)
            if isinstance(value, (int, float)):
                self._logger.store({key: float(value)})
        # Everything the aggregates were computed from: per-probe predictions, per-repeat
        # returns and their variances. An aggregate cannot be un-averaged later, so a
        # calibration curve or a reliability diagram has to come from these.
        path = save_eval_data(
            self._logger.log_dir,
            epoch,
            {'mc_study': {'stats': stats, 'raw': raw},
             'probe_seeds': list(self._mc_probe_seeds),
             'epoch': epoch,
             'total_env_steps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch},
            subdir='mc_eval_data',
        )
        log_eval_data_to_wandb(path, epoch, name_prefix='mc-eval-data')

    def _eval_value_function(self, epoch: int) -> None:
        """Evaluate the Q-critics against Monte-Carlo returns of the current policy.

        The off-policy counterpart of the value-function evaluation :class:`PolicyGradient` runs
        each epoch, with the same config knobs and the same ``Eval_s0`` / ``Eval_all`` wandb
        keys: ``test_estimate`` turns it off, ``early_eval_freq`` sets the cadence for the first
        100 epochs and ``value_eval_freq`` thereafter, and ``value_eval_episodes`` sets how many
        complete episodes each evaluation rolls out.

        Skipped while the agent is still taking uniform-random warm-up actions, where the
        rollout would measure the critics against a policy that is not yet driving the
        environment.

        Unlike the on-policy version, ``test_estimate`` defaults to *off* when the config does
        not mention it, so only configs that opt in (currently ``SACPID.yaml``) pay for it. An
        off-policy epoch collects far fewer env steps than an on-policy one -- 2000 against
        20000 -- so a 100-episode evaluation would cost many times the epoch it is measuring.

        Args:
            epoch (int): Current epoch.
        """
        if not getattr(self._cfgs.algo_cfgs, 'test_estimate', False):
            return
        if not self._is_value_eval_epoch(epoch):
            return
        if (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch <= (
            self._cfgs.algo_cfgs.start_learning_steps
        ):
            return

        if getattr(self._cfgs.algo_cfgs, 'mc_value_study', True):
            # Repeated same-layout rollouts: the reference is a sample mean with a reported
            # variance, not a single draw from the return distribution.
            self._run_mc_value_study(epoch)
            return

        gamma = self._cfgs.algo_cfgs.gamma
        stats, extras = self._env.eval_value_function(
            agent=self._actor_critic,
            # The off-policy cost critic bootstraps with ``gamma`` too (see
            # :meth:`_update_cost_critic`), so ``cost_gamma`` only takes effect if a config
            # sets it; it is read for parity with the on-policy call site.
            discount_r=gamma,
            discount_c=getattr(self._cfgs.algo_cfgs, 'cost_gamma', gamma),
            eval_episodes=getattr(self._cfgs.algo_cfgs, 'value_eval_episodes', 100),
            epoch=epoch,
        )
        # value_eval reports only to wandb, so with wandb off the evaluation would run, pay
        # for its rollouts and discard every number. It returns the same figures as a flat
        # tuple; name them here so they reach progress.csv as well.
        names = self._value_eval_keys()
        if stats is not None and len(stats) == len(names):
            for key, value in zip(names, stats):
                self._logger.store({key: float(value)})
        for key, value in extras.get('stats', {}).items():
            self._logger.store({key: float(value)})

        # Persist the per-probe arrays and push them to wandb, the same way the on-policy
        # path does: the scalars above are an aggregate, and a calibration diagram or a
        # reliability curve needs the probes themselves, which no aggregate can reconstruct.
        raw = extras.get('raw')
        if raw:
            bundle = {
                'q_study': {
                    'stats': dict(zip(names, (float(v) for v in stats))) if stats else {},
                    'spearman': extras.get('stats', {}),
                    'raw': raw,
                },
                'epoch': epoch,
                'total_env_steps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
            }
            path = save_eval_data(self._logger.log_dir, epoch, bundle)
            log_eval_data_to_wandb(path, epoch)

    def _update(self) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. note::

            +----------+---------------------------------------+
            | obs      | ``observaion`` stored in buffer.      |
            +==========+=======================================+
            | act      | ``action`` stored in buffer.          |
            +----------+---------------------------------------+
            | reward   | ``reward`` stored in buffer.          |
            +----------+---------------------------------------+
            | cost     | ``cost`` stored in buffer.            |
            +----------+---------------------------------------+
            | next_obs | ``next observaion`` stored in buffer. |
            +----------+---------------------------------------+
            | done     | ``terminated`` stored in buffer.      |
            +----------+---------------------------------------+

        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the mini-batch data from buffer.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the ``update_iters`` times.
        """
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
            )

            # update_cycle=1 runs this whole block once per environment *step*, so every
            # successor-representation hook below is paid steps_per_epoch times an epoch.
            # The on-policy path it was ported from runs them once per epoch. sr_update_every
            # restores that cadence: 1 keeps the per-step behaviour, steps_per_epoch matches
            # on-policy. The ridge read-out is a closed-form fit on a fresh batch, so solving
            # it less often costs accuracy only to the extent the trunk moved in between.
            sr_every = int(self._cfgs.model_cfgs.sr_cfgs.get('sr_update_every', 1))
            self._sr_tick = getattr(self, '_sr_tick', 0) + 1
            sr_due = sr_every <= 1 or self._sr_tick % sr_every == 1

            if self._sr_td_ridge and self._sr_readout == 'ridge' and sr_due:
                # Refresh w_r / w_c before the critic losses consume them, mirroring how the
                # on-policy path solves the ridge on the fresh batch before its minibatch loop.
                self._ridge_update_successor_weights(obs, act, reward, cost)

            self._update_reward_critic(obs, act, reward, done, next_obs)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)
            if self._sr_td_ridge and sr_due:
                self._update_successor_features(obs, act, done, next_obs)
                if self._sr_readout == 'sgd':
                    # Take one gradient step on the SGD read-out weights. Order relative to the
                    # critic losses is irrelevant: w is detached in the critic read-out, so it is
                    # trained only here, and the target trunk tracks it via polyak_update.
                    self._sgd_update_readout_weights()

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

    def _ridge_update_successor_weights(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
    ) -> None:
        """Refresh the ``td_ridge`` successor-representation read-out weights.

        Solves ``w_r`` / ``w_c`` in closed form on the replay minibatch, once per
        :meth:`_update` iteration. The on-policy counterpart solves them once per epoch on the
        freshly collected batch; here the natural analogue is once per fresh env-step batch,
        since off-policy ``_update`` is called every rollout step.

        Unlike the successor-feature TD target below, this solve needs no trajectory structure
        at all -- it is a regression of the *immediate* reward/cost onto the *immediate* feature
        ``phi(s, a)``, so an i.i.d. replay sample is a perfectly valid design matrix. The result
        is synced onto the target trunk, whose ``w`` buffers ``polyak_update`` cannot reach.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            reward (torch.Tensor): The ``reward`` sampled from buffer.
            cost (torch.Tensor): The ``cost`` sampled from buffer.
        """
        sr_cfgs = self._cfgs.model_cfgs.sr_cfgs
        phi, _ = self._actor_critic.sr_features(obs, act)
        stats = self._actor_critic.sr_trunk.ridge_update(
            phi,
            reward,
            cost,
            ridge_kappa=sr_cfgs.get('ridge_kappa', 1e-3),
            ema_tau=sr_cfgs.get('ema_tau', 1.0),
            ridge_kappa_cost=sr_cfgs.get('ridge_kappa_cost', None),
            cond_every=int(sr_cfgs.get('cond_every', 1)),
        )
        self._actor_critic.sync_sr_readout_weights()
        self._logger.store(stats)

    def _sgd_update_readout_weights(self) -> None:
        """Take one SGD step on the ``readout='sgd'`` read-out weights ``w_r`` / ``w_c``.

        The gradient-descent analogue of :meth:`_ridge_update_successor_weights`: instead of
        re-solving the ``phi -> reward/cost`` regression from scratch each batch, it fits the same
        regression by gradient descent, so ``w`` accumulates over the whole replay buffer. Because
        the reward/cost read-out is policy-independent, every past transition is valid training
        data -- this is the setting the mode is meant for, most cleanly with a frozen ``phi``
        (``phi_source`` ``random`` / ``separate``), where the regression target is stationary.

        A fresh minibatch of ``sr_cfgs.readout_batch_size`` transitions is sampled from the replay
        buffer (independent of the critics' batch, so the read-out can use a different size).
        ``phi`` is taken from :meth:`sr_features`, i.e. detached, so this loss trains ``w`` only,
        never the representation. Unlike the ridge path there is no target sync: ``w`` is now a
        parameter, so ``polyak_update`` tracks it onto the target trunk like any other.
        """
        sr_cfgs = self._cfgs.model_cfgs.sr_cfgs
        batch_size = sr_cfgs.get('readout_batch_size', None) or self._cfgs.algo_cfgs.batch_size
        data = self._buf.sample_batch(batch_size)
        obs, act, reward, cost = data['obs'], data['act'], data['reward'], data['cost']
        phi, _ = self._actor_critic.sr_features(obs, act)  # detached: trains w only
        loss, stats = self._actor_critic.sr_trunk.regression_loss(phi, reward, cost)

        self._actor_critic.sr_readout_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                [self._actor_critic.sr_trunk.w_r, self._actor_critic.sr_trunk.w_c],
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.sr_readout_optimizer.step()
        self._logger.store(stats)

    def _update_successor_features(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        r"""Update the ``td_ridge`` successor-representation trunk on a replay minibatch.

        Trains every ``psi`` head to satisfy the successor-representation Bellman recursion

        .. math::

            \psi_i(s, a) \approx \phi(s, a)
                + \gamma_{sr} (1 - d) \, \bar\psi_i(s', a'),\quad a' \sim \pi(\cdot|s')

        via MSE, summed over heads exactly as SAC sums the MSE over its twin reward critics.

        This is where the port necessarily departs from the on-policy version, which regresses
        ``psi`` onto ``target_sr`` -- a discounted lambda-return over the ``phi`` stream, built by
        the buffer from complete on-policy trajectories. A replay sample has no trajectory
        structure to build a lambda-return from, so the target is the one-step bootstrap above,
        which is the same substitution the reward and cost critics themselves make when going
        from on-policy (GAE targets) to off-policy (one-step Bellman targets). Consequently
        ``sr_cfgs.lam_sr`` has no meaning here and is not read.

        The immediate ``phi`` term comes from the *live* trunk (detached), matching the
        on-policy convention where ``phi`` was computed by the live network at collection time
        and is the same ``phi`` the ridge solve above regresses against; only the bootstrap term
        is taken from the target trunk.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """
        gamma_sr = self._cfgs.model_cfgs.sr_cfgs.get('gamma_sr', None)
        if gamma_sr is None:
            gamma_sr = self._cfgs.algo_cfgs.gamma

        with torch.no_grad():
            phi, _ = self._actor_critic.sr_features(obs, act)
            next_act = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_psi = self._actor_critic.target_sr_trunk.psi(next_obs, next_act)
            not_done = (1 - done).unsqueeze(-1)
            target_psi = [phi + gamma_sr * not_done * next_psi_i for next_psi_i in next_psi]

        psi = self._actor_critic.sr_trunk.psi(obs, act)
        loss = torch.stack(
            [nn.functional.mse_loss(psi_i, target_i) for psi_i, target_i in zip(psi, target_psi)],
        ).sum()

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.sr_trunk.parameters():
                if not param.requires_grad:
                    # frozen phi network (sr_cfgs.phi_source); a no-op otherwise
                    continue
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.sr_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.sr_trunk.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.sr_optimizer.step()

        self._logger.store({'Loss/Loss_sr': loss.mean().item()})

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update reward critic.

        - Get the TD loss of reward critic.
        - Update critic network by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            reward (torch.Tensor): The ``reward`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=True)
            next_q_value_r = self._actor_critic.target_reward_critic(next_obs, next_action)[0]
            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r
        q_value_r = self._actor_critic.reward_critic(obs, action)[0]
        loss = nn.functional.mse_loss(q_value_r, target_q_value_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                if not param.requires_grad:
                    # frozen phi network (sr_cfgs.phi_source); a no-op otherwise
                    continue
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff
        self._logger.store(
            {
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic': q_value_r.mean().item(),
            },
        )
        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.reward_critic_optimizer.step()

    def _aggregate_cost_q(self, values: list[torch.Tensor]) -> torch.Tensor:
        """Combine the cost critic's ensemble members into one conservative estimate.

        Mirrors what SAC already does on the reward side, in the opposite direction. There
        the twins are combined with ``min``, so a reward the members disagree about is
        treated as the lower of their estimates. Cost is the same argument reflected: the
        conservative reading of a disputed cost is the *higher* one, because a cost critic
        that reads low feeds the Lagrange multiplier a violation smaller than the real one,
        and the constraint is under-enforced without anything in the loss noticing.

        ``cost_ensemble_method`` selects the rule ('none', 'cdq', 'gpl', 'top'); the shared
        aggregator applies the cost-side direction for each. With one member, or 'none',
        this is the first member and the behaviour is exactly what it was.

        Args:
            values (list of torch.Tensor): Per-member ``(B,)`` predictions, as the critic
                returns them.

        Returns:
            The aggregated ``(B,)`` cost estimate.
        """
        method = str(getattr(self._cfgs.algo_cfgs, 'cost_ensemble_method', 'none'))
        if method == 'none' or len(values) == 1:
            return values[0]
        beta = float(getattr(self._cfgs.algo_cfgs, 'cost_ensemble_beta', 1.0))
        return aggregate_critic_ensemble(torch.stack(values, dim=0), method, 'c', beta)

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update cost critic.

        - Get the TD loss of cost critic.
        - Update critic network by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            cost (torch.Tensor): The ``cost`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=True)
            next_q_value_c = self._aggregate_cost_q(
                self._actor_critic.target_cost_critic(next_obs, next_action),
            )
            target_q_value_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_c
        q_values_c = self._actor_critic.cost_critic(obs, action)
        # Every member is regressed onto the same target, exactly as SAC trains its reward
        # twins: summed, not averaged, so each member's gradient is unchanged by how many
        # members there are.
        loss = sum(nn.functional.mse_loss(q, target_q_value_c) for q in q_values_c)
        q_value_c = self._aggregate_cost_q(q_values_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                if not param.requires_grad:
                    # frozen phi network (sr_cfgs.phi_source); a no-op otherwise
                    continue
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store(
            {
                'Loss/Loss_cost_critic': loss.mean().item(),
                'Value/cost_critic': q_value_c.mean().item(),
            },
        )

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
    ) -> None:
        """Update actor.

        - Get the loss of actor.
        - Update actor by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
        """
        loss = self._loss_pi(obs)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.actor_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_pi': loss.mean().item(),
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing ``pi/actor`` loss.

        The loss function in DDPG is defined as:

        .. math::

            L = -Q^V (s, \pi (s))

        where :math:`Q^V` is the reward critic network, and :math:`\pi` is the policy network.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=True)
        return -self._actor_critic.reward_critic(obs, action)[0].mean()

    def _log_when_not_update(self) -> None:
        """Log default value when not update."""
        self._logger.store(
            {
                'Loss/Loss_reward_critic': 0.0,
                'Loss/Loss_pi': 0.0,
                'Value/reward_critic': 0.0,
            },
        )
        if self._cfgs.algo_cfgs.use_cost:
            self._logger.store(
                {
                    'Loss/Loss_cost_critic': 0.0,
                    'Value/cost_critic': 0.0,
                },
            )
        if self._sr_td_ridge:
            self._logger.store(
                {
                    'Loss/Loss_sr': 0.0,
                    'Misc/RidgeResidualReward': 0.0,
                    'Misc/RidgeResidualCost': 0.0,
                    'Misc/WrNorm': 0.0,
                    'Misc/WcNorm': 0.0,
                    'Misc/GramCond': 0.0,
                    'Misc/GramCondCost': 0.0,
                },
            )
