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
"""Implementation of the Policy Gradient algorithm."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter import OnPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import distributed
from omnisafe.utils.value_eval import estimate_true_value


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods,line-too-long
class PolicyGradient(BaseAlgo):
    """The Policy Gradient algorithm.

    References:
        - Title: Policy Gradient Methods for Reinforcement Learning with Function Approximation
        - Authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour.
        - URL: `PG <https://proceedings.neurips.cc/paper/1999/file64d828b85b0bed98e80ade0a5c43b0f-Paper.pdf>`_
    """

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OnPolicyAdapter` to adapt the environment to the
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
        """
        self._env: OnPolicyAdapter = OnPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init_model(self) -> None:
        """Initialize the model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_critic.ConstraintActorCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorCritic()
        """
        self._actor_critic: ConstraintActorCritic = ConstraintActorCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        use_sr = bool(self._cfgs.model_cfgs.get('use_successor_representation', False))
        sr_mode = self._cfgs.model_cfgs.sr_cfgs.get('sr_mode', 'shared_trunk') if use_sr else None
        self._sr_td_ridge: bool = sr_mode == 'td_ridge'
        self._sr_fb: bool = sr_mode == 'fb'
        # Set by :meth:`_update`; :meth:`learn` uses it to catch algorithms whose own ``_update``
        # never reaches PolicyGradient's loop, where the FB training step lives.
        self._fb_update_ran: bool = False

        if self._sr_fb:
            self._check_fb_supported()

        self._buf: VectorOnPolicyBuffer = VectorOnPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
            cost_gamma=getattr(self._cfgs.algo_cfgs, 'cost_gamma', None),
            cost_advantage_estimator=getattr(self._cfgs.algo_cfgs, 'cost_adv_estimation_method', None),
            sr_dim=self._cfgs.model_cfgs.sr_cfgs.sr_dim if self._sr_td_ridge else None,
            lam_sr=self._cfgs.model_cfgs.sr_cfgs.get('lam_sr', 0.95) if self._sr_td_ridge else 0.95,
            gamma_sr=self._cfgs.model_cfgs.sr_cfgs.get('gamma_sr', None) if self._sr_td_ridge else None,
            fb_dim=self._cfgs.model_cfgs.sr_cfgs.sr_dim if self._sr_fb else None,
        )

    def _check_fb_supported(self) -> None:
        r"""Validate the discount setup for ``sr_mode: fb``.

        FB represents *one* successor measure and therefore one discount factor. If
        ``cost_gamma`` differs from ``gamma_sr`` (which itself defaults to ``gamma``), ``V_r``
        and ``V_c`` cannot both be read out from the same ``F``, and the cost critic would be
        quietly wrong rather than obviously broken.

        The other way ``fb`` can silently do nothing -- an algorithm whose ``_update`` replaces
        :meth:`PolicyGradient._update` instead of delegating to it -- cannot be detected
        reliably from the class alone (:class:`PPOLag`, for instance, overrides ``_update`` but
        still calls ``super()._update()``), so it is checked at runtime in :meth:`learn` via
        :attr:`_fb_update_ran`.
        """
        gamma = self._cfgs.algo_cfgs.gamma
        gamma_sr = self._cfgs.model_cfgs.sr_cfgs.get('gamma_sr', None)
        gamma_sr = gamma if gamma_sr is None else gamma_sr
        cost_gamma = getattr(self._cfgs.algo_cfgs, 'cost_gamma', None)
        cost_gamma = gamma if cost_gamma is None else cost_gamma
        if abs(cost_gamma - gamma_sr) > 1e-12:
            raise ValueError(
                f'model_cfgs.sr_cfgs.sr_mode = "fb" requires a single discount factor, but '
                f'algo_cfgs.cost_gamma ({cost_gamma}) != gamma_sr ({gamma_sr}). FB factorizes '
                'one successor measure, so the reward and cost values are both read out from '
                'the same F(s) at the same gamma; two different discounts would make the cost '
                'critic quietly wrong. Set them equal, or use sr_mode "td_ridge".',
            )

    def _init_log(self) -> None:
        """Log info about epoch.

        +-----------------------+----------------------------------------------------------------------+
        | Things to log         | Description                                                          |
        +=======================+======================================================================+
        | Train/Epoch           | Current epoch.                                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpCost        | Average cost of the epoch.                                           |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpRet         | Average return of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpLen         | Average length of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Values/reward         | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-----------------------+----------------------------------------------------------------------+
        | Values/cost           | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-----------------------+----------------------------------------------------------------------+
        | Values/Adv            | Average reward advantage of the epoch.                               |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi          | Loss of the policy network.                                          |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic | Loss of the cost critic network.                                     |
        +-----------------------+----------------------------------------------------------------------+
        | Train/Entropy         | Entropy of the policy network.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Train/StopIters       | Number of iterations of the policy network.                          |
        +-----------------------+----------------------------------------------------------------------+
        | Train/PolicyRatio     | Ratio of the policy network.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Train/LR              | Learning rate of the policy network.                                 |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/Seed             | Seed of the experiment.                                              |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps    | Total steps of the experiment.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Time                  | Total time.                                                          |
        +-----------------------+----------------------------------------------------------------------+
        | FPS                   | Frames per second of the epoch.                                      |
        +-----------------------+----------------------------------------------------------------------+
        """
        self._logger = Logger(
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

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/Entropy')
        self._logger.register_key('Train/KL')
        self._logger.register_key('Train/StopIter')
        self._logger.register_key('Train/PolicyRatio', min_and_max=True)
        self._logger.register_key('Train/LR')
        if self._cfgs.model_cfgs.actor_type == 'gaussian_learning':
            self._logger.register_key('Train/PolicyStd')

        self._logger.register_key('TotalEnvSteps')

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)
        self._logger.register_key('Value/Adv')

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward')
        _n_val = getattr(self._cfgs.algo_cfgs, 'n_val_episodes', 0)
        _splits = ('Train', 'Val') if _n_val > 0 else ('Train',)
        for split in _splits:
            for stage in ('BeforeUpdate', 'AfterUpdate'):
                self._logger.register_key(f'Value/{split}/{stage}/RewardCriticCorr')
                self._logger.register_key(f'Value/{split}/{stage}/RewardPredTrueCorr')
            self._logger.register_key(f'Value/{split}/RewardTargetTrueCorr')

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost')
            for split in _splits:
                for stage in ('BeforeUpdate', 'AfterUpdate'):
                    self._logger.register_key(f'Value/{split}/{stage}/CostCriticCorr')
                    self._logger.register_key(f'Value/{split}/{stage}/CostPredTrueCorr')
                self._logger.register_key(f'Value/{split}/CostTargetTrueCorr')

        if self._sr_td_ridge:
            # log information about the td_ridge successor-representation critic
            self._logger.register_key('Loss/Loss_sr', delta=True)
            self._logger.register_key('Misc/RidgeResidualReward')
            self._logger.register_key('Misc/RidgeResidualCost')
            self._logger.register_key('Misc/WrNorm')
            self._logger.register_key('Misc/WcNorm')
            self._logger.register_key('Misc/GramCond')

        if self._sr_fb:
            # log information about the forward-backward successor-representation critic
            self._logger.register_key('Loss/Loss_fb', delta=True)
            self._logger.register_key('Loss/Loss_fb_ortho')
            self._logger.register_key('Misc/RidgeResidualReward')
            self._logger.register_key('Misc/RidgeResidualCost')
            self._logger.register_key('Misc/ZrNorm')
            self._logger.register_key('Misc/ZcNorm')
            self._logger.register_key('Misc/BCovErr')
            self._logger.register_key('Misc/GramCond')
            self._logger.register_key('Misc/FNorm')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

        self._logger.register_key('Metrics/TotalCost')

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
            ep_ret: Average episode return in final epoch.
            ep_cost: Average episode cost in final epoch.
            ep_len: Average episode length in final epoch.
        """
        start_time = time.time()
        self._logger.log('INFO: Start training')
        total_cost: float = 0.0

        for epoch in range(self._cfgs.train_cfgs.epochs):
            epoch_time = time.time()

            rollout_time = time.time()
            self._env.rollout(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=self._logger,
            )

            eval_freq = getattr(self._cfgs.algo_cfgs, 'value_eval_freq', 50)
            early_eval_freq = getattr(self._cfgs.algo_cfgs, 'early_eval_freq', 5)
            effective_eval_freq = early_eval_freq if epoch < 100 else eval_freq
            eval_episodes = getattr(self._cfgs.algo_cfgs, 'value_eval_episodes', 100)
            if getattr(self._cfgs.algo_cfgs, 'test_estimate', True) and epoch % effective_eval_freq == 0:
                estimate_true_value(
                    agent=self._actor_critic,
                    env=self._env._env,
                    cfgs=self._cfgs,
                    discount_r=self._cfgs.algo_cfgs.gamma,
                    discount_c=getattr(self._cfgs.algo_cfgs, 'cost_gamma', self._cfgs.algo_cfgs.gamma),
                    eval_episodes=eval_episodes,
                    epoch=epoch,
                )
            self._logger.store({'Time/Rollout': time.time() - rollout_time})

            update_time = time.time()
            self._current_epoch = epoch
            self._update()
            if self._sr_fb and not self._fb_update_ran:
                raise NotImplementedError(
                    f'model_cfgs.sr_cfgs.sr_mode = "fb" is not supported by '
                    f'{type(self).__name__}: its _update() never called _sr_extra_columns() / '
                    '_sr_minibatch_update(), so F and B would stay at their initialization and '
                    'both critics would be meaningless. Any algorithm with its own _update() '
                    'must route the successor-representation work through those two hooks (see '
                    'PolicyGradient._update, NaturalPG._update or FOCOPS._update for the '
                    'pattern). Until then, use sr_mode "shared_trunk" or "td_ridge".',
                )
            total_cost += self._env._epoch_cost_sum
            self._logger.store({'Metrics/TotalCost': total_cost})
            self._logger.store({'Time/Update': time.time() - update_time})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)

            if self._cfgs.model_cfgs.actor.lr is not None:
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': (
                        0.0
                        if self._cfgs.model_cfgs.actor.lr is None
                        else self._actor_critic.actor_scheduler.get_last_lr()[0]
                    ),
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                epoch + 1
            ) == self._cfgs.train_cfgs.epochs:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. hint::

            +----------------+------------------------------------------------------------------+
            | obs            | ``observation`` sampled from buffer.                             |
            +================+==================================================================+
            | act            | ``action`` sampled from buffer.                                  |
            +----------------+------------------------------------------------------------------+
            | target_value_r | ``target reward value`` sampled from buffer.                     |
            +----------------+------------------------------------------------------------------+
            | target_value_c | ``target cost value`` sampled from buffer.                       |
            +----------------+------------------------------------------------------------------+
            | logp           | ``log probability`` sampled from buffer.                         |
            +----------------+------------------------------------------------------------------+
            | adv_r          | ``estimated advantage`` (e.g. **GAE**) sampled from buffer.      |
            +----------------+------------------------------------------------------------------+
            | adv_c          | ``estimated cost advantage`` (e.g. **GAE**) sampled from buffer. |
            +----------------+------------------------------------------------------------------+


        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the data from buffer.
        #. Shuffle the data and split it into mini-batch data.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the number of mini-batch data is used up.
        #. Repeat steps 2, 3, 4 until the KL divergence violates the limit.
        """
        data = self._buf.get()
        train_data, val_data = self._make_train_val_split(data)

        sr_extras = self._sr_extra_columns(train_data)

        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            train_data['obs'],
            train_data['act'],
            train_data['logp'],
            train_data['target_value_r'],
            train_data['target_value_c'],
            train_data['adv_r'],
            train_data['adv_c'],
        )

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        dataloader = DataLoader(
            dataset=TensorDataset(
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
                *sr_extras,
            ),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for batch in dataloader:
                obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = batch[:7]
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)
                self._sr_minibatch_update(obs, batch[7:], original_obs)
                self._update_actor(obs, act, logp, adv_r, adv_c)

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)

            final_kl = kl.item()
            update_counts += 1

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': final_kl,
            },
        )

        self._log_critic_diagnostics_splits(train_data, val_data)

    def _make_train_val_split(
        self, data: dict,
    ) -> tuple[dict, dict | None]:
        """Split buffer data into train/val subsets by episode.

        Returns ``(train_data, val_data)``.  ``val_data`` is ``None`` when
        ``n_val_episodes == 0`` (default), preserving the existing behaviour.
        """
        n_val = getattr(self._cfgs.algo_cfgs, 'n_val_episodes', 0)
        if n_val <= 0:
            return data, None

        episode_slices = self._buf.get_episode_slices()
        n_total = len(episode_slices)
        n_val = min(n_val, n_total - 1)  # always keep at least 1 episode for training

        val_ep_idx = set(
            torch.randperm(n_total)[:n_val].tolist()
        )
        n = data['obs'].shape[0]
        device = data['obs'].device
        train_mask = torch.ones(n, dtype=torch.bool, device=device)
        val_mask = torch.zeros(n, dtype=torch.bool, device=device)
        for i, (s, e) in enumerate(episode_slices):
            if i in val_ep_idx:
                train_mask[s:e] = False
                val_mask[s:e] = True

        def _apply(mask: torch.Tensor) -> dict:
            return {
                k: v[mask] if isinstance(v, torch.Tensor) and v.shape[0] == n else v
                for k, v in data.items()
            }

        return _apply(train_mask), _apply(val_mask)

    def _log_critic_diagnostics_splits(
        self,
        train_data: dict,
        val_data: dict | None,
    ) -> None:
        """Call ``_log_critic_diagnostics`` for Train and optionally Val splits."""
        self._log_critic_diagnostics(
            train_data['obs'],
            train_data['target_value_r'],
            train_data['target_value_c'],
            train_data['discounted_ret'],
            train_data['discounted_cost_ret'],
            preupdate_pred_r=train_data['value_r'].flatten(),
            preupdate_pred_c=train_data['value_c'].flatten(),
            split='Train',
        )
        if val_data is not None:
            self._log_critic_diagnostics(
                val_data['obs'],
                val_data['target_value_r'],
                val_data['target_value_c'],
                val_data['discounted_ret'],
                val_data['discounted_cost_ret'],
                preupdate_pred_r=val_data['value_r'].flatten(),
                preupdate_pred_c=val_data['value_c'].flatten(),
                split='Val',
            )

    def _log_critic_diagnostics(
        self,
        obs: torch.Tensor,
        target_value_r: torch.Tensor,
        target_value_c: torch.Tensor,
        discounted_ret: torch.Tensor,
        discounted_cost_ret: torch.Tensor,
        preupdate_pred_r: torch.Tensor | None = None,
        preupdate_pred_c: torch.Tensor | None = None,
        split: str = 'Train',
    ) -> None:
        """Log scatter plots and Pearson correlations for both BeforeUpdate and AfterUpdate stages.

        ``preupdate_pred_r``/``preupdate_pred_c`` should be ``data['value_r/c'].flatten()``
        (critic predictions collected during rollout, before any gradient step).  The AfterUpdate
        predictions are always re-queried from the live network.
        """
        epoch = getattr(self, '_current_epoch', 0)
        eval_freq = getattr(self._cfgs.algo_cfgs, 'value_eval_freq', 50)
        early_eval_freq = getattr(self._cfgs.algo_cfgs, 'early_eval_freq', 5)
        effective_eval_freq = early_eval_freq if epoch < 100 else eval_freq
        if epoch % effective_eval_freq != 0:
            return

        target_r = target_value_r.flatten()
        true_r = discounted_ret.flatten()

        with torch.no_grad():
            post_pred_r = self._actor_critic.reward_critic(obs)[0].flatten()

        n = post_pred_r.shape[0]
        idx = torch.randperm(n, device=post_pred_r.device)[:min(n, 2000)]

        stages_r = []
        if preupdate_pred_r is not None:
            stages_r.append(('BeforeUpdate', preupdate_pred_r.flatten()))
        stages_r.append(('AfterUpdate', post_pred_r))

        corr_target_true_r = torch.corrcoef(torch.stack([target_r, true_r]))[0, 1].item()
        self._logger.store({f'Value/{split}/RewardTargetTrueCorr': corr_target_true_r})
        self._logger.log_scatter_image(
            f'Value/{split}/RewardTargetTrueScatter',
            x_values=true_r[idx],
            y_values=target_r[idx],
            xlabel='True G_r',
            ylabel='Target V_r',
        )

        for label, pred_r in stages_r:
            corr_pred_target_r = torch.corrcoef(torch.stack([pred_r, target_r]))[0, 1].item()
            corr_pred_true_r = torch.corrcoef(torch.stack([pred_r, true_r]))[0, 1].item()
            self._logger.store({
                f'Value/{split}/{label}/RewardCriticCorr': corr_pred_target_r,
                f'Value/{split}/{label}/RewardPredTrueCorr': corr_pred_true_r,
            })
            self._logger.log_scatter_image(
                f'Value/{split}/{label}/RewardCriticScatter',
                x_values=target_r[idx],
                y_values=pred_r[idx],
                xlabel='Target V_r',
                ylabel='Predicted V_r',
                c_values=true_r[idx],
                c_label='True G_r',
            )
            self._logger.log_scatter_image(
                f'Value/{split}/{label}/RewardPredTrueScatter',
                x_values=true_r[idx],
                y_values=pred_r[idx],
                xlabel='True G_r',
                ylabel='Predicted V_r',
            )

        if self._cfgs.algo_cfgs.use_cost:
            target_c = target_value_c.flatten()
            true_c = discounted_cost_ret.flatten()

            with torch.no_grad():
                post_pred_c = self._actor_critic.cost_critic(obs)[0].flatten()

            stages_c = []
            if preupdate_pred_c is not None:
                stages_c.append(('BeforeUpdate', preupdate_pred_c.flatten()))
            stages_c.append(('AfterUpdate', post_pred_c))

            corr_target_true_c = torch.corrcoef(torch.stack([target_c, true_c]))[0, 1].item()
            self._logger.store({f'Value/{split}/CostTargetTrueCorr': corr_target_true_c})
            self._logger.log_scatter_image(
                f'Value/{split}/CostTargetTrueScatter',
                x_values=true_c[idx],
                y_values=target_c[idx],
                xlabel='True G_c',
                ylabel='Target V_c',
            )

            for label, pred_c in stages_c:
                corr_pred_target_c = torch.corrcoef(torch.stack([pred_c, target_c]))[0, 1].item()
                corr_pred_true_c = torch.corrcoef(torch.stack([pred_c, true_c]))[0, 1].item()
                self._logger.store({
                    f'Value/{split}/{label}/CostCriticCorr': corr_pred_target_c,
                    f'Value/{split}/{label}/CostPredTrueCorr': corr_pred_true_c,
                })
                self._logger.log_scatter_image(
                    f'Value/{split}/{label}/CostCriticScatter',
                    x_values=target_c[idx],
                    y_values=pred_c[idx],
                    xlabel='Target V_c',
                    ylabel='Predicted V_c',
                    c_values=true_c[idx],
                    c_label='True G_c',
                )
                self._logger.log_scatter_image(
                    f'Value/{split}/{label}/CostPredTrueScatter',
                    x_values=true_c[idx],
                    y_values=pred_c[idx],
                    xlabel='True G_c',
                    ylabel='Predicted V_c',
                )

    def _update_reward_critic(self, obs: torch.Tensor, target_value_r: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_r (torch.Tensor): The ``target_value_r`` sampled from buffer.
        """
        if self._skip_value_mse_step():
            self._log_value_mse_only(
                self._actor_critic.reward_critic,
                obs,
                target_value_r,
                'Loss/Loss_reward_critic',
            )
            return

        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.reward_critic(obs)[0], target_value_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.reward_critic)
        self._actor_critic.reward_critic_optimizer.step()

        self._logger.store({'Loss/Loss_reward_critic': loss.mean().item()})

    def _update_cost_critic(self, obs: torch.Tensor, target_value_c: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_c (torch.Tensor): The ``target_value_c`` sampled from buffer.
        """
        if self._skip_value_mse_step():
            self._log_value_mse_only(
                self._actor_critic.cost_critic,
                obs,
                target_value_c,
                'Loss/Loss_cost_critic',
            )
            return

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.cost_critic(obs)[0], target_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store({'Loss/Loss_cost_critic': loss.mean().item()})

    def _skip_value_mse_step(self) -> bool:
        """Whether to skip the ordinary value-target MSE optimizer step.

        In ``fb`` mode the reward/cost values are read out as ``F(s)^T z``, so a value-MSE
        gradient would flow into ``F`` and fight the measure-Bellman objective that defines it
        -- the same "one trunk pulled by three losses" problem the ``td_ridge`` mode has. By
        default FB therefore trains ``F`` from its own loss only. Set
        ``sr_cfgs.fb_train_critic_mse: True`` to restore the ``td_ridge``-style behaviour for
        ablation.
        """
        if not self._sr_fb:
            return False
        return not bool(self._cfgs.model_cfgs.sr_cfgs.get('fb_train_critic_mse', False))

    def _log_value_mse_only(
        self,
        critic: torch.nn.Module,
        obs: torch.Tensor,
        target_value: torch.Tensor,
        key: str,
    ) -> None:
        """Compute the value-target MSE for logging only -- no ``backward``, no ``step``.

        Keeps ``Loss/Loss_reward_critic`` / ``Loss/Loss_cost_critic`` populated and comparable
        across ``sr_mode`` settings (the logger raises on a registered-but-unfilled key) while
        the critic itself is trained by another objective.

        Args:
            critic (torch.nn.Module): The critic to evaluate.
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value (torch.Tensor): The value target for ``obs``.
            key (str): The logger key to store the MSE under.
        """
        with torch.no_grad():
            loss = nn.functional.mse_loss(critic(obs)[0], target_value)
        self._logger.store({key: loss.mean().item()})

    def _sr_extra_columns(self, train_data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Per-epoch successor-representation work, plus the extra minibatch columns it needs.

        Called once per ``_update()``, before the minibatch loop, by every algorithm that has
        its own update loop. Returns the tensors to append to that loop's ``TensorDataset``;
        :meth:`_sr_minibatch_update` consumes them in the same order. Returns ``()`` when no
        successor-representation mode is active, which is what keeps the non-SR and
        ``shared_trunk`` paths byte-for-byte unchanged.

        Args:
            train_data (dict[str, torch.Tensor]): The training-split epoch batch.

        Returns:
            The extra ``TensorDataset`` columns for the active mode.
        """
        if self._sr_td_ridge:
            self._ridge_update_successor_weights(train_data)
            return (train_data['target_sr'],)
        if self._sr_fb:
            self._fb_update_ran = True
            self._fb_update_task_vectors(train_data)
            return (train_data['next_obs'], train_data['terminated'])
        return ()

    def _sr_minibatch_update(
        self,
        obs: torch.Tensor,
        sr_batch: tuple[torch.Tensor, ...],
        all_obs: torch.Tensor,
    ) -> None:
        """Run the active successor-representation mode's per-minibatch training step.

        The counterpart to :meth:`_sr_extra_columns`: ``sr_batch`` is that method's return
        value sliced out of the current minibatch, in the same order.

        Args:
            obs (torch.Tensor): The minibatch ``observation``.
            sr_batch (tuple of torch.Tensor): The extra minibatch columns for the active mode.
            all_obs (torch.Tensor): The full training-split ``obs``, used by ``fb`` to draw a
                marginal-state batch independently of this minibatch.
        """
        if self._sr_td_ridge:
            (target_sr,) = sr_batch
            self._update_successor_features(obs, target_sr)
        elif self._sr_fb:
            next_obs, terminated = sr_batch
            self._update_fb_representation(
                obs,
                next_obs,
                terminated,
                self._fb_sample_marginal(all_obs),
            )

    def _fb_sample_marginal(self, all_obs: torch.Tensor) -> torch.Tensor:
        r"""Draw a marginal-state batch from the on-policy distribution ``rho``.

        Sampled from the full training split, independently of whichever transition minibatch
        it will be paired with -- that independence is what makes the FB loss an unbiased
        estimate of the measure-Bellman residual against ``rho``.

        Args:
            all_obs (torch.Tensor): The full training-split ``obs``.

        Returns:
            ``fb_marginal_batch_size`` rows of ``all_obs`` (default ``algo_cfgs.batch_size``).
        """
        size = self._cfgs.model_cfgs.sr_cfgs.get('fb_marginal_batch_size', None)
        size = size or self._cfgs.algo_cfgs.batch_size
        idx = torch.randint(
            0,
            all_obs.shape[0],
            (min(size, all_obs.shape[0]),),
            device=all_obs.device,
        )
        return all_obs[idx]

    def _fb_update_task_vectors(self, train_data: dict[str, torch.Tensor]) -> None:
        r"""Refresh the ``fb`` read-out vectors ``z_r`` / ``z_c``.

        ``z_g = E_{s \sim \rho}[B(s) g(s)]`` is a plain expectation over the on-policy state
        distribution, evaluated once per :meth:`_update` call on the fresh epoch's *training*
        split -- the FB analogue of :meth:`_ridge_update_successor_weights`, and like it kept
        away from the held-out validation episodes.

        Args:
            train_data (dict[str, torch.Tensor]): The training-split epoch batch (post
                :meth:`_make_train_val_split`), including the ``obs``, ``reward`` and ``cost``
                fields.
        """
        sr_cfgs = self._cfgs.model_cfgs.sr_cfgs
        with torch.no_grad():
            b = self._actor_critic.sr_trunk.backward_map(train_data['obs'])
        stats = self._actor_critic.sr_trunk.update_task_vectors(
            b,
            train_data['reward'],
            train_data['cost'],
            ema_tau=sr_cfgs.get('ema_tau', 1.0),
            estimator=sr_cfgs.get('fb_z_estimator', 'expectation'),
            ridge_kappa=sr_cfgs.get('ridge_kappa', 1e-3),
        )
        self._logger.store(stats)

    def _update_fb_representation(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        terminated: torch.Tensor,
        marginal_obs: torch.Tensor,
    ) -> None:
        r"""Update the forward-backward maps by the measure-Bellman residual.

        The successor measure is taken in the ``t >= 0`` convention,

        .. math::

            M^\pi(s, X) = \sum_{t \ge 0} \gamma^t \Pr(s_t \in X \mid s_0 = s)
                        = \mathbb{1}[s \in X] + \gamma\, \mathbb{E}_{s^+}[M^\pi(s^+, X)],

        i.e. the atom sits on ``s`` itself rather than on its successor. This differs from the
        ``t >= 1`` convention of Touati & Ollivier, and it is the one that matters here: it makes
        the read-out :math:`F(s)^T z_g` *equal* to the value function
        :math:`V_g(s) = \mathbb{E}[\sum_t \gamma^t g(s_t)]` that GAE consumes. The ``t >= 1``
        convention would instead yield :math:`(V_g(s) - g(s)) / \gamma`, which is not a drop-in
        critic. Concretely, the only difference in the loss is that the "diagonal" term below is
        evaluated at :math:`s_i` rather than at :math:`s_i^+`.

        With a transition minibatch :math:`\{(s_i, s_i^+, d_i)\}` of size ``n`` and an
        independent marginal minibatch :math:`\{s'_j\} \sim \rho` of size ``m``:

        .. math::

            M &= F(S) B(S')^T \\
            \bar{M} &= \gamma (1 - d) \odot \big( \bar{F}(S^+) \bar{B}(S')^T \big) \\
            L_{fb} &= \mathrm{mean}\big((M - \bar{M})^2\big)
                      - 2\, \mathrm{mean}_i\big(F(s_i)^T B(s_i)\big) \\
            L_{ortho} &= \big\| \tfrac{1}{m} B(S')^T B(S') - I_d \big\|_F^2

        The second term of :math:`L_{fb}` is the atom :math:`\mathbb{1}[s \in X]`, which the
        outer-product residual cannot see; it is what puts mass on each state's own occupancy.
        The ``(1 - d)`` factor makes a terminal state's measure the atom alone, with no future
        mass -- time-limit truncations are stored as ``0.0`` in ``terminated``, so they still
        bootstrap.

        :math:`L_{ortho}` is not optional. Only the product :math:`F B^T` is identified, so
        without it ``F`` and ``B`` drift along the :math:`F \to FA,\; B \to BA^{-T}`
        degeneracy and the read-out :math:`z_g = E_\rho[B g]` -- which assumes
        :math:`E_\rho[B B^T] \approx I` -- stops being correct.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            next_obs (torch.Tensor): The successor of ``obs`` (AutoReset-corrected in the
                adapter, so it is the true final observation at episode boundaries).
            terminated (torch.Tensor): 1.0 where the episode genuinely terminated at ``obs``
                (time-limit truncations are 0.0 and therefore still bootstrap).
            marginal_obs (torch.Tensor): States drawn independently from the on-policy
                distribution ``rho``.
        """
        sr_cfgs = self._cfgs.model_cfgs.sr_cfgs
        trunk = self._actor_critic.sr_trunk
        gamma = sr_cfgs.get('gamma_sr', None)
        gamma = self._cfgs.algo_cfgs.gamma if gamma is None else gamma

        self._actor_critic.sr_optimizer.zero_grad()

        f = trunk.forward_map(obs)  # (n, d)
        b_marginal = trunk.backward_map(marginal_obs)  # (m, d)
        b_diag = trunk.backward_map(obs)  # (n, d), the atom at s (see the t >= 0 convention above)

        with torch.no_grad():
            f_next_target = trunk.forward_map(next_obs, target=True)
            b_marginal_target = trunk.backward_map(marginal_obs, target=True)
            not_done = (1.0 - terminated).unsqueeze(-1)  # (n, 1)
            m_target = gamma * not_done * (f_next_target @ b_marginal_target.T)  # (n, m)

        m_online = f @ b_marginal.T  # (n, m)
        loss_fb = (m_online - m_target).pow(2).mean() - 2.0 * (f * b_diag).sum(-1).mean()

        cov = b_marginal.T @ b_marginal / b_marginal.shape[0]
        loss_ortho = (cov - torch.eye(trunk.sr_dim, device=cov.device)).pow(2).sum()

        loss = loss_fb + sr_cfgs.get('fb_ortho_coef', 1.0) * loss_ortho

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in trunk.parameters():
                if param.requires_grad:
                    loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                [p for p in trunk.parameters() if p.requires_grad],
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(trunk)
        self._actor_critic.sr_optimizer.step()
        trunk.soft_update_targets(sr_cfgs.get('fb_target_tau', 0.01))

        # Log the measure-Bellman term alone, so that Loss/Loss_fb and Loss/Loss_fb_ortho are
        # independent diagnostics rather than one containing the other.
        self._logger.store(
            {
                'Loss/Loss_fb': loss_fb.item(),
                'Loss/Loss_fb_ortho': loss_ortho.item(),
                'Misc/FNorm': f.norm(dim=-1).mean().item(),
            },
        )

    def _ridge_update_successor_weights(self, train_data: dict[str, torch.Tensor]) -> None:
        """Refresh the ``td_ridge`` successor-representation read-out weights.

        Solves ``w_r`` / ``w_c`` in closed form on the fresh epoch's *training* split, once per
        :meth:`_update` call (not per minibatch) -- mirroring how the buffer's GAE targets are
        also computed once per epoch from the just-collected data. Uses ``train_data`` (not the
        full pre-split batch) so the ridge fit never sees the held-out validation episodes.

        Args:
            train_data (dict[str, torch.Tensor]): The training-split epoch batch (post
                :meth:`_make_train_val_split`), including the ``phi``, ``reward``, and ``cost``
                fields used for the ridge solve.
        """
        sr_cfgs = self._cfgs.model_cfgs.sr_cfgs
        stats = self._actor_critic.sr_trunk.ridge_update(
            train_data['phi'],
            train_data['reward'],
            train_data['cost'],
            ridge_kappa=sr_cfgs.get('ridge_kappa', 1e-3),
            ema_tau=sr_cfgs.get('ema_tau', 1.0),
        )
        self._logger.store(stats)

    def _update_successor_features(self, obs: torch.Tensor, target_sr: torch.Tensor) -> None:
        r"""Update the ``td_ridge`` successor-representation trunk under a double for loop.

        Trains ``psi`` to satisfy the successor-representation Bellman recursion via MSE
        against ``target_sr``, the buffer's discounted lambda-target built with the exact same
        GAE / GAE-RTG / V-trace / Plain / Reinforce / TD(0) machinery used for the reward and
        cost value targets (see
        :meth:`omnisafe.common.buffer.onpolicy_buffer.OnPolicyBuffer.finish_path`), applied to
        the vector-valued feature stream ``phi`` instead of a scalar reward/cost stream.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_sr (torch.Tensor): The ``target_sr`` sampled from buffer.
        """
        self._actor_critic.sr_optimizer.zero_grad()
        psi = self._actor_critic.sr_trunk.psi(obs)
        loss = nn.functional.mse_loss(psi, target_sr)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.sr_trunk.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.sr_trunk.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.sr_trunk)
        self._actor_critic.sr_optimizer.step()

        self._logger.store({'Loss/Loss_sr': loss.mean().item()})

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network under a double for loop.

        #. Compute the loss function.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        .. warning::
            For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
            the ``KL divergence`` between the old policy and the new policy is calculated.
            And the ``KL divergence`` is used to determine whether the update is successful.
            If the ``KL divergence`` is too large, the update will be terminated.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log_p`` sampled from buffer.
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.
        """
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss = self._loss_pi(obs, act, logp, adv)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()

    def _compute_adv_surrogate(  # pylint: disable=unused-argument
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surrogate loss.

        Policy Gradient only use reward advantage.

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function of reward to update policy network.
        """
        return adv_r

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing pi/actor loss.

        In Policy Gradient, the loss is defined as:

        .. math::

            L = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} [
                \sum_{t=0}^T ( \frac{\pi^{'}_{\theta}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)} )
                 A^{R}_{\pi_{\theta}}(s_t, a_t)
            ]

        where :math:`\pi_{\theta}` is the policy network, :math:`\pi^{'}_{\theta}`
        is the new policy network, :math:`A^{R}_{\pi_{\theta}}(s_t, a_t)` is the advantage.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv (torch.Tensor): The ``advantage`` processed. ``reward_advantage`` here.

        Returns:
            The loss of pi/actor.
        """
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)
        loss = -(ratio * adv).mean()
        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss
