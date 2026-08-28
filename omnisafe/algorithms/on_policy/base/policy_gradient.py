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
from omnisafe.common.buffer.readout_buffer import ReadoutReplayBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import clearning, contrastive, distributed, laplacian, sr_diagnostics
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

    # Successor-representation state, declared at class level rather than only in :meth:`_init`
    # because subclasses may replace ``_init`` outright to build their own buffer (MICE does) and
    # still inherit ``_init_log`` / ``_update``, which read these. Without the defaults such an
    # algorithm dies with an AttributeError before its first rollout.
    _sr_td_ridge: bool = False
    _sr_readout: str = 'ridge'
    _sr_ridge_data: str = 'epoch'
    _sr_readout_buf: ReadoutReplayBuffer | None = None
    _sr_cost_readout_buf: ReadoutReplayBuffer | None = None
    _sr_probe_size: int = 0
    _sr_diag_max_samples: int = 0
    _sr_probe_fixed_obs: torch.Tensor | None = None
    _sr_probe_fresh_obs: torch.Tensor | None = None
    _sr_snapshot: dict[str, torch.Tensor] | None = None
    # phi_source='contrastive' state; see _contrastive_update_phi / _relabel_after_phi_update.
    _sr_phi_source: str = 'trunk'
    _sr_phi_pretrain_steps: int = 0
    _sr_phi_steps_per_epoch: int = 0
    # Set by learn() right before each epoch's _update() call; declared here (rather than only in
    # learn()) so the getattr(self, '_current_epoch', 0) fallback used elsewhere in this class has
    # a real class-level default to fall back to.
    _current_epoch: int = 0

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
        self._sr_td_ridge = use_sr and self._cfgs.model_cfgs.sr_cfgs.get(
            'sr_mode',
            'shared_trunk',
        ) == 'td_ridge'
        # How the td_ridge read-out weights w_r / w_c are learned: 'ridge' (closed-form on the
        # fresh epoch) or 'sgd' (gradient descent over a persistent cross-epoch replay buffer).
        self._sr_readout = (
            self._cfgs.model_cfgs.sr_cfgs.get('readout', 'ridge') if self._sr_td_ridge else 'ridge'
        )
        # Which transitions the closed-form ridge is solved on (readout='ridge' only): 'epoch' --
        # the fresh epoch's training split, 'buffer' -- the persistent cross-epoch history, or
        # 'cost_replay' -- the epoch plus an equal number of past cost-inducing transitions. This
        # is deliberately a separate axis from ``readout`` above, which selects only *how* w is fit;
        # holding one fixed while varying the other is what makes the two comparable.
        self._sr_ridge_data = (
            self._cfgs.model_cfgs.sr_cfgs.get('ridge_data', 'epoch')
            if self._sr_td_ridge
            else 'epoch'
        )
        if self._sr_ridge_data not in ('epoch', 'buffer', 'cost_replay'):
            raise NotImplementedError(
                f'Unknown sr_cfgs.ridge_data "{self._sr_ridge_data}". '
                'Available ridge data sources are: "epoch", "buffer", "cost_replay".',
            )
        # Where phi comes from. Only the *trained* sources are read here (every other value is
        # handled entirely inside the trunk/critic construction): they are the ones needing a
        # gradient loop of their own, driven by _maybe_update_trained_phi's pretrain-then-continual
        # cadence -- see _contrastive_update_phi / _laplacian_update_phi.
        self._sr_phi_source = (
            self._cfgs.model_cfgs.sr_cfgs.get('phi_source', 'trunk') if self._sr_td_ridge else 'trunk'
        )
        self._sr_phi_trained = self._sr_phi_source in ('contrastive', 'laplacian')
        # The cadence knobs are per-source (phi_contrastive_* / phi_laplacian_*) rather than one
        # shared set: the two objectives want quite different budgets, and a sweep row reading
        # `phi_contrastive_pretrain_steps: 500` while `phi_source: laplacian` would be a trap for
        # whoever reads the results back. The prefix keeps the dispatch to one line regardless.
        phi_cfg_prefix = f'phi_{self._sr_phi_source}_'
        self._sr_phi_pretrain_steps = (
            int(self._cfgs.model_cfgs.sr_cfgs.get(f'{phi_cfg_prefix}pretrain_steps', 500))
            if self._sr_phi_trained
            else 0
        )
        self._sr_phi_steps_per_epoch = (
            int(self._cfgs.model_cfgs.sr_cfgs.get(f'{phi_cfg_prefix}steps_per_epoch', 50))
            if self._sr_phi_trained
            else 0
        )
        # How psi is fitted: 'td' bootstraps it against a lambda-target built from phi (the
        # original and the default), 'contrastive' drops the bootstrap entirely and fits phi and
        # psi together as the two factors of a successor-measure critic -- see
        # _contrastive_update_successor_features and omnisafe.utils.clearning.
        self._sr_psi_objective = (
            self._cfgs.model_cfgs.sr_cfgs.get('psi_objective', 'td') if self._sr_td_ridge else 'td'
        )
        if self._sr_psi_objective not in ('td', 'contrastive'):
            raise NotImplementedError(
                f'Unknown sr_cfgs.psi_objective "{self._sr_psi_objective}". '
                'Available psi objectives are: "td", "contrastive".',
            )
        if self._sr_psi_objective == 'contrastive' and self._sr_phi_source != 'joint':
            # Not a stylistic preference. Every other source either freezes phi or gives it a
            # separate objective and optimizer, and both are incoherent here: this loss trains phi
            # as one of its own two factors, so a frozen phi cannot be fitted and a separately
            # trained one would be pulled by two objectives whose scales have nothing in common
            # (the bilinear critic's magnitude *is* the estimate, so a phi normalized or
            # orthonormalized by another loss silently rescales the successor measure).
            raise NotImplementedError(
                'sr_cfgs.psi_objective="contrastive" requires sr_cfgs.phi_source="joint" '
                f'(phi is fitted as a factor of the same critic), got "{self._sr_phi_source}".',
            )
        self._sr_psi_pretrain_steps = (
            int(self._cfgs.model_cfgs.sr_cfgs.get('psi_contrastive_pretrain_steps', 2000))
            if self._sr_psi_objective == 'contrastive'
            else 0
        )
        self._sr_psi_steps_per_epoch = (
            int(self._cfgs.model_cfgs.sr_cfgs.get('psi_contrastive_steps_per_epoch', 300))
            if self._sr_psi_objective == 'contrastive'
            else 0
        )

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
            adv_norm_mode=getattr(self._cfgs.algo_cfgs, 'adv_norm_mode', 'batch'),
            adv_norm_timestep_min_count=getattr(
                self._cfgs.algo_cfgs, 'adv_norm_timestep_min_count', 4,
            ),
        )

        # Persistent cross-epoch buffer for the read-out regression. The rollout buffer above is
        # wiped each epoch; this one accumulates the whole history so w_r / w_c can be fit over all
        # past experience (the read-out is policy-independent). Used by readout='sgd' always, and
        # by readout='ridge' under ridge_data='buffer'.
        self._sr_readout_buf = None
        if self._sr_td_ridge and (self._sr_readout == 'sgd' or self._sr_ridge_data == 'buffer'):
            self._sr_readout_buf = ReadoutReplayBuffer(
                obs_dim=self._env.observation_space.shape[0],
                size=int(self._cfgs.model_cfgs.sr_cfgs.get('readout_buffer_size', 1_000_000)),
                device=self._device,
            )

        # Persistent cross-epoch buffer holding only *cost-inducing* transitions (cost > 0), used
        # by readout='ridge' under ridge_data='cost_replay'. Costs are typically sparse, so the
        # fresh epoch alone rarely carries enough cost-positive transitions to fit w_c well; this
        # buffer accumulates them across epochs so a matched number can be resampled back in.
        self._sr_cost_readout_buf = None
        if self._sr_td_ridge and self._sr_ridge_data == 'cost_replay':
            self._sr_cost_readout_buf = ReadoutReplayBuffer(
                obs_dim=self._env.observation_space.shape[0],
                size=int(self._cfgs.model_cfgs.sr_cfgs.get('cost_replay_buffer_size', 1_000_000)),
                device=self._device,
            )

        # --- SR diagnostic state (td_ridge only); see _log_sr_diagnostics -------------------
        # Two probe sets of observations on which phi / psi drift is measured. The Fixed probe is
        # drawn once and reused for the whole run, so its drift series is a clean measure of how
        # much the feature map itself moved; the Fresh probe is redrawn from each epoch's rollout,
        # so its drift is measured on the states the policy actually visits now. Neither alone is
        # enough: the Fixed one slides off-distribution as the policy improves, the Fresh one
        # confounds feature drift with distribution shift. Their divergence is the signal.
        # Read behind the mode check: some on-policy configs (MICE.yaml) omit the sr_cfgs block
        # entirely, and it is only reachable once use_successor_representation is on.
        self._sr_probe_size = (
            int(self._cfgs.model_cfgs.sr_cfgs.get('probe_size', 512)) if self._sr_td_ridge else 0
        )
        self._sr_diag_max_samples = (
            int(self._cfgs.model_cfgs.sr_cfgs.get('diag_max_samples', 4096))
            if self._sr_td_ridge
            else 0
        )
        self._sr_probe_fixed_obs = None
        self._sr_probe_fresh_obs = None
        # Pre-update snapshots, taken in _make_train_val_split and consumed in _log_sr_diagnostics.
        self._sr_snapshot = None

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

        # How much of the return-to-go is explained by the episode timestep alone; see
        # :func:`~omnisafe.common.buffer.onpolicy_buffer.timestep_baseline_diagnostics`.  Logged
        # in both advantage-normalization modes, so a run can be used to decide whether
        # ``algo_cfgs.adv_norm_mode='timestep'`` has anything to remove.
        self._logger.register_key('Value/TimestepBaseline/RewardCorr')
        self._logger.register_key('Value/TimestepBaseline/CostCorr')

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
            self._logger.register_key('Misc/GramCondCost')
            # How many transitions w_r / w_c were actually fitted over this epoch -- constant under
            # ridge_data='epoch', but rising then plateauing at the buffer capacity otherwise.
            self._logger.register_key('Misc/RidgeFitSamples')
            if self._sr_ridge_data == 'cost_replay':
                # Size of the cost-inducing replay buffer -- rises then plateaus at
                # cost_replay_buffer_size; also caps how close 'try to sample steps_per_epoch
                # replay rows' gets to actually matching the epoch size early in training.
                self._logger.register_key('Misc/CostReplayBufferSize')
            if self._sr_phi_source == 'contrastive':
                # Keys match the f'Misc/Contrastive{k}' format _contrastive_update_phi's stats
                # dict produces (see contrastive.info_nce_loss). SR/{split}/PhiEffRank (registered
                # below by _register_sr_diagnostic_keys) already reflects the post-pretrain phi
                # for free, since train_data['phi'] is relabelled before _log_sr_diagnostics runs.
                self._logger.register_key('Misc/ContrastiveLoss')
                self._logger.register_key('Misc/ContrastivePosSim')
                self._logger.register_key('Misc/ContrastiveNegSim')
            if self._sr_phi_source == 'laplacian':
                # Keys match the f'Misc/Laplacian{k}' format _laplacian_update_phi's stats dict
                # produces (see laplacian.allo_loss). The two error series are the ones to watch:
                # Misc/GramCond should approach 1 as OffDiagErr falls, since a phi that satisfies
                # E[phi phi^T] = I hands the ridge solve an N * I design matrix.
                self._logger.register_key('Misc/LaplacianLoss')
                self._logger.register_key('Misc/LaplacianDirichlet')
                self._logger.register_key('Misc/LaplacianDiagErr')
                self._logger.register_key('Misc/LaplacianOffDiagErr')
                self._logger.register_key('Misc/LaplacianDualNorm')
            if self._sr_psi_objective == 'contrastive':
                # Keys match the f'Misc/CLearning{k}' format
                # _contrastive_update_successor_features' stats dict produces (see
                # clearning.density_ratio_loss). RatioMass against RatioMassTarget is the one to
                # read: the fitted ratio must integrate to sum_t gamma^t = 1 / (1 - gamma) under
                # rho, and it drifts off that long before the loss (which has no scale of its own)
                # shows anything.
                self._logger.register_key('Misc/CLearningLoss')
                self._logger.register_key('Misc/CLearningPosRatio')
                self._logger.register_key('Misc/CLearningNegSq')
                self._logger.register_key('Misc/CLearningRatioMass')
                self._logger.register_key('Misc/CLearningRatioMassTarget')
            self._register_sr_diagnostic_keys(_splits)

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

        self._logger.register_key('Metrics/TotalCost')

        # register environment specific keys
        for env_spec_key in self._env.env_spec_keys:
            self.logger.register_key(env_spec_key)

    def _register_sr_diagnostic_keys(self, splits: tuple[str, ...]) -> None:
        """Register the ``td_ridge`` successor-representation diagnostic keys.

        The metrics themselves are described in :meth:`_log_sr_diagnostics`; this only declares
        them with the logger. Like the value-critic diagnostics they are throttled to eval epochs,
        so the un-evaluated epochs record ``nan`` -- the same convention the ``Value/...`` keys
        already follow.

        Args:
            splits (tuple of str): The data splits in use -- ``('Train',)``, or
                ``('Train', 'Val')`` when ``algo_cfgs.n_val_episodes`` is positive.
        """
        for split in splits:
            # Layer 0 -- can phi's span express the one-step reward/cost, and how many
            # directions does it actually use?
            self._logger.register_key(f'SR/{split}/RidgeR2Reward')
            self._logger.register_key(f'SR/{split}/RidgeR2Cost')
            self._logger.register_key(f'SR/{split}/PhiEffRank')
            self._logger.register_key(f'SR/{split}/PhiStableRank')
            self._logger.register_key(f'SR/{split}/PhiDeadDims')
            self._logger.register_key(f'SR/{split}/PsiEffRank')

            # Layer 1 -- does psi match its target, before and after this epoch's update?
            for stage in ('BeforeUpdate', 'AfterUpdate'):
                self._logger.register_key(f'SR/{split}/{stage}/TDExplainedVar')
                self._logger.register_key(f'SR/{split}/{stage}/MCExplainedVar')
                self._logger.register_key(f'SR/{split}/{stage}/CosSim')
                self._logger.register_key(f'SR/{split}/{stage}/NormRatio')
                self._logger.register_key(f'SR/{split}/{stage}/PerDimEVMin')
                self._logger.register_key(f'SR/{split}/{stage}/PerDimEVMedian')
            self._logger.register_key(f'SR/{split}/TargetTrueEV')

            # Layer 3 -- the psi x w attribution grid (see _log_sr_diagnostics).
            for stream in ('Reward', 'Cost'):
                self._logger.register_key(f'SR/{split}/V{stream}Corr')
                self._logger.register_key(f'SR/{split}/PsiOracle{stream}Corr')
                self._logger.register_key(f'SR/{split}/WOracle{stream}Corr')
                self._logger.register_key(f'SR/{split}/Ceiling{stream}Corr')

        if 'Val' in splits:
            # Train - Val, logged directly so overfitting is one series rather than a
            # subtraction the reader has to do by eye.
            for key in (
                'RidgeR2Reward',
                'RidgeR2Cost',
                'BeforeUpdate/TDExplainedVar',
                'AfterUpdate/TDExplainedVar',
                'AfterUpdate/MCExplainedVar',
                'CeilingRewardCorr',
                'CeilingCostCorr',
            ):
                self._logger.register_key(f'SR/Gap/{key}')

        # Layer 2 -- non-stationarity of the pieces the fit assumes are fixed.
        for probe in ('Fixed', 'Fresh'):
            for feature in ('Phi', 'Psi'):
                self._logger.register_key(f'SR/{feature}Drift/{probe}/Rel')
                self._logger.register_key(f'SR/{feature}Drift/{probe}/Cos')
                self._logger.register_key(f'SR/{feature}Drift/{probe}/Subspace')
        for stream in ('R', 'C'):
            self._logger.register_key(f'SR/WDrift/Rel{stream}')
            self._logger.register_key(f'SR/WDrift/Cos{stream}')

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

        self._maybe_update_trained_phi(train_data, val_data)

        if self._sr_td_ridge:
            if self._sr_readout == 'ridge':
                self._ridge_update_successor_weights(train_data)
            else:  # 'sgd': fit w_r / w_c over the persistent cross-epoch buffer
                self._sgd_update_readout_weights(data)
            target_sr = train_data['target_sr']

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

        if self._sr_td_ridge:
            dataloader = DataLoader(
                dataset=TensorDataset(
                    obs,
                    act,
                    logp,
                    target_value_r,
                    target_value_c,
                    adv_r,
                    adv_c,
                    target_sr,
                ),
                batch_size=self._cfgs.algo_cfgs.batch_size,
                shuffle=True,
            )
        else:
            dataloader = DataLoader(
                dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
                batch_size=self._cfgs.algo_cfgs.batch_size,
                shuffle=True,
            )

        update_counts = 0
        final_kl = 0.0

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for batch in dataloader:
                if self._sr_td_ridge:
                    obs, act, logp, target_value_r, target_value_c, adv_r, adv_c, target_sr = batch
                else:
                    obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = batch
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)
                if self._sr_td_ridge:
                    self._update_successor_features(obs, target_sr)
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

        Under ``td_ridge`` this also carries two extra pieces of bookkeeping for the SR
        diagnostics, because it is the one hook every ``_update`` implementation calls exactly
        once, immediately after ``self._buf.get()`` and before any gradient step:

        * each returned dict gains an ``'_episode_lengths'`` entry (a plain list, not a tensor, so
          it passes through the mask untouched), needed to rebuild the Monte-Carlo successor
          feature per episode after the update;
        * :meth:`_sr_capture_pre_update` snapshots ``phi`` / ``psi`` / ``w`` while they are still
          at their pre-update values, which is what the drift metrics are measured against.
        """
        self._logger.store(self._buf.timestep_baseline_stats)

        n_val = getattr(self._cfgs.algo_cfgs, 'n_val_episodes', 0)
        need_slices = self._sr_td_ridge or n_val > 0
        episode_slices = self._buf.get_episode_slices() if need_slices else []

        if self._sr_td_ridge:
            self._sr_capture_pre_update(data)

        if n_val <= 0:
            if self._sr_td_ridge:
                data['_episode_lengths'] = [e - s for s, e in episode_slices]
            return data, None

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

        train_data, val_data = _apply(train_mask), _apply(val_mask)
        if self._sr_td_ridge:
            # Episodes are contiguous blocks assigned whole to one split, so masking preserves
            # their order and contiguity -- the lengths alone are enough to recover the boundaries.
            train_data['_episode_lengths'] = [
                e - s for i, (s, e) in enumerate(episode_slices) if i not in val_ep_idx
            ]
            val_data['_episode_lengths'] = [
                e - s for i, (s, e) in enumerate(episode_slices) if i in val_ep_idx
            ]
        return train_data, val_data

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
        if self._sr_td_ridge:
            self._log_sr_diagnostics(train_data, val_data)

    def _is_sr_eval_epoch(self) -> bool:
        """Whether this epoch runs the (expensive) diagnostic pass.

        Reuses the same throttle as the value-critic diagnostics -- ``early_eval_freq`` for the
        first 100 epochs, ``value_eval_freq`` after -- so every diagnostic series in the run
        shares one set of sample points and can be read side by side.
        """
        epoch = getattr(self, '_current_epoch', 0)
        eval_freq = getattr(self._cfgs.algo_cfgs, 'value_eval_freq', 50)
        early_eval_freq = getattr(self._cfgs.algo_cfgs, 'early_eval_freq', 5)
        effective_eval_freq = early_eval_freq if epoch < 100 else eval_freq
        return epoch % effective_eval_freq == 0

    @torch.no_grad()
    def _sr_capture_pre_update(self, data: dict) -> None:
        """Snapshot ``phi`` / ``psi`` / ``w`` before this epoch's update, for the drift metrics.

        Called from :meth:`_make_train_val_split`, i.e. after the rollout but before both the
        ridge re-solve and the gradient loop, so the recorded ``w`` is the one the epoch *started*
        with and ``SR/WDrift/*`` measures the movement this epoch's re-solve caused.

        Args:
            data (dict): The full (pre train/val split) epoch batch, used to draw the probe
                observations.
        """
        if not self._is_sr_eval_epoch():
            self._sr_snapshot = None
            return

        obs = data['obs']
        n_probe = min(self._sr_probe_size, obs.shape[0])
        if self._sr_probe_fixed_obs is None:
            self._sr_probe_fixed_obs = obs[torch.randperm(obs.shape[0])[:n_probe]].clone()
        self._sr_probe_fresh_obs = obs[torch.randperm(obs.shape[0])[:n_probe]].clone()

        phi_fixed, psi_fixed = self._actor_critic.sr_features(self._sr_probe_fixed_obs)
        phi_fresh, psi_fresh = self._actor_critic.sr_features(self._sr_probe_fresh_obs)
        self._sr_snapshot = {
            'phi_fixed': phi_fixed.clone(),
            'psi_fixed': psi_fixed.clone(),
            'phi_fresh': phi_fresh.clone(),
            'psi_fresh': psi_fresh.clone(),
            'w_r': self._actor_critic.sr_trunk.w_r.detach().clone(),
            'w_c': self._actor_critic.sr_trunk.w_c.detach().clone(),
        }

    @torch.no_grad()
    def _sr_prepare_split(self, data: dict) -> dict[str, torch.Tensor] | None:
        """Assemble every tensor the SR metrics for one split are computed from.

        The one non-obvious piece is ``mc_after``. ``psi`` is *defined* as the discounted sum of
        the ``phi`` stream, so scoring the post-update ``psi`` against a Monte-Carlo target built
        from the pre-update ``phi`` would fold ``phi``'s drift into what is meant to be a
        regression-quality number. Recomputing the target from the current ``phi`` keeps Layer 1
        (does ``psi`` fit?) and Layer 2 (did ``phi`` move?) measuring separate things.

        Args:
            data (dict): One split of the epoch batch, carrying ``'_episode_lengths'``.

        Returns:
            A dict of aligned, subsampled tensors, or ``None`` if the split is too small to
            produce meaningful statistics.
        """
        lengths = data.get('_episode_lengths')
        obs = data['obs']
        if lengths is None or obs.shape[0] < 2:
            return None

        # Same fallback rule as OnPolicyBuffer, so the recomputed Monte-Carlo target uses exactly
        # the discount the buffer's stored one did.
        gamma_sr = self._cfgs.model_cfgs.sr_cfgs.get('gamma_sr', None)
        gamma_sr = self._cfgs.algo_cfgs.gamma if gamma_sr is None else gamma_sr
        phi_after, psi_after = self._actor_critic.sr_features(obs)
        mc_after = sr_diagnostics.discount_cumsum_segments(phi_after, lengths, gamma_sr)

        prepared = {
            'phi_roll': data['phi'],
            'psi_roll': data['psi'],
            'target_sr': data['target_sr'],
            'mc_roll': data['discounted_sr'],
            'phi_after': phi_after,
            'psi_after': psi_after,
            'mc_after': mc_after,
            'reward': data['reward'].flatten(),
            'cost': data['cost'].flatten(),
            'true_r': data['discounted_ret'].flatten(),
            'true_c': data['discounted_cost_ret'].flatten(),
        }

        n = obs.shape[0]
        if n > self._sr_diag_max_samples:
            idx = torch.randperm(n, device=obs.device)[: self._sr_diag_max_samples]
            prepared = {k: v[idx] for k, v in prepared.items()}
        return prepared

    def _sr_stage_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """Score one ``psi`` prediction against one target, as four complementary numbers.

        Explained variance is the headline (the vector-valued stand-in for the scalar critics'
        correlation), but on its own it cannot say *how* a fit fails: the cosine and norm-ratio
        pair separates a wrong direction from a merely mis-scaled one, and the per-dimension
        spread separates a representation that is uniformly mediocre from one carrying all its
        accuracy in a few coordinates.

        Args:
            pred (torch.Tensor): Predicted successor features of shape ``(N, sr_dim)``.
            target (torch.Tensor): Target successor features of the same shape.

        Returns:
            A dict of metric-suffix to value, to be prefixed with the split and stage.
        """
        cos, norm_ratio = sr_diagnostics.cosine_norm_stats(pred, target)
        per_dim = sr_diagnostics.per_dim_explained_variance(pred, target)
        finite = per_dim[torch.isfinite(per_dim)]
        return {
            'TDExplainedVar': sr_diagnostics.explained_variance(pred, target),
            'CosSim': cos,
            'NormRatio': norm_ratio,
            'PerDimEVMin': finite.min().item() if finite.numel() else float('nan'),
            'PerDimEVMedian': finite.median().item() if finite.numel() else float('nan'),
        }

    def _log_sr_diagnostics(self, train_data: dict, val_data: dict | None) -> None:
        r"""Measure each link of the SR factorization separately, on both splits.

        The critic factors the value function as
        ``V(s) = psi(s) . w`` with ``psi(s) ~= sum_k gamma^k phi(s_{t+k})``, and the composite
        ``Value/{split}/...`` correlations cannot say which of the three fitted pieces -- the
        basis ``phi``, the successor regression ``psi``, or the read-out ``w`` -- is responsible
        when ``V`` is poor. These metrics can, in four groups:

        **Layer 0, basis quality** (``SR/{split}/RidgeR2*``, ``Phi*Rank``, ``PhiDeadDims``).
        Whether ``phi``'s span expresses the one-step reward/cost out of sample, and how many of
        its ``sr_dim`` directions carry signal. The ridge is solved on the training split only, so
        the ``Train`` / ``Val`` gap in ``RidgeR2*`` is the read-out's generalization gap directly.

        **Layer 1, regression quality** (``SR/{split}/{stage}/*``). How well ``psi`` matches its
        bootstrapped target and the Monte-Carlo truth, on both splits and both before and after
        the epoch's gradient steps. ``Train`` improving while ``Val`` does not is ``psi``
        memorizing the on-policy batch -- the failure this whole pass exists to detect.

        **Layer 2, non-stationarity** (``SR/PhiDrift/*``, ``PsiDrift/*``, ``WDrift/*``). ``psi``'s
        target and the ridge's basis are both defined in terms of ``phi``, so a ``phi`` that moves
        during the fit makes both stale. ``Subspace`` is the discriminating one: ``w`` is solved
        against ``phi``'s *span*, so a rotation inside a fixed span is recoverable by a re-solve
        and a tilt of the span itself is not.

        **Layer 3, attribution** (``SR/{split}/{V,PsiOracle,WOracle,Ceiling}*Corr``). A 2x2 over
        {current ``psi``, Monte-Carlo ``psi``} x {current ``w``, oracle ``w`` fitted on the
        training split}, each correlated against the true discounted return:

        +----------------+--------------------+--------------------+
        |                | current w          | oracle w*          |
        +================+====================+====================+
        | current psi    | ``V*Corr``         | ``WOracle*Corr``   |
        +----------------+--------------------+--------------------+
        | Monte-Carlo psi| ``PsiOracle*Corr`` | ``Ceiling*Corr``   |
        +----------------+--------------------+--------------------+

        ``Ceiling*Corr`` is the ceiling of the factorization itself -- perfect successor features
        and the best possible read-out on top of them. If it is low, ``phi`` cannot express the
        value function and no amount of fitting ``psi`` or ``w`` will help; if it is high while
        ``V*Corr`` is low, whichever oracle recovers the correlation names the broken piece.

        Args:
            train_data (dict): The training-split epoch batch.
            val_data (dict or None): The held-out split, or ``None`` when ``n_val_episodes`` is 0.
        """
        if not self._is_sr_eval_epoch():
            return

        prepared: dict[str, dict[str, torch.Tensor]] = {}
        train_prepared = self._sr_prepare_split(train_data)
        if train_prepared is None:
            return
        prepared['Train'] = train_prepared
        if val_data is not None:
            val_prepared = self._sr_prepare_split(val_data)
            if val_prepared is not None:
                prepared['Val'] = val_prepared

        # Oracle read-outs are fitted on the training split and applied to both, so the Val
        # column is a genuine held-out test of the read-out rather than an in-sample refit.
        # Each stream is fitted at the same kappa the training solve uses for it, so WOracle* /
        # Ceiling* stay a measure of the read-out rather than of a regularization mismatch.
        kappa_r = self._cfgs.model_cfgs.sr_cfgs.get('ridge_kappa', 1e-3)
        kappa_c = self._cfgs.model_cfgs.sr_cfgs.get('ridge_kappa_cost', None)
        if kappa_c is None:
            kappa_c = kappa_r
        oracle_w = {
            f'{feat}_{stream}': sr_diagnostics.ridge_fit(
                train_prepared[feat],
                train_prepared[target],
                kappa,
            )
            for feat in ('psi_after', 'mc_after')
            for stream, target, kappa in (('r', 'true_r', kappa_r), ('c', 'true_c', kappa_c))
        }

        scalars: dict[str, dict[str, float]] = {}
        for split, prep in prepared.items():
            scalars[split] = self._sr_split_scalars(prep, oracle_w)
            self._logger.store({f'SR/{split}/{k}': v for k, v in scalars[split].items()})

        if 'Val' in scalars:
            for key in (
                'RidgeR2Reward',
                'RidgeR2Cost',
                'BeforeUpdate/TDExplainedVar',
                'AfterUpdate/TDExplainedVar',
                'AfterUpdate/MCExplainedVar',
                'CeilingRewardCorr',
                'CeilingCostCorr',
            ):
                self._logger.store({f'SR/Gap/{key}': scalars['Train'][key] - scalars['Val'][key]})

        self._log_sr_drift()
        self._log_sr_scatters(prepared, oracle_w)

    def _sr_split_scalars(
        self,
        prep: dict[str, torch.Tensor],
        oracle_w: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Compute the Layer 0, 1 and 3 metrics for a single split.

        Args:
            prep (dict): The split's tensors, as returned by :meth:`_sr_prepare_split`.
            oracle_w (dict): Oracle read-out weights fitted on the training split, keyed
                ``'{psi_after,mc_after}_{r,c}'``.

        Returns:
            A dict of key-suffix to value, to be prefixed with ``SR/{split}/``.
        """
        w_r = self._actor_critic.sr_trunk.w_r.detach()
        w_c = self._actor_critic.sr_trunk.w_c.detach()
        out: dict[str, float] = {}

        # --- Layer 0: is phi a usable basis? -------------------------------------------------
        # Measured on the rollout-time phi and the live w, which is exactly the pair the ridge
        # solved for this epoch, so Train is in-sample fit and Val is its generalization.
        out['RidgeR2Reward'] = sr_diagnostics.r2_score(prep['phi_roll'] @ w_r, prep['reward'])
        out['RidgeR2Cost'] = sr_diagnostics.r2_score(prep['phi_roll'] @ w_c, prep['cost'])
        out['PhiEffRank'] = sr_diagnostics.effective_rank(prep['phi_roll'])
        out['PhiStableRank'] = sr_diagnostics.stable_rank(prep['phi_roll'])
        out['PhiDeadDims'] = sr_diagnostics.dead_dim_fraction(prep['phi_roll'])
        out['PsiEffRank'] = sr_diagnostics.effective_rank(prep['psi_roll'])

        # --- Layer 1: does psi fit the successor recursion? ----------------------------------
        # BeforeUpdate reads psi straight off the rollout (the SR counterpart of value_r/value_c);
        # AfterUpdate re-queries the live network. Both are scored against the same fixed
        # target_sr, so the pair isolates what this epoch's gradient steps achieved.
        for stage, psi, mc_target in (
            ('BeforeUpdate', prep['psi_roll'], prep['mc_roll']),
            ('AfterUpdate', prep['psi_after'], prep['mc_after']),
        ):
            for name, value in self._sr_stage_metrics(psi, prep['target_sr']).items():
                out[f'{stage}/{name}'] = value
            out[f'{stage}/MCExplainedVar'] = sr_diagnostics.explained_variance(psi, mc_target)
        out['TargetTrueEV'] = sr_diagnostics.explained_variance(prep['target_sr'], prep['mc_roll'])

        # --- Layer 3: which piece is responsible for V's error? ------------------------------
        for stream, weight, true in (('Reward', w_r, 'true_r'), ('Cost', w_c, 'true_c')):
            suffix = 'r' if stream == 'Reward' else 'c'
            out[f'V{stream}Corr'] = sr_diagnostics.correlation(
                prep['psi_after'] @ weight,
                prep[true],
            )
            out[f'PsiOracle{stream}Corr'] = sr_diagnostics.correlation(
                prep['mc_after'] @ weight,
                prep[true],
            )
            out[f'WOracle{stream}Corr'] = sr_diagnostics.correlation(
                prep['psi_after'] @ oracle_w[f'psi_after_{suffix}'],
                prep[true],
            )
            out[f'Ceiling{stream}Corr'] = sr_diagnostics.correlation(
                prep['mc_after'] @ oracle_w[f'mc_after_{suffix}'],
                prep[true],
            )
        return out

    @torch.no_grad()
    def _log_sr_drift(self) -> None:
        """Log how far ``phi`` / ``psi`` / ``w`` moved over this epoch's update.

        Consumes the snapshot :meth:`_sr_capture_pre_update` took before the gradient loop. The
        two probe sets are reported separately because they answer different questions: ``Fixed``
        holds the input distribution constant so its drift is purely the feature map moving, while
        ``Fresh`` tracks the states the current policy visits.
        """
        snapshot = self._sr_snapshot
        if snapshot is None:
            return

        for probe, probe_obs in (
            ('Fixed', self._sr_probe_fixed_obs),
            ('Fresh', self._sr_probe_fresh_obs),
        ):
            if probe_obs is None:
                continue
            phi_now, psi_now = self._actor_critic.sr_features(probe_obs)
            for feature, now, before in (
                ('Phi', phi_now, snapshot[f'phi_{probe.lower()}']),
                ('Psi', psi_now, snapshot[f'psi_{probe.lower()}']),
            ):
                self._logger.store(
                    {
                        f'SR/{feature}Drift/{probe}/Rel': sr_diagnostics.relative_change(
                            now,
                            before,
                        ),
                        f'SR/{feature}Drift/{probe}/Cos': sr_diagnostics.mean_cosine(now, before),
                        f'SR/{feature}Drift/{probe}/Subspace': sr_diagnostics.subspace_distance(
                            now,
                            before,
                        ),
                    },
                )

        for stream, name in (('R', 'w_r'), ('C', 'w_c')):
            now = getattr(self._actor_critic.sr_trunk, name).detach()
            before = snapshot[name]
            self._logger.store(
                {
                    f'SR/WDrift/Rel{stream}': sr_diagnostics.relative_change(now, before),
                    f'SR/WDrift/Cos{stream}': sr_diagnostics.mean_cosine(now, before),
                },
            )

    def _log_sr_scatters(
        self,
        prepared: dict[str, dict[str, torch.Tensor]],
        oracle_w: dict[str, torch.Tensor],
    ) -> None:
        """Log the scatter plots behind the SR scalars.

        Three per split. The first two show the ``psi`` fit flattened over ``(sample, dimension)``
        pairs -- against its bootstrapped target and against the Monte-Carlo truth -- which
        separate a residual that is uniform noise from one that is structured bias, something no
        single explained-variance number can. The third plots the ceiling prediction against the
        true return, the picture behind ``SR/{split}/CeilingRewardCorr``: a cloud that fails to
        line up there is ``phi`` failing, not ``psi`` or ``w``.

        Args:
            prepared (dict): Per-split tensors, as returned by :meth:`_sr_prepare_split`.
            oracle_w (dict): Oracle read-out weights fitted on the training split.
        """
        for split, prep in prepared.items():
            psi, target = prep['psi_after'], prep['target_sr']
            flat_n = psi.numel()
            idx = torch.randperm(flat_n, device=psi.device)[: min(flat_n, 4000)]
            self._logger.log_scatter_image(
                f'SR/{split}/PsiTargetScatter',
                x_values=target.reshape(-1)[idx],
                y_values=psi.reshape(-1)[idx],
                xlabel='Target psi (per dim)',
                ylabel='Predicted psi (per dim)',
            )
            self._logger.log_scatter_image(
                f'SR/{split}/PsiMonteCarloScatter',
                x_values=prep['mc_after'].reshape(-1)[idx],
                y_values=psi.reshape(-1)[idx],
                xlabel='Monte-Carlo psi (per dim)',
                ylabel='Predicted psi (per dim)',
            )

            n = prep['true_r'].shape[0]
            row_idx = torch.randperm(n, device=psi.device)[: min(n, 2000)]
            self._logger.log_scatter_image(
                f'SR/{split}/CeilingRewardScatter',
                x_values=prep['true_r'][row_idx],
                y_values=(prep['mc_after'] @ oracle_w['mc_after_r'])[row_idx],
                xlabel='True G_r',
                ylabel='Monte-Carlo psi . oracle w_r',
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
        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.reward_critic(obs)[0], target_value_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            excluded_ids = getattr(self._actor_critic, '_sr_critic_norm_excluded_ids', set())
            for param in self._actor_critic.reward_critic.parameters():
                if not param.requires_grad or id(param) in excluded_ids:
                    # frozen phi network (sr_cfgs.phi_source), or -- under phi_source='contrastive'
                    # -- phi_net, trainable but fit only by its own InfoNCE loss/optimizer, never
                    # this one; a no-op under every other phi_source.
                    continue
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
        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.cost_critic(obs)[0], target_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            # Cost gets its own coefficient (null falls back to critic_norm_coef), the same
            # null-fallback convention as sr_cfgs.ridge_kappa_cost / w_weight_decay_cost --
            # a sparser, differently-scaled cost target generally wants different shrinkage
            # than the dense reward critic it shares no parameters with.
            critic_norm_coef_cost = getattr(self._cfgs.algo_cfgs, 'critic_norm_coef_cost', None)
            if critic_norm_coef_cost is None:
                critic_norm_coef_cost = self._cfgs.algo_cfgs.critic_norm_coef
            excluded_ids = getattr(self._actor_critic, '_sr_critic_norm_excluded_ids', set())
            for param in self._actor_critic.cost_critic.parameters():
                if not param.requires_grad or id(param) in excluded_ids:
                    # frozen phi network (sr_cfgs.phi_source), or -- under phi_source='contrastive'
                    # -- phi_net, trainable but fit only by its own InfoNCE loss/optimizer, never
                    # this one; a no-op under every other phi_source.
                    continue
                loss += param.pow(2).sum() * critic_norm_coef_cost

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store({'Loss/Loss_cost_critic': loss.mean().item()})

    def _ridge_update_successor_weights(self, train_data: dict[str, torch.Tensor]) -> None:
        """Refresh the ``td_ridge`` successor-representation read-out weights.

        Solves ``w_r`` / ``w_c`` in closed form once per :meth:`_update` call (not per minibatch)
        -- mirroring how the buffer's GAE targets are also computed once per epoch from the
        just-collected data. Either way the fit sees only ``train_data``, never the held-out
        validation episodes, so the ``SR/Gap/RidgeR2*`` series stays a genuine held-out test.

        ``sr_cfgs.ridge_data`` selects the design matrix:

        * ``'epoch'`` (default): the fresh epoch's training split, using the ``phi`` the rollout
            already stored.
        * ``'buffer'``: the persistent cross-epoch history. The read-out ``r(s) ~= phi(s) . w`` is
            policy-independent, so stale-policy transitions remain correct regression data, and
            pooling them widens the state coverage the Gram matrix is built from -- which is what
            conditions the solve when ``phi`` is rank-deficient (``phi_source='random'``). ``phi``
            is recomputed from the stored observations rather than cached, so the basis is always
            the current one even when it drifts under ``phi_source='trunk'``.
        * ``'cost_replay'``: the fresh epoch's training split, plus an equal number of past
            cost-inducing (``cost > 0``) transitions resampled from a dedicated replay buffer --
            e.g. at ``steps_per_epoch=5000`` this tries for 5000 fresh + 5000 replayed rows. Costs
            are typically sparse, so an ``'epoch'``-only fit sees very few cost-positive rows per
            update; retaining and reinjecting only those (rather than the whole history, as
            ``'buffer'`` does) concentrates the replay on the signal ``w_c`` actually needs without
            diluting it with the many cost-zero transitions that already dominate the fresh epoch.
            Both ``w_r`` and ``w_c`` are solved on the same combined design matrix, so the reward
            read-out sees the wider state coverage too. ``phi`` is recomputed from the combined
            observations, as under ``'buffer'``.

        Args:
            train_data (dict[str, torch.Tensor]): The training-split epoch batch (post
                :meth:`_make_train_val_split`), including the ``phi``, ``reward``, and ``cost``
                fields used for the ridge solve.
        """
        sr_cfgs = self._cfgs.model_cfgs.sr_cfgs
        if self._sr_ridge_data == 'buffer':
            assert self._sr_readout_buf is not None
            # Only the training split is retained, so held-out episodes never enter the fit --
            # neither this epoch's nor, once they have aged into the history, any earlier one's.
            self._sr_readout_buf.add(train_data['obs'], train_data['reward'], train_data['cost'])
            n_samples = sr_cfgs.get('ridge_buffer_samples', None)
            if n_samples is None or int(n_samples) >= len(self._sr_readout_buf):
                obs, reward, cost = self._sr_readout_buf.all()
            else:
                obs, reward, cost = self._sr_readout_buf.sample(int(n_samples))
            phi, _ = self._actor_critic.sr_features(obs)  # detached; recomputed under current phi
        elif self._sr_ridge_data == 'cost_replay':
            assert self._sr_cost_readout_buf is not None
            epoch_obs, epoch_reward, epoch_cost = (
                train_data['obs'],
                train_data['reward'],
                train_data['cost'],
            )
            # Only the training split's cost-inducing rows are retained -- same held-out
            # discipline as the 'buffer' path above, and consistent across epochs since a row
            # added while cost-inducing stays in the buffer even if later re-evaluated as 0.
            cost_mask = epoch_cost.reshape(-1) > 0
            if bool(cost_mask.any()):
                self._sr_cost_readout_buf.add(
                    epoch_obs[cost_mask],
                    epoch_reward[cost_mask],
                    epoch_cost[cost_mask],
                )
            n_replay_avail = len(self._sr_cost_readout_buf)
            n_target = epoch_obs.shape[0]  # try to match the epoch 1:1, e.g. 5000 + 5000
            if n_replay_avail == 0:
                obs, reward, cost = epoch_obs, epoch_reward, epoch_cost
            else:
                # Fewer stored than requested: take them all rather than sample-with-replacement,
                # which would just duplicate the same handful of rows early in training.
                if n_replay_avail >= n_target:
                    replay_obs, replay_reward, replay_cost = self._sr_cost_readout_buf.sample(
                        n_target,
                    )
                else:
                    replay_obs, replay_reward, replay_cost = self._sr_cost_readout_buf.all()
                obs = torch.cat([epoch_obs, replay_obs], dim=0)
                reward = torch.cat([epoch_reward, replay_reward], dim=0)
                cost = torch.cat([epoch_cost, replay_cost], dim=0)
            phi, _ = self._actor_critic.sr_features(obs)  # detached; recomputed under current phi
        else:
            phi, reward, cost = train_data['phi'], train_data['reward'], train_data['cost']

        stats = self._actor_critic.sr_trunk.ridge_update(
            phi,
            reward,
            cost,
            ridge_kappa=sr_cfgs.get('ridge_kappa', 1e-3),
            ema_tau=sr_cfgs.get('ema_tau', 1.0),
            ridge_kappa_cost=sr_cfgs.get('ridge_kappa_cost', None),
        )
        stats['Misc/RidgeFitSamples'] = float(phi.shape[0])
        if self._sr_ridge_data == 'cost_replay':
            assert self._sr_cost_readout_buf is not None
            stats['Misc/CostReplayBufferSize'] = float(len(self._sr_cost_readout_buf))
        self._logger.store(stats)

    def _sgd_update_readout_weights(self, data: dict[str, torch.Tensor]) -> None:
        r"""Fit the ``readout='sgd'`` read-out weights ``w_r`` / ``w_c`` by gradient descent.

        The on-policy analogue of :meth:`_ridge_update_successor_weights`. Rather than re-solving
        the ``phi -> reward/cost`` regression in closed form on the fresh epoch (and then throwing
        that data away), it appends the epoch to a persistent replay buffer and takes several SGD
        steps over the whole accumulated history. This is valid precisely because the read-out
        ``r(s) \approx phi(s) . w`` is policy-independent, so stale-policy transitions are still
        correct regression targets -- most cleanly when ``phi`` is frozen (``phi_source`` ``random``
        / ``separate``), where the target is fully stationary.

        ``phi`` is recomputed from the stored observations via :meth:`sr_features` (detached), so
        this loss trains ``w`` only, never the representation.

        Args:
            data (dict[str, torch.Tensor]): The full (pre train/val split) epoch batch, providing
                the ``obs`` / ``reward`` / ``cost`` fields added to the persistent buffer.
        """
        assert self._sr_readout_buf is not None
        sr_cfgs = self._cfgs.model_cfgs.sr_cfgs
        # The read-out regression is policy-independent, so all of the epoch's data is valid.
        self._sr_readout_buf.add(data['obs'], data['reward'], data['cost'])

        grad_steps = int(sr_cfgs.get('readout_grad_steps', 50))
        batch_size = sr_cfgs.get('readout_batch_size', None) or self._cfgs.algo_cfgs.batch_size
        stats: dict[str, float] = {}
        for _ in range(grad_steps):
            obs_b, reward_b, cost_b = self._sr_readout_buf.sample(batch_size)
            phi_b, _ = self._actor_critic.sr_features(obs_b)  # detached: trains w only
            loss, stats = self._actor_critic.sr_trunk.regression_loss(phi_b, reward_b, cost_b)
            self._actor_critic.sr_readout_optimizer.zero_grad()
            loss.backward()
            if self._cfgs.algo_cfgs.use_max_grad_norm:
                clip_grad_norm_(
                    [self._actor_critic.sr_trunk.w_r, self._actor_critic.sr_trunk.w_c],
                    self._cfgs.algo_cfgs.max_grad_norm,
                )
            self._actor_critic.sr_readout_optimizer.step()
        if stats:
            # Counterpart of the same key on the ridge path: the history w was fitted over.
            stats['Misc/RidgeFitSamples'] = float(len(self._sr_readout_buf))
            self._logger.store(stats)

    def _maybe_update_trained_phi(
        self,
        train_data: dict[str, torch.Tensor],
        val_data: dict[str, torch.Tensor] | None = None,
    ) -> None:
        r"""Train ``phi`` by its own loss, if this ``phi_source`` has one, then relabel.

        The single entry point every ``_update`` implementation calls, immediately after
        :meth:`_make_train_val_split` and strictly *before* the ridge solve and the ``psi`` TD
        update -- "before updating psi for epoch 0 we should first train phi." A no-op under every
        ``phi_source`` that is frozen at init or read off the shared trunk.

        The cadence is pretrain-then-continually-adapt, and both halves matter: epoch 0 takes a
        large one-time burst (``sr_cfgs.phi_{source}_pretrain_steps``) to move ``phi`` out of its
        random init before anything downstream is fit against it, and every epoch after takes a
        small one (``..._steps_per_epoch``) so ``phi`` keeps up with the shifting on-policy state
        distribution instead of describing a policy that no longer exists.

        Factored out of the three ``_update`` implementations that need it rather than duplicated
        into each. :class:`~omnisafe.algorithms.on_policy.base.natural_pg.NaturalPG` (with its
        TRPO / CPO / PCPO descendants) and
        :class:`~omnisafe.algorithms.on_policy.first_order.focops.FOCOPS` reimplement ``_update``
        independently rather than calling ``super()``, so anything that lives inline in
        :meth:`PolicyGradient._update` alone silently never runs for them -- which is exactly the
        bug that shipped when the contrastive hook was written inline three times.

        Args:
            train_data (dict[str, torch.Tensor]): The training-split epoch batch, mutated in place
                by the relabelling.
            val_data (dict[str, torch.Tensor] or None, optional): The held-out split, relabelled
                identically when present. Passing it is not optional in spirit: the SR diagnostics
                read ``phi`` from both splits and difference them (``SR/Gap/*``), so relabelling
                only the training split would leave the two sides describing *different networks*
                and turn the generalization gap into a measure of how far ``phi`` moved this epoch.
                Defaults to ``None``.
        """
        if not self._sr_td_ridge:
            return
        epoch = getattr(self, '_current_epoch', 0)

        if self._sr_psi_objective == 'contrastive':
            # phi has no objective of its own here -- it is fitted together with psi, and
            # _update_successor_features is a no-op for the duration. relabel_target_sr is False
            # for the same reason: there is no TD target left to be stale.
            n_steps = self._sr_psi_pretrain_steps if epoch == 0 else self._sr_psi_steps_per_epoch
            if n_steps <= 0:
                return
            self._contrastive_update_successor_features(train_data, n_steps)
        elif self._sr_phi_trained:
            n_steps = self._sr_phi_pretrain_steps if epoch == 0 else self._sr_phi_steps_per_epoch
            if n_steps <= 0:
                return
            if self._sr_phi_source == 'contrastive':
                self._contrastive_update_phi(train_data, n_steps)
            else:
                self._laplacian_update_phi(train_data, n_steps)
        else:
            return

        relabel_target_sr = epoch == 0 and self._sr_psi_objective == 'td'
        for split in (train_data, val_data):
            if split is not None:
                self._relabel_after_phi_update(split, relabel_target_sr=relabel_target_sr)

    def _laplacian_update_phi(self, train_data: dict[str, torch.Tensor], n_steps: int) -> None:
        r"""Train ``phi`` by the Augmented Lagrangian Laplacian Objective (``'laplacian'`` only).

        Drives ``phi`` to the bottom ``sr_dim`` eigenvectors of the transition graph's Laplacian --
        the basis in which an arbitrary state signal is best approximated linearly at that
        dimensionality, which is precisely what the ridge read-outs ``w_r`` / ``w_c`` are asked to
        do. See :mod:`omnisafe.utils.laplacian` for the objective and why it needs an augmented
        Lagrangian rather than a fixed penalty.

        Each step draws three index sets from the epoch batch: a set of graph edges, and *two
        independent* sets of states. The two state sets are not redundant -- the barrier term's
        gradient carries the constraint residual as a factor, and a single minibatch cannot
        estimate that factor and the differentiable term without correlating them into a biased
        product (see :func:`~omnisafe.utils.laplacian.allo_loss`). All three are drawn without
        replacement via ``randperm``: a duplicated row would appear in the covariance estimate
        twice and be counted as evidence of a correlation it does not have.

        Trains only ``self._actor_critic.sr_trunk.phi_net`` (via its own ``sr_phi_optimizer``) --
        never ``psi_head`` or the trunk body, which this loss's forward pass never even reads. The
        Lagrange multipliers are *not* parameters and are not touched by that optimizer: they are a
        buffer on ``phi_net`` updated by explicit dual ascent after each primal step.

        Args:
            train_data (dict[str, torch.Tensor]): The training-split epoch batch, providing ``obs``
                and ``_episode_lengths`` (always present when ``self._sr_td_ridge``, see
                :meth:`_make_train_val_split`) that the edges and states are drawn from.
            n_steps (int): Number of gradient steps to take.
        """
        sr_cfgs = self._cfgs.model_cfgs.sr_cfgs
        barrier = float(sr_cfgs.get('phi_laplacian_barrier', 2.0))
        dual_lr = float(sr_cfgs.get('phi_laplacian_dual_lr', 0.01))
        batch_size = int(
            sr_cfgs.get('phi_laplacian_batch_size', None) or self._cfgs.algo_cfgs.batch_size,
        )

        obs = train_data['obs']
        # horizon=1 is not a tunable here: the Laplacian is defined on the *one-step* transition
        # graph, and a wider window would decompose a different (k-step-averaged) operator whose
        # eigenvectors are no longer the ones the argument above is about.
        src_pool, dst_pool = contrastive.sample_temporal_pairs(
            train_data['_episode_lengths'],
            horizon=1,
            device=obs.device,
        )
        if src_pool.numel() == 0:
            # e.g. every episode this epoch has length 1 -- the graph has no edges yet.
            return

        phi_net = self._actor_critic.sr_trunk.phi_net
        n_rows, n_edges = obs.shape[0], src_pool.shape[0]
        edge_batch = min(batch_size, n_edges)
        # Two disjoint halves, so the covariance estimates really are independent.
        state_batch = min(batch_size, n_rows // 2)
        stats: dict[str, float] = {}
        for _ in range(n_steps):
            edges = torch.randperm(n_edges, device=obs.device)[:edge_batch]
            rows = torch.randperm(n_rows, device=obs.device)[: 2 * state_batch]
            loss, violation, stats = laplacian.allo_loss(
                phi_s=phi_net(obs[src_pool[edges]]),
                phi_s_next=phi_net(obs[dst_pool[edges]]),
                phi_rho_a=phi_net(obs[rows[:state_batch]]),
                phi_rho_b=phi_net(obs[rows[state_batch:]]),
                dual=phi_net.dual,
                barrier=barrier,
            )
            self._actor_critic.sr_phi_optimizer.zero_grad()
            loss.backward()
            if self._cfgs.algo_cfgs.use_max_grad_norm:
                clip_grad_norm_(phi_net.parameters(), self._cfgs.algo_cfgs.max_grad_norm)
            distributed.avg_grads(phi_net)
            self._actor_critic.sr_phi_optimizer.step()
            # Dual ascent, after the primal step and on the *averaged* violation: the multipliers
            # are a buffer, not a parameter, so nothing else keeps them identical across ranks,
            # and ranks whose multipliers diverged would pull one shared phi in different
            # directions.
            laplacian.dual_ascent_(phi_net.dual, distributed.dist_avg(violation), dual_lr)
        if stats:
            self._logger.store({f'Misc/Laplacian{k}': v for k, v in stats.items()})

    def _contrastive_update_phi(self, train_data: dict[str, torch.Tensor], n_steps: int) -> None:
        r"""Train ``phi`` by a time-contrastive InfoNCE loss (``phi_source='contrastive'`` only).

        Pulls together states visited close in time within the same rollout episode and pushes
        apart temporally-distant / other-episode ones (see :mod:`omnisafe.utils.contrastive`).
        Called from :meth:`_maybe_update_trained_phi`, which owns the pretrain-then-continual
        cadence ``n_steps`` comes from and the relabelling that follows.

        Trains only ``self._actor_critic.sr_trunk.phi_net`` (via its own ``sr_phi_optimizer``) --
        never ``psi_head`` or the trunk body, which this loss's forward pass never even reads.

        Args:
            train_data (dict[str, torch.Tensor]): The training-split epoch batch, providing
                ``obs`` and ``_episode_lengths`` (always present when ``self._sr_td_ridge``, see
                :meth:`_make_train_val_split`) that the temporal pairs are sampled from.
            n_steps (int): Number of gradient steps to take.
        """
        sr_cfgs = self._cfgs.model_cfgs.sr_cfgs
        horizon = int(sr_cfgs.get('phi_contrastive_horizon', 5))
        temperature = sr_cfgs.get('phi_contrastive_temperature', 0.1)
        batch_size = sr_cfgs.get('phi_contrastive_batch_size', None) or self._cfgs.algo_cfgs.batch_size

        obs = train_data['obs']
        anchor_pool, positive_pool = contrastive.sample_temporal_pairs(
            train_data['_episode_lengths'],
            horizon,
            device=obs.device,
        )
        if anchor_pool.numel() == 0:
            # e.g. every episode this epoch has length 1 -- nothing to contrast against yet.
            return

        phi_net = self._actor_critic.sr_trunk.phi_net
        stats: dict[str, float] = {}
        for _ in range(n_steps):
            idx = torch.randint(
                anchor_pool.shape[0],
                (min(batch_size, anchor_pool.shape[0]),),
                device=obs.device,
            )
            anchor_feat = phi_net(obs[anchor_pool[idx]])
            positive_feat = phi_net(obs[positive_pool[idx]])
            loss, stats = contrastive.info_nce_loss(anchor_feat, positive_feat, temperature)
            self._actor_critic.sr_phi_optimizer.zero_grad()
            loss.backward()
            if self._cfgs.algo_cfgs.use_max_grad_norm:
                clip_grad_norm_(phi_net.parameters(), self._cfgs.algo_cfgs.max_grad_norm)
            distributed.avg_grads(phi_net)
            self._actor_critic.sr_phi_optimizer.step()
        if stats:
            self._logger.store({f'Misc/Contrastive{k}': v for k, v in stats.items()})

    @torch.no_grad()
    def _relabel_after_phi_update(
        self,
        train_data: dict[str, torch.Tensor],
        relabel_target_sr: bool,
    ) -> None:
        r"""Recompute ``train_data``'s ``phi`` (and, at epoch 0, ``target_sr``) after phi moved.

        ``phi`` was cached by the rollout under the network as it stood *before* this epoch's
        phi update, so once :meth:`_contrastive_update_phi` / :meth:`_laplacian_update_phi` has
        moved it, that cached value -- and anything computed from it -- is stale. Two of the three
        stale tensors are cheap and *exact* to fix, so they are fixed unconditionally:

        * ``phi``, a single forward pass -- the same idea
          :meth:`_ridge_update_successor_weights`'s ``ridge_data='buffer'`` / ``'cost_replay'``
          paths already use;
        * ``discounted_sr``, the Monte-Carlo successor feature the SR diagnostics score
          ``target_sr`` against. It is a plain discounted sum of the ``phi`` stream, so recomputing
          it from the relabelled ``phi`` is exact. Leaving it alone would score a relabelled
          ``target_sr`` against a ground truth built from a feature map that no longer exists,
          which is a comparison between two different quantities rather than a measurement of
          anything.

        ``target_sr`` is the third, and unlike those it is only relabelled at epoch 0 -- where the
        pretraining burst moves ``phi`` far more than a normal epoch's drift. Smaller continual
        drift in later epochs is left alone, since it self-corrects the same way
        ``phi_source='trunk'``'s per-epoch drift already does without special-casing; the
        resulting ``SR/*/TargetTrueEV`` (stale TD target vs. fresh MC target) is then a direct
        read-out of how much that residual staleness costs.

        Args:
            train_data (dict[str, torch.Tensor]): One split of the epoch batch, mutated in place.
            relabel_target_sr (bool): Whether to also recompute ``target_sr`` (epoch 0 only).
        """
        sr_cfgs = self._cfgs.model_cfgs.sr_cfgs
        # Same fallback rule as OnPolicyBuffer and _sr_prepare_split, so every discounted sum of
        # the phi stream in this codebase uses one discount.
        gamma_sr = sr_cfgs.get('gamma_sr', None)
        gamma_sr = self._cfgs.algo_cfgs.gamma if gamma_sr is None else gamma_sr

        train_data['phi'] = self._actor_critic.sr_trunk.phi(train_data['obs'])
        if 'discounted_sr' in train_data:
            train_data['discounted_sr'] = sr_diagnostics.discount_cumsum_segments(
                train_data['phi'],
                train_data['_episode_lengths'],
                gamma_sr,
            )
        if not relabel_target_sr:
            return
        # gae_lambda_targets_segments only reimplements the 'gae' branch of the buffer's own
        # estimator; target_sr was originally built with whatever algo_cfgs.adv_estimation_method
        # selects (finish_path passes no explicit estimator, so it defaults to that same setting),
        # so silently mismatching estimators here would be wrong rather than merely approximate.
        # Under any other estimator, leave target_sr as its (epoch-0-stale) rollout-time value --
        # the same accepted staleness _update_successor_features already tolerates within an
        # epoch under phi_source='trunk'.
        if self._cfgs.algo_cfgs.adv_estimation_method != 'gae':
            self._logger.log(
                f'phi_source="{self._sr_phi_source}": target_sr left un-relabelled after the '
                f'epoch-0 phi pretraining, because adv_estimation_method is '
                f'"{self._cfgs.algo_cfgs.adv_estimation_method}" and only "gae" has a segment-wise '
                'reimplementation here. psi spends epoch 0 fitting a target built from the '
                'pre-pretraining phi.',
            )
            return
        train_data['target_sr'] = sr_diagnostics.gae_lambda_targets_segments(
            values=train_data['psi'],  # rollout-time psi -- untouched by the contrastive step
            rewards=train_data['phi'],  # just-relabelled above
            lengths=train_data['_episode_lengths'],
            lam=sr_cfgs.get('lam_sr', 0.95),
            gamma=gamma_sr,
        )

    def _contrastive_update_successor_features(
        self,
        train_data: dict[str, torch.Tensor],
        n_steps: int,
    ) -> None:
        r"""Fit ``phi`` and ``psi`` jointly as a successor-measure critic (C-learning).

        The replacement for :meth:`_update_successor_features` under
        ``sr_cfgs.psi_objective='contrastive'``. Instead of bootstrapping ``psi`` against a target
        built from ``phi``, both are trained as the two factors of one bilinear critic
        ``psi(s)^T phi(g)`` fitted to the discounted successor measure's density ratio -- positives
        drawn from each anchor's own geometrically-discounted future, negatives from the batch at
        large. See :mod:`omnisafe.utils.clearning` for the objective, and why fitting the ratio in
        its linear (not log) scale is what keeps the ``value(s) = psi(s) . w`` read-out valid.

        Runs as its own loop over the epoch rather than inside the policy's minibatch loop, the way
        :meth:`_laplacian_update_phi` does. That is not merely convenient: the negatives are the
        ``rho`` sample, and their count is what the ``E_rho`` estimate's quality rests on, so this
        loss wants a batch sized for *it* rather than whatever ``algo_cfgs.batch_size`` the policy
        update happens to use. :meth:`_update_successor_features` becomes a no-op for the duration,
        so ``psi`` is fitted here and nowhere else.

        Args:
            train_data (dict[str, torch.Tensor]): The training-split epoch batch, providing ``obs``
                and ``_episode_lengths``.
            n_steps (int): Number of gradient steps to take.
        """
        sr_cfgs = self._cfgs.model_cfgs.sr_cfgs
        gamma_sr = sr_cfgs.get('gamma_sr', None)
        gamma_sr = self._cfgs.algo_cfgs.gamma if gamma_sr is None else gamma_sr
        batch_size = int(
            sr_cfgs.get('psi_contrastive_batch_size', None) or self._cfgs.algo_cfgs.batch_size,
        )

        obs = train_data['obs']
        anchor_pool, future_pool = clearning.sample_geometric_futures(
            train_data['_episode_lengths'],
            gamma=gamma_sr,
            device=obs.device,
        )
        if anchor_pool.numel() == 0:
            # Every episode this epoch has length 1 -- no row has a future within its own episode.
            return

        trunk = self._actor_critic.sr_trunk
        n_rows, n_pairs = obs.shape[0], anchor_pool.shape[0]
        pair_batch = min(batch_size, n_pairs)
        neg_batch = min(batch_size, n_rows)
        stats: dict[str, float] = {}
        for _ in range(n_steps):
            # Anchor/positive pairs and the rho sample are drawn independently: a negative that
            # happened to be a given anchor's own sampled future is not an error (rho genuinely
            # contains those states), but drawing them from one shared index set would correlate
            # the two expectations the loss differences against each other.
            pairs = torch.randperm(n_pairs, device=obs.device)[:pair_batch]
            negatives = torch.randperm(n_rows, device=obs.device)[:neg_batch]
            loss, stats = clearning.density_ratio_loss(
                psi_anchor=trunk.psi(obs[anchor_pool[pairs]]),
                phi_future=trunk.phi(obs[future_pool[pairs]]),
                phi_random=trunk.phi(obs[negatives]),
                gamma=gamma_sr,
            )
            if self._cfgs.algo_cfgs.use_critic_norm:
                for param in trunk.parameters():
                    if param.requires_grad:
                        loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

            self._actor_critic.sr_optimizer.zero_grad()
            loss.backward()
            if self._cfgs.algo_cfgs.use_max_grad_norm:
                clip_grad_norm_(trunk.parameters(), self._cfgs.algo_cfgs.max_grad_norm)
            distributed.avg_grads(trunk)
            self._actor_critic.sr_optimizer.step()
        if stats:
            self._logger.store({f'Misc/CLearning{k}': v for k, v in stats.items()})
            # Loss/Loss_sr is registered unconditionally for td_ridge and is the series a reader
            # will look at first; fill it with this objective's loss so it stays the "is psi being
            # fitted" line whichever objective is fitting it.
            self._logger.store({'Loss/Loss_sr': stats['Loss']})

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
        if self._sr_psi_objective != 'td':
            # psi was already fitted for this epoch by
            # _contrastive_update_successor_features, which owns it end to end. Guarding here
            # rather than at the three call sites keeps the check in one place -- NaturalPG and
            # FOCOPS reimplement _update independently and would each need their own otherwise.
            return
        self._actor_critic.sr_optimizer.zero_grad()
        psi = self._actor_critic.sr_trunk.psi(obs)
        loss = nn.functional.mse_loss(psi, target_sr)

        if self._cfgs.algo_cfgs.use_critic_norm:
            excluded_ids = getattr(self._actor_critic, '_sr_critic_norm_excluded_ids', set())
            for param in self._actor_critic.sr_trunk.parameters():
                if not param.requires_grad or id(param) in excluded_ids:
                    # frozen phi network (sr_cfgs.phi_source), or -- under phi_source='contrastive'
                    # -- phi_net, trainable but fit only by its own InfoNCE loss/optimizer, never
                    # this one; a no-op under every other phi_source.
                    continue
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
