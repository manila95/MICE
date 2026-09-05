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
"""Implementation of ConstraintActorCritic."""

from __future__ import annotations

import itertools
from typing import Any

import torch
from torch import optim

from omnisafe.models.actor_critic.actor_critic import ActorCritic
from omnisafe.models.base import Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.models.critic.successor_representation_critic import (
    SuccessorRepresentationLinearReadout,
    SuccessorRepresentationReadout,
    SuccessorRepresentationTrunk,
    TDRidgeSuccessorRepresentationTrunk,
)
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig


class ConstraintActorCritic(ActorCritic):
    """ConstraintActorCritic is a wrapper around ActorCritic that adds a cost critic to the model.

    In OmniSafe, we combine the actor and critic into one this class.

    +-----------------+-----------------------------------------------+
    | Model           | Description                                   |
    +=================+===============================================+
    | Actor           | Input is observation. Output is action.       |
    +-----------------+-----------------------------------------------+
    | Reward V Critic | Input is observation. Output is reward value. |
    +-----------------+-----------------------------------------------+
    | Cost V Critic   | Input is observation. Output is cost value.   |
    +-----------------+-----------------------------------------------+

    .. note::
        When ``model_cfgs.use_successor_representation`` is ``True``, ``reward_critic`` and
        ``cost_critic`` are both read-out heads over a single shared successor-representation
        value function (see ``model_cfgs.sr_cfgs.sr_mode``) instead of two independent
        networks. Every other consumer of ``reward_critic`` / ``cost_critic`` (rollout, GAE,
        the critic training loop) is unaffected, since both modes still expose the standard
        ``Critic`` interface (``forward(obs) -> [value]``).

        When additionally ``model_cfgs.sr_cfgs.cost_only`` is ``True`` (``sr_mode: 'td_ridge'``
        only), ``reward_critic`` is *not* part of the successor representation at all: it is
        built and trained exactly as it would be with ``use_successor_representation: False``
        (its own network and optimizer), while ``cost_critic`` alone reads out the SR trunk. The
        trunk still fits its reward read-out ``w_r`` every update for diagnostic parity with the
        non-``cost_only`` run; it is simply never consulted by the actual reward critic.

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        model_cfgs (ModelConfig): The model configurations.
        epochs (int): The number of epochs.

    Attributes:
        actor (Actor): The actor network.
        reward_critic (Critic): The critic network.
        cost_critic (Critic): The critic network.
        std_schedule (Schedule): The schedule for the standard deviation of the Gaussian distribution.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        """Initialize an instance of :class:`ConstraintActorCritic`."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)

        self._use_sr: bool = bool(model_cfgs.get('use_successor_representation', False))
        self._sr_mode: str | None = model_cfgs.sr_cfgs.get('sr_mode', 'shared_trunk') if self._use_sr else None
        # sr_cfgs.cost_only: train reward_critic the normal way (a plain critic, no trunk
        # sharing) and give the SR trunk to cost_critic alone. Only meaningful -- and only read
        # -- under sr_mode == 'td_ridge'; see _build_successor_representation_critics.
        self._sr_cost_only: bool = (
            bool(model_cfgs.sr_cfgs.get('cost_only', False)) if self._use_sr else False
        )
        if self._sr_cost_only and self._sr_mode != 'td_ridge':
            raise NotImplementedError(
                'model_cfgs.sr_cfgs.cost_only is only supported under sr_mode == "td_ridge", '
                f'got sr_mode "{self._sr_mode}".',
            )

        if not self._use_sr:
            # Cost gets its own dropout/layer-norm/spectral-norm/weight-decay, each falling back
            # to the shared (reward-side) value when null -- same null-fallback convention as
            # critic_norm_coef_cost / ridge_kappa_cost / w_weight_decay_cost: a sparser,
            # differently-scaled cost target generally wants different regularization strength
            # than the dense reward critic it shares no parameters with.
            def _cost_or_shared(name: str, default: Any) -> Any:
                cost_val = model_cfgs.critic.get(f'{name}_cost', None)
                return model_cfgs.critic.get(name, default) if cost_val is None else cost_val

            # Critic-ensemble bias-correction knobs, each falling back to the shared (reward-
            # side) value when null -- same null-fallback convention as dropout_cost /
            # use_layer_norm_cost above, so the cost critic can run an ensemble method (e.g.
            # 'cdq') while the reward critic stays a plain single critic ('none'), or vice
            # versa, rather than being forced to share one method across both streams. stream='c'
            # is what actually gives the cost critic the opposite (conservative, not pessimistic)
            # aggregation direction once it does run an ensemble method.
            ensemble_method = str(_cost_or_shared('ensemble_method', 'none'))
            ensemble_size = (
                int(_cost_or_shared('ensemble_size', 2)) if ensemble_method != 'none' else 1
            )
            beta_init = float(_cost_or_shared('beta_init', 0.0) or 0.0)
            self.cost_critic: Critic = CriticBuilder(
                obs_space=obs_space,
                act_space=act_space,
                hidden_sizes=model_cfgs.critic.hidden_sizes,
                activation=model_cfgs.critic.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
                num_critics=ensemble_size,
                use_obs_encoder=False,
                dropout=float(_cost_or_shared('dropout', 0.0) or 0.0),
                use_layer_norm=bool(_cost_or_shared('use_layer_norm', False)),
                use_spectral_norm=bool(_cost_or_shared('use_spectral_norm', False)),
                ensemble_method=ensemble_method,
                stream='c',
                beta_init=beta_init,
            ).build_critic('v')
            self.add_module('cost_critic', self.cost_critic)
            # See ActorCritic.__init__'s matching call for why this defaults to eval() and is
            # only flipped to train() around _update_cost_critic's own backward/step call.
            self.cost_critic.eval()

            if model_cfgs.critic.lr is not None:
                self.cost_critic_optimizer: optim.Optimizer
                self.cost_critic_optimizer = optim.AdamW(
                    self.cost_critic.parameters(),
                    lr=model_cfgs.critic.lr,
                    weight_decay=float(_cost_or_shared('weight_decay', 0.0) or 0.0),
                )
            return

        self._build_successor_representation_critics(obs_space, act_space, model_cfgs)

    def _build_successor_representation_critics(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
    ) -> None:
        """Replace the plain reward/cost critics with a shared successor-representation critic.

        Discards the plain ``reward_critic`` built by :class:`ActorCritic` in favor of a
        successor-representation-based one; ``reward_critic_optimizer`` and
        ``cost_critic_optimizer`` are set to the *same* optimizer instance so that the shared
        trunk's Adam momentum state is not split across two independent optimizers (each of
        :meth:`PolicyGradient._update_reward_critic` / :meth:`_update_cost_critic` still does
        its own ``zero_grad`` / ``backward`` / ``step`` call, so the shared trunk still only
        ever moves along one loss's gradient at a time).

        Args:
            obs_space (OmnisafeSpace): The observation space.
            act_space (OmnisafeSpace): The action space.
            model_cfgs (ModelConfig): The model configurations.
        """
        sr_cfgs = model_cfgs.sr_cfgs
        obs_dim = obs_space.shape[0]
        # hidden_sizes sets the trunk's own hidden widths verbatim; sr_dim only sets the width of
        # the final phi/psi projection -- the two are independent knobs. (Previously every hidden
        # width was silently overridden to sr_dim as well, coupling trunk capacity to sr_dim.)
        #
        # Caution: the layer that produces phi/psi off the trunk (phi_head / psi_head, or
        # SuccessorRepresentationTrunk's own final layer in shared_trunk mode) is a plain linear
        # map with no nonlinearity of its own, so its output can never carry more independent
        # information than the last hidden layer feeding it. Setting hidden_sizes[-1] < sr_dim is
        # now possible and silently caps psi/phi's effective rank at hidden_sizes[-1] -- wasted
        # capacity in shared_trunk mode; a rank-deficient, exactly-singular-without-regularization
        # ridge Gram matrix in td_ridge mode. Keep hidden_sizes[-1] >= sr_dim unless that
        # bottleneck is deliberate.
        hidden_sizes = list(sr_cfgs.hidden_sizes)
        # Same independence for the standalone phi network of sr_cfgs.phi_source='separate' /
        # 'contrastive': phi_hidden_sizes sets its hidden widths verbatim, sr_dim only sets its
        # output width. Its depth (and now its width) is independent of the trunk's, which is the
        # point of that mode.
        phi_hidden_sizes = list(sr_cfgs.get('phi_hidden_sizes', []) or [])

        # SR-trunk regularization (dropout/LayerNorm/spectral-norm/weight_decay): shared across
        # both reward and cost since the trunk itself is shared between them (see the class
        # docstring's cost_only note for the one case where the trunk is cost-exclusive instead --
        # regularizing it there already only affects cost, with no separate "_cost" knob needed).
        # Out of scope: the frozen/contrastive/laplacian phi sources (phi_head/phi_net), which
        # have their own separate training paths and hyperparameter surfaces already.
        sr_dropout = float(sr_cfgs.get('dropout', 0.0) or 0.0)
        sr_use_layer_norm = bool(sr_cfgs.get('use_layer_norm', False))
        sr_use_spectral_norm = bool(sr_cfgs.get('use_spectral_norm', False))

        if self._sr_mode == 'shared_trunk':
            trunk = SuccessorRepresentationTrunk(
                obs_dim=obs_dim,
                hidden_sizes=hidden_sizes,
                sr_dim=sr_cfgs.sr_dim,
                activation=sr_cfgs.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
                dropout=sr_dropout,
                use_layer_norm=sr_use_layer_norm,
                use_spectral_norm=sr_use_spectral_norm,
            )
            self.reward_critic = SuccessorRepresentationReadout(
                obs_space,
                act_space,
                trunk,
                sr_cfgs.sr_dim,
                model_cfgs.weight_initialization_mode,
            )
            self.cost_critic: Critic = SuccessorRepresentationReadout(
                obs_space,
                act_space,
                trunk,
                sr_cfgs.sr_dim,
                model_cfgs.weight_initialization_mode,
            )
            self.sr_trunk: SuccessorRepresentationTrunk | TDRidgeSuccessorRepresentationTrunk = trunk
            trainable_params = itertools.chain(
                trunk.parameters(),
                self.reward_critic.head.parameters(),
                self.cost_critic.head.parameters(),
            )
        elif self._sr_mode == 'td_ridge':
            # Read-out weight learning: 'ridge' (closed-form buffers) or 'sgd' (learned params
            # fit over a persistent replay buffer, see PolicyGradient._sgd_update_readout_weights).
            self._sr_readout: str = sr_cfgs.get('readout', 'ridge')
            learnable_readout = self._sr_readout == 'sgd'
            self._sr_phi_source: str = sr_cfgs.get('phi_source', 'trunk')
            # The phi sources that are *trained* rather than frozen or read off the shared trunk.
            # Both need the same three pieces of special-casing below -- exclusion from
            # sr_optimizer, exclusion from the critic-norm penalty, and an optimizer of their own
            # -- so they are named once here rather than tested for individually three times.
            self._sr_phi_trained: bool = self._sr_phi_source in ('contrastive', 'laplacian')
            trunk = TDRidgeSuccessorRepresentationTrunk(
                obs_dim=obs_dim,
                hidden_sizes=hidden_sizes,
                sr_dim=sr_cfgs.sr_dim,
                activation=sr_cfgs.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
                phi_source=self._sr_phi_source,
                phi_hidden_sizes=phi_hidden_sizes,
                learnable_readout=learnable_readout,
                phi_orthogonal_init=sr_cfgs.get('phi_orthogonal_init', False),
                phi_rff_bandwidth=sr_cfgs.get('phi_rff_bandwidth', 1.0),
                phi_ensemble_sources=sr_cfgs.get('phi_ensemble_sources', None),
                dropout=sr_dropout,
                use_layer_norm=sr_use_layer_norm,
                use_spectral_norm=sr_use_spectral_norm,
            )
            if self._sr_cost_only:
                # Reward critic is a plain, independent V critic -- see the class docstring's
                # "cost_only" note. The trunk still fits w_r each ridge/sgd update (see
                # PolicyGradient._ridge_update_successor_weights), it just never backs an actual
                # critic. Being a genuine standalone VCritic (not part of the SR trunk at all),
                # it takes the plain model_cfgs.critic.* regularization knobs, same as the
                # not-self._use_sr path in ActorCritic.__init__ -- not sr_cfgs's, which govern
                # the trunk/cost side only.
                self.reward_critic = CriticBuilder(
                    obs_space=obs_space,
                    act_space=act_space,
                    hidden_sizes=model_cfgs.critic.hidden_sizes,
                    activation=model_cfgs.critic.activation,
                    weight_initialization_mode=model_cfgs.weight_initialization_mode,
                    num_critics=1,
                    use_obs_encoder=False,
                    dropout=float(model_cfgs.critic.get('dropout', 0.0) or 0.0),
                    use_layer_norm=bool(model_cfgs.critic.get('use_layer_norm', False)),
                    use_spectral_norm=bool(model_cfgs.critic.get('use_spectral_norm', False)),
                ).build_critic('v')
                self.reward_critic.eval()
            else:
                self.reward_critic = SuccessorRepresentationLinearReadout(
                    obs_space,
                    act_space,
                    trunk,
                    'w_r',
                    model_cfgs.weight_initialization_mode,
                )
            # cost_value_clip folds in a prior the ridge solve has no way to know about: a
            # single-occurrence, discounted cost (cost=terminated, as in plain-Mujoco
            # environments) is bounded in [0, 1], but psi(s).w is an unconstrained linear
            # functional -- see SuccessorRepresentationLinearReadout.value_clip's docstring.
            # None (the default) reproduces the unclamped behavior exactly.
            cost_value_clip = sr_cfgs.get('cost_value_clip', None)
            self.cost_critic = SuccessorRepresentationLinearReadout(
                obs_space,
                act_space,
                trunk,
                'w_c',
                model_cfgs.weight_initialization_mode,
                value_clip=tuple(cost_value_clip) if cost_value_clip is not None else None,
            )
            self.sr_trunk = trunk
            # The trunk parameters (trunk body + phi_head + psi_head) are trained by the value /
            # SR-feature losses through the shared sr_optimizer. Under readout='sgd', w_r / w_c are
            # also parameters but are fit by their own regression loss on a separate optimizer, so
            # they are excluded here. The requires_grad filter additionally drops the frozen phi
            # network under phi_source='random' / 'separate' / 'rff' / 'ensemble' (a no-op under
            # 'trunk'). Under the trained sources ('contrastive' / 'laplacian') phi_net is
            # trainable (unlike every frozen source) but must still be excluded: it is fit by its
            # own loss on its own optimizer (sr_phi_optimizer, below), never by the value/SR-feature
            # losses this optimizer trains against -- so its params would otherwise sit in
            # sr_optimizer's param group forever accumulating unrelated gradients no step of this
            # optimizer should apply.
            readout_weight_ids = (
                {id(trunk.w_r), id(trunk.w_c)} if learnable_readout else set()
            )
            phi_param_ids = (
                {id(p) for p in trunk.phi_net.parameters()} if self._sr_phi_trained else set()
            )
            excluded_ids = readout_weight_ids | phi_param_ids
            # Read by PolicyGradient's use_critic_norm loops (_update_reward_critic /
            # _update_cost_critic / _update_successor_features), which otherwise fold phi_net's
            # norm into losses no optimizer of phi_net ever steps from -- see those methods.
            self._sr_critic_norm_excluded_ids: set[int] = excluded_ids
            trainable_params = [
                param
                for param in trunk.parameters()
                if param.requires_grad and id(param) not in excluded_ids
            ]
        else:
            raise NotImplementedError(
                f'Unknown sr_cfgs.sr_mode "{self._sr_mode}". '
                'Available successor-representation modes are: "shared_trunk", "td_ridge".',
            )

        self.add_module('reward_critic', self.reward_critic)
        self.add_module('cost_critic', self.cost_critic)
        self.add_module('sr_trunk', self.sr_trunk)
        # See ActorCritic.__init__'s reward_critic.eval() call: sr_trunk defaults to eval() so
        # dropout (if sr_cfgs.dropout > 0) never affects a value read (rollout GAE, Eval_s0/
        # Eval_all, the MC value study, the intermediate-state study). Calling .eval() on either
        # readout wrapper recurses into sr_trunk (it's a registered submodule of both), but both
        # calls are kept for symmetry with the not-self._use_sr path and because in
        # sr_cfgs.cost_only mode reward_critic is a *different* module (the plain VCritic above,
        # already eval()'d there) that doesn't touch sr_trunk at all.
        self.reward_critic.eval()
        self.cost_critic.eval()

        if sr_cfgs.lr is not None:
            # AdamW rather than Adam: decoupled weight decay -- see ActorCritic.__init__'s
            # matching comment. sr_cfgs.weight_decay=0.0 (the default) makes this identical to the
            # previous plain Adam, a no-op unless set. Shared across reward+cost since the trunk
            # itself is shared between them (except under cost_only, where the trunk is
            # cost-exclusive already and reward_critic_optimizer below is a wholly separate
            # optimizer over the plain reward critic instead).
            sr_optimizer = optim.AdamW(
                list(trainable_params),
                lr=sr_cfgs.lr,
                weight_decay=float(sr_cfgs.get('weight_decay', 0.0) or 0.0),
            )
            self.cost_critic_optimizer: optim.Optimizer = sr_optimizer
            self.sr_optimizer: optim.Optimizer = sr_optimizer
            self.reward_critic_optimizer: optim.Optimizer
            if self._sr_cost_only:
                # Trained the normal way, on the standard critic learning rate -- not sr_cfgs.lr,
                # which governs the SR trunk/cost-critic optimizer above. AdamW for the same
                # decoupled-weight-decay reason as sr_optimizer above.
                if model_cfgs.critic.lr is not None:
                    self.reward_critic_optimizer = optim.AdamW(
                        self.reward_critic.parameters(),
                        lr=model_cfgs.critic.lr,
                        weight_decay=float(model_cfgs.critic.get('weight_decay', 0.0) or 0.0),
                    )
            else:
                self.reward_critic_optimizer = sr_optimizer
            # Under readout='sgd', w_r / w_c get their own optimizer so their regression loss
            # never shares Adam state with (or steps) the representation parameters. Its
            # weight_decay is the SGD analogue of the ridge kappa -- explicit L2 on w_r / w_c.
            if self._sr_mode == 'td_ridge' and self._sr_readout == 'sgd':
                w_lr = sr_cfgs.get('w_lr', None) or sr_cfgs.lr
                # Separate param groups so w_c can carry its own decay, mirroring
                # ridge_kappa_cost on the closed-form path; None keeps both on one value.
                wd_r = sr_cfgs.get('w_weight_decay', 0.0)
                wd_c = sr_cfgs.get('w_weight_decay_cost', None)
                self.sr_readout_optimizer: optim.Optimizer = optim.Adam(
                    [
                        {'params': [self.sr_trunk.w_r], 'weight_decay': wd_r},
                        {
                            'params': [self.sr_trunk.w_c],
                            'weight_decay': wd_r if wd_c is None else wd_c,
                        },
                    ],
                    lr=w_lr,
                )
            # phi_net's own optimizer (the trained sources only): their losses run on a
            # different cadence (a handful of steps per epoch, see
            # PolicyGradient._contrastive_update_phi / _laplacian_update_phi) than the
            # per-minibatch value/SR-feature losses sr_optimizer trains against, so phi_net gets
            # independent Adam momentum state rather than sharing sr_optimizer's -- the same
            # reasoning as sr_readout_optimizer above, applied to phi_net instead of w_r / w_c.
            if self._sr_mode == 'td_ridge' and self._sr_phi_trained:
                # A source-specific rate wins over the shared phi_lr, which in turn wins over the
                # trunk's lr. The two objectives are not interchangeable here: ALLO has an
                # orthonormality constraint to satisfy from scratch and needs an order of
                # magnitude more step than the InfoNCE loss does, so one shared default cannot
                # serve both. Absent keys fall straight through, leaving 'contrastive' exactly as
                # it was.
                phi_lr = (
                    sr_cfgs.get(f'phi_{self._sr_phi_source}_lr', None)
                    or sr_cfgs.get('phi_lr', None)
                    or sr_cfgs.lr
                )
                self.sr_phi_optimizer: optim.Optimizer = optim.Adam(
                    trunk.phi_net.parameters(),
                    lr=phi_lr,
                )

    def sr_features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(phi, psi)`` successor-representation features (``td_ridge`` mode only).

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            phi: The one-step feature of the observation.
            psi: The successor feature of the observation.
        """
        assert self._sr_mode == 'td_ridge', (
            'sr_features() is only available when model_cfgs.sr_cfgs.sr_mode == "td_ridge".'
        )
        with torch.no_grad():
            z = self.sr_trunk.features(obs)
            phi = self.sr_trunk.phi(obs, z=z)
            psi = self.sr_trunk.psi(obs, z=z)
        return phi, psi

    def step(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_c: The cost value of the observation.
            log_prob: The log probability of the action.
        """
        with torch.no_grad():
            value_r = self.reward_critic(obs)
            value_c = self.cost_critic(obs)

            action = self.actor.predict(obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(action)

        return action, value_r[0], value_c[0], log_prob

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_c: The cost value of the observation.
            log_prob: The log probability of the action.
        """
        return self.step(obs, deterministic=deterministic)
