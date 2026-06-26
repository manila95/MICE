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
"""Implementation of CriticBuilder."""

from __future__ import annotations

from omnisafe.models.base import Critic
from omnisafe.models.critic.q_critic import QCritic
from omnisafe.models.critic.v_critic import DistributionalVCritic, IQNVCritic, VCritic
from omnisafe.typing import Activation, CriticType, InitFunction, OmnisafeSpace


# pylint: disable-next=too-few-public-methods
class CriticBuilder:
    """Build critic networks of different types.

    Supported critic_type values:
    - ``'q'``     — standard Q-critic (action-value, off-policy)
    - ``'v'``     — standard V-critic (state-value, scalar output)
    - ``'v_qr'``  — quantile-regression V-critic (fixed quantile levels)
    - ``'v_tqc'`` — truncated quantile critics (multiple QR networks + top-atom truncation)
    - ``'v_iqn'`` — implicit quantile networks (sampled quantile levels via cosine embedding)
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        use_obs_encoder: bool = False,
    ) -> None:
        """Initialize an instance of :class:`CriticBuilder`."""
        self._obs_space = obs_space
        self._act_space = act_space
        self._weight_initialization_mode = weight_initialization_mode
        self._activation = activation
        self._hidden_sizes = hidden_sizes
        self._num_critics = num_critics
        self._use_obs_encoder = use_obs_encoder

    def build_critic(
        self,
        critic_type: CriticType,
        n_quantiles: int = 50,
        n_top_quantiles_to_drop: int = 0,
        iqn_embed_dim: int = 64,
        iqn_n_cos: int = 64,
        iqn_n_tau_eval: int = 32,
    ) -> Critic:
        """Build and return a critic of the requested type.

        Args:
            critic_type: One of ``'q'``, ``'v'``, ``'v_qr'``, ``'v_tqc'``, ``'v_iqn'``.
                The legacy alias ``'v_dist'`` maps to ``'v_qr'``.
            n_quantiles: Atoms per network for QR/TQC; train-time samples for IQN.
            n_top_quantiles_to_drop: TQC only — top atoms dropped at value-estimation time.
            iqn_embed_dim: IQN only — embedding dimension for obs and quantile encoders.
            iqn_n_cos: IQN only — cosine features per quantile level.
            iqn_n_tau_eval: IQN only — quantile samples at eval / rollout time.
        """
        if critic_type == 'q':
            return QCritic(
                obs_space=self._obs_space,
                act_space=self._act_space,
                hidden_sizes=self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
                num_critics=self._num_critics,
                use_obs_encoder=self._use_obs_encoder,
            )
        if critic_type == 'v':
            return VCritic(
                obs_space=self._obs_space,
                act_space=self._act_space,
                hidden_sizes=self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
                num_critics=self._num_critics,
            )
        if critic_type in ('v_qr', 'v_dist'):  # v_dist kept as legacy alias
            return DistributionalVCritic(
                obs_space=self._obs_space,
                act_space=self._act_space,
                hidden_sizes=self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
                num_critics=1,
                n_quantiles=n_quantiles,
                n_top_quantiles_to_drop=0,
            )
        if critic_type == 'v_tqc':
            return DistributionalVCritic(
                obs_space=self._obs_space,
                act_space=self._act_space,
                hidden_sizes=self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
                num_critics=self._num_critics,  # K independent networks
                n_quantiles=n_quantiles,
                n_top_quantiles_to_drop=n_top_quantiles_to_drop,
            )
        if critic_type == 'v_iqn':
            return IQNVCritic(
                obs_space=self._obs_space,
                act_space=self._act_space,
                hidden_sizes=self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
                n_quantiles=n_quantiles,
                iqn_embed_dim=iqn_embed_dim,
                iqn_n_cos=iqn_n_cos,
                iqn_n_tau_eval=iqn_n_tau_eval,
            )

        raise NotImplementedError(
            f'critic_type "{critic_type}" is not implemented. '
            'Available: "q", "v", "v_qr", "v_tqc", "v_iqn".',
        )
