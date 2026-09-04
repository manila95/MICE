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
from omnisafe.models.critic.v_critic import VCritic
from omnisafe.typing import Activation, CriticType, InitFunction, OmnisafeSpace


# pylint: disable-next=too-few-public-methods
class CriticBuilder:
    """Implementation of CriticBuilder.

    .. note::
        A :class:`CriticBuilder` is a class for building a critic network. In OmniSafe, instead of
        building the critic network directly, we build it by integrating various types of critic
        networks into the :class:`CriticBuilder`. The advantage of this is that each type of critic
        has a uniform way of passing parameters. This makes it easy for users to use existing
        critics, and also facilitates the extension of new critic types.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        num_critics (int, optional): Number of critics. Defaults to 1.
        use_obs_encoder (bool, optional): Whether to use observation encoder, only used in q critic.
            Defaults to False.
        dropout (float, optional): Dropout probability for the ``v`` critic's hidden layers
            (ignored by ``q``). Defaults to ``0.0``.
        use_layer_norm (bool, optional): Whether to LayerNorm the ``v`` critic's hidden layers
            (ignored by ``q``). Defaults to ``False``.
        use_spectral_norm (bool, optional): Whether to spectral-normalize the ``v`` critic's
            hidden layers (ignored by ``q``). Defaults to ``False``.
        ensemble_method (str, optional): ``'v'`` critic only -- see :class:`VCritic`. Ignored by
            ``'q'``. Defaults to ``'none'``.
        stream (str, optional): ``'v'`` critic only -- see :class:`VCritic`. Defaults to ``'r'``.
        beta_init (float, optional): ``'v'`` critic only -- see :class:`VCritic`. Defaults to 0.0.
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
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_spectral_norm: bool = False,
        ensemble_method: str = 'none',
        stream: str = 'r',
        beta_init: float = 0.0,
    ) -> None:
        """Initialize an instance of :class:`CriticBuilder`."""
        self._obs_space: OmnisafeSpace = obs_space
        self._act_space: OmnisafeSpace = act_space
        self._weight_initialization_mode: InitFunction = weight_initialization_mode
        self._activation: Activation = activation
        self._hidden_sizes: list[int] = hidden_sizes
        self._num_critics: int = num_critics
        self._use_obs_encoder: bool = use_obs_encoder
        # v-critic-only regularization / ensemble-bias-correction options (see VCritic /
        # build_mlp_network); QCritic doesn't take these, so build_critic('q') below never
        # passes them on.
        self._dropout: float = dropout
        self._use_layer_norm: bool = use_layer_norm
        self._use_spectral_norm: bool = use_spectral_norm
        self._ensemble_method: str = ensemble_method
        self._stream: str = stream
        self._beta_init: float = beta_init

    def build_critic(
        self,
        critic_type: CriticType,
    ) -> Critic:
        """Build critic.

        Currently, we support two types of critics: ``q`` and ``v``.
        If you want to add a new critic type, you can simply add it here.

        Args:
            critic_type (str): Critic type.

        Returns:
            An instance of V-Critic or Q-Critic

        Raises:
            NotImplementedError: If the critic type is not ``q`` or ``v``.
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
                dropout=self._dropout,
                use_layer_norm=self._use_layer_norm,
                use_spectral_norm=self._use_spectral_norm,
                ensemble_method=self._ensemble_method,
                stream=self._stream,
                beta_init=self._beta_init,
            )

        raise NotImplementedError(
            f'critic_type "{critic_type}" is not implemented.'
            'Available critic types are: "q", "v".',
        )
