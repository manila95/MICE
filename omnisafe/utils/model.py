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
"""This module contains the helper functions for the model."""

from __future__ import annotations

import numpy as np
from torch import nn

from omnisafe.typing import Activation, InitFunction


def initialize_layer(init_function: InitFunction, layer: nn.Linear) -> None:
    """Initialize the layer with the given initialization function.

    The ``init_function`` can be chosen from: ``kaiming_uniform``, ``xavier_normal``, ``glorot``,
    ``xavier_uniform``, ``orthogonal``.

    Args:
        init_function (InitFunction): The initialization function.
        layer (nn.Linear): The layer to be initialized.
    """
    if init_function == 'kaiming_uniform':
        nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    elif init_function == 'xavier_normal':
        nn.init.xavier_normal_(layer.weight)
    elif init_function in ['glorot', 'xavier_uniform']:
        nn.init.xavier_uniform_(layer.weight)
    elif init_function == 'orthogonal':
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    else:
        raise TypeError(f'Invalid initialization function: {init_function}')


def get_activation(
    activation: Activation,
) -> type[nn.Identity | nn.ReLU | nn.Sigmoid | nn.Softplus | nn.Tanh]:
    """Get the activation function.

    The ``activation`` can be chosen from: ``identity``, ``relu``, ``sigmoid``, ``softplus``,
    ``tanh``.

    Args:
        activation (Activation): The activation function.

    Returns:
        The activation function, ranging from ``nn.Identity``, ``nn.ReLU``, ``nn.Sigmoid``,
        ``nn.Softplus`` to ``nn.Tanh``.
    """
    activations = {
        'identity': nn.Identity,
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'softplus': nn.Softplus,
        'tanh': nn.Tanh,
    }
    assert activation in activations
    return activations[activation]


def build_mlp_network(
    sizes: list[int],
    activation: Activation,
    output_activation: Activation = 'identity',
    weight_initialization_mode: InitFunction = 'kaiming_uniform',
    dropout: float = 0.0,
    use_layer_norm: bool = False,
    use_spectral_norm: bool = False,
) -> nn.Module:
    """Build the MLP network.

    Examples:
        >>> build_mlp_network([64, 64, 64], 'relu', 'tanh')
        Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): Linear(in_features=64, out_features=64, bias=True)
            (5): Tanh()
        )

    Args:
        sizes (list of int): The sizes of the layers.
        activation (Activation): The activation function.
        output_activation (Activation, optional): The output activation function. Defaults to
            ``identity``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        dropout (float, optional): Dropout probability applied after each hidden layer's
            activation. ``0.0`` (the default) omits the ``nn.Dropout`` module entirely rather
            than adding a no-op ``p=0`` one, so existing callers/checkpoints are byte-for-byte
            unaffected unless they opt in. Never applied after the output layer -- regularizing
            away part of the network's own output distribution (as opposed to its internal
            representation) is a different, not-requested thing.
        use_layer_norm (bool, optional): If ``True``, insert ``nn.LayerNorm`` between each hidden
            layer's affine transform and its activation (i.e. pre-activation norm: Linear ->
            LayerNorm -> activation -> Dropout), standard placement for stabilizing a small MLP's
            layer input distributions. Not applied to the output layer, for the same reason as
            ``dropout`` above -- a critic's output is an unconstrained scalar (a value, not an
            activation feeding another layer), and LayerNorm's mean/variance normalization would
            work against exactly the free scale that scalar needs to have. Defaults to ``False``.
        use_spectral_norm (bool, optional): If ``True``, wrap each hidden layer's ``nn.Linear``
            with spectral normalization (``nn.utils.parametrizations.spectral_norm``), bounding
            its spectral norm to 1 and so the whole hidden stack's Lipschitz constant -- a
            regularizer on how sharply the network's internal representation can respond to a
            small input perturbation, distinct from (and stackable with) weight-decay/L2, which
            only bounds parameter magnitude, not sensitivity. Not applied to the output layer, so
            the critic's output scale stays unconstrained, matching ``dropout``/``use_layer_norm``
            above. Defaults to ``False``.

    Returns:
        The MLP network.
    """
    activation_fn = get_activation(activation)
    output_activation_fn = get_activation(output_activation)
    layers = []
    for j in range(len(sizes) - 1):
        is_output_layer = j == len(sizes) - 2
        act_fn = output_activation_fn if is_output_layer else activation_fn
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        initialize_layer(weight_initialization_mode, affine_layer)
        if use_spectral_norm and not is_output_layer:
            affine_layer = nn.utils.parametrizations.spectral_norm(affine_layer)
        layers.append(affine_layer)
        if use_layer_norm and not is_output_layer:
            layers.append(nn.LayerNorm(sizes[j + 1]))
        layers.append(act_fn())
        if dropout > 0.0 and not is_output_layer:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)
