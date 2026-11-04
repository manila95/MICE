"""Risk-conditioned actor and critic networks for SRPL.

The risk estimate (see :mod:`omnisafe.utils.risk`) is fused into the policy and value networks
the way the reference implementation does it, rather than by appending it to the observation:
observation and risk each get their own affine layer and ``tanh``, and the two are concatenated
*after* that first layer, so risk enters through a narrow dedicated bottleneck (12 units) instead
of competing with the observation's features at the input layer.

Ported from ``RiskNet`` / ``build_risk_mlp_network`` in ``manila95/Safe-Policy-Optimization``
(``risk_aware_exploration`` branch, ``safepo/common/model.py``), adapted to omnisafe's
:func:`~omnisafe.utils.model.build_mlp_network` so the configured activation, weight
initialization, dropout and norm options still apply to everything after the fusion point.

The ``risk`` argument is required, not optional, on every forward here: a silently-missing risk
would quietly turn a risk-aware run into a differently-initialized ordinary one, which is a
failure mode worth having raise instead.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Distribution, Normal

from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.models.critic.v_critic import VCritic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


#: Width of the risk embedding fused into each network, from the reference implementation.
RISK_EMBED_DIM = 12


class RiskNet(nn.Module):
    """An MLP whose first layer takes observation and risk through separate affine maps.

    Args:
        sizes (list of int): Layer sizes as the equivalent plain MLP would take them,
            ``[in_dim, *hidden, out_dim]``.
        risk_size (int): Width of the risk vector.
        activation (Activation): Activation for the layers after the fusion point.
        weight_initialization_mode (InitFunction): Weight initialization for those layers.
    """

    def __init__(
        self,
        sizes: list[int],
        risk_size: int,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize an instance of :class:`RiskNet`."""
        super().__init__()
        self.affine_obs = nn.Linear(sizes[0], sizes[1])
        self.affine_risk = nn.Linear(risk_size, RISK_EMBED_DIM)
        # tanh regardless of the configured activation, matching the reference: it bounds both
        # branches to the same scale before they are concatenated, so a risk vector of
        # log-probabilities cannot swamp the observation features.
        self.fuse_activation = nn.Tanh()
        self.rest = build_mlp_network(
            sizes=[sizes[1] + RISK_EMBED_DIM, *sizes[2:]],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def forward(self, obs: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
        """Map an observation and its risk estimate to the network's output."""
        obs_embed = self.fuse_activation(self.affine_obs(obs))
        risk_embed = self.fuse_activation(self.affine_risk(risk))
        return self.rest(torch.cat([obs_embed, risk_embed], dim=-1))


def _check_risk(risk: torch.Tensor | None, who: str) -> torch.Tensor:
    if risk is None:
        raise ValueError(
            f'{who} is risk-conditioned (model_cfgs.use_risk) but was called without a risk '
            'tensor. Every call site reached during a risk-aware run has to pass one -- see '
            'OnPolicyAdapter.rollout for where it is produced and the buffer for where the '
            'per-step values are kept for the update.',
        )
    return risk


class RiskGaussianLearningActor(GaussianLearningActor):
    """:class:`GaussianLearningActor` whose mean network is conditioned on the risk estimate.

    Only the mean is conditioned; ``log_std`` stays a free parameter, as in the reference.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        risk_size: int = 10,
    ) -> None:
        """Initialize an instance of :class:`RiskGaussianLearningActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        self.mean = RiskNet(
            sizes=[self._obs_dim, *self._hidden_sizes, self._act_dim],
            risk_size=risk_size,
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def _distribution(  # type: ignore[override]
        self,
        obs: torch.Tensor,
        risk: torch.Tensor | None = None,
    ) -> Normal:
        """The action distribution at ``obs`` given its risk estimate."""
        mean = self.mean(obs, _check_risk(risk, 'actor'))
        return Normal(mean, torch.exp(self.log_std))

    def predict(  # type: ignore[override]
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        risk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample (or take the mean of) the action distribution at ``obs``."""
        self._current_dist = self._distribution(obs, risk)
        self._after_inference = True
        if deterministic:
            return self._current_dist.mean
        return self._current_dist.rsample()

    def forward(  # type: ignore[override]
        self,
        obs: torch.Tensor,
        risk: torch.Tensor | None = None,
    ) -> Distribution:
        """The action distribution at ``obs`` given its risk estimate."""
        self._current_dist = self._distribution(obs, risk)
        self._after_inference = True
        return self._current_dist


class RiskVCritic(VCritic):
    """:class:`VCritic` whose value networks are conditioned on the risk estimate.

    Subclassed rather than reimplemented so the ensemble aggregation, ``beta`` buffer and every
    other behaviour of the ordinary critic carry over untouched; only how each member maps an
    input to a value changes.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        risk_size: int = 10,
        **kwargs: object,
    ) -> None:
        """Initialize an instance of :class:`RiskVCritic`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics,
            **kwargs,  # type: ignore[arg-type]
        )
        self.net_lst = []
        for idx in range(self._num_critics):
            net = RiskNet(
                sizes=[self._obs_dim, *self._hidden_sizes, 1],
                risk_size=risk_size,
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
            )
            self.net_lst.append(net)
            # Replaces the plain net registered under this name by VCritic.__init__.
            self.add_module(f'critic_{idx}', net)

    def raw_values(  # type: ignore[override]
        self,
        obs: torch.Tensor,
        risk: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Each ensemble member's own prediction, un-aggregated."""
        risk = _check_risk(risk, 'critic')
        return [torch.squeeze(critic(obs, risk), -1) for critic in self.net_lst]

    def forward(  # type: ignore[override]
        self,
        obs: torch.Tensor,
        risk: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """The value at ``obs`` given its risk estimate, aggregated across the ensemble."""
        from omnisafe.utils.critic_ensemble import aggregate as aggregate_ensemble  # noqa: PLC0415

        raw = self.raw_values(obs, risk)
        if self._ensemble_method == 'none' or len(raw) == 1:
            return [raw[0]]
        stacked = torch.stack(raw, dim=0)
        return [aggregate_ensemble(stacked, self._ensemble_method, self._stream, self.beta.item())]
