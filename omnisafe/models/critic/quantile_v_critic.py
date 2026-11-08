"""A distributional V critic that predicts quantiles of the return instead of its mean.

The ordinary :class:`~omnisafe.models.critic.v_critic.VCritic` regresses a single number onto the
value target, so everything downstream -- the advantage, the constraint, the value studies -- sees
only the mean of the return. Two states with the same expected cost but very different spreads
are indistinguishable to it, which is exactly the distinction a safety constraint cares about.

This critic predicts ``num_quantiles`` quantiles of the same quantity and is trained by quantile
regression. ``forward`` still returns a single number -- the mean over the quantiles -- so every
existing consumer (GAE, the actor's advantages, the Monte-Carlo value studies, the eval dumps)
works unchanged and a run differs from its non-distributional counterpart only in how the critic
is parameterized and trained. The full set is available through :meth:`quantiles` for anything
that wants the spread: a CVaR-style constraint, or scoring the predicted *distribution* against
realized returns rather than just its mean.

**The value target must be a return sample.** Quantile regression learns the quantiles of
whatever distribution its regression targets are drawn from. Under ``value_target='mc'`` the
target is the realized discounted return-to-go, a genuine sample of the return distribution, and
the learned quantiles mean what they say. Under a bootstrapped target (``td_lambda``,
``td_zero``) the target is a critic-dependent mixture rather than a return sample, and the
quantiles would describe that mixture instead -- a different and much less useful object. The
algorithm checks this and refuses rather than quietly fitting the wrong thing; see
``PolicyGradient._init_model``.
"""

from __future__ import annotations

import torch
from torch import nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class QuantileVCritic(Critic):
    """V critic predicting ``num_quantiles`` quantiles of the return.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): Hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode.
        num_quantiles (int, optional): How many quantiles to predict. Defaults to 32.
        dropout (float, optional): Dropout probability, as for :class:`VCritic`.
        use_layer_norm (bool, optional): Layer norm between affine and activation.
        use_spectral_norm (bool, optional): Spectral norm on the hidden layers.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_quantiles: int = 32,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_spectral_norm: bool = False,
    ) -> None:
        """Initialize an instance of :class:`QuantileVCritic`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        )
        if num_quantiles < 2:
            raise ValueError(f'num_quantiles must be at least 2, got {num_quantiles}.')
        self._num_quantiles = num_quantiles
        self.net: nn.Module = build_mlp_network(
            sizes=[self._obs_dim, *self._hidden_sizes, num_quantiles],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_spectral_norm=use_spectral_norm,
        )
        # Midpoints of equal-width bins: the quantile levels the i-th output is regressed onto.
        # A buffer, not a parameter -- they are fixed by construction, and they have to move with
        # .to(device) and survive checkpointing like any other fixed tensor.
        self.register_buffer(
            'tau',
            (torch.arange(num_quantiles, dtype=torch.float32) + 0.5) / num_quantiles,
        )

    @property
    def num_quantiles(self) -> int:
        """How many quantiles this critic predicts."""
        return self._num_quantiles

    def quantiles(self, obs: torch.Tensor) -> torch.Tensor:
        """The predicted quantiles, shape ``(B, num_quantiles)``.

        Returned in the network's own output order, which is *not* guaranteed to be sorted:
        quantile regression pushes them toward monotonicity but does not enforce it, and
        silently sorting here would hide a critic that has not learned the ordering. Call
        :meth:`sorted_quantiles` where an ordered set is actually needed.
        """
        return self.net(obs)

    def sorted_quantiles(self, obs: torch.Tensor) -> torch.Tensor:
        """The predicted quantiles, sorted ascending -- for reading off a specific level."""
        return torch.sort(self.quantiles(obs), dim=-1).values

    def cvar(self, obs: torch.Tensor, alpha: float = 0.1, upper: bool = True) -> torch.Tensor:
        """Mean of the worst ``alpha`` tail, as a ``(B,)`` tensor.

        ``upper`` takes the top tail, which is the risk-relevant end for a cost; ``False`` takes
        the bottom, the pessimistic end for a reward.
        """
        if not 0 < alpha <= 1:
            raise ValueError(f'alpha must be in (0, 1], got {alpha}.')
        ordered = self.sorted_quantiles(obs)
        k = max(1, int(round(alpha * self._num_quantiles)))
        tail = ordered[..., -k:] if upper else ordered[..., :k]
        return tail.mean(dim=-1)

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """The mean of the predicted quantiles, in :class:`Critic`'s list-of-one form.

        Deliberately the mean and not a risk-adjusted statistic: this is what GAE, the
        advantages and the value studies consume, and keeping it the plain expectation is what
        makes a distributional run comparable to its non-distributional counterpart. Anything
        wanting the tail asks for it explicitly via :meth:`cvar` or :meth:`quantiles`.
        """
        return [self.quantiles(obs).mean(dim=-1)]


def quantile_huber_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    tau: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile regression loss with a Huber core.

    For each predicted quantile ``theta_i`` at level ``tau_i`` and each target sample ``y``, the
    pinball loss ``|tau_i - 1{y < theta_i}|`` weights under- and over-prediction asymmetrically,
    which is what makes ``theta_i`` converge on the ``tau_i``-quantile rather than the mean. The
    Huber core bounds the gradient for large residuals, as in QR-DQN -- without it a single
    outlier return dominates the update.

    Args:
        predicted: ``(B, N)`` predicted quantiles.
        target: ``(B,)`` or ``(B, 1)`` target samples.
        tau: ``(N,)`` quantile levels matching ``predicted``'s columns.
        kappa (float): Huber threshold. ``0`` gives the plain (non-Huber) pinball loss.

    Returns:
        A scalar: summed over quantiles, averaged over the batch -- the standard reduction, so
        the loss scale does not change with ``num_quantiles``.
    """
    target = target.reshape(-1, 1)
    # (B, N): residual of every quantile against the sample.
    diff = target - predicted
    if kappa <= 0:
        element_wise = diff.abs()
    else:
        abs_diff = diff.abs()
        element_wise = torch.where(
            abs_diff <= kappa,
            0.5 * diff.pow(2) / kappa,
            abs_diff - 0.5 * kappa,
        )
    # The indicator is of the *residual's* sign, so no gradient flows through it.
    weight = (tau.reshape(1, -1) - (diff.detach() < 0).float()).abs()
    return (weight * element_wise).sum(dim=-1).mean()
