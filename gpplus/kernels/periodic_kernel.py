import math
from typing import Optional

import torch
from gpytorch.priors import Prior

from ..constraints import SoftClamp
from .unconstrained_kernel import UnconstrainedKernel


class PeriodicKernel(UnconstrainedKernel):
    r"""
    Periodic kernel with base-10 log parameterization for both the inverse squared
    lengthscale and the period.

    For inputs :math:`x, x' \in \mathbb{R}^D`, the kernel is

    .. math::

        k(x, x')
        = \exp\Big(
            - 2 \sum_{d=1}^D
                \sin^2\!\Big(\pi \frac{x_d - x'_d}{p_d}\Big)
                \, \ell_d^{-2}
          \Big),

    where we parameterize

    .. math::

        \ell_d^{-2} = 10^{\,\text{lengthscale}_d}, \qquad
        p_d        = 10^{\,\text{period}_d}.

    This matches the repository's convention for :class:`GaussianKernel` /
    :class:`PowerExponentialKernel`, where ``lengthscale`` behaves like a
    base-10 log *inverse squared* lengthscale.
    """

    has_lengthscale = True
    is_stationary = True

    def __init__(
        self,
        period_prior: Optional[Prior] = None,
        period_constraint: Optional[SoftClamp] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Default: allow log10(period) in roughly [1e-1, 1e1] (periods from 0.1 to 10)
        # Tightened from [-2, 2] to prevent extreme periods that can occur during optimization
        if period_constraint is None:
            period_constraint = SoftClamp(lower_bound=-1.0, upper_bound=1.0, margin=1e-2)

        period_num_dims = 1 if self.ard_num_dims is None else self.ard_num_dims

        # raw_period parameter lives in unconstrained space; after applying the constraint
        # we interpret it as log10(period), and exponentiate to get the actual period.
        self.register_parameter(
            name="raw_period",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, period_num_dims)),
        )

        if period_prior is not None:
            if not isinstance(period_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(period_prior).__name__)
            self.register_prior(
                "period_prior",
                period_prior,
                self._period_param,
                self._period_closure,
            )

        self.register_constraint("raw_period", period_constraint)

    # --- Prior hooks for period ---
    def _period_param(self, m):
        return m.period

    def _period_closure(self, m, v):
        m._set_period(v)

    # --- Period: property + setter ---
    @property
    def period(self):
        # First map raw -> constrained (log10-period), then exponentiate base-10
        log10_period = self.raw_period_constraint.transform(self.raw_period)
        return torch.pow(10.0, log10_period)

    @period.setter
    def period(self, value):
        self._set_period(value)

    def _set_period(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period)
        if torch.any(value <= 0):
            raise ValueError("period must be strictly positive.")

        # Work in log10 space for the constrained parameter
        log10_value = torch.log10(value)
        self.initialize(raw_period=self.raw_period_constraint.inverse_transform(log10_value))

    # --- Core kernel logic ---
    def forward(self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False, **params):
        if last_dim_is_batch:
            raise NotImplementedError("PeriodicKernel does not currently support last_dim_is_batch=True.")

        # Diagonal shortcut: k(x, x) = 1 for all x
        if diag:
            if torch.equal(x1, x2):
                return x1.new_ones(x1.shape[:-1])

        # lengthscale: base-10 log inverse-squared lengthscale
        lengthscale = self.lengthscale  # shape: (*batch, 1, D_l)
        period = self.period  # shape: (*batch, 1, D_p)

        D = x1.size(-1)

        # Expand isotropic parameters (D_l or D_p == 1) across input dimensions if needed
        if lengthscale.shape[-1] == 1 and D > 1:
            lengthscale = lengthscale.expand(*lengthscale.shape[:-1], D)
        if period.shape[-1] == 1 and D > 1:
            period = period.expand(*period.shape[:-1], D)

        # Convert log10 inverse-squared lengthscale into actual scale factors
        inv_sq_lengthscale = torch.pow(10.0, lengthscale)  # (*batch, 1, D)

        if diag:
            # x1, x2: (*batch, N, D); compute elementwise k(x1[i], x2[i])
            diff = x1 - x2  # (*batch, N, D)
            # Broadcast inv_sq_lengthscale, period over N
            arg = (math.pi * diff) / period  # (*batch, N, D)
            sin2 = torch.sin(arg).pow(2)
            exponent = -2.0 * sin2 * inv_sq_lengthscale  # (*batch, N, D)
            return exponent.sum(dim=-1).exp()  # (*batch, N)

        # Full covariance: pairwise between all points in x1 and x2
        x1_ = x1.unsqueeze(-2)  # (*batch, N1, 1, D)
        x2_ = x2.unsqueeze(-3)  # (*batch, 1, N2, D)

        # Broadcast inv_sq_lengthscale and period to (*batch, 1, 1, D)
        inv_sq_lengthscale = inv_sq_lengthscale.unsqueeze(-2)
        period = period.unsqueeze(-2)

        diff = x1_ - x2_  # (*batch, N1, N2, D)
        arg = (math.pi * diff) / period
        sin2 = torch.sin(arg).pow(2)
        exponent = -2.0 * sin2 * inv_sq_lengthscale  # (*batch, N1, N2, D)

        return exponent.sum(dim=-1).exp()


