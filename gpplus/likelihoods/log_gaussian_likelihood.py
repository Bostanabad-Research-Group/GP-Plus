#!/usr/bin/env python3

import warnings
from typing import Any, Optional

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import _HomoskedasticNoiseBase
from gpytorch.priors import Prior
from torch import Tensor

from ..constraints import SoftClamp


class LogScaleHomoskedasticNoise(_HomoskedasticNoiseBase):
    """
    Homoskedastic noise model with log-scale parameterization.

    This is similar to GPyTorch's HomoskedasticNoise but uses log-scale
    parameterization with SoftClamp constraints for better numerical stability.
    """

    def __init__(
        self,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[SoftClamp] = None,
        batch_shape: torch.Size = torch.Size(),
        **kwargs: Any,
    ) -> None:
        # Default constraint for log noise (allows noise from 0.0000001 to 1000)
        if noise_constraint is None:
            noise_constraint = SoftClamp(lower_bound=-7.0, upper_bound=3.0)

        # Call parent constructor with our custom constraint
        # We'll override the constraint after initialization
        super().__init__(noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape, **kwargs)

    def _noise_param(self, m):
        """Get the noise parameter from the model."""
        return m.noise

    def _noise_closure(self, m, v):
        """Set the noise parameter in the model."""
        return m._set_noise(v)

    @property
    def noise(self):
        """Get the actual noise parameter (10^raw_noise after constraint)."""
        return torch.pow(10, self.raw_noise_constraint.transform(self.raw_noise))

    @noise.setter
    def noise(self, value):
        """Set the noise parameter (will be converted to log scale internally)."""
        self._set_noise(value)

    def _set_noise(self, value):
        """Internal method to set noise parameter (converts to log scale)."""
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        # Convert to log scale
        log_value = torch.log10(value)
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(log_value))


class LogGaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    Custom Gaussian likelihood with log-scale noise parameterization.

    This is similar to GPyTorch's GaussianLikelihood but uses log-scale
    parameterization for the noise parameter, providing better numerical
    stability and consistency with custom kernels.

    Assumes a standard homoskedastic noise model:

    .. math::
        p(y \mid f) = f + \epsilon, \quad \epsilon \sim \mathcal N (0, \sigma^2)

    where :math:`\sigma^2` is a noise parameter parameterized in log scale internally.

    .. note::
        This likelihood can be used for exact or approximate inference.

    .. note::
        LogGaussianLikelihood has an analytic marginal distribution.

    Args:
        noise_prior: Prior for noise parameter :math:`\sigma^2`.
        noise_constraint: Constraint for raw_noise parameter. Default: `SoftClamp(-7.0, 3.0)`
                         (log scale from 0.0000001 to 1000).
        batch_shape: The batch shape of the learned noise parameter (default: []).

    Attributes:
        raw_noise (Tensor): Raw noise parameter (unconstrained)
        noise (Tensor): :math:`\sigma^2` parameter (actual noise value = 10^(constrained raw_noise))
    """

    has_analytic_marginal = True

    def __init__(
        self,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[SoftClamp] = None,
        batch_shape: torch.Size = torch.Size(),
        **kwargs: Any,
    ) -> None:
        # Handle deprecated param_transform argument
        param_transform = kwargs.get("param_transform")
        if param_transform is not None:
            warnings.warn(
                "The 'param_transform' argument is now deprecated. If you want to use a different "
                "transformation, specify a different 'noise_constraint' instead.",
                DeprecationWarning,
            )

        # Use our custom log-scale noise model
        noise_covar = LogScaleHomoskedasticNoise(
            noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
        )

        # Call the base class constructor with the noise covariance
        super().__init__(noise_covar=noise_covar, **kwargs)

    @property
    def noise(self) -> Tensor:
        """Get the actual noise parameter (10^(constrained raw_noise))."""
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        """Set the noise parameter (will be converted to log scale internally)."""
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        """Get the raw noise parameter (unconstrained)."""
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        """Set the raw noise parameter (unconstrained)."""
        self.noise_covar.initialize(raw_noise=value)

    def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> MultivariateNormal:
        r"""
        Compute the marginal distribution :math:`p(\mathbf y)`.

        Returns:
            Analytic marginal distribution.
        """
        return super().marginal(function_dist, *args, **kwargs)
