from typing import Any, List, Optional, Union

import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import _HomoskedasticNoiseBase
from linear_operator.operators import ConstantDiagLinearOperator, DiagLinearOperator
from torch import Tensor

from gpplus.constraints import SoftClamp


class MultiLikelihood(_GaussianLikelihoodBase):
    """
    Multifidelity likelihood that allows different noise levels for each fidelity source.

    Args:
        source_cols: Either:
            - Tensor, list, or numpy array of column indices for one-hot encoded fidelity levels, OR
            - Single integer indicating the column index containing fidelity indicators
        noise_prior: Prior distribution for noise parameters
        noise_constraint: Constraint for noise parameters
        batch_shape: Batch shape for the likelihood
        training_x: Training data to determine fidelity order
    """

    def __init__(
        self,
        source_cols: Union[int, List[int], Tensor],
        noise_prior=None,
        noise_constraint=None,
        batch_shape=torch.Size(),
        training_data: Optional[Tensor] = None,
        log_scale=True,
        **kwargs,
    ):
        # Store source_cols and determine if it's one-hot or single column
        self.source_cols = source_cols
        self.is_onehot = isinstance(source_cols, (list, tuple, Tensor, np.ndarray))

        if self.is_onehot:
            self.source_cols = torch.tensor(source_cols, dtype=torch.long)
            self.num_fidelities = len(source_cols)
        else:
            self.source_col = int(source_cols)
            if training_data is not None:
                fidel_indices = training_data[:, self.source_col].long()
                self.num_fidelities = len(torch.unique(fidel_indices))
            else:
                raise ValueError(
                    "For single-column fidelity encoding, training_data must be provided "
                    "at initialization to determine the number of fidelities for noise model."
                )

        # Initialize noise model with log-scale parameterization for better performance
        if log_scale:
            noise_covar = LogScaleMultiNoise(
                noise_prior=noise_prior,
                noise_constraint=noise_constraint,
                batch_shape=batch_shape,
                num_noises=self.num_fidelities,
            )
        else:
            # Should use only if using linear-scale parameterization for modules (gpytorch default)
            noise_covar = MultiNoise(
                noise_prior=noise_prior,
                noise_constraint=noise_constraint,
                batch_shape=batch_shape,
                num_noises=self.num_fidelities,
            )

        # Initialize parent class with noise_covar FIRST (needed before registering buffers)
        super().__init__(noise_covar=noise_covar)

        # Buffers to persist fidelity/noise indices across state_dict save/load
        self.register_buffer("fidel_indices", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("noise_indices", torch.empty(0, dtype=torch.long), persistent=True)
        # test_x will be created lazily when set_fidelity_indices(x, is_test=True) is called

        # Optionally set fidelity indices from provided training data to fix source order
        if training_data is not None:
            try:
                self.set_fidelity_indices(training_data)
            except KeyError:
                pass

    def _extract_fidelity_indices(self, x: Tensor) -> Tensor:
        """
        Extract fidelity indices from input data based on source_cols.

        Args:
            x: Input tensor of shape (N, D) where N is number of samples, D is number of features

        Returns:
            Tensor of shape (N,) containing fidelity indices for each sample
        """
        if self.is_onehot:
            # One-hot encoded case: find which column has value 1
            onehot_cols = x[:, self.source_cols]  # Shape: (N, num_fidelities)
            fidel_indices = torch.argmax(onehot_cols, dim=1)  # Shape: (N,)
        else:
            # Single column case: use values directly as fidelity indices
            fidel_indices = x[:, self.source_col].long()  # Shape: (N,)

        return fidel_indices

    def set_fidelity_indices(self, x: Tensor, is_test: bool = False) -> None:
        """
        Set the fidelity indices from input data.

        Args:
            x: Input tensor of shape (N, D) where N is number of samples, D is number of features
            is_test: If True, stores test data for later extraction during prediction.
                     If False, stores as training fidelity indices (default, for training).
        """
        fidel = self._extract_fidelity_indices(x)
        # Ensure on same device
        fidel = fidel.to(device=self.noise_covar.noise.device)

        if is_test:
            # Store test data so we can extract fidelity indices on-demand during prediction
            self.register_buffer("test_x", x.detach(), persistent=False)
        else:
            # Store as training fidelity indices (persistent)
            if self.fidel_indices.numel() == 0 or self.fidel_indices.shape != fidel.shape:
                # Resize/replace buffer
                self.register_buffer("fidel_indices", fidel.detach(), persistent=True)
            else:
                self.fidel_indices.copy_(fidel)
            if self.noise_indices.numel() == 0:
                # Extract unique fidelities from the data to set noise_indices
                unique_fidelities = torch.unique(fidel)
                ni = unique_fidelities.to(device=self.noise_covar.noise.device, dtype=torch.long)
                self.register_buffer("noise_indices", ni.detach(), persistent=True)

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        """Get the noise covariance matrix for the given parameters.
        Automatically determines which fidelity indices to use based on shape matching.
        """
        if "fidel_indices" in kwargs and kwargs["fidel_indices"] is not None:
            pass
        else:
            # Determine which fidelity indices to use based on shape
            # base_shape is mean.shape, which is typically (n_samples,) or (batch_shape..., n_samples)
            n_samples = base_shape[-1] if len(base_shape) > 0 else 0

            # If we have test data stored and the shape matches, extract indices from test data
            if hasattr(self, "test_x") and self.test_x.numel() > 0 and self.test_x.shape[0] == n_samples:
                # Extract fidelity indices from stored test data
                test_fidel = self._extract_fidelity_indices(self.test_x)
                kwargs["fidel_indices"] = test_fidel.to(device=self.noise_covar.noise.device)
            elif self.fidel_indices.numel() > 0:
                # Use training fidelity indices (for prediction strategy setup with training data)
                kwargs["fidel_indices"] = self.fidel_indices
            else:
                raise ValueError(
                    "Fidelity indices are required to build the noise matrix. "
                    "Call likelihood.set_fidelity_indices(x) beforehand or provide fidel_indices."
                )

            if self.noise_indices.numel() > 0:
                kwargs["noise_indices"] = self.noise_indices.tolist()

        return self.noise_covar.forward(*params, shape=base_shape, **kwargs)

    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        """Compute the marginal distribution by adding noise covariance."""
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)

        # Add noise covariance to the function covariance
        full_covar = covar + noise_covar

        return function_dist.__class__(mean, full_covar)


class LogScaleMultiNoise(_HomoskedasticNoiseBase):
    """
    Multifidelity noise model with log-scale parameterization.

    This applies different noise levels based on fidelity indices, but uses log-scale
    parameterization (10^raw_noise) for better numerical stability and optimization,
    similar to LogScaleHomoskedasticNoise.

    Args:
        noise_prior: Prior distribution for noise parameters
        noise_constraint: Constraint for noise parameters (default: SoftClamp(-7.0, 3.0))
        batch_shape: Batch shape for the noise model
        num_noises: Number of different noise levels to learn
    """

    def __init__(self, noise_prior=None, noise_constraint=None, batch_shape=torch.Size(), num_noises=1):
        # Default constraint for log noise (allows noise from 0.0000001 to 1000)
        if noise_constraint is None:
            noise_constraint = SoftClamp(lower_bound=-7.0, upper_bound=3.0)

        super().__init__(noise_prior, noise_constraint, batch_shape, num_tasks=num_noises)

    @property
    def noise(self):
        """Get the actual noise parameter (10^raw_noise after constraint)."""
        # The parent class stores the constraint as raw_noise_constraint
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
        # The parent class stores the constraint as raw_noise_constraint
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(log_value))

    def forward(
        self,
        *params: Any,
        shape: Optional[torch.Size] = None,
        fidel_indices: Optional[Tensor] = None,
        noise_indices: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> DiagLinearOperator:
        """
        Compute a diagonal noise covariance where each point's variance
        is selected by its fidelity index. Uses log-scale parameterization
        (noise values are 10^raw_noise after constraint).

        Args:
            *params: Parameters from the base noise model
            shape: Unused—shape is inferred from fidel_indices
            fidel_indices: Tensor of shape (N,) mapping each point to a fidelity level
            noise_indices: Which fidelity levels correspond to each learned noise parameter
            **kwargs: Additional keyword arguments

        Returns:
            DiagLinearOperator: A (N×N) diagonal covariance tensor where entry i,i
                = noise[fidelity_indices[i]] (with log-scale transformation applied)
        """
        if fidel_indices is None or len(fidel_indices) == 0:
            raise ValueError("fidel_indices must be provided and non-empty")

        if noise_indices is None:
            noise_indices = list(range(self.num_tasks))

        # Get noise variances from parent class (uses log-scale parameterization)
        covar = super().forward(*params, shape=fidel_indices.shape, **kwargs)

        # Handle different covariance dimensions
        if covar.dim() > 2:
            if covar.shape[1] != len(noise_indices):
                raise ValueError(
                    f"Number of noise parameters ({covar.shape[1]}) does not \
                        match number of noise indices ({len(noise_indices)})"
                )

        if covar.dim() == 4:  # batch case
            covar = covar.squeeze(0)
        elif covar.dim() == 5:  # batch case
            covar = covar.squeeze(1)

        # Initialize diagonal matrix with zeros
        temp = ConstantDiagLinearOperator(torch.tensor([0.0]), len(fidel_indices))
        temp = temp.to(dtype=covar.dtype, device=covar.device)

        # For each noise level, create a diagonal mask and accumulate
        for i, noise_idx in enumerate(noise_indices):
            # Create diagonal mask for points with this fidelity level
            # Ensure the mask is on the same device as the covariance matrix
            mask_values = (fidel_indices == noise_idx).float()
            if covar.dim() >= 3:
                # Get device from the first noise parameter
                device = covar[i, ...].device if covar.dim() == 3 else covar[0, i, ...].device
                mask_values = mask_values.to(device)
            diag_mask = DiagLinearOperator(mask_values)

            # Select the appropriate noise variance and multiply by mask
            if covar.dim() == 4:  # batch case
                temp += diag_mask * covar[:, i, ...]
            elif covar.dim() == 3:  # no batch
                temp += diag_mask * covar[i, ...]
            elif covar.dim() == 2:  # single noise parameter
                temp += diag_mask * covar
            else:
                raise ValueError(f"Unexpected covariance dimension: {covar.dim()}")

        return temp


class MultiNoise(_HomoskedasticNoiseBase):
    """
    Multifidelity noise model that applies different noise levels based on fidelity indices.

    NOTE: This uses linear-scale parameterization. For better performance, consider using
    LogScaleMultiNoise instead, which uses log-scale parameterization.

    Args:
        noise_prior: Prior distribution for noise parameters
        noise_constraint: Constraint for noise parameters
        batch_shape: Batch shape for the noise model
        num_noises: Number of different noise levels to learn
    """

    def __init__(self, noise_prior=None, noise_constraint=None, batch_shape=torch.Size(), num_noises=1):
        super().__init__(noise_prior, noise_constraint, batch_shape, num_tasks=num_noises)

    def forward(
        self,
        *params: Any,
        shape: Optional[torch.Size] = None,
        fidel_indices: Optional[Tensor] = None,
        noise_indices: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> DiagLinearOperator:
        """
        Compute a diagonal noise covariance where each point's variance
        is selected by its fidelity index.

        Args:
            *params: Parameters from the base noise model
            shape: Unused—shape is inferred from fidel_indices
            fidel_indices: Tensor of shape (N,) mapping each point to a fidelity level
            noise_indices: Which fidelity levels correspond to each learned noise parameter
            **kwargs: Additional keyword arguments

        Returns:
            DiagLinearOperator: A (N×N) diagonal covariance tensor where entry i,i
                = noise[fidelity_indices[i]]
        """
        if fidel_indices is None or len(fidel_indices) == 0:
            raise ValueError("fidel_indices must be provided and non-empty")

        if noise_indices is None:
            noise_indices = list(range(self.num_tasks))

        # Get noise variances from parent class (uses linear-scale parameterization)
        covar = super().forward(*params, shape=fidel_indices.shape, **kwargs)

        # Handle different covariance dimensions
        if covar.dim() > 2:
            if covar.shape[1] != len(noise_indices):
                raise ValueError(
                    f"Number of noise parameters ({covar.shape[1]}) does not \
                        match number of noise indices ({len(noise_indices)})"
                )

        if covar.dim() == 4:  # batch case
            covar = covar.squeeze(0)
        elif covar.dim() == 5:  # batch case
            covar = covar.squeeze(1)

        # Initialize diagonal matrix with zeros
        temp = ConstantDiagLinearOperator(torch.tensor([0.0]), len(fidel_indices))
        temp = temp.to(dtype=covar.dtype, device=covar.device)

        # For each noise level, create a diagonal mask and accumulate
        for i, noise_idx in enumerate(noise_indices):
            # Create diagonal mask for points with this fidelity level
            # Ensure the mask is on the same device as the covariance matrix
            mask_values = (fidel_indices == noise_idx).float()
            if covar.dim() >= 3:
                # Get device from the first noise parameter
                device = covar[i, ...].device if covar.dim() == 3 else covar[0, i, ...].device
                mask_values = mask_values.to(device)
            diag_mask = DiagLinearOperator(mask_values)

            # Select the appropriate noise variance and multiply by mask
            if covar.dim() == 4:  # batch case
                temp += diag_mask * covar[:, i, ...]
            elif covar.dim() == 3:  # no batch
                temp += diag_mask * covar[i, ...]
            elif covar.dim() == 2:  # single noise parameter
                temp += diag_mask * covar
            else:
                raise ValueError(f"Unexpected covariance dimension: {covar.dim()}")

        return temp
