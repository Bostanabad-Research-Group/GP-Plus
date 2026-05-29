import torch
from gpytorch.constraints import Interval

# from gpytorch.kernels import Kernel
from ..kernels import UnconstrainedKernel


def postprocess_(dist_mat):
    return dist_mat.mul_(-1).exp_()


class PowerExponentialKernelFixed(UnconstrainedKernel):
    has_lengthscale = True  # Enable lengthscale functionality
    is_stationary = True  # The kernel is stationary

    def __init__(self, power=2.0, **kwargs):
        # Validate that power is between 1 and 2
        if not (1 <= power <= 2):
            raise ValueError("The 'power' parameter must be between 1 and 2.")
        # power is the exponent parameter (κ)
        super().__init__(**kwargs)
        self.power = power

    def forward(self, x1, x2, diag=False, **params):
        scaling_factors = torch.pow(10, self.lengthscale / self.power)

        x1_ = x1.mul(scaling_factors)
        x2_ = x2.mul(scaling_factors)

        return postprocess_(self.covar_dist(x1_, x2_, ord=self.power, diag=diag, **params))


class PowerExponentialKernel(UnconstrainedKernel):
    has_lengthscale = True  # Enable lengthscale functionality
    is_stationary = True  # The kernel is stationary

    def __init__(self, **kwargs):
        # power is the exponent parameter (κ)
        super().__init__(**kwargs)
        # Register the power parameter (as a raw parameter to allow constraints)
        self.register_parameter(
            name="raw_power",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )

        # Constrain raw_power to be in the interval [1, 2]
        self.register_constraint("raw_power", Interval(1.0, 2.0))

    @property
    def power(self):
        # Transform raw_power via the constraint's transformation.
        return self.raw_power_constraint.transform(self.raw_power)

    @power.setter
    def power(self, value):
        self._set_power(value)

    def _set_power(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_power)
        # Use the inverse transform to set the raw parameter value
        self.initialize(raw_power=self.raw_power_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        scaling_factors = torch.pow(10, self.lengthscale / self.power)

        x1_ = x1.mul(scaling_factors)
        x2_ = x2.mul(scaling_factors)

        return postprocess_(self.covar_dist(x1_, x2_, ord=self.power.item(), diag=diag, **params))
