import torch

# from gpytorch.kernels import Kernel
from ..kernels import UnconstrainedKernel


def postprocess_gaussian(dist_mat):
    return dist_mat.mul_(-1).exp_()


class GaussianKernel(UnconstrainedKernel):
    has_lengthscale = True  # Enable lengthscale functionality
    is_stationary = True  # The kernel is stationary

    def forward(self, x1, x2, diag=False, **params):
        power = 2.0

        scaling_factors = torch.pow(10, self.lengthscale / power)

        x1_ = x1.mul(scaling_factors)
        x2_ = x2.mul(scaling_factors)

        return postprocess_gaussian(self.covar_dist(x1_, x2_, ord=power, diag=diag, **params))
