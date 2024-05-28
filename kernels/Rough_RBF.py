from gpytorch.kernels import RBFKernel
from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode
import torch

def postprocess_rbf(dist_mat):
    return dist_mat.div_(-1).exp_()


class Rough_RBF(RBFKernel):
    """_summary_

    Args:
        RBFKernel (_type_): _description_
    """
    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            ten_power_omega_sqrt = self.lengthscale.sqrt()
            x1_ = x1.mul(ten_power_omega_sqrt)
            x2_ = x2.mul(ten_power_omega_sqrt)
            return self.covar_dist(
                x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rbf, postprocess=True, **params
            )
        return RBFCovariance.apply(
            x1,
            x2,
            self.lengthscale,
            lambda x1, x2: self.covar_dist(
                x1, x2, square_dist=True, diag=False, dist_postprocess_func=postprocess_rbf, postprocess=False, **params
            ),
        )