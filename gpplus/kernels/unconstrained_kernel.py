import warnings
from typing import Optional, Tuple

import torch
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from torch import Tensor

from ..constraints import SoftClamp


class UnconstrainedKernel(Kernel):
    has_lengthscale = False

    def __init__(
        self,
        ard_num_dims: Optional[int] = None,
        batch_shape: Optional[torch.Size] = None,
        active_dims: Optional[Tuple[int, ...]] = None,
        lengthscale_prior: Optional[Prior] = None,
        lengthscale_constraint: Optional[SoftClamp] = None,
        eps: float = 1e-6,
        **kwargs,
    ):
        super(Kernel, self).__init__()  # Call Module's __init__ and Skip Kernel's __init__
        self._batch_shape = torch.Size([]) if batch_shape is None else batch_shape
        if active_dims is not None and not torch.is_tensor(active_dims):
            active_dims = torch.tensor(active_dims, dtype=torch.long)
        self.register_buffer("active_dims", active_dims)
        self.ard_num_dims = ard_num_dims

        self.eps = eps

        param_transform = kwargs.get("param_transform")

        if lengthscale_constraint is None:
            lengthscale_constraint = SoftClamp(lower_bound=-5, upper_bound=3, margin=1e-2)
            # lengthscale_constraint = gpytorch.constraints.Interval(lower_bound=-5, upper_bound=3)

        if param_transform is not None:
            # warnings.warn(
            #     "The 'param_transform' argument is now deprecated. If you want to use a different "
            #     "transformation, specify a different 'lengthscale_constraint' instead.",
            #     DeprecationWarning,
            # )
            warnings.warn(
                "The 'param_transform' argument is now deprecated.",
                DeprecationWarning,
            )

        if self.has_lengthscale:
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
            self.register_parameter(
                name="raw_lengthscale",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims)),
            )
            if lengthscale_prior is not None:
                if not isinstance(lengthscale_prior, Prior):
                    raise TypeError("Expected gpytorch.priors.Prior but got " + type(lengthscale_prior).__name__)
                self.register_prior(
                    "lengthscale_prior", lengthscale_prior, self._lengthscale_param, self._lengthscale_closure
                )
                # self.register_prior("lengthscale_prior", lengthscale_prior, "lengthscale")

            self.register_constraint("raw_lengthscale", lengthscale_constraint)

        self.distance_module = None
        # TODO: Remove this on next official PyTorch release.
        self.__pdist_supports_batch = True

    def covar_dist(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        # square_dist: bool = False,
        ord: float = 2,
        **params,
    ) -> Tensor:
        r"""
        This is a helper method for computing the Euclidean distance between
        all pairs of points in :math:`\mathbf x_1` and :math:`\mathbf x_2`.

        :param x1: First set of data (... x N x D).
        :param x2: Second set of data (... x M x D).
        :param diag: Should the Kernel compute the whole kernel, or just the diag?
            If True, it must be the case that `x1 == x2`. (Default: False.)
        :param last_dim_is_batch: If True, treat the last dimension
            of `x1` and `x2` as another batch dimension.
            (Useful for additive structure over the dimensions). (Default: False.)
        :param square_dist:
            If True, returns the squared distance rather than the standard distance. (Default: False.)
        :return: The kernel matrix or vector. The shape depends on the kernel's evaluation mode:

            * `full_covar`: `... x N x M`
            * `full_covar` with `last_dim_is_batch=True`: `... x K x N x M`
            * `diag`: `... x N`
            * `diag` with `last_dim_is_batch=True`: `... x K x N`
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)
        res = None

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                return torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
            else:
                res = torch.linalg.norm(x1 - x2, dim=-1, ord=ord)
                return res.pow(ord)
        else:
            res = torch.cdist(x1, x2, p=ord).clamp_min(1e-15)
            return res.pow(ord)
