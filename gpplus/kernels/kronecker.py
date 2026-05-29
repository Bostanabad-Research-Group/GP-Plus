from typing import List

import numpy as np
import torch
from gpytorch.kernels import Kernel
from linear_operator.operators import KroneckerProductLinearOperator


class KroneckerKernel(Kernel):
    """
    Computes the covariance matrix based on a Kronecker structure.

    Args:
        ard_num_dims (int, optional): Set this if you want a separate lengthscale for
            each input dimension. It should be `d` if :math:`\mathbf{x_1}` is a `n x d`
            matrix. Defaults to `None`.
        kernels (list): Ordered list of kernels to perform the Kronecker product.
        column_indices (list): List with start and end indices for the input columns
            corresponding to each kernel features.

    Returns:
        CovarianceMatrix: The resulting covariance matrix.
    """

    def __init__(
        self,
        ard_num_dims,
        kernels: List[Kernel],
        column_indices: List[List[int]],
        **kwargs,
    ):
        super(KroneckerKernel, self).__init__(ard_num_dims=ard_num_dims, **kwargs)

        self._kernels = kernels
        self._xi = column_indices

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """
        Compute the covariance matrix using a Kronecker structure.

        This method computes the covariance matrix by combining multiple kernels
        through a Kronecker product. Each kernel operates on specific features
        of the input tensors `x1` and `x2`. The method supports diagonal extraction
        if `diag` is set to `True`.

        Args:
            x1 (torch.Tensor): The first input tensor of shape `(n, d1, ...)`,
                where `d1` is the number of dimensions used by the first kernel.
            x2 (torch.Tensor): The second input tensor of shape `(m, d2, ...)`,
                where `d2` is the number of dimensions used by the first kernel.
            diag (bool, optional): If `True`, only the diagonal of the covariance matrix is returned.
                Defaults to `False`.
            last_dim_is_batch (bool, optional): If `True`, treats the last dimension of inputs as batch dimensions.
                This argument is not supported and will raise a `RuntimeError` if used. Defaults to `False`.
            **params: Additional keyword arguments for compatibility (currently unused).

        Returns:
            torch.Tensor or KroneckerProductLinearOperator:
                - If `diag` is `True`, returns the diagonal of the covariance matrix as a tensor.
                - Otherwise, returns a `KroneckerProductLinearOperator` representing the full covariance matrix.

        Raises:
            RuntimeError: If `last_dim_is_batch` is set to `True`.
        """
        if last_dim_is_batch:
            raise RuntimeError("KroneckerKernel does not accept the last_dim_is_batch argument.")
        covar = []
        for i, kernel in enumerate(self._kernels):
            unique_x1 = self._unique_rows_in_order(x1[:, :, self._xi[i]])
            unique_x2 = self._unique_rows_in_order(x2[:, :, self._xi[i]])
            kres = kernel(unique_x1, unique_x2)
            covar.append(kres)
        res = KroneckerProductLinearOperator(*covar)
        return res.diagonal(dim1=-1, dim2=-2) if diag else res

    def _unique_rows_in_order(self, a: torch.tensor) -> torch.tensor:
        a_np = np.array(a[0])
        _, unique_indices = np.unique(a_np, axis=0, return_index=True)
        unique_rows = a_np[np.sort(unique_indices)]
        res = torch.tensor(unique_rows).unsqueeze(0)
        return res

    def get_kernel(self, idx: int) -> Kernel:
        """
        Retrieve a Kronecker kernel by its index.

        This method returns the kernel corresponding to the specified order index
        in the Kronecker structure.

        Args:
            idx (int): The index of the kernel to retrieve. Must be within the range
                `[0, len(self._kernels) - 1]`.

        Returns:
            Kernel: The kernel at the specified index.

        Raises:
            IndexError: If the provided index is out of range.
        """
        return self._kernels[idx]
