import torch
from gpytorch.kernels import Kernel

# from linear_operator.operators import DenseLinearOperator  # Convert tensors back to LazyTensor
from ..utils import InputTransformNet

################################


class CompositeKernel(Kernel):
    """
    Custom kernel where inputs are transformed using a provided transformation module
    before applying a base kernel.

    The transformation (`input_transform`) can be any valid mapping function, such as:
      - Polynomial expansion
      - Neural network module
      - Other custom transformations

    As long as `input_transform` is a callable or `torch.nn.Module` that
    takes in `x` and returns a transformed tensor, it can be used here.
    """

    def __init__(self, base_kernel, input_transform, **kwargs):
        """
        Args:
            base_kernel (Kernel): The base kernel to apply on the transformed inputs.
            input_transform (callable or torch.nn.Module): A user-defined module/function
                for transforming inputs (e.g., polynomial feature expansion, MLP, etc.).
        """
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.input_transform = input_transform  # Predefined network

    def forward(self, x1, x2, **params):
        """
        Transforms inputs and computes the base kernel matrix.

        Args:
            x1 (torch.Tensor): First input tensor.
            x2 (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Kernel matrix after transforming the inputs.
        """
        x1_transformed = self.input_transform(x1)
        x2_transformed = self.input_transform(x2)
        return self.base_kernel.forward(x1_transformed, x2_transformed, **params)


################################


class NeuralKernel(Kernel):
    """
    Custom kernel where inputs are transformed using an internally constructed neural network.

    For example, a valid `layer_config` could look like:

        layer_config = {
            0: {"dims": 64, "activation": nn.ReLU},
            1: {"dims": 32, "activation": nn.ReLU},
            2: {"dims": 16, "activation": nn.Tanh},
        }

    This will create a feedforward network with the specified layer sizes
    and activations in sequence.
    """

    def __init__(self, base_kernel, input_dim, layer_config, **kwargs):
        """
        Args:
            base_kernel (Kernel): The base kernel.
            input_dim (int): Input dimension if constructing a network.
            layer_config (dict): Layer configuration to construct a network.
        """
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.input_transform = InputTransformNet(input_dim, layer_config)

    def forward(self, x1, x2, **params):
        """
        Transforms inputs and computes the base kernel matrix.

        Args:
            x1 (torch.Tensor): First input tensor.
            x2 (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Kernel matrix after transforming the inputs.
        """
        x1_transformed = self.input_transform(x1)
        x2_transformed = self.input_transform(x2)
        return self.base_kernel.forward(x1_transformed, x2_transformed, **params)


################################


class NeuralScaleKernel(Kernel):
    """
    Custom kernel that constructs an InputTransformNet internally based on a layer configuration
    and computes the dot product between transformed inputs.

    The kernel function is defined as:
        k(x, x') = f(x) . f(x')
    where f is the internally constructed input transformation network.

    Args:
        input_dim (int): Input dimension of the data.
        layer_config (dict): Configuration dictionary for constructing the InputTransformNet.
        **kwargs: Additional keyword arguments for the Kernel base class.
    """

    def __init__(self, input_dim, layer_config, **kwargs):
        super().__init__(**kwargs)
        self.input_transform = InputTransformNet(input_dim, layer_config)

    def forward(self, x1, x2, **params):
        """
        Compute the dot product kernel matrix between transformed inputs.

        Args:
            x1 (torch.Tensor): First input tensor of shape (n1, d).
            x2 (torch.Tensor): Second input tensor of shape (n2, d).
            **params: Additional parameters (unused).

        Returns:
            torch.Tensor: Kernel matrix of shape (n1, n2).
        """
        # Apply the input transformation
        f_x1 = self.input_transform(x1)  # Shape: (n1, m)
        f_x2 = self.input_transform(x2)  # Shape: (n2, m)

        # Compute the dot product between transformed inputs
        return torch.mm(f_x1, f_x2.t())


################################


class CompositeScaleKernel(Kernel):
    """
    Custom kernel that applies an input transformation and computes the dot product
    between transformed inputs.

    The kernel function is defined as:
        k(x, x') = f(x) . f(x')
    where f is the input_transform neural network.

    Args:
        input_transform (callable or torch.nn.Module): The transformation to apply to inputs.
        **kwargs: Additional keyword arguments for the Kernel base class.
    """

    def __init__(self, input_transform, **kwargs):
        super().__init__(**kwargs)
        self.input_transform = input_transform  # Predefined transformation module

    def forward(self, x1, x2, **params):
        """
        Compute the dot product kernel matrix between transformed inputs.

        Args:
            x1 (torch.Tensor): First input tensor of shape (n1, d).
            x2 (torch.Tensor): Second input tensor of shape (n2, d).
            **params: Additional parameters (unused).

        Returns:
            torch.Tensor: Kernel matrix of shape (n1, n2).
        """
        # Apply the input transformation
        f_x1 = self.input_transform(x1)  # Shape: (n1, m)
        f_x2 = self.input_transform(x2)  # Shape: (n2, m)

        # Compute the dot product between transformed inputs
        return torch.mm(f_x1, f_x2.t())


################################


class GibbsKernel(Kernel):
    """
    A custom Gibbs (nonstationary) kernel in D dimensions with ARD.
    Each input x in R^D maps to a lengthscale vector ell(x) in R^D.
    """

    has_lengthscale = False  # We'll handle our own lengthscales

    def __init__(self, input_dim=None, layer_config=None, input_transform=None, **kwargs):
        super().__init__(**kwargs)

        # If no net is provided, define a small default MLP that outputs D dims:
        # We'll use Softplus to ensure positivity.
        self.input_dim = input_dim
        if input_dim is not None and layer_config is not None and input_transform is None:
            self.input_transform = InputTransformNet(input_dim, layer_config)
        elif input_dim is None and layer_config is None and input_transform is not None:
            self.input_transform = input_transform
        else:
            raise ValueError("Must pass either input_dim/layer_config or a pre-built input_transform.")

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """
        Computes the ARD Gibbs covariance between sets of points x1 and x2.
        x1, x2: [..., N, D] or [N, D]
        """
        if diag:
            # Diagonal means x1 == x2, so k(x, x) = 1 for all x (by design).
            # Return a vector of shape [..., N].
            return x1.new_ones(x1.shape[:-1])

        # Evaluate ell(x) for each point in x1 and x2:
        #    ell_1 -> shape [..., N1, D]
        #    ell_2 -> shape [..., N2, D]
        ell_1 = self.input_transform(x1)
        ell_2 = self.input_transform(x2)

        # Expand for pairwise distance:
        # x1_: [..., N1, 1, D], x2_: [..., 1, N2, D]
        x1_ = x1.unsqueeze(-2)
        x2_ = x2.unsqueeze(-3)

        # ell_1_: [..., N1, 1, D], ell_2_: [..., 1, N2, D]
        ell_1_ = ell_1.unsqueeze(-2)
        ell_2_ = ell_2.unsqueeze(-3)

        # Sum of squares of the lengthscales across each dimension
        ell_sum = ell_1_.pow(2) + ell_2_.pow(2)  # [..., N1, N2, D]
        ell_sum = ell_sum.clamp_min(1e-9)  # avoid division by zero

        # Distance term in the exponent: sum_d [ (x_d - x'_d)^2 / ell_d^2(x) + ell_d^2(x') ]
        sq_diff = (x1_ - x2_).pow(2)  # [..., N1, N2, D]
        exponent_term = -sq_diff / ell_sum
        exponent_term = exponent_term.sum(dim=-1)  # sum over D => [..., N1, N2]

        # Prefactor: product_d sqrt( 2 * ell_d(x1) * ell_d(x2) / [ell_d^2(x1)+ell_d^2(x2)] )
        # We'll do this in log-space for numerical stability:
        log_prefactor = 0.5 * (  # because of the sqrt
            torch.log(torch.tensor(2.0)) + torch.log(ell_1_ * ell_2_ + 1e-9) - torch.log(ell_sum)
        )
        # sum over dimension D
        log_prefactor = log_prefactor.sum(dim=-1)  # => [..., N1, N2]

        # Combine
        covar = torch.exp(log_prefactor + exponent_term)  # => [..., N1, N2]

        return covar


################################


class ExponentialKernel(Kernel):
    # has_lengthscale = True  # Enables automatic hyperparameter handling

    def __init__(self, base_kernel=None, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel

    def forward(self, x1, x2, diag=False, **params):
        # Compute the base kernel (this gives you either a Tensor for diag
        # or a LazyEvaluatedKernelTensor / LinearOperator for full covar).
        base_output = self.base_kernel(x1, x2, diag=diag, **params)  # Returns a LazyTensor

        if diag:
            # If diagonal, base_output is already a tensor
            # No need for LazyTensor wrapping on diag case
            return torch.exp(base_output)

        # Full-covar case: first get a dense Tensor
        dense_base = base_output.to_dense()

        return torch.exp(dense_base)  # Apply exp function; GPyTorch will lazily wrap this


class SinhKernel(Kernel):
    # has_lengthscale = True  # Enables automatic hyperparameter handling

    def __init__(self, base_kernel=None, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel

    def forward(self, x1, x2, diag=False, **params):
        # Compute the base kernel (this gives you either a Tensor for diag
        # or a LazyEvaluatedKernelTensor / LinearOperator for full covar).
        base_output = self.base_kernel(x1, x2, diag=diag, **params)  # Returns a LazyTensor

        if diag:
            # If diagonal, base_output is already a tensor
            # No need for LazyTensor wrapping on diag case
            return torch.sinh(base_output)

        # Full-covar case: first get a dense Tensor
        dense_base = base_output.to_dense()

        return torch.sinh(dense_base)  # Apply sinh function; GPyTorch will lazily wrap this


class CoshKernel(Kernel):
    # has_lengthscale = True  # Enables automatic hyperparameter handling

    def __init__(self, base_kernel=None, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel

    def forward(self, x1, x2, diag=False, **params):
        # Compute the base kernel (this gives you either a Tensor for diag
        # or a LazyEvaluatedKernelTensor / LinearOperator for full covar).
        base_output = self.base_kernel(x1, x2, diag=diag, **params)  # Returns a LazyTensor

        if diag:
            # If diagonal, base_output is already a tensor
            # No need for LazyTensor wrapping on diag case
            return torch.cosh(base_output)

        # Full-covar case: first get a dense Tensor
        dense_base = base_output.to_dense()

        return torch.cosh(dense_base)  # Apply cosh function; GPyTorch will lazily wrap this
