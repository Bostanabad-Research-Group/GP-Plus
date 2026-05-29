from typing import Optional

import torch
from gpytorch.priors import Prior
from linear_operator.operators import to_dense

from ..constraints import SoftClamp

# from .kernel import UnconstrainedKernel
from ..kernels import UnconstrainedKernel


class LogScaleKernel(UnconstrainedKernel):
    r"""
    Decorates a base kernel with a learnable output scale, using a base-10 log parameterization.

    .. math::

    \begin{equation*}
        K_{\text{scaled}} \;=\; \big(10^{\,\text{outputscale}}\big)\; K_{\text{base}}
    \end{equation*}

    Here, ``outputscale`` is the learnable *log10*-scale parameter; the actual multiplicative factor is
    ``10**outputscale``. This keeps the underlying parameter unconstrained while ensuring the
    applied scale is positive. By default, the parameter is constrained with a
    :class:`~gpytorch.constraints.SoftClamp` (e.g., ``[-5, 3]`` ⇒ scale in ``[1e-5, 1e3]``),
    and you may optionally place a prior on the constrained parameter.

    Batching
    --------
    In batch settings (e.g., when ``x1`` and ``x2`` contain batches of inputs), a separate
    ``outputscale`` can be learned per batch by setting ``batch_shape`` accordingly.

    Args:
        base_kernel (UnconstrainedKernel):
            The kernel whose covariance will be scaled.
        batch_shape (torch.Size, optional):
            If provided, learns one ``outputscale`` per batch. For inputs of shape
            ``b × n × d``, set this to ``torch.Size([b])``. Default: ``torch.Size([])``.
        outputscale_prior (gpytorch.priors.Prior, optional):
            Prior over the constrained ``outputscale`` parameter (log10 space). Default: ``None``.
        outputscale_constraint (gpytorch.constraints.Constraint, optional):
            Constraint applied to the raw parameter. Default: :class:`SoftClamp`.

    Attributes:
        base_kernel (UnconstrainedKernel):
            The wrapped kernel module.
        outputscale (Tensor):
            The *log10* output-scale parameter. Its shape matches ``batch_shape``.
            The effective multiplier used at runtime is ``10**outputscale``.

    Example:
        >>> x = torch.randn(10, 5)
        >>> base = gpplus.kernels.GaussianKernel()
        >>> covar_module = LogScaleKernel(base)  # Applies 10**outputscale * K_base
        >>> covar = covar_module(x)                # LinearOperator with shape (10 x 10)
    """

    @property
    def is_stationary(self) -> bool:
        # Stationarity is inherited from the base kernel
        return self.base_kernel.is_stationary

    def __init__(
        self,
        base_kernel: UnconstrainedKernel,
        outputscale_prior: Optional[Prior] = None,
        outputscale_constraint: Optional[SoftClamp] = None,
        **kwargs,
    ):
        # Preserve active_dims of the wrapped base kernel
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims

        super().__init__(**kwargs)

        # default to SoftClamp if user didn't provide a constraint
        if outputscale_constraint is None:
            outputscale_constraint = SoftClamp(lower_bound=-5, upper_bound=4, margin=1e-2)

        self.base_kernel = base_kernel
        # Start at log_outputscale = 0 -> multiplicative scale of 10**0 = 1.0
        self.register_parameter(
            name="raw_outputscale",
            parameter=torch.nn.Parameter(
                torch.zeros(*self.batch_shape) if len(self.batch_shape) else torch.tensor(0.0)
            ),
        )

        # Optional prior over the *constrained* parameter
        if outputscale_prior is not None:
            if not isinstance(outputscale_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(outputscale_prior).__name__)
            self.register_prior(
                "outputscale_prior", outputscale_prior, self._outputscale_param, self._outputscale_closure
            )

        # Register how to transform raw <-> constrained
        self.register_constraint("raw_outputscale", outputscale_constraint)

    # --- Prior hooks ---
    def _outputscale_param(self, m):
        return m.outputscale

    def _outputscale_closure(self, m, v):
        m._set_outputscale(v)

    # --- properties: view + setter ---
    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    # --- Core scaling logic ---
    def forward(self, x1, x2, last_dim_is_batch: bool = False, diag: bool = False, **params):
        orig_output = self.base_kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)

        # use 10**s instead of s
        multiplier = torch.pow(10.0, self.outputscale)

        # Align shapes for broadcasting
        if last_dim_is_batch:
            multiplier = multiplier.unsqueeze(-1)

        if diag:
            multiplier = multiplier.unsqueeze(-1)  # (batch..., N)
            return to_dense(orig_output) * multiplier
        else:
            multiplier = multiplier.view(*multiplier.shape, 1, 1)  # (batch..., 1, 1)
            return orig_output.mul(multiplier)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel.prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)
