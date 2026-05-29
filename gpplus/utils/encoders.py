"""
Encoder module.

Public API
----------
- ``BaseEncoder``: abstract base class. Subclass it and implement ``forward``.
- ``MatrixEncoder``, ``NeuralEncoder``: shipped backends.

Adding a new encoder is just::

    class MyEncoder(BaseEncoder):
        def __init__(self, input_dim, z_dim=2, ...):
            super().__init__(input_dim=input_dim, z_dim=z_dim)
            ...
        def forward(self, x, **kwargs):
            ...
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any, Optional

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseEncoder(gpytorch.Module):
    """
    Abstract base class for all encoders.

    Inherits from ``gpytorch.Module`` (a ``torch.nn.Module`` with prior support)
    so subclasses can register Bayesian priors when they need to, without
    paying any cost when they don't.

    Subclasses MUST:
        - call ``super().__init__(input_dim=..., z_dim=...)``
        - implement ``forward(self, x, **kwargs)``

    Subclasses MAY:
        - set the class attribute ``is_probabilistic = True``
        - override ``encode_pair`` for custom pairwise behavior (e.g. shared
          stochastic sampling)
    """

    is_probabilistic: bool = False

    def __init__(self, input_dim: int, z_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.z_dim = int(z_dim)

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Encode ``x`` into the latent space."""
        raise NotImplementedError

    def encode_pair(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        shared_sampling: bool = True,
        **kwargs: Any,
    ):
        """
        Encode a pair of inputs.

        Default implementation calls ``forward`` twice. Probabilistic
        encoders should override this to share noise between ``x1`` and
        ``x2`` so pairwise kernel evaluations are not corrupted by
        sampling-mismatch noise.
        """
        if x2 is None:
            x2 = x1
        return self(x1, **kwargs), self(x2, **kwargs)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, z_dim={self.z_dim}"


# ---------------------------------------------------------------------------
# MatrixEncoder
# ---------------------------------------------------------------------------


class MatrixEncoder(BaseEncoder):
    """
    Matrix-based encoder for categorical variables (Section 4 of the GP+ paper).

    Maps one-hot encoded categorical inputs to a latent space using a single
    learnable projection matrix. Interpretable and cheap.

    Args:
        input_dim: Dimension of input (one-hot vectors).
        z_dim: Dimension of latent space (default=2).
        initialization: 'normal', 'uniform', or 'orthogonal'.
        init_std: Spread parameter for the chosen initialization.
        seed: Optional RNG seed for reproducible initialization.
    """

    is_probabilistic = False

    def __init__(
        self,
        input_dim: int,
        z_dim: int = 2,
        initialization: str = "normal",
        init_std: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__(input_dim=input_dim, z_dim=z_dim)
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        self.seed = int(seed)

        self.projection_matrix = nn.Parameter(torch.empty(input_dim, z_dim))

        # Bookkeeping consumed by an external parameter initializer.
        self._param_init_types = {"projection_matrix": initialization}
        self._param_init_params = {"projection_matrix": {"init_std": init_std}}

        generator = torch.Generator().manual_seed(self.seed)
        if initialization == "normal":
            nn.init.normal_(self.projection_matrix, mean=0.0, std=init_std, generator=generator)
        elif initialization == "uniform":
            nn.init.uniform_(self.projection_matrix, -init_std, init_std, generator=generator)
        elif initialization == "orthogonal":
            nn.init.orthogonal_(self.projection_matrix, gain=init_std, generator=generator)
        else:
            raise ValueError(f"Unknown initialization method: {initialization}")

    def forward(self, x_onehot: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if x_onehot.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x_onehot.shape[-1]}")
        x_onehot = x_onehot.to(
            device=self.projection_matrix.device,
            dtype=self.projection_matrix.dtype,
        )
        return torch.matmul(x_onehot, self.projection_matrix)

    def get_projection_matrix(self) -> torch.Tensor:
        return self.projection_matrix.data.clone()

    def set_projection_matrix(self, matrix: torch.Tensor) -> None:
        if matrix.shape != (self.input_dim, self.z_dim):
            raise ValueError(f"Expected matrix shape {(self.input_dim, self.z_dim)}, got {matrix.shape}")
        self.projection_matrix.data = matrix.clone()


# ---------------------------------------------------------------------------
# NeuralEncoder
# ---------------------------------------------------------------------------


class NeuralEncoder(BaseEncoder):
    """
    Neural encoder for one-hot categorical inputs.

    Two modes:
        - Deterministic: a feed-forward MLP producing a z_dim-dim embedding.
        - Probabilistic: outputs the parameters of a 2D Gaussian per unique
          category (mu1, mu2, L21, L11, L22) and samples a latent point via
          the reparameterization trick. Currently restricted to z_dim=2.

    Args:
        input_dim: Number of one-hot categories.
        architecture_config: dict with keys ``hidden_dims`` (list[int]),
            ``activation`` (str), and ``dropout`` (float or None).
        z_dim: Latent dimension (must be 2 for probabilistic mode).
        num_passes: Forward passes during training (probabilistic mode).
        num_passes_pred: Forward passes during prediction. Defaults to
            ``num_passes``.
        probabilistic: Whether to use the probabilistic head.
    """

    def __init__(
        self,
        input_dim: int,
        architecture_config: Optional[dict] = None,
        z_dim: int = 2,
        num_passes: int = 1,
        num_passes_pred: Optional[int] = None,
        probabilistic: bool = False,
    ):
        super().__init__(input_dim=input_dim, z_dim=z_dim)
        self.num_passes = int(num_passes)
        # Instance-level override of the class default.
        self.is_probabilistic = bool(probabilistic)

        if self.is_probabilistic and self.z_dim != 2:
            raise NotImplementedError("Probabilistic NeuralEncoder currently only supports 2D embeddings.")

        # Bug fix: previously both branches assigned the same value.
        if num_passes_pred is None or num_passes == 1:
            self.num_passes_pred = self.num_passes
        else:
            self.num_passes_pred = int(num_passes_pred)

        if architecture_config is None:
            architecture_config = {"hidden_dims": [], "activation": "tanh", "dropout": None}
        self._architecture_config = architecture_config

        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "hardtanh": nn.Hardtanh(),
        }
        activation = activation_map.get(architecture_config["activation"].lower(), nn.ReLU())

        if self.is_probabilistic:
            self._build_probabilistic(input_dim, architecture_config)
        else:
            self._build_deterministic(input_dim, architecture_config, activation)

    # ---- builders ---------------------------------------------------------

    def _build_probabilistic(self, input_dim: int, architecture_config: dict) -> None:
        vae_output_dim = 5  # mu1, mu2, L21, L11, L22
        self.fci = Linear_VAE(in_features=input_dim, out_features=vae_output_dim, bias=True, name="fci")
        self.hidden_layers = list(architecture_config["hidden_dims"])
        if self.hidden_layers:
            prev_dim = vae_output_dim
            for i, hidden_dim in enumerate(self.hidden_layers):
                name = f"h{i + 1}"
                setattr(self, name, Linear_VAE(prev_dim, hidden_dim, name=name))
                prev_dim = hidden_dim
            self.fce = Linear_VAE(in_features=prev_dim, out_features=vae_output_dim, bias=True, name="fce")
        else:
            self.fce = None

    def _build_deterministic(self, input_dim: int, architecture_config: dict, activation: nn.Module) -> None:
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in architecture_config["hidden_dims"]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            if architecture_config.get("dropout"):
                layers.append(nn.Dropout(architecture_config["dropout"]))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.z_dim))
        self.fci = nn.Sequential(*layers)

    # ---- forward ----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        epsilon: Optional[torch.Tensor] = None,
        visualize: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not self.is_probabilistic:
            return self.fci(x)

        # Probabilistic path.
        if x.dim() <= 1:
            # Single-sample 1D input: return distribution parameters directly.
            return self.fci(x)

        # Multi-sample input: deduplicate categories before pushing through
        # the head, then project back onto the one-hot inputs.
        xtemp, _inverse = torch.unique(x, dim=0, return_inverse=True)
        if self.hidden_layers:
            h = self.fci(xtemp)
            for i in range(len(self.hidden_layers)):
                h = F.relu(getattr(self, f"h{i + 1}")(h))
            out = self.fce(h)
        else:
            out = self.fci(xtemp)

        # Sample latent points per unique category via reparameterization.
        # epsilon shape: [num_unique_categories, z_dim]. Caller may pass a
        # larger epsilon and we will slice it; if smaller, we resample.
        num_unique = out.shape[0]
        if epsilon is None or epsilon.shape[0] < num_unique:
            epsilon = torch.normal(
                mean=0.0,
                std=1.0,
                size=(num_unique, self.z_dim),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            epsilon = epsilon[:num_unique]

        Mu_1 = out[:, 0:1]
        Mu_2 = out[:, 1:2]
        L21 = out[:, 2:3]
        L11 = torch.abs(out[:, 3:4])
        L22 = torch.abs(out[:, 4:5])
        eps_1 = epsilon[:, 0:1]
        eps_2 = epsilon[:, 1:2]

        z1 = Mu_1 + L11 * eps_1
        z2 = Mu_2 + L21 * eps_1 + L22 * eps_2
        z = torch.cat([z1, z2], dim=1)  # [num_unique, z_dim]

        if visualize:
            return z
        # Project x (one-hot over original categories) onto per-category latents.
        return x @ z

    # ---- pairwise encoding with shared noise ------------------------------

    def encode_pair(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        shared_sampling: bool = True,
        **kwargs: Any,
    ):
        """
        Probabilistic version shares epsilon across the pair so kernel
        evaluations are not contaminated by independent sampling noise.
        """
        if x2 is None:
            x2 = x1

        local_kwargs = dict(kwargs)
        if self.is_probabilistic and shared_sampling and local_kwargs.get("epsilon") is None:
            n_unique = max(_num_unique_rows(x1), _num_unique_rows(x2))
            local_kwargs["epsilon"] = torch.normal(
                mean=0.0,
                std=1.0,
                size=(n_unique, self.z_dim),
                device=x1.device,
                dtype=x1.dtype,
            )

        return self(x1, **local_kwargs), self(x2, **local_kwargs)


def _num_unique_rows(x: torch.Tensor) -> int:
    """How many unique rows would torch.unique(x, dim=0) produce?"""
    if x.dim() <= 1:
        return x.shape[0] if x.dim() == 1 else 1
    return torch.unique(x, dim=0).shape[0]


# ---------------------------------------------------------------------------
# Linear_VAE — used by NeuralEncoder's probabilistic mode
# ---------------------------------------------------------------------------


class Linear_VAE(gpytorch.Module):
    """
    Linear layer with NormalPrior on weights (and bias, if present).

    Args:
        in_features: Input size.
        out_features: Output size.
        bias: Whether to include a learnable bias.
        name: Prefix used when registering priors with GPyTorch.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.name = str(name) if name is not None else ""

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_prior(
            f"{self.name}_prior_weight",
            gpytorch.priors.NormalPrior(0.0, 0.2),
            lambda m: m.weight,
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.register_prior(
                f"{self.name}_prior_bias",
                gpytorch.priors.NormalPrior(0.0, 0.05),
                lambda m: m.bias,
            )
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.weight.dtype)
        return nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"
