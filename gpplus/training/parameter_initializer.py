from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch.quasirandom import SobolEngine

from ..config import logger
from .parameter_init_utils import get_initialization_config, initialize_parameter


def _make_generator_for_device(device: torch.device, seed: int) -> torch.Generator:
    """Create a seeded RNG generator on the same device as initialized tensors."""
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    try:
        return torch.Generator(device=generator_device).manual_seed(seed)
    except TypeError:
        # Backward compatibility for torch builds without device argument support.
        return torch.Generator().manual_seed(seed)


class ParameterInitializer(ABC):
    """Abstract base class for parameter initializers."""

    @abstractmethod
    def __init__(self, num_inits: int, seed: int = None):
        pass

    @abstractmethod
    def setup(self, model: torch.nn.Module):
        raise NotImplementedError

    @abstractmethod
    def initialize(self, model: torch.nn.Module, run_index: int):
        raise NotImplementedError


class DefaultParameterInitializer(ParameterInitializer):
    """
    Parameter initializer that looks at parameter names to determine initialization strategies.

    Features:
    - Parameter type detection based on parameter names only
    - Conservative initialization values for numerical stability
    - Allows the user to specify custom initialization strategies for specific parameters.
    """

    def __init__(
        self,
        num_inits: int,
        seed: int = None,
        parameter_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize the default parameter initializer.

        Args:
            num_inits: Total number of initialization runs.
            seed: Random seed for reproducibility.
            parameter_configs: Optional custom parameter configurations.

        Note:
            This initializer looks at parameter names to determine initialization strategies.
            All constraints are handled by the model's kernel and likelihood classes.
        """
        self.num_inits = num_inits
        self.seed = seed
        self.num_params = None
        self.sobol_samples = None
        self.parameter_configs = parameter_configs or {}

    def setup(self, model: torch.nn.Module):
        """
        Calculate the total number of learnable parameters and precompute Sobol samples.

        Excludes '.weight' and '.bias' parameters from Sobol sampling count,
        as these are initialized separately (Xavier uniform for weights, zeros for biases).
        """
        # Count only parameters that will use Sobol samples (exclude .weight and .bias)
        self.num_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad and ".weight" not in name and ".bias" not in name:
                self.num_params += param.numel()

        if self.num_params > 0:
            sobol_engine = SobolEngine(dimension=self.num_params, scramble=True, seed=self.seed)
            self.sobol_samples = sobol_engine.draw(self.num_inits)
            logger.debug(f"Sobol samples generated: {self.sobol_samples.shape}")
        else:
            self.sobol_samples = None
            logger.info("No non-weight/bias parameters found; Sobol sampling skipped.")

        logger.info("Using DefaultParameterInitializer")
        logger.info("All constraints are now built into kernel and likelihood classes - no manual setup needed")
        logger.debug("Excluding .weight and .bias parameters from Sobol sampling (initialized separately)")

    def initialize(self, model: torch.nn.Module, run_index: int):
        """
        Initialize the model parameters for a specific run.

        We exclude '.weight' and '.bias' parameters from Sobol sampling:
        - '.weight': Xavier uniform
        - '.bias': zeros
        Other parameters are mapped from Sobol [0,1] samples into configured ranges.

        :param model: The model whose parameters need initialization.
        :param run_index: The run index corresponding to the precomputed Sobol sample.
        """
        idx = 0

        with torch.no_grad():
            all_params = list(model.named_parameters())
            for i, (name, param) in enumerate(all_params):
                param_length = param.numel()

                # Skip parameters with zero elements or non-learnable parameters
                if param_length == 0 or not param.requires_grad:
                    logger.debug(f"Skipping parameter: {name}")
                    continue

                # Get initialization configuration
                config = get_initialization_config(
                    name=name,
                    param=param,
                    parameter_configs=self.parameter_configs,
                    model=model,
                )

                # Handle weight parameters with Xavier uniform (exclude from Sobol sampling)
                # Only apply Xavier to parameters with at least 2 dimensions (required for fan_in/fan_out calculation)
                if ".weight" in name:
                    # reproducible per-run, on CPU
                    generator_seed = (self.seed + run_index) if self.seed is not None else run_index
                    g = _make_generator_for_device(param.device, generator_seed)
                    if param.dim() >= 2:
                        # Standard neural network weight matrix: use Xavier uniform
                        torch.nn.init.xavier_uniform_(param, generator=g)
                        logger.debug(f"Initialized weight parameter '{name}' with Xavier uniform (shape={param.shape})")
                    else:
                        # 1D or scalar weight: use uniform initialization instead
                        torch.nn.init.uniform_(param, -0.1, 0.1)
                        logger.debug(f"Initialized 1D weight parameter '{name}' with uniform (shape={param.shape})")
                    continue

                # Handle bias parameters with zeros (exclude from Sobol sampling)
                if ".bias" in name:
                    torch.nn.init.zeros_(param)
                    logger.debug(f"Initialized bias parameter '{name}' with zeros")
                    continue

                # Slice the sobol_samples for the current parameter
                sample = self.sobol_samples[run_index, idx : idx + param_length]
                sample = sample.reshape(param.shape)
                sample = sample.to(device=param.device, dtype=param.dtype)

                # Initialize the parameter
                old_value = param.data.clone()
                initialize_parameter(
                    param=param,
                    sample=sample,
                    config=config,
                    seed=self.seed,
                    name=name,
                    run_index=run_index,
                )
                new_value = param.data.clone()

                logger.debug(
                    f"Initialized {name}: {config['description']} (shape={param.shape}, method={config['method']})"
                )
                # logger.debug(f"  Old value: {old_value}")
                logger.debug(f"  New value: {new_value}")

                # Special debug for key parameters
                if "cont_kernel.lengthscale" in name or "raw_noise" in name:
                    logger.info(f"KEY PARAMETER INITIALIZED: {name}")
                    logger.info(f"  Method: {config['method']}")
                    logger.info(f"  Old: {old_value}")
                    logger.info(f"  New: {new_value}")
                    logger.info(f"  Config: {config}")

                idx += param_length

        logger.info(f"Model parameters initialized with run #{run_index}")
