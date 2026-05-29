from typing import Any, Dict, Optional

import torch

from ..config import logger


def _make_generator_for_device(device: torch.device, seed: int) -> torch.Generator:
    """Create a seeded RNG generator on the same device as initialized tensors."""
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    try:
        return torch.Generator(device=generator_device).manual_seed(seed)
    except TypeError:
        # Backward compatibility for torch builds without device argument support.
        return torch.Generator().manual_seed(seed)


def get_parameter_type(name: str, param: torch.Tensor) -> str:
    """Determine parameter type from name/shape."""
    if "projection_matrix" in name:
        return "projection_matrix"
    if "raw_lengthscale" in name:
        return "raw_lengthscale"
    if "raw_outputscale" in name:
        return "raw_outputscale"
    if "raw_noise" in name:
        return "raw_noise"
    if "weight" in name and param.dim() >= 2:
        return "weight"
    if "bias" in name and param.dim() == 1:
        return "bias"
    if "power" in name:
        return "power"
    if "constant" in name:
        return "constant"
    return "unknown"


def _projection_matrix_config(name: str, model: Optional[torch.nn.Module]) -> Dict[str, Any]:
    init_type = "orthogonal"
    init_std = 0.1
    if model is not None:
        for module_name, module in model.named_modules():
            param_init_types = getattr(module, "_param_init_types", None)
            if isinstance(param_init_types, dict) and "projection_matrix" in param_init_types:
                if name.startswith(module_name + "."):
                    init_type = param_init_types["projection_matrix"]
                    param_init_params = getattr(module, "_param_init_params", None)
                    if isinstance(param_init_params, dict) and "projection_matrix" in param_init_params:
                        init_std = param_init_params["projection_matrix"]["init_std"]
                    break

    if init_type == "orthogonal":
        return {
            "method": "orthogonal_matrix",
            "gain": init_std,
            "description": "Matrix encoder projection matrix (orthogonal)",
        }
    if init_type == "normal":
        return {
            "method": "normal_matrix",
            "mean": 0.0,
            "std": init_std,
            "description": "Matrix encoder projection matrix (normal)",
        }
    if init_type == "uniform":
        return {
            "method": "uniform_matrix",
            "lower": -init_std,
            "upper": init_std,
            "description": "Matrix encoder projection matrix (uniform)",
        }
    return {
        "method": "orthogonal_matrix",
        "gain": init_std,
        "description": "Matrix encoder projection matrix (default orthogonal)",
    }


def get_initialization_config(
    name: str,
    param: torch.Tensor,
    parameter_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    model: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    """Get initialization configuration by parameter type."""
    param_type = get_parameter_type(name, param)
    parameter_configs = parameter_configs or {}
    if param_type in parameter_configs:
        config = parameter_configs[param_type].copy()
        config["description"] = f"{param_type} parameter (custom config)"
        return config

    if param_type == "raw_lengthscale":
        is_ard = param.dim() == 2 and param.shape[1] > 1
        return {
            "method": "normal",
            "mean": -2.0,
            "std": 2.0,
            "description": f"Lengthscale parameter {'(ARD)' if is_ard else '(single)'} - log scale",
        }
    if param_type == "raw_outputscale":
        return {
            "method": "normal",
            "mean": -2.0,
            "std": 0.5,
            "description": "Outputscale parameter - log scale",
        }
    if param_type == "raw_noise":
        return {
            "method": "uniform",
            "lower": -7.0,
            "upper": -1.0,
            "description": "Noise parameter - uniform scale",
        }
    if param_type == "constant":
        return {
            "method": "normal",
            "mean": 0.0,
            "std": 1.0,
            "description": "Mean constant parameter",
        }
    if param_type == "weight":
        fan_in, fan_out = param.size(1), param.size(0)
        limit = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
        return {
            "method": "xavier_uniform",
            "limit": limit.item(),
            "description": f"Neural network weight ({fan_in}->{fan_out})",
        }
    if param_type == "bias":
        return {
            "method": "zeros",
            "description": "Neural network bias",
        }
    if param_type == "power":
        return {
            "method": "uniform",
            "lower": 1.0,
            "upper": 2.0,
            "description": "Power kernel parameter (uniform)",
        }
    if param_type == "projection_matrix":
        return _projection_matrix_config(name, model)
    return {
        "method": "orthogonal_matrix",
        "gain": 0.1,
        "description": "Unknown parameter",
    }


def _generate_normal_samples(sample: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Generate normal samples using inverse CDF from Sobol samples."""
    z = torch.erfinv(2.0 * sample - 1.0) * torch.sqrt(torch.tensor(2.0, dtype=sample.dtype, device=sample.device))
    return mean + std * z


def initialize_parameter(
    param: torch.Tensor,
    sample: torch.Tensor,
    config: Dict[str, Any],
    seed: int = None,
    name: str = "",
    run_index: int = 0,
) -> None:
    """Initialize a single parameter according to its config."""
    # pylint: disable=broad-exception-caught
    method = config["method"]

    if method == "orthogonal_matrix":
        temp_param = torch.empty_like(param, dtype=param.dtype, device=param.device)
        try:
            generator_seed = (seed + run_index * 1000) if seed is not None else (run_index * 1000)
            torch.nn.init.orthogonal_(
                temp_param,
                gain=config.get("gain", 1.0),
                generator=_make_generator_for_device(temp_param.device, generator_seed),
            )
            if torch.isnan(temp_param).any():
                logger.error("NaN detected in orthogonal initialization for %s", name)
                logger.error("temp_param: %s", temp_param)
                logger.error("param shape: %s, dtype: %s", param.shape, param.dtype)
            param.data = temp_param
            logger.debug("Orthogonal initialization successful for %s with seed %s", name, generator_seed)
        except Exception as e:
            logger.error("Orthogonal initialization failed for %s: %s", name, e)
            torch.nn.init.normal_(temp_param, mean=0.0, std=0.1)
            param.data = temp_param
        return

    if method == "orthogonal":
        torch.nn.init.orthogonal_(param, gain=config.get("gain", 1.0))
        return

    if method == "normal_matrix":
        generator_seed = (seed + run_index * 1000) if seed is not None else (run_index * 1000)
        torch.nn.init.normal_(
            param,
            mean=config.get("mean", 0.0),
            std=config.get("std", 0.1),
            generator=_make_generator_for_device(param.device, generator_seed),
        )
        return

    if method == "uniform_matrix":
        generator_seed = (seed + run_index * 1000) if seed is not None else (run_index * 1000)
        torch.nn.init.uniform_(
            param,
            a=config.get("lower", -0.1),
            b=config.get("upper", 0.1),
            generator=_make_generator_for_device(param.device, generator_seed),
        )
        return

    if method == "xavier_uniform":
        generator_seed = (seed + run_index) if seed is not None else run_index
        g = _make_generator_for_device(param.device, generator_seed)
        torch.nn.init.xavier_uniform_(param, generator=g)
        logger.debug("Initialized weight parameter '%s' with Xavier uniform (seed=%s)", name, generator_seed)
        return

    if method == "uniform":
        lower = config.get("lower", -6.0)
        upper = config.get("upper", 3.0)
        raw_value = lower + (upper - lower) * sample
        param.data = raw_value.to(dtype=param.dtype)
        return

    if method == "normal":
        mean = config.get("mean", -2.0)
        std = config.get("std", 1.5)
        raw_value = _generate_normal_samples(sample, mean, std)
        param.data = raw_value.to(dtype=param.dtype)
        logger.debug("Direct initialization: %s = %s (constraints built into kernel classes)", name, raw_value)
        return

    if method == "zeros":
        torch.nn.init.zeros_(param)
        return

    if method == "constant":
        value = config.get("value", 0.0)
        param.data = torch.full_like(param, value, dtype=param.dtype)
        return

    if method == "skip":
        return

    raw_value = 0.1 * (sample * 2 - 1)
    param.data = raw_value.to(dtype=param.dtype)
