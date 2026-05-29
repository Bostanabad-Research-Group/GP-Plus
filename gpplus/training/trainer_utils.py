import inspect
import os
from typing import Any, Callable, Optional, TypedDict

import torch
from joblib import Parallel, delayed

from ..config import logger
from .optimizers import LBFGSScipy


class SingleRunResult(TypedDict):
    loss: float
    state_dict: dict[str, Any]


class RunResult(TypedDict, total=False):
    run_index: int
    loss: Optional[float]
    state_dict: Optional[dict[str, Any]]
    error: str


def build_run_error(run_index: int, exc: Exception) -> RunResult:
    """Return a consistent error payload for failed runs."""
    logger.exception(f"Error in training run #{run_index}: {exc}")
    return {
        "run_index": run_index,
        "state_dict": None,
        "loss": None,
        "error": str(exc),
    }


def get_optimizer_default_kwargs(optimizer_class) -> dict:
    """Best-effort extraction of constructor defaults excluding params/self."""
    try:
        signature = inspect.signature(optimizer_class.__init__)
    except (TypeError, ValueError):
        return {}

    defaults = {}
    for name, parameter in signature.parameters.items():
        if name in {"self", "params"}:
            continue
        if parameter.default is not inspect.Signature.empty:
            defaults[name] = parameter.default
    return defaults


def get_effective_optimizer_kwargs(optimizer_class, user_kwargs: Optional[dict]) -> dict:
    """Merge optimizer defaults with user overrides for logging."""
    effective = get_optimizer_default_kwargs(optimizer_class)
    if user_kwargs:
        effective.update(user_kwargs)
    return effective


def select_best_run(results: list[RunResult]) -> Optional[RunResult]:
    """Return the best successful run by minimum loss."""
    valid_runs = [
        run_result
        for run_result in results
        if run_result.get("loss") is not None and run_result.get("state_dict") is not None
    ]
    if not valid_runs:
        return None
    return min(valid_runs, key=lambda run_result: run_result["loss"])


def _cpu_parallel_jobs(num_inits: int, n_jobs: Optional[int] = None) -> int:
    if n_jobs is not None:
        return min(num_inits, max(1, n_jobs))
    cpu_count = os.cpu_count() or 1
    # TODO: expose reserved cores as config for cluster-specific tuning.
    reserved_cores = 2
    return min(num_inits, max(1, cpu_count - reserved_cores))


def run_parallel_initializations(
    num_inits: int,
    trainer_device: torch.device,
    run_callable: Callable[[int, torch.device], RunResult],
    n_jobs: Optional[int] = None,
) -> list[RunResult]:
    """Execute initialization runs across CPU cores or available GPUs."""
    if trainer_device.type == "cpu":
        max_jobs = _cpu_parallel_jobs(num_inits, n_jobs=n_jobs)
        logger.info(f"Running {num_inits} runs using {max_jobs} parallel jobs on {os.cpu_count()} available CPU cores.")
        return Parallel(n_jobs=max_jobs, backend="loky")(
            delayed(run_callable)(run_index, trainer_device) for run_index in range(num_inits)
        )

    if trainer_device.type == "cuda":
        torch.cuda.empty_cache()
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            logger.warning("CUDA device selected but no GPUs were detected. Falling back to CPU.")
            cpu_device = torch.device("cpu")
            max_jobs = _cpu_parallel_jobs(num_inits, n_jobs=n_jobs)
            return Parallel(n_jobs=max_jobs, backend="loky")(
                delayed(run_callable)(run_index, cpu_device) for run_index in range(num_inits)
            )

        max_jobs = min(num_inits, num_gpus)
        if n_jobs is not None:
            max_jobs = min(max_jobs, max(1, n_jobs))
        logger.info(f"Running {num_inits} runs distributed across {num_gpus} GPUs.")
        return Parallel(n_jobs=max_jobs, backend="threading")(
            delayed(run_callable)(run_index, torch.device(f"cuda:{run_index % num_gpus}"))
            for run_index in range(num_inits)
        )

    raise ValueError(f"Unsupported training device: {trainer_device}")


def select_epoch_train_fn(
    optimizer,
    standard_epoch_fn: Callable,
    lbfgs_epoch_fn: Callable,
    scipy_lbfgs_epoch_fn: Callable,
) -> Callable:
    """Choose the per-epoch training function based on optimizer class."""
    if isinstance(optimizer, torch.optim.LBFGS):
        return lbfgs_epoch_fn
    if isinstance(optimizer, LBFGSScipy):
        return scipy_lbfgs_epoch_fn
    return standard_epoch_fn


def check_early_stop(
    stop_conditions: list,
    stop_context: dict,
    epoch: int,
    best_loss: float,
    min_epochs: int = 0,
) -> bool:
    """Evaluate stop conditions and log the first stop signal batch."""
    if epoch + 1 < min_epochs:
        return False
    reasons = []
    for stop_condition in stop_conditions:
        stop_now, reason = stop_condition.should_stop(stop_context)
        if stop_now and reason:
            reasons.append(reason)
        if stop_now and not reason:
            reasons.append("Stop condition met")
    if reasons:
        logger.info(
            "Early stopping triggered at epoch %s. Reason: %s. Best loss: %.6f",
            epoch + 1,
            " OR ".join(reasons),
            best_loss,
        )
        return True
    return False
