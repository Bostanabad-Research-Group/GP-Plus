import copy
from typing import List, Optional

import gpytorch
import torch

from ..config import logger
from .callbacks import Callback
from .optimizers import LBFGSScipy
from .parameter_initializer import DefaultParameterInitializer, ParameterInitializer
from .stop_conditions import StopCondition
from .trainer_utils import (
    RunResult,
    build_run_error,
    get_effective_optimizer_kwargs,
    run_parallel_initializations,
    select_best_run,
)
from .training_single_run import GPTrainerSingleProcess


class GPTrainer:
    def __init__(
        self,
        model,
        optimizer_class: torch.optim.Optimizer = None,
        optimizer_kwargs: dict = None,
        scheduler_class: torch.optim.lr_scheduler.LRScheduler = None,
        scheduler_kwargs: dict = None,
        num_epochs: int = 1000,
        seed: int = None,
        num_inits: int = 64,
        mll_class: gpytorch.mlls.MarginalLogLikelihood = None,
        cholesky_jitter: float = 1e-6,
        callbacks: Optional[List[Callback]] = None,
        initializer_class: ParameterInitializer = None,
        initializer_kwargs: dict = None,
        device: str = "cpu",
        stop_conditions: Optional[List[StopCondition]] = None,
        min_epochs: int = 0,
        n_jobs: Optional[int] = None,
        inner_max_num_threads: Optional[int] = 1,
        dtype: torch.dtype = torch.float64,
    ):
        #! TODO: Update so LBFGS and adam use different trainers to minimize 'if' lines
        """
        Initialize the multi-run GP trainer.

        Args:
            model: GP model instance with `train_inputs` and `train_targets`.
            optimizer_class: Optimizer class used for each run. Defaults to `LBFGSScipy`.
            optimizer_kwargs: Optimizer kwargs (without `params`).
            scheduler_class: Optional learning-rate scheduler class.
            scheduler_kwargs: Optional scheduler kwargs.
            num_epochs: Number of epochs per run.
            seed: Random seed for parameter initialization.
            num_inits: Number of initialization runs to evaluate.
            mll_class: Marginal log likelihood class. Defaults to exact MLL.
            cholesky_jitter: Cholesky jitter used during training.
            callbacks: Optional callback instances applied during training.
            initializer_class: Parameter initializer class for per-run starts.
            initializer_kwargs: Optional kwargs for `initializer_class`.
            device: Target device string (falls back to CPU if CUDA is unavailable).
            stop_conditions: Optional early-stop conditions. Defaults are applied when omitted.
            min_epochs: Minimum epochs before stop conditions can terminate a run.
            n_jobs: Optional parallel job cap used by run dispatch.
            inner_max_num_threads: Optional torch thread cap per run worker.
            dtype: Tensor dtype used for model and training data.
        """
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)
        logger.info("Using device: %s", self.device)

        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}.")
        self.dtype = dtype
        self._prepare_model_and_data(model)

        self.num_epochs = num_epochs
        self.num_inits = num_inits
        self.seed = seed
        self.callbacks = callbacks or []
        self.cholesky_jitter = cholesky_jitter
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.min_epochs = min_epochs
        self.n_jobs = n_jobs
        self.inner_max_num_threads = inner_max_num_threads

        if stop_conditions is None:
            from .stop_conditions import ConvergencePatienceStopCondition, MinLossChangeStopCondition

            self.stop_conditions = [
                ConvergencePatienceStopCondition(patience=20),
                MinLossChangeStopCondition(min_loss_change=1e-7),
            ]
        else:
            self.stop_conditions = stop_conditions

        if initializer_class is None:
            self.initializer = DefaultParameterInitializer(num_inits=self.num_inits, seed=self.seed)
        else:
            self.initializer = initializer_class(
                num_inits=self.num_inits,
                seed=self.seed,
                **(initializer_kwargs or {}),
            )
        self.initializer.setup(self.model)

        if optimizer_class is None:
            self.optimizer_class = LBFGSScipy
            logger.warning(
                "No optimizer class passed (input=%s). Defaulting to optimizer class=%s.",
                optimizer_class,
                self.optimizer_class.__name__,
            )
        else:
            self.optimizer_class = optimizer_class

        self.optimizer_kwargs = optimizer_kwargs or {}
        if optimizer_kwargs is None:
            logger.warning("No optimizer kwargs passed (input=%s). Using optimizer class defaults.", optimizer_kwargs)
        else:
            logger.info("Optimizer class: %s, kwargs: %s", self.optimizer_class.__name__, optimizer_kwargs)

        if mll_class is None:
            self.mll_class = gpytorch.mlls.ExactMarginalLogLikelihood
            logger.warning("No MLL class passed. Defaulting to ExactMarginalLogLikelihood.")
        else:
            self.mll_class = mll_class

        is_lbfgs_like = (
            self.optimizer_class is LBFGSScipy
            or (isinstance(self.optimizer_class, type) and issubclass(self.optimizer_class, LBFGSScipy))
            or self.optimizer_class is torch.optim.LBFGS
            or (isinstance(self.optimizer_class, type) and issubclass(self.optimizer_class, torch.optim.LBFGS))
        )
        if is_lbfgs_like and self.num_epochs != 1:
            logger.info("Overriding num_epochs=%s to 1 for LBFGS-style optimizer.", self.num_epochs)
            self.num_epochs = 1

        optimizer_name = getattr(self.optimizer_class, "__name__", str(self.optimizer_class))
        effective_optimizer_kwargs = get_effective_optimizer_kwargs(self.optimizer_class, self.optimizer_kwargs)
        logger.info(
            "Trainer optimizer configured: class=%s, effective_kwargs=%s",
            optimizer_name,
            effective_optimizer_kwargs,
        )

    def _prepare_model_and_data(self, model) -> None:
        if not hasattr(model, "train_inputs") or not hasattr(model, "train_targets"):
            raise AttributeError("model must expose train_inputs and train_targets before training.")

        train_x = model.train_inputs[0]
        train_y = model.train_targets
        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            raise TypeError("train_inputs and train_targets must be torch.Tensor instances.")
        if train_x.dtype != train_y.dtype:
            raise TypeError(f"Training data dtype mismatch: train_x is {train_x.dtype}, train_y is {train_y.dtype}.")

        if train_x.dtype != self.dtype:
            logger.info("Converting model training data from %s to %s on %s.", train_x.dtype, self.dtype, self.device)

        self.model = model.to(self.device, dtype=self.dtype)
        self.model.set_train_data(
            train_x.to(self.device, dtype=self.dtype),
            train_y.to(self.device, dtype=self.dtype),
            strict=False,
        )
        self.model.dtype = self.dtype
        self.train_x = self.model.train_inputs[0]
        self.train_y = self.model.train_targets

    def train_single_process(self, run_index: int, run_device: Optional[torch.device] = None) -> RunResult:
        target_device = run_device or self.device
        base_model = copy.deepcopy(self.model)
        self.initializer.initialize(base_model, run_index)
        base_model = base_model.to(target_device, dtype=self.dtype)

        if self.num_inits == 1:
            # Preserve callback state for single-run workflows (e.g., plotting callbacks in examples).
            callbacks_copy = self.callbacks
            stop_conditions_copy = self.stop_conditions
        else:
            callbacks_copy = [copy.deepcopy(cb) for cb in self.callbacks] if self.callbacks else []
            stop_conditions_copy = [copy.deepcopy(sc) for sc in self.stop_conditions] if self.stop_conditions else None

        run = GPTrainerSingleProcess(
            model=base_model,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            mll_class=self.mll_class,
            num_epochs=self.num_epochs,
            cholesky_jitter=self.cholesky_jitter,
            callbacks=callbacks_copy,
            device=target_device,
            scheduler_class=self.scheduler_class,
            scheduler_kwargs=self.scheduler_kwargs,
            stop_conditions=stop_conditions_copy,
            min_epochs=self.min_epochs,
            dtype=self.dtype,
        )
        train_result = run.train()
        return {"run_index": run_index, **train_result}

    def _train_single_process_safe(self, run_index: int, run_device: torch.device) -> RunResult:
        previous_num_threads = None
        try:
            if self.inner_max_num_threads is not None:
                previous_num_threads = torch.get_num_threads()
                torch.set_num_threads(max(1, self.inner_max_num_threads))
            return self.train_single_process(run_index, run_device=run_device)
        except Exception as exc:
            return build_run_error(run_index, exc)
        finally:
            if previous_num_threads is not None:
                torch.set_num_threads(previous_num_threads)

    def train_multiple_process_parallel(self) -> list[RunResult]:
        results = run_parallel_initializations(
            num_inits=self.num_inits,
            trainer_device=self.device,
            run_callable=self._train_single_process_safe,
            n_jobs=self.n_jobs,
        )
        logger.info("Training completed.")
        return results

    def train(self) -> list[RunResult]:
        results = self.train_multiple_process_parallel()
        failed_runs = [result for result in results if result.get("error")]
        if failed_runs:
            logger.warning(
                "%s/%s runs failed. Check run-level error payloads for details.",
                len(failed_runs),
                len(results),
            )

        best_run = select_best_run(results)
        if best_run is not None:
            best_loss = best_run["loss"]
            self.model.load_state_dict(best_run["state_dict"])
            logger.info("Best run found: #%s with loss=%.4f.", best_run["run_index"], best_loss)
        else:
            logger.warning("No valid best run found. Model was not updated.")
        return results
