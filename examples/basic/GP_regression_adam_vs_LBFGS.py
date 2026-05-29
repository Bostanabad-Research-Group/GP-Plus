# Import needed libraries
import copy
import logging
import time

import gpytorch
import matplotlib.pyplot as plt
import torch

import gpplus
from gpplus.config import configure_logger
from gpplus.training import (
    ConvergencePatienceStopCondition,
    GPTrainer,
    MinLossChangeStopCondition,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils.set_seed import set_seed

configure_logger(logging.INFO)
set_seed(42)

train_x = torch.linspace(0, 1, 10)
train_y = torch.sin(train_x * (2 * torch.pi)) + 0.1 * torch.randn(train_x.size())

# Plot the training data
plt.figure(figsize=(6, 4))
plt.scatter(train_x.numpy(), train_y.numpy(), color="red", label="Train Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training Data")
plt.legend()
plt.show()

# DEfine the GP model and likelihood

base_likelihood = gpytorch.likelihoods.GaussianLikelihood()
base_model = gpplus.models.GPR(train_x, train_y, base_likelihood)


class PrintLossCallback(gpplus.training.callbacks.Callback):
    def __init__(self):
        self._is_lbfgs = False

    def on_train_start(self, context: dict):
        optimizer_class = context["trainer"].optimizer_class
        self._is_lbfgs = (
            optimizer_class is LBFGSScipy
            or (isinstance(optimizer_class, type) and issubclass(optimizer_class, LBFGSScipy))
            or optimizer_class is torch.optim.LBFGS
            or (isinstance(optimizer_class, type) and issubclass(optimizer_class, torch.optim.LBFGS))
        )

    def register_with_optimizer(self, optimizer, model=None, trainer=None):
        if not self._is_lbfgs:
            return

        def on_lbfgs_iteration(iter_idx, loss):
            print(f"Iteration {iter_idx} - Loss: {loss:.4f}")

        if hasattr(optimizer, "iteration_callback"):
            previous_callback = optimizer.iteration_callback

            def chained_iteration_callback(iter_idx, loss):
                if previous_callback is not None:
                    previous_callback(iter_idx, loss)
                on_lbfgs_iteration(iter_idx, loss)

            optimizer.iteration_callback = chained_iteration_callback

    def on_epoch_end(self, context: dict):
        if self._is_lbfgs:
            return
        print(f"Epoch {context['epoch']} - Loss: {context['loss']:.4f}")


class LossCallback(gpplus.training.callbacks.Callback):
    def __init__(self):
        self.loss = []
        self.steps = []
        self.x_label = "Epoch"
        self._is_lbfgs = False

    def on_train_start(self, context: dict):
        optimizer_class = context["trainer"].optimizer_class
        self._is_lbfgs = (
            optimizer_class is LBFGSScipy
            or (isinstance(optimizer_class, type) and issubclass(optimizer_class, LBFGSScipy))
            or optimizer_class is torch.optim.LBFGS
            or (isinstance(optimizer_class, type) and issubclass(optimizer_class, torch.optim.LBFGS))
        )
        self.x_label = "LBFGS Iteration" if self._is_lbfgs else "Epoch"

    def register_with_optimizer(self, optimizer, model=None, trainer=None):
        if not self._is_lbfgs:
            return

        def on_lbfgs_iteration(iter_idx, loss):
            self.steps.append(iter_idx)
            self.loss.append(loss)

        if hasattr(optimizer, "iteration_callback"):
            previous_callback = optimizer.iteration_callback

            def chained_iteration_callback(iter_idx, loss):
                if previous_callback is not None:
                    previous_callback(iter_idx, loss)
                on_lbfgs_iteration(iter_idx, loss)

            optimizer.iteration_callback = chained_iteration_callback

    def on_epoch_end(self, context: dict):
        if self._is_lbfgs:
            return
        self.steps.append(context["epoch"])
        self.loss.append(context["loss"])


printCallback = PrintLossCallback()


def train_and_plot(optimizer_class, optimizer_kwargs=None, title_suffix=None):
    if title_suffix is None:
        title_suffix = optimizer_class.__name__

    model = copy.deepcopy(base_model)
    model.train()
    model.likelihood.train()

    lossCallback = LossCallback()
    cllbcks = [printCallback, lossCallback]

    trainer = GPTrainer(
        model=model,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        num_inits=1,
        callbacks=cllbcks,
        device="cuda",
        stop_conditions=[
            ConvergencePatienceStopCondition(patience=150),
            MinLossChangeStopCondition(min_loss_change=1e-7),
        ],
    )
    start_time = time.perf_counter()
    trainer.train()
    elapsed_time = time.perf_counter() - start_time
    print(f"{title_suffix} training finished. Time = {elapsed_time:.2f} (s).")
    return {
        "title": title_suffix,
        "steps": lossCallback.steps,
        "loss": lossCallback.loss,
        "x_label": lossCallback.x_label,
        "elapsed_time": elapsed_time,
    }


# Train with LBFGSScipy (iteration-based x-axis)
lbfgs_result = train_and_plot(
    optimizer_class=LBFGSScipy,
    optimizer_kwargs={"max_iter": 2000, "max_eval": 5000, "tolerance_grad": 1e-5, "tolerance_change": 1e-9},
    title_suffix="LBFGSScipy",
)

# Train with Adam (epoch-based x-axis)
adam_result = train_and_plot(
    optimizer_class=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.1},
    title_suffix="Adam",
)

# Plot both runs side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, result in zip(axes, [lbfgs_result, adam_result]):
    ax.plot(result["steps"], result["loss"], label="Loss")
    ax.plot([], [], " ", label=f"Time = {result['elapsed_time']:.2f} (s).")
    ax.set_xlabel(result["x_label"])
    ax.set_ylabel("Loss")
    ax.set_title(f"{result['title']}: Loss over {result['x_label']}s")
    ax.legend()

plt.tight_layout()
plt.show()
