# Import needed libraries
import logging
import time

import gpytorch
import matplotlib.pyplot as plt
import torch

import gpplus
from gpplus.config import configure_logger
from gpplus.training.optimizers import LBFGSScipy
from gpplus.training.stop_conditions import ConvergencePatienceStopCondition, MinLossChangeStopCondition
from gpplus.training.trainer import GPTrainer

configure_logger(logging.WARNING)

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

# Define the GP model and likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = gpplus.models.GPR(train_x, train_y, likelihood)
model.train()
likelihood.train()


class PrintLossCallback(gpplus.training.callbacks.Callback):
    def register_with_optimizer(self, optimizer, model=None, trainer=None):
        def on_lbfgs_iteration(iter_idx, loss):
            print(f"Iteration {iter_idx} - Loss: {loss:.4f}")

        if hasattr(optimizer, "iteration_callback"):
            previous_callback = optimizer.iteration_callback

            def chained_iteration_callback(iter_idx, loss):
                if previous_callback is not None:
                    previous_callback(iter_idx, loss)
                on_lbfgs_iteration(iter_idx, loss)

            optimizer.iteration_callback = chained_iteration_callback


class LossCallback(gpplus.training.callbacks.Callback):
    def __init__(self):
        self.steps = []
        self.loss = []

    def register_with_optimizer(self, optimizer, model=None, trainer=None):
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


printCallback = PrintLossCallback()
lossCallback = LossCallback()
cllbcks = [printCallback, lossCallback]

trainer = GPTrainer(
    model=model,
    num_inits=1,
    callbacks=cllbcks,
    device="cuda",
    optimizer_class=LBFGSScipy,
    optimizer_kwargs={"max_iter": 2000, "max_eval": 5000, "tolerance_grad": 1e-5, "tolerance_change": 1e-9},
    stop_conditions=[
        ConvergencePatienceStopCondition(patience=150),
        MinLossChangeStopCondition(min_loss_change=1e-7),
    ],
)
start_time = time.perf_counter()
trainer.train()
elapsed_time = time.perf_counter() - start_time
print(f"LBFGSScipy training finished. Time = {elapsed_time:.2f} (s).")

# Plot training loss vs LBFGS internal iteration
plt.figure(figsize=(6, 4))
plt.plot(lossCallback.steps, lossCallback.loss, label="Loss")
plt.plot([], [], " ", label=f"Time = {elapsed_time:.2f} (s).")
plt.xlabel("LBFGS Iteration")
plt.ylabel("Loss")
plt.title("LBFGSScipy: Training Loss over Iterations")
plt.legend()
plt.show()
