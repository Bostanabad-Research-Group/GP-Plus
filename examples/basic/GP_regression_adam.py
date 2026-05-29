# Import needed libraries
import logging
import time

import gpytorch
import matplotlib.pyplot as plt
import torch

import gpplus
from gpplus.config import configure_logger
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

# DEfine the GP model and likelihood

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = gpplus.models.GPR(train_x, train_y, likelihood)

# Train

training_iter = 50
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


class PrintLossCallback(gpplus.training.callbacks.Callback):
    def on_epoch_end(self, context: dict):
        print(f"Epoch {context['epoch']} - Loss: {context['loss']:.4f}")


class LossCallback(gpplus.training.callbacks.Callback):
    def __init__(self):
        self.loss = []

    def on_epoch_end(self, context: dict):
        self.loss.append(context["loss"])


printCallback = PrintLossCallback()
lossCallback = LossCallback()
cllbcks = [printCallback, lossCallback]

trainer = GPTrainer(
    model=model,
    num_inits=1,
    callbacks=cllbcks,
    device="cuda",
    optimizer_class=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.1},
    stop_conditions=[ConvergencePatienceStopCondition(patience=150), MinLossChangeStopCondition(min_loss_change=1e-7)],
)
start_time = time.perf_counter()
trainer.train()
elapsed_time = time.perf_counter() - start_time
print(f"Adam training finished. Time = {elapsed_time:.2f} (s).")

# Plot training loss
plt.figure(figsize=(6, 4))
plt.plot(range(len(lossCallback.loss)), lossCallback.loss, label="Loss")
plt.plot([], [], " ", label=f"Time = {elapsed_time:.2f} (s).")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations")
plt.legend()
plt.show()
