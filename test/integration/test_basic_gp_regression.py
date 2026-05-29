import logging

import gpytorch
import torch

from gpplus.config import configure_logger
from gpplus.models import GPR
from gpplus.training import evaluate_gp_model
from gpplus.training.trainer import GPTrainer

configure_logger(logging.WARNING)


def test_gpr_training_single_job_cpu():
    # Define toy dataset
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.sin(train_x * (2 * torch.pi)) + 0.1 * torch.randn(train_x.size())

    # Define the model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPR(train_x, train_y, likelihood)

    # Training loop
    device = "cpu"
    trainer = GPTrainer(
        model=model, optimizer_class=torch.optim.Adam, optimizer_kwargs={"lr": 0.1}, num_inits=1, device=device
    )
    print("training")
    res = trainer.train()
    print("hey")
    print(res)

    assert model.train_inputs[0].to(device).equal(train_x.unsqueeze(1).to(device)), (
        "Model's stored train_x does not match input!"
    )
    assert model.train_targets.to(device).equal(train_y.to(device)), "Model's stored train_y does not match input!"

    # Evaluate
    test_x = torch.linspace(0, 1, 51)
    mean, _, _, _ = evaluate_gp_model(model, test_x.to(device))

    # Assertions to verify inference
    assert mean.shape == test_x.shape, "Prediction mean has incorrect shape!"
    assert torch.isfinite(mean).all(), "Predictions contain NaN or Inf!"


def test_gpr_training_parallel_job_cpu():
    # Define toy dataset
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.sin(train_x * (2 * torch.pi)) + 0.1 * torch.randn(train_x.size())

    # Define the model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPR(train_x, train_y, likelihood)

    # Training loop
    device = "cpu"
    trainer = GPTrainer(model=model, optimizer_class=torch.optim.Adam, optimizer_kwargs={"lr": 0.1}, device=device)
    trainer.train()

    assert model.train_inputs[0].to(device).equal(train_x.unsqueeze(1).to(device)), (
        "Model's stored train_x does not match input!"
    )
    assert model.train_targets.to(device).equal(train_y.to(device)), "Model's stored train_y does not match input!"

    # Evaluate
    test_x = torch.linspace(0, 1, 51)
    mean, _, _, _ = evaluate_gp_model(model, test_x.to(device))

    # Assertions to verify inference
    assert mean.shape == test_x.shape, "Prediction mean has incorrect shape!"
    assert torch.isfinite(mean).all(), "Predictions contain NaN or Inf!"


if __name__ == "__main__":
    test_gpr_training_single_job_cpu()
