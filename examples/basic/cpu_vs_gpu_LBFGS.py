import time

import torch
from gpytorch.likelihoods import GaussianLikelihood

from gpplus.models import GPR
from gpplus.training import GPTrainer, optimizers

# -------------------------------
# Create toy training data
# -------------------------------
# Simple 1D regression: y = sin(2*pi*x) with noise
train_x = torch.linspace(0, 1, 400).unsqueeze(-1)
train_y = torch.sin(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.2

# Ensure train_y is 1D (shape: [n]) rather than 2D ([n,1])
train_y = train_y.squeeze(-1)

# -------------------------------
# CPU Training
# -------------------------------
print("Training on CPU...")

# Create likelihood and model on CPU
likelihood_cpu = GaussianLikelihood()
model_cpu = GPR(train_x, train_y, likelihood_cpu)

# Instantiate GPTrainer for CPU
trainer_cpu = GPTrainer(
    model=model_cpu,
    optimizer_class=optimizers.LBFGSScipy,
    num_epochs=50,
    num_inits=16,
    seed=123,
    device="cpu",  # Use CPU
)

start_time = time.time()
results_cpu = trainer_cpu.train()
cpu_time = time.time() - start_time

# Collect valid losses
losses_cpu = [result["loss"] for result in results_cpu if result["loss"] is not None]
if losses_cpu:
    best_loss_cpu = min(losses_cpu)
    print(f"CPU training - Best Loss: {best_loss_cpu:.4f}, Time: {cpu_time:.2f} seconds")
else:
    print("No successful CPU runs; check logs for errors.")

# Check that model parameters are on CPU
for name, param in model_cpu.named_parameters():
    print(f"[CPU] {name} device: {param.device}")

# -------------------------------
# GPU Training (if available)
# -------------------------------
if torch.cuda.is_available():
    print("\nTraining on GPU...")

    # Move training data to GPU
    train_x_gpu = train_x.to("cuda")
    train_y_gpu = train_y.to("cuda")

    # Create likelihood and model on GPU
    likelihood_gpu = GaussianLikelihood().to("cuda")
    model_gpu = GPR(train_x_gpu, train_y_gpu, likelihood_gpu)

    # Instantiate GPTrainer for GPU
    trainer_gpu = GPTrainer(
        model=model_gpu,
        optimizer_class=optimizers.LBFGSScipy,
        num_epochs=50,
        num_inits=16,
        seed=123,
        device="cuda",  # Use GPU
    )

    start_time = time.time()
    results_gpu = trainer_gpu.train()
    gpu_time = time.time() - start_time

    losses_gpu = [result["loss"] for result in results_gpu if result["loss"] is not None]
    if losses_gpu:
        best_loss_gpu = min(losses_gpu)
        print(f"GPU training - Best Loss: {best_loss_gpu:.4f}, Time: {gpu_time:.2f} seconds")
    else:
        print("No successful GPU runs; check logs for errors.")

    # Check that model parameters are on GPU
    for name, param in model_gpu.named_parameters():
        print(f"[GPU] {name} device: {param.device}")
else:
    print("\nGPU not available; skipping GPU training test.")

# -------------------------------
# Additional Checks & Comparisons
# -------------------------------
# - Verify that the loss decreases relative to initial values.
# - Check that model parameters are on the intended device.
# - Compare training times and loss values between CPU and GPU.
# - Ensure that no runtime errors occur during parallel processing.
