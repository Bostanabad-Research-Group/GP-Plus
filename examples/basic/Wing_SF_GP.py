import time

import gpytorch
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import gpplus
from examples.data.data_gen import scale, wing_mixed_variables
from gpplus.models import GPR
from gpplus.training.eval import evaluate_gp_model
from gpplus.utils import set_seed


def compute_metrics(y_true, y_hat, output_std=None, start_time=None):
    """
    Compute basic metrics for predictions.

    Args:
        y_true: True values (1D array)
        y_hat: Predicted values (1D array)
        output_std: Standard deviation of predictions (optional)
        start_time: Start time for timing (optional)

    Returns:
        dict: Dictionary with computed metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().reshape(-1)
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)
    if output_std is not None and isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy().reshape(-1)

    # Only add time if start_time is provided
    if start_time is not None:
        metrics = {
            "Time": time.time() - start_time,
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
        }
    else:
        metrics = {
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
        }

    # Add NIS if output_std is provided
    if output_std is not None:
        z = 1.96
        L = y_hat - z * output_std
        U = y_hat + z * output_std
        width = U - L
        below = (L - y_true) * (y_true < L)
        above = (y_true - U) * (y_true > U)
        interval_score = width + (2 / 0.05) * below + (2 / 0.05) * above
        NIS = interval_score.mean() / y_true.std()
        metrics["NIS"] = NIS
        return metrics

    return metrics


# Generate Sobol sequence for X inputs
seed = 42
set_seed(seed)

# Generate training and test data for all fidelity levels s0-s3
sources = "s0"
num_train = 400
num_test = 10000  # 10k total / 4 sources = 2.5k per source

l_bound = torch.tensor([150.0, 220.0, 6.0, -10.0, 16.0, 0.5, 0.08, 2.5, 1700.0, 0.025])
u_bound = torch.tensor([200.0, 300.0, 10.0, 10.0, 45.0, 1.0, 0.18, 6.0, 2500.0, 0.08])


# Generate test data
print("\nGenerating test data...")

sobol = torch.quasirandom.SobolEngine(dimension=10, scramble=True)
X_sobol_raw = sobol.draw(num_test).to(dtype=torch.float64)

# Scale Sobol samples to the proper bounds
X_sobol = scale(X_sobol_raw, l_bound, u_bound)

print(f"Test source: {sources}")
result = wing_mixed_variables(
    X=X_sobol,
    source=sources,
)


X_test = X_sobol
y_test = result
print(f"  Test samples: {X_test.shape[0]}, Result range: [{result.min():.4f}, {result.max():.4f}]")


sobol = torch.quasirandom.SobolEngine(dimension=10, scramble=True)
X_sobol_raw = sobol.draw(num_train).to(dtype=torch.float64)

# Scale Sobol samples to the proper bounds
X_sobol = scale(X_sobol_raw, l_bound, u_bound)

print(f"Training source: {sources}")
result = wing_mixed_variables(
    X=X_sobol,
    source=sources,
)


X_train = X_sobol
y_train = result
print(f"  Training samples: {X_train.shape[0]}, Result range: [{result.min():.4f}, {result.max():.4f}]")

cont_cols = np.arange(0, 10)

# Shuffle both training and test datasets
print("Shuffling datasets...")
train_indices = torch.randperm(X_train.shape[0])
test_indices = torch.randperm(X_test.shape[0])

X_train = X_train[train_indices]
y_train = y_train[train_indices]
X_test = X_test[test_indices]
y_test = y_test[test_indices]

print(f"\nFinal training dataset shape: X={X_train.shape}, y={y_train.shape}")
print(f"Final test dataset shape: X={X_test.shape}, y={y_test.shape}")

# Print ranges for each of the 10 original features (excluding source one-hot)
print("\nX feature ranges (10 original features):")
for i in range(10):
    print(f"  Feature {i}: [{X_train[:, i].min():.4f}, {X_train[:, i].max():.4f}]")

print(f"\ny range: [{y_train.min():.4f}, {y_train.max():.4f}]")

# Standardize training and test inputs (continuous features)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train[:, :10])
X_test_scaled = scaler_X.transform(X_test[:, :10])

# SF setup: use only the 10 scaled continuous features.
# Do not append any source columns.
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float64)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float64)


# Standardize y
scaler_y = StandardScaler()
y_train_scaled = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten())

print("Standardized data shapes:")
print(f"  X_train: {X_train_scaled.shape}, y_train: {y_train_scaled.shape}")
print(f"  X_test: {X_test_scaled.shape}, y_test: {y_test.shape}")

t1 = time.time()

kernel = gpplus.kernels.LogScaleKernel(
    gpplus.kernels.MVMFKernel(
        cont_cols=cont_cols.tolist(),
        cat_cols=None,
        source_cols=None,
    )
)

model = GPR(
    X_train_scaled,
    y_train_scaled,
    kernel_module=kernel,
    # kernel_module=gpplus.kernels.GaussianKernel(),
    mean_module=gpytorch.means.ConstantMean(),
    likelihood=gpytorch.likelihoods.GaussianLikelihood(),
)

num_inits = 16

print(model)
# Create trainer
trainer = gpplus.training.GPTrainer(
    model=model,
    # num_epochs=10000, # Not required if not using Adam
    seed=seed,
    num_inits=num_inits,
    stop_conditions=[
        gpplus.training.ConvergencePatienceStopCondition(patience=50),
        gpplus.training.MinLossChangeStopCondition(min_loss_change=1e-7),
    ],
    # optimizer_class=torch.optim.Adam,
    device="cpu",
)

print("Training model...")
results = trainer.train()

# Evaluate on standardized test data
y_pred_scaled, pred_lower_scaled, pred_upper_scaled, output_std_scaled = evaluate_gp_model(model, X_test_scaled)

# Transform predictions back to original scale for proper metrics
y_test_orig = y_test  # Already in original scale
y_pred_orig = scaler_y.inverse_transform(y_pred_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
output_std_orig = output_std_scaled.detach().cpu().numpy() * scaler_y.scale_[0]  # Scale the uncertainty


# Compute metrics on original scale

metric = compute_metrics(y_test_orig, y_pred_orig, output_std_orig, start_time=t1)

print("Metrics:")
for k, v in metric.items():
    print(f"  {k}: {v:.4f}")
