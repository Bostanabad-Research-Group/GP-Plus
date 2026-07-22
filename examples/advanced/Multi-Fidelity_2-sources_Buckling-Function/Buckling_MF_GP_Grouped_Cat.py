import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

import gpplus
from examples.data.data_gen import load_data_buckling_MF
from gpplus.models import GPR
from gpplus.training.callbacks import PrintInitialParametersCallback
from gpplus.training.eval import evaluate_gp_model
from gpplus.utils import set_seed
from gpplus.utils.latent_reps import get_latent_representations, plot_encoders


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


def rrmse_for_source(y_true, y_pred, source_test, source_idx: int) -> float:
    """RRMSE on test rows where source index matches (0 = s0, 1 = s1)."""
    if isinstance(source_test, torch.Tensor):
        mask = (source_test == source_idx).cpu().numpy()
    else:
        mask = np.asarray(source_test) == source_idx
    yt = np.asarray(y_true).reshape(-1)[mask]
    yp = np.asarray(y_pred).reshape(-1)[mask]
    if yt.size == 0 or float(np.std(yt)) == 0.0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(yt, yp)) / np.std(yt))


start_seed = 42
num_seeds = 1

# Training loop below
full_metrics = []
t0 = time.time()
for seed in range(start_seed, start_seed + num_seeds):
    set_seed(seed)

    # Generate training and test data for all fidelity levels s0-s3
    sources = ["s0", "s1"]
    num_train_per_source = {"s0": 250, "s1": 250}
    num_test_per_source = {"s0": 5000, "s1": 5000}  # 10k total

    # Generate test data
    print("\nGenerating test data...")
    X_test_data = []
    y_test_data = []

    data = load_data_buckling_MF(
        n_train=num_train_per_source,
        n_test=num_test_per_source,
        noise_levels=[0.0, 0.0],
        return_one_hot=True,
        shuffle=True,
        seed=seed,
    )

    # Concatenate all test data (10,000x14)
    X_test = data["x_test_full"]
    y_test = data["y_test_full"]
    # Concatenate all training data (400x14)
    X_train = data["x_train_full"]
    y_train = data["y_train_full"]

    cont_cols = [11]
    cat_cols = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    source_cols = [0, 1]

    print(f"\nFinal training dataset shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Final test dataset shape: X={X_test.shape}, y={y_test.shape}")

    # Standardize the data
    scalerX = gpplus.utils.StandardScaler()
    scalerX.fit(X_train[:, cont_cols])
    X_train[:, cont_cols] = scalerX.transform(X_train[:, cont_cols])
    X_test[:, cont_cols] = scalerX.transform(X_test[:, cont_cols])
    # scaler.transform(X_test[:,cont_cols])

    scalerY = gpplus.utils.StandardScaler()
    scalerY.fit(y_train)
    y_train = scalerY.transform(y_train)

    t1 = time.time()

    # source_encoder = gpplus.utils.MatrixEncoder(
    #     input_dim=len(source_cols), initialization="normal", init_std=0.1, z_dim=2
    # )

    # source_encoder2 = gpplus.utils.NeuralEncoder(
    #     input_dim=len(source_cols),
    #     architecture_config={"hidden_dims": [], "activation": "relu", "dropout": 0.0},
    #     z_dim=2,
    # )

    # cat_encoder = gpplus.utils.NeuralEncoder(
    #     input_dim=len(cat_cols),
    #     architecture_config={"hidden_dims": [], "activation": "relu", "dropout": 0.0},
    #     z_dim=2,
    # )

    # Create model
    kernel = gpplus.kernels.LogScaleKernel(
        gpplus.kernels.MVMFKernel(
            cont_cols=cont_cols,
            cat_cols=cat_cols,
            source_cols=source_cols,
            # cat_encoder=cat_encoder,
            # source_encoder=source_encoder, # Can uncomment encoders above to test differnt encodings.
            # source_encoder=source_encoder2,
        )
    )

    model = GPR(
        X_train,
        y_train,
        kernel_module=kernel,
        mean_module=gpplus.means.MultiMean(source_cols=source_cols),
        likelihood=gpplus.likelihoods.MultiLikelihood(source_cols=source_cols, training_data=X_train),
        # likelihood=gpytorch.likelihoods.GaussianLikelihood(),
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
            gpplus.training.ConvergencePatienceStopCondition(patience=10),
            gpplus.training.MinLossChangeStopCondition(min_loss_change=1e-7),
        ],
        # optimizer_class=torch.optim.Adam, # Problem is slow with Adam defaults kwargs
        device="cpu",  # Problem is slow with cuda
        callbacks=[PrintInitialParametersCallback()],
    )

    print("Training model...")
    results = trainer.train()

    # Evaluate on standardized test data
    y_pred_scaled, pred_lower_scaled, pred_upper_scaled, output_std_scaled = evaluate_gp_model(model, X_test)

    # Transform predictions back to original scale for proper metrics
    y_pred_orig = scalerY.inverse_transform(y_pred_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
    # Scale predictive std to original y units
    output_std_orig = output_std_scaled.detach().cpu().numpy() * scalerY.std.detach().cpu().numpy().squeeze()

    # Compute metrics on original scale

    metric = compute_metrics(y_test, y_pred_orig, output_std_orig, start_time=t1)

    source_test = data["source_test_full"]
    metric["RRMSE_s0"] = rrmse_for_source(y_test, y_pred_orig, source_test, 0)
    metric["RRMSE_s1"] = rrmse_for_source(y_test, y_pred_orig, source_test, 1)

    print(f"Metrics for seed {seed}:")
    for k, v in metric.items():
        if isinstance(v, float) and (v != v):  # nan
            print(f"  {k}: nan")
        else:
            print(f"  {k}: {v:.4f}")

    full_metrics.append(metric)

# Calculate average metrics
avg_metrics = {}
std_metrics = {}
min_metrics = {}
max_metrics = {}
median_metrics = {}

for key in full_metrics[0].keys():  # Get keys from first metric dict
    values = [metric[key] for metric in full_metrics]
    avg_metrics[key] = sum(values) / len(values)
    std_metrics[key] = (sum((x - avg_metrics[key]) ** 2 for x in values) / len(values)) ** 0.5
    min_metrics[key] = min(values)
    max_metrics[key] = max(values)
    # Calculate median
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 0:
        median_metrics[key] = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        median_metrics[key] = sorted_values[n // 2]
print("Buckling, Matrix, grouped OH")
print("\n=== FINAL RESULTS ===")
print(f"Total time: {time.time() - t0:.2f} s\n({num_inits} restarts)")
print(f"Average metrics across {num_seeds} seeds (± std):")
for metric in avg_metrics.keys():
    mean_val = avg_metrics[metric]
    std_val = std_metrics[metric]
    print(f"  {metric}: {mean_val:.6f} ± {std_val:.6f}")

print(f"\nMin/Max/Median metrics across {num_seeds} runs:")
for metric in avg_metrics.keys():
    min_val = min_metrics[metric]
    max_val = max_metrics[metric]
    median_val = median_metrics[metric]
    print(f"  {metric}: min={min_val:.6f}, max={max_val:.6f}, median={median_val:.6f}")

BUCKLING_QUAL_DICT = {0: 2, 1: 4, 2: 3}
encoder_data_dict = get_latent_representations(model, qual_dict=BUCKLING_QUAL_DICT)
print(f"Encoder data dictionary:\n{encoder_data_dict}")
plot_encoders(model, qual_dict=BUCKLING_QUAL_DICT)
