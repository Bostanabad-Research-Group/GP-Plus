import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

import gpplus
from examples.data.data_gen import load_data_wing_MV_MF
from gpplus.models import GPR
from gpplus.training.callbacks import PrintInitialParametersCallback
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
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().reshape(-1)
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)
    if output_std is not None and isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy().reshape(-1)

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

    if output_std is not None:
        z = 1.96
        L = y_hat - z * output_std
        U = y_hat + z * output_std
        width = U - L
        below = (L - y_true) * (y_true < L)
        above = (y_true - U) * (y_true > U)
        interval_score = width + (2 / 0.05) * below + (2 / 0.05) * above
        metrics["NIS"] = interval_score.mean() / y_true.std()
        return metrics

    return metrics


seed = 42
set_seed(seed)
num_inits = 16
encoder_modes = ("matrix", "neural")  # default MatrixEncoder, then NeuralEncoder

n_train = {"s0": 100, "s1": 100, "s2": 100, "s3": 100}
n_test = {"s0": 2500, "s1": 2500, "s2": 2500, "s3": 2500}

print("\nGenerating data using load_data_wing_MV_MF...")
data = load_data_wing_MV_MF(
    seed=seed,
    n_train=n_train,
    n_test=n_test,
    noise_levels=[0.0, 0.0, 0.0, 0.0],
    shuffle=True,
    qual_dict={},
    return_one_hot=False,
)

X_train_raw = data["x_train_full"]
y_train_raw = data["y_train_full"]
X_test = data["x_test_full"]
y_test = data["y_test_full"]

cont_cols = list(range(4, 14))
source_cols = list(range(0, 4))

print(f"Xtrainshape: {X_train_raw.shape}")
print(f"Xtestshape: {X_test.shape}")
print(f"\nFinal training dataset shape: X={X_train_raw.shape}, y={y_train_raw.shape}")
print(f"Final test dataset shape: X={X_test.shape}, y={y_test.shape}")

print(f"\nX feature ranges ({len(cont_cols)} continuous features):")
for i, col in enumerate(cont_cols):
    print(f"  Feature {i}: [{X_train_raw[:, col].min():.4f}, {X_train_raw[:, col].max():.4f}]")

print(f"\ny range: [{y_train_raw.min():.4f}, {y_train_raw.max():.4f}]")

scalerX = gpplus.utils.StandardScaler()
scalerX.fit(X_train_raw[:, cont_cols])
X_train = X_train_raw.clone()
X_test_scaled = X_test.clone()
X_train[:, cont_cols] = scalerX.transform(X_train_raw[:, cont_cols])
X_test_scaled[:, cont_cols] = scalerX.transform(X_test[:, cont_cols])

scalerY = gpplus.utils.StandardScaler()
scalerY.fit(y_train_raw)
y_train = scalerY.transform(y_train_raw)

print("Standardized data shapes:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_test: {X_test_scaled.shape}, y_test: {y_test.shape}")

for encoder_mode in encoder_modes:
    print(f"\n=== Encoder: {encoder_mode} ===")
    t1 = time.time()

    if encoder_mode == "matrix":
        # MatrixEncoder is the default for MVMFKernel; passing it explicitly is clearer.
        print("---------------------------------------------------")
        print("Using MatrixEncoder for source encoder")
        print("---------------------------------------------------")
        source_encoder = gpplus.utils.MatrixEncoder(
            input_dim=len(source_cols),
            initialization="normal",
            init_std=0.1,
            z_dim=2,
        )
    else:  # neural
        print("---------------------------------------------------")
        print("Using NeuralEncoder for source encoder")
        print("---------------------------------------------------")
        source_encoder = gpplus.utils.NeuralEncoder(
            input_dim=len(source_cols),
            architecture_config={"hidden_dims": [], "activation": "relu", "dropout": 0.0},
            z_dim=2,
        )

    kernel = gpplus.kernels.LogScaleKernel(
        gpplus.kernels.MVMFKernel(
            cont_cols=cont_cols,
            cat_cols=None,
            source_cols=source_cols,
            source_encoder=source_encoder,
        )
    )

    model = GPR(
        X_train.clone(),
        y_train.clone(),
        kernel_module=kernel,
        mean_module=gpplus.means.MultiMean(source_cols=source_cols),
        likelihood=gpplus.likelihoods.MultiLikelihood(source_cols=source_cols, training_data=X_train),
    )

    print(model)
    trainer = gpplus.training.GPTrainer(
        model=model,
        seed=seed,
        num_epochs=10000,
        num_inits=num_inits,
        stop_conditions=[
            gpplus.training.ConvergencePatienceStopCondition(patience=50),
            gpplus.training.MinLossChangeStopCondition(min_loss_change=1e-7),
        ],
        device="cpu",
        callbacks=[PrintInitialParametersCallback()],
    )

    print("Training model...")
    trainer.train()

    y_pred_scaled, _, _, output_std_scaled = evaluate_gp_model(model, X_test_scaled)
    y_pred_orig = scalerY.inverse_transform(y_pred_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
    output_std_orig = output_std_scaled.detach().cpu().numpy() * scalerY.std.detach().cpu().numpy().squeeze()

    metric = compute_metrics(y_test, y_pred_orig, output_std_orig, start_time=t1)
    run_label = "Matrix" if encoder_mode == "matrix" else "Neural"
    print(f"Metrics ({run_label}Encoder):")
    for k, v in metric.items():
        print(f"  {k}: {v:.4f}")
