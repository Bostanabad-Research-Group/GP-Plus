"""
Single-fidelity Buckling GP: one source (s0) only; E / K / I as separate cat groups.

Same training stack as A2_buckling_SF_GPvsPFN.py: float64, LBFGSScipy, cholesky_jitter.
"""

import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

import gpplus
from examples.data.data_gen import load_data_buckling_MF
from gpplus.models import GPR
from gpplus.training.eval import evaluate_gp_model
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import set_seed
from gpplus.utils.latent_reps import get_latent_representations, plot_encoders


def compute_metrics(y_true, y_hat, output_std=None, start_time=None):
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


num_train = 120
num_test = 5000
num_inits = 16

start_seed = 42
num_seeds = 1

full_metrics = []
t0 = time.time()
for seed in range(start_seed, start_seed + num_seeds):
    set_seed(seed)

    data = load_data_buckling_MF(
        n_train={"s0": num_train},
        n_test={"s0": num_test},
        noise_levels=[0.0],
        return_one_hot=True,
        shuffle=True,
        seed=seed,
    )

    X_train = data["x_train_full"].clone()
    y_train = data["y_train_full"].clone()
    X_test = data["x_test_full"].clone()
    y_test = data["y_test_full"].clone()

    n_src = len(data["column_indices"]["source"])
    X_train = X_train[:, n_src:]
    X_test = X_test[:, n_src:]

    cont_cols = [9]
    cat_cols = [[0, 1], [2, 3, 4, 5], [6, 7, 8]]

    print(f"Training shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shape: X={X_test.shape}, y={y_test.shape}")

    scalerX = gpplus.utils.StandardScaler()
    scalerX.fit(X_train[:, cont_cols])
    X_train[:, cont_cols] = scalerX.transform(X_train[:, cont_cols])
    X_test[:, cont_cols] = scalerX.transform(X_test[:, cont_cols])

    scalerY = gpplus.utils.StandardScaler()
    scalerY.fit(y_train)
    y_train = scalerY.transform(y_train)

    X_train = X_train.to(dtype=torch.float64)
    X_test = X_test.to(dtype=torch.float64)
    y_train = y_train.to(dtype=torch.float64)
    y_test = y_test.to(dtype=torch.float64)

    t1 = time.time()

    kernel = gpplus.kernels.LogScaleKernel(
        gpplus.kernels.MVMFKernel(
            cont_cols=cont_cols,
            cat_cols=cat_cols,
        )
    )

    model = GPR(X_train, y_train, kernel_module=kernel)
    print(model)

    trainer = gpplus.training.GPTrainer(
        model=model,
        seed=seed,
        num_inits=num_inits,
        stop_conditions=[
            gpplus.training.ConvergencePatienceStopCondition(patience=10),
            gpplus.training.MinLossChangeStopCondition(min_loss_change=1e-7),
        ],
        device="cpu",
        optimizer_class=LBFGSScipy,
    )

    print("Training...")
    trainer.train()

    y_pred_s, _, _, output_std_s = evaluate_gp_model(model, X_test)
    y_pred_orig = scalerY.inverse_transform(y_pred_s.detach().cpu().numpy().reshape(-1, 1)).flatten()
    output_std_orig = output_std_s.detach().cpu().numpy() * scalerY.std.detach().cpu().numpy().squeeze()

    metric = compute_metrics(y_test, y_pred_orig, output_std_orig, start_time=t1)
    print(f"Seed {seed} metrics:")
    for k, v in metric.items():
        print(f"  {k}: {v:.4f}")
    full_metrics.append(metric)

avg_metrics = {}
std_metrics = {}
for key in full_metrics[0].keys():
    vals = [m[key] for m in full_metrics]
    avg_metrics[key] = float(np.mean(vals))
    std_metrics[key] = float(np.std(vals))

print("\n=== Buckling SF (separate E/K/I groups) ===")
print(
    f"Wall time: {time.time() - t0:.2f} s | {num_inits} inits | "
    f"{num_train} train / {num_test} test | dtype={torch.float64}"
)
for k in avg_metrics:
    print(f"  {k}: {avg_metrics[k]:.6f} ± {std_metrics[k]:.6f}")

encoder_data_dict = get_latent_representations(model)  # Obtains the latent representations of last model trained
print(f"Encoder data dictionary:\n{encoder_data_dict}")  # Prints the latent representations of last model trained
plot_encoders(model)  # Plots the latent representations of last model trained
