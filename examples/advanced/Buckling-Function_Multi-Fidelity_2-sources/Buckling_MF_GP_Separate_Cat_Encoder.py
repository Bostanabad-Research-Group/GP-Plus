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


def rrmse_for_source(y_true, y_pred, source_labels, source_idx: int) -> float:
    """RRMSE on test rows where source index matches (0 = s0, 1 = s1)."""
    if isinstance(source_labels, torch.Tensor):
        mask = (source_labels == source_idx).cpu().numpy()
    else:
        mask = np.asarray(source_labels) == source_idx
    yt = np.asarray(y_true).reshape(-1)[mask]
    yp = np.asarray(y_pred).reshape(-1)[mask]
    if yt.size == 0 or float(np.std(yt)) == 0.0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(yt, yp)) / np.std(yt))


def print_metric_summary(metrics_list, encoder_label, n_seeds, n_inits, wall_time):
    avg_metrics = {}
    std_metrics = {}
    min_metrics = {}
    max_metrics = {}
    median_metrics = {}

    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        avg_metrics[key] = sum(values) / len(values)
        std_metrics[key] = (sum((x - avg_metrics[key]) ** 2 for x in values) / len(values)) ** 0.5
        min_metrics[key] = min(values)
        max_metrics[key] = max(values)
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            median_metrics[key] = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        else:
            median_metrics[key] = sorted_values[n // 2]

    print(f"Buckling, {encoder_label}, separate OH")
    print("\n=== FINAL RESULTS ===")
    print(f"Total time: {wall_time:.2f} s\n({n_inits} restarts)")
    print(f"Average metrics across {n_seeds} seeds (± std):")
    for key in avg_metrics:
        print(f"  {key}: {avg_metrics[key]:.6f} ± {std_metrics[key]:.6f}")

    print(f"\nMin/Max/Median metrics across {n_seeds} runs:")
    for key in avg_metrics:
        print(f"  {key}: min={min_metrics[key]:.6f}, max={max_metrics[key]:.6f}, median={median_metrics[key]:.6f}")


start_seed = 42
num_seeds = 1
num_inits = 16
encoder_modes = ("matrix", "neural")  # MatrixEncoder, then NeuralEncoder

metrics_by_encoder = {mode: [] for mode in encoder_modes}
models_by_encoder = {}
t0 = time.time()

for seed in range(start_seed, start_seed + num_seeds):
    set_seed(seed)

    num_train_per_source = {"s0": 250, "s1": 250}
    num_test_per_source = {"s0": 5000, "s1": 5000}

    print("\nGenerating test data...")
    data = load_data_buckling_MF(
        n_train=num_train_per_source,
        n_test=num_test_per_source,
        noise_levels=[0.0, 0.0],
        return_one_hot=True,
        shuffle=True,
        seed=seed,
    )

    X_test_raw = data["x_test_full"]
    y_test = data["y_test_full"]
    X_train_raw = data["x_train_full"]
    y_train_raw = data["y_train_full"]

    cont_cols = [11]
    cat_cols = [[2, 3], [4, 5, 6, 7], [8, 9, 10]]
    source_cols = [0, 1]

    print(f"\nFinal training dataset shape: X={X_train_raw.shape}, y={y_train_raw.shape}")
    print(f"Final test dataset shape: X={X_test_raw.shape}, y={y_test.shape}")

    scalerX = gpplus.utils.StandardScaler()
    scalerX.fit(X_train_raw[:, cont_cols])
    X_train_scaled = X_train_raw.clone()
    X_test_scaled = X_test_raw.clone()
    X_train_scaled[:, cont_cols] = scalerX.transform(X_train_raw[:, cont_cols])
    X_test_scaled[:, cont_cols] = scalerX.transform(X_test_raw[:, cont_cols])

    scalerY = gpplus.utils.StandardScaler()
    scalerY.fit(y_train_raw)
    y_train_scaled = scalerY.transform(y_train_raw)

    source_test = data["source_test_full"]

    for encoder_mode in encoder_modes:
        print(f"\n=== Seed {seed} | Encoder: {encoder_mode} ===")
        t1 = time.time()

        if encoder_mode == "matrix":
            # MatrixEncoder is the default for MVMFKernel; passing it explicitly is clearer.
            print("---------------------------------------------------")
            print("Using MatrixEncoder for cat and source encoders")
            print("---------------------------------------------------")
            cat_encoder = [
                gpplus.utils.MatrixEncoder(
                    input_dim=len(group),
                    initialization="normal",
                    init_std=0.1,
                    z_dim=2,
                )
                for group in cat_cols
            ]
            source_encoder = gpplus.utils.MatrixEncoder(
                input_dim=len(source_cols),
                initialization="normal",
                init_std=0.1,
                z_dim=2,
            )
        else:  # neural
            print("---------------------------------------------------")
            print("Using NeuralEncoder for cat and source encoders")
            print("---------------------------------------------------")
            cat_encoder = [
                gpplus.utils.NeuralEncoder(
                    input_dim=len(group),
                    architecture_config={"hidden_dims": [], "activation": "relu", "dropout": 0.0},
                    z_dim=2,
                )
                for group in cat_cols
            ]
            source_encoder = gpplus.utils.NeuralEncoder(
                input_dim=len(source_cols),
                architecture_config={"hidden_dims": [], "activation": "relu", "dropout": 0.0},
                z_dim=2,
            )

        kernel = gpplus.kernels.LogScaleKernel(
            gpplus.kernels.MVMFKernel(
                cont_cols=cont_cols,
                cat_cols=cat_cols,
                source_cols=source_cols,
                cat_encoder=cat_encoder,
                source_encoder=source_encoder,
            )
        )

        model = GPR(
            X_train_scaled.clone(),
            y_train_scaled.clone(),
            kernel_module=kernel,
            mean_module=gpplus.means.MultiMean(source_cols=source_cols),
            likelihood=gpplus.likelihoods.MultiLikelihood(source_cols=source_cols, training_data=X_train_scaled),
        )
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
            callbacks=[PrintInitialParametersCallback()],
        )

        print("Training model...")
        trainer.train()

        y_pred_scaled, _, _, output_std_scaled = evaluate_gp_model(model, X_test_scaled)
        y_pred_orig = scalerY.inverse_transform(y_pred_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
        output_std_orig = output_std_scaled.detach().cpu().numpy() * scalerY.std.detach().cpu().numpy().squeeze()

        metric = compute_metrics(y_test, y_pred_orig, output_std_orig, start_time=t1)
        metric["RRMSE_s0"] = rrmse_for_source(y_test, y_pred_orig, source_test, 0)
        metric["RRMSE_s1"] = rrmse_for_source(y_test, y_pred_orig, source_test, 1)

        print(f"Metrics for seed {seed} ({encoder_mode}):")
        for k, v in metric.items():
            if isinstance(v, float) and (v != v):
                print(f"  {k}: nan")
            else:
                print(f"  {k}: {v:.4f}")

        metrics_by_encoder[encoder_mode].append(metric)
        models_by_encoder[encoder_mode] = model

for encoder_mode, run_metrics in metrics_by_encoder.items():
    run_label = "Matrix" if encoder_mode == "matrix" else "Neural"
    print_metric_summary(run_metrics, run_label, num_seeds, num_inits, time.time() - t0)
    model = models_by_encoder[encoder_mode]
    encoder_data_dict = get_latent_representations(model)
    print(f"Encoder data dictionary ({run_label}):\n{encoder_data_dict}")
    plot_encoders(model)
