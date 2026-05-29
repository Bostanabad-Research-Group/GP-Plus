import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _cpu_numpy(x):
    """Matplotlib expects CPU NumPy arrays; CUDA tensors crash in scatter/plot."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _to_cpu_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def get_latent_representations(model, qual_dict=None):
    """
    Generate latent representations with robust column index handling

    Args:
        model: Trained GP model
        qual_dict: Dictionary to create proper grouped OH encodings for checking latent representations.
                   If provided, will be used to generate proper combinations for grouped categorical encodings.
                   If there is only one categorical variable in your encoding, this is not necessary.
                   Not necessary if using separate categorical encoders.


    Returns:
        dict: encoder_data_dict mapping encoder names to their data. Keys include
              'cat_encoder', 'cat_encoder_i', and optionally 'source_encoder'.
              If no encoders are found, returns None.
    """
    model.eval()
    # Handle both direct CombinedKernel and LogScaleKernel wrapping CombinedKernel
    if hasattr(model.covar_module, "base_kernel"):
        combined_kernel = model.covar_module.base_kernel
    else:
        combined_kernel = model.covar_module

    # Resolve categorical encoders into (name, module) pairs.
    # Supports legacy "cat_encoder_i", single "cat_encoder", and ModuleList "cat_encoder".
    cat_encoder_entries = []
    numbered_attrs = sorted([attr for attr in dir(combined_kernel) if attr.startswith("cat_encoder_")])
    for attr in numbered_attrs:
        latent_net = getattr(combined_kernel, attr)
        cat_encoder_entries.append((attr, latent_net))

    if not cat_encoder_entries and hasattr(combined_kernel, "cat_encoder"):
        cat_encoder_obj = getattr(combined_kernel, "cat_encoder")
        if isinstance(cat_encoder_obj, torch.nn.ModuleList):
            for i, latent_net in enumerate(cat_encoder_obj):
                cat_encoder_entries.append((f"cat_encoder_{i}", latent_net))
        elif cat_encoder_obj is not None:
            cat_encoder_entries.append(("cat_encoder", cat_encoder_obj))

    # It's possible there are no categorical encoders but a source encoder exists.
    # So do not early-return here; we'll build whatever encoders exist.

    # Dictionary to store encoder data with names
    encoder_data_dict = {}

    # Loop through all categorical encoders
    for i, (encoder_name, latent_net) in enumerate(cat_encoder_entries):
        # Determine encoder dimensions
        if qual_dict is not None and len(cat_encoder_entries) == 1:
            # For grouped categorical encoding, use qual_dict to get proper dimensions
            cat_vars = sorted(qual_dict.items())
            encoder_dims = [dim for _, dim in cat_vars]
        else:
            # For individual encoders, get dimension from the encoder itself
            if hasattr(latent_net, "input_dim"):
                encoder_dims = [latent_net.input_dim]
            else:
                print(f"Warning: Encoder {encoder_name} has no input_dim attribute, skipping")
                continue

        # Skip if no dimensions found
        if not encoder_dims:
            print(f"Warning: No dimensions found for encoder {encoder_name}, skipping")
            continue

        # Generate all possible combinations for this encoder's variables
        indices = torch.cartesian_prod(*[torch.arange(dim) for dim in encoder_dims])

        # Ensure indices is 2D (add dimension if it's 1D)
        if indices.dim() == 1:
            indices = indices.unsqueeze(1)

        # Create one-hot encoded combinations
        one_hots = [F.one_hot(indices[:, j], num_classes=dim) for j, dim in enumerate(encoder_dims)]
        encoder_combinations = torch.cat(one_hots, dim=1)

        # Get latent representations
        with torch.no_grad():
            latent_reps = latent_net(encoder_combinations.to(dtype=model.train_inputs[0].dtype))

        # Store in dictionary with encoder name
        encoder_data_dict[encoder_name] = {
            "combinations": _to_cpu_tensor(encoder_combinations),
            "indices": _to_cpu_tensor(indices),
            "latent_reps": _to_cpu_tensor(latent_reps),
            "input_dim": getattr(latent_net, "input_dim", encoder_combinations.shape[1]),
        }

    # Optionally include source encoder, if present and there are multiple sources
    if hasattr(combined_kernel, "source_encoder") and combined_kernel.source_encoder is not None:
        try:
            n_sources = len(getattr(combined_kernel, "source_cols", []) or [])
        except Exception:
            n_sources = 0
        if n_sources and n_sources > 1:
            source_net = combined_kernel.source_encoder
            source_indices = torch.arange(n_sources).unsqueeze(1)
            source_combinations = torch.eye(n_sources)
            with torch.no_grad():
                source_latent = source_net(source_combinations.to(dtype=model.train_inputs[0].dtype))
            encoder_data_dict["source_encoder"] = {
                "combinations": _to_cpu_tensor(source_combinations),
                "indices": _to_cpu_tensor(source_indices),
                "latent_reps": _to_cpu_tensor(source_latent),
                "input_dim": getattr(source_net, "input_dim", n_sources),
            }

    if len(encoder_data_dict) == 0:
        return None

    return encoder_data_dict


def plot_encoders(model, qual_dict=None, save_path=None):
    """
    Plot latent representations for all categorical encoders in the model.

    Args:
        model: Trained GP model
        qual_dict: Dictionary mapping categorical variable names to their dimensions
        save_path: Optional path to save the plot
    """
    # Get all encoder data from your fixed function
    encoder_data_dict = get_latent_representations(model, qual_dict)

    if encoder_data_dict is None:
        print("No encoder data found")
        return

    # Create subplots for each encoder
    encoder_names = list(encoder_data_dict.keys())
    num_encoders = len(encoder_names)
    fig, axes = plt.subplots(1, num_encoders, figsize=(5 * num_encoders, 5))
    if num_encoders == 1:
        axes = [axes]

    for i, encoder_name in enumerate(encoder_names):
        encoder_data = encoder_data_dict[encoder_name]
        indices = _to_cpu_tensor(encoder_data["indices"])
        latent_reps = _to_cpu_tensor(encoder_data["latent_reps"])
        latent_xy = _cpu_numpy(latent_reps)

        # Plot this encoder
        ax = axes[i]

        # Get unique categories for coloring
        unique_cats = indices.unique(dim=0)
        n_colors = max(len(unique_cats), 1)
        colors = plt.cm.tab10(np.arange(len(unique_cats), dtype=np.float64) / n_colors)

        for j, cat_combo in enumerate(unique_cats):
            # Find indices that match this category combination
            mask = (indices == cat_combo).all(dim=1)
            pts_xy = latent_xy[_cpu_numpy(mask)]

            ax.scatter(
                pts_xy[:, 0],
                pts_xy[:, 1],
                label=f"Cat {j}: {cat_combo.tolist()}",
                alpha=0.7,
                s=50,
                c=np.atleast_2d(colors[j]),
            )
            # Annotate each point with its index
            point_indices = torch.where(mask)[0]
            for idx in point_indices:
                ii = int(idx.item())
                x, y = float(latent_xy[ii, 0]), float(latent_xy[ii, 1])
                ax.text(x, y, str(ii), fontsize=8, ha="center", va="bottom")

        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")
        ax.set_title(f"{encoder_name}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
