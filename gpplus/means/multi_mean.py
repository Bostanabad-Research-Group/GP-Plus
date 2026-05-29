import torch
import torch.nn as nn
from gpytorch.means import ConstantMean, Mean, ZeroMean


class MultiMean(Mean):
    """
    A mean function for grouped / multi-fidelity data.

    Each source (fidelity level) has its own mean module. At forward time the
    source of each input row is read from a one-hot block of columns, and the
    appropriate per-source mean is applied.

    Args:
        source_cols (list[int]):
            Column indices of the one-hot encoded source block in the input
            tensor. ``len(source_cols)`` defines the number of sources.
            Required and must be a list of ints with no duplicates.
        means (list[gpytorch.means.Mean], optional):
            One mean module per source. If None, the first source defaults to
            ``ZeroMean`` and the remaining sources to fresh ``ConstantMean``
            instances. Length must equal ``len(source_cols)``.

    Attributes:
        source_cols (list[int]): Stored one-hot column indices.
        num_sources (int): Number of fidelity sources (== ``len(source_cols)``).
        means (nn.ModuleList): Per-source mean modules, registered as
            submodules so parameters are tracked, moved with ``.to(device)``,
            and saved/loaded via ``state_dict``.

    Example:
        >>> # 3 fidelities, one-hot encoded in columns 10, 11, 12
        >>> mm = MultiMean(source_cols=[10, 11, 12])
        >>> # Or with explicit per-source means:
        >>> mm = MultiMean(
        ...     source_cols=[10, 11, 12],
        ...     means=[ZeroMean(), ConstantMean(), ConstantMean()],
        ... )
    """

    def __init__(self, source_cols, means=None):
        super().__init__()

        # ---- validate source_cols ---------------------------------------
        if source_cols is None:
            raise ValueError("source_cols is required.")
        if not isinstance(source_cols, list):
            raise TypeError(
                f"source_cols must be a list of ints, got {type(source_cols).__name__}. "
                "Convert numpy arrays / tensors / ranges with list(...) at the call site."
            )
        if len(source_cols) == 0:
            raise ValueError("source_cols must be non-empty.")
        if not all(isinstance(c, int) for c in source_cols):
            raise TypeError("source_cols must contain only ints.")
        if len(set(source_cols)) != len(source_cols):
            raise ValueError(f"source_cols contains duplicates: {source_cols}")

        self.source_cols = list(source_cols)
        self.num_sources = len(self.source_cols)

        # ---- validate / build means -------------------------------------
        if means is None:
            # Default: first source is ZeroMean, rest are independent ConstantMeans.
            built = [ZeroMean()]
            built.extend(ConstantMean() for _ in range(self.num_sources - 1))
            means = built
        else:
            if not isinstance(means, list):
                raise TypeError(f"means must be a list of gpytorch.means.Mean, got {type(means).__name__}.")
            if len(means) != self.num_sources:
                raise ValueError(
                    f"Got {len(means)} means but {self.num_sources} sources. "
                    "Lengths must match exactly — duplicate explicitly if you want sharing."
                )
            for i, m in enumerate(means):
                if not isinstance(m, Mean):
                    raise TypeError(f"means[{i}] must be a gpytorch.means.Mean, got {type(m).__name__}.")
            # Guard against accidental parameter sharing from passing the
            # same instance twice — fail loudly rather than silently tie params.
            if len({id(m) for m in means}) != len(means):
                raise ValueError(
                    "means contains the same module instance more than once, which would "
                    "share parameters across sources. Pass independent instances."
                )

        self.means = nn.ModuleList(means)

    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Apply the per-source mean to each row of ``x``.

        Args:
            x: Tensor of shape ``[..., num_features]``. The columns indexed
               by ``self.source_cols`` must form a one-hot block.

        Returns:
            Tensor of shape ``[...]`` with the mean evaluated row-wise.
        """
        if x.shape[-1] <= max(self.source_cols):
            raise ValueError(
                f"Input has {x.shape[-1]} features but source_cols references index {max(self.source_cols)}."
            )

        # Resolve source index per row from the one-hot block.
        source_onehot = x[..., self.source_cols]
        source_ids = source_onehot.argmax(dim=-1)  # shape [...]

        output = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for i, mean_i in enumerate(self.means):
            mask = source_ids == i
            if mask.any():
                output[mask] = mean_i(x[mask]).to(dtype=output.dtype)
        return output
