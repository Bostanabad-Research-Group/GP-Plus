import gpytorch
import torch

from ..utils.encoders import BaseEncoder, MatrixEncoder
from .gaussian_kernel import GaussianKernel


def _validate_int_list(cols, name):
    """Validate that ``cols`` is a list of ints. Returns a copy."""
    if cols is None:
        return []
    if not isinstance(cols, list):
        raise TypeError(
            f"{name} must be a list of ints, got {type(cols).__name__}. "
            "Convert numpy arrays / tensors / ranges with list(...) at the call site."
        )
    if not all(isinstance(c, int) and not isinstance(c, bool) for c in cols):
        raise TypeError(f"{name} must contain only ints.")
    if len(set(cols)) != len(cols):
        raise ValueError(f"{name} contains duplicate indices: {cols}")
    return list(cols)


def _validate_cat_cols(cat_cols):
    """
    Normalize ``cat_cols`` into a ``list[list[int]]`` (one inner list per
    categorical group).

    Accepts:
        - None → []
        - A flat list of ints → wrapped as a single group: [[...]]
        - A list of lists of ints → returned as-is (validated)

    Rejects numpy arrays, tensors, ranges, tuples — convert at the call site.
    """
    if cat_cols is None:
        return []
    if not isinstance(cat_cols, list):
        raise TypeError(
            f"cat_cols must be a list, got {type(cat_cols).__name__}. "
            "Convert numpy arrays / tensors / ranges with list(...)."
        )
    if len(cat_cols) == 0:
        return []

    first = cat_cols[0]
    if isinstance(first, int) and not isinstance(first, bool):
        # Flat list of ints → single group.
        return [_validate_int_list(cat_cols, "cat_cols")]
    if isinstance(first, list):
        # Nested: each entry must be a list of ints.
        groups = []
        for i, group in enumerate(cat_cols):
            if not isinstance(group, list):
                raise TypeError(f"cat_cols[{i}] must be a list of ints, got {type(group).__name__}.")
            groups.append(_validate_int_list(group, f"cat_cols[{i}]"))
        # Reject overlap across groups — almost always a bug.
        seen = set()
        for i, group in enumerate(groups):
            overlap = seen & set(group)
            if overlap:
                raise ValueError(f"cat_cols[{i}] overlaps with earlier groups on indices {sorted(overlap)}.")
            seen.update(group)
        return groups
    raise TypeError(
        "cat_cols must be either a flat list of ints (one group) or "
        f"a list of lists of ints (grouped); got first element of type {type(first).__name__}."
    )


class MVMFKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        cont_cols: list = None,
        cat_cols: list = None,
        source_cols: list = None,
        cont_kernel: gpytorch.kernels.Kernel = None,
        cat_kernel: gpytorch.kernels.Kernel = None,
        source_kernel: gpytorch.kernels.Kernel = None,
        cat_encoder=None,
        source_encoder=None,
        z_dim=2,
        fix_lengthscale_cat=False,
        fix_lengthscale_source=False,
        **kwargs,
    ):
        """
        Multi-Variable Multi-Fidelity Combined Kernel.

        Args:
            cont_cols: list[int] of continuous feature column indices.
            cat_cols: Either a flat list[int] (one categorical group) or a
                list[list[int]] (multiple groups, one inner list per group).
            source_cols: list[int] of source/fidelity one-hot column indices.
            cont_kernel: Kernel for continuous features (default: GaussianKernel
                with ARD).
            cat_kernel: Kernel for categorical features (default: GaussianKernel
                on the concatenated latent space).
            source_kernel: Kernel for source features (default: GaussianKernel
                on the latent space).
            cat_encoder: A BaseEncoder or a list of BaseEncoders (one per
                group). Defaults to one MatrixEncoder per group.
            source_encoder: A BaseEncoder. Defaults to MatrixEncoder.
            z_dim: Latent dimension for default encoders.
            fix_lengthscale_cat: Whether to freeze the categorical kernel
                lengthscale.
            fix_lengthscale_source: Whether to freeze the source kernel
                lengthscale.
        """
        super().__init__(**kwargs)

        self.cont_cols = _validate_int_list(cont_cols, "cont_cols")
        self.cat_cols = _validate_cat_cols(cat_cols)
        self.source_cols = _validate_int_list(source_cols, "source_cols")
        self.z_dim = z_dim
        self.fix_lengthscale_cat = fix_lengthscale_cat
        self.fix_lengthscale_source = fix_lengthscale_source

        self._process_cont(cont_kernel)
        self._process_cat(cat_encoder, cat_kernel)
        self._process_source(source_encoder, source_kernel)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x1, x2=None, diag=False, last_dim_is_batch=False, **kwargs):
        device = x1.device
        if x2 is None:
            x2 = x1

        # Continuous kernel
        if self.cont_kernel is not None:
            cont_idx = torch.as_tensor(self.cont_cols, device=device)
            k_cont = self.cont_kernel(
                x1.index_select(-1, cont_idx),
                x2.index_select(-1, cont_idx),
                diag=diag,
                last_dim_is_batch=last_dim_is_batch,
                **kwargs,
            )
            result = k_cont
        else:
            result = torch.ones(x1.shape[0], x2.shape[0], device=device)

        # Categorical kernel
        if self.cat_kernel is not None and self.cat_encoder is not None:
            encoders = self.cat_encoder if isinstance(self.cat_encoder, torch.nn.ModuleList) else [self.cat_encoder]
            z1_cat_list, z2_cat_list = [], []
            for encoder, col_group in zip(encoders, self.cat_cols):
                cat_idx = torch.as_tensor(col_group, device=device)
                x1_group = x1.index_select(-1, cat_idx)
                x2_group = x2.index_select(-1, cat_idx)
                z1_c, z2_c = encoder.encode_pair(x1_group, x2_group)
                z1_cat_list.append(z1_c)
                z2_cat_list.append(z2_c)
            z1_cat_concat = torch.cat(z1_cat_list, dim=-1)
            z2_cat_concat = torch.cat(z2_cat_list, dim=-1)
            result_cat = self.cat_kernel(
                z1_cat_concat,
                z2_cat_concat,
                diag=diag,
                last_dim_is_batch=last_dim_is_batch,
                **kwargs,
            )
            result = result.mul(result_cat) if self.cont_kernel is not None else result_cat

        # Source kernel
        if self.source_kernel is not None and self.source_encoder is not None:
            source_idx = torch.as_tensor(self.source_cols, device=device)
            x1_source = x1.index_select(-1, source_idx)
            x2_source = x2.index_select(-1, source_idx)
            z1_s, z2_s = self.source_encoder.encode_pair(x1_source, x2_source)
            k_source = self.source_kernel(
                z1_s,
                z2_s,
                diag=diag,
                last_dim_is_batch=last_dim_is_batch,
                **kwargs,
            )
            result = result.mul(k_source)

        return result

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------

    def _process_cont(self, cont_kernel):
        if cont_kernel is not None or len(self.cont_cols) > 0:
            if cont_kernel is None:
                self.cont_kernel = GaussianKernel(ard_num_dims=len(self.cont_cols))
            else:
                self.cont_kernel = cont_kernel
        else:
            self.cont_kernel = None

    def _make_default_kernel(self, kernel, z_dim, fix_lengthscale):
        if kernel is not None:
            return kernel
        gauss_k = GaussianKernel(ard_num_dims=z_dim)
        if fix_lengthscale:
            gauss_k.raw_lengthscale.requires_grad_(False)
            gauss_k.raw_lengthscale.data = torch.ones(z_dim) * 0.0
        else:
            gauss_k.raw_lengthscale.requires_grad_(True)
        return gauss_k

    def _resolve_encoder_list(self, encoder_spec, col_groups, name):
        """Coerce ``encoder_spec`` into a list of BaseEncoder, one per group."""
        if encoder_spec is None:
            encoders = [MatrixEncoder(input_dim=len(group), z_dim=self.z_dim) for group in col_groups]
        elif isinstance(encoder_spec, BaseEncoder):
            if len(col_groups) != 1:
                raise ValueError(
                    f"A single {name} can only be used with one group. "
                    f"For grouped columns, pass a list of BaseEncoder objects."
                )
            encoders = [encoder_spec]
        elif isinstance(encoder_spec, list):
            encoders = list(encoder_spec)
        else:
            raise TypeError(
                f"{name} must be a BaseEncoder or a list of BaseEncoder; got {type(encoder_spec).__name__}."
            )

        if len(encoders) != len(col_groups):
            raise ValueError(
                f"Number of {name} entries ({len(encoders)}) must match number of groups ({len(col_groups)})."
            )
        for i, (encoder, group) in enumerate(zip(encoders, col_groups)):
            if not isinstance(encoder, BaseEncoder):
                raise TypeError(f"{name}[{i}] must be a BaseEncoder instance")
            if encoder.input_dim != len(group):
                raise ValueError(f"{name}[{i}].input_dim={encoder.input_dim} does not match group size={len(group)}")
        return encoders

    def _resolve_single_encoder(self, encoder_spec, input_dim, name):
        if encoder_spec is None:
            encoder = MatrixEncoder(input_dim=input_dim, z_dim=self.z_dim)
        elif isinstance(encoder_spec, BaseEncoder):
            encoder = encoder_spec
        else:
            raise TypeError(f"{name} must be a BaseEncoder; got {type(encoder_spec).__name__}.")
        if encoder.input_dim != input_dim:
            raise ValueError(f"{name}.input_dim={encoder.input_dim} does not match size={input_dim}")
        return encoder

    def _process_cat(self, cat_encoder, cat_kernel):
        if len(self.cat_cols) > 0:
            encoders = self._resolve_encoder_list(cat_encoder, self.cat_cols, "cat_encoder")
            if len(encoders) == 1:
                self.cat_encoder = encoders[0]
            else:
                self.cat_encoder = torch.nn.ModuleList(encoders)
            total_z_dim = sum(encoder.z_dim for encoder in encoders)
            self.cat_kernel = self._make_default_kernel(
                kernel=cat_kernel,
                z_dim=total_z_dim,
                fix_lengthscale=self.fix_lengthscale_cat,
            )
        elif cat_kernel is not None or cat_encoder is not None:
            raise ValueError("cat_cols must be non-empty when cat_encoder or cat_kernel is provided")
        else:
            self.cat_kernel = None
            self.cat_encoder = None

    def _process_source(self, source_encoder, source_kernel):
        if len(self.source_cols) > 0:
            encoder = self._resolve_single_encoder(
                encoder_spec=source_encoder,
                input_dim=len(self.source_cols),
                name="source_encoder",
            )
            self.source_kernel = self._make_default_kernel(
                kernel=source_kernel,
                z_dim=encoder.z_dim,
                fix_lengthscale=self.fix_lengthscale_source,
            )
            self.source_encoder = encoder
        elif source_kernel is not None or source_encoder is not None:
            raise ValueError("source_cols must be non-empty when source_encoder or source_kernel is provided")
        else:
            self.source_kernel = None
            self.source_encoder = None
