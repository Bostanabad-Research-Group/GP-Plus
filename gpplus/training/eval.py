import gpytorch
import torch

from ..config import logger
from ..likelihoods import MultiLikelihood


def evaluate_gp_model(
    model,
    test_x: torch.Tensor,
):
    """
    Evaluates the Gaussian Process model on test data.

    Args:
        model (GPModel):
            The Gaussian Process model to evaluate.
        test_x (torch.Tensor):
            Test data features.

    Returns:
        tuple:
            A tuple containing:
                - **mean** (torch.Tensor): Predictive mean for each test point.
                - **lower** (torch.Tensor): Lower confidence bound for each test point.
                - **upper** (torch.Tensor): Upper confidence bound for each test point.
                - **stddev** (torch.Tensor): Standard deviation of the predictions.
    """
    # Align GPyTorch / settings with training, then evaluate with
    # slower stable fast_computations(False...) to reduce host-to-host variance.
    model.eval()

    with (
        torch.no_grad(),
        gpytorch.settings.fast_computations(
            covar_root_decomposition=False,
            log_prob=False,
            solves=False,
        ),
    ):  # gpytorch.settings.fast_pred_var():
        # Make predictions
        # Option 1: Without nugget (latent function f) - follows Equation 29b structure
        # observed_pred = model(test_x)

        # Option 2: With nugget (noisy observations y) - follows Equation 31b structure
        # This adds +δI to the predictive covariance: +δ* = K_test_test - ... + +δI
        # For MultiLikelihood, we need to set test fidelity indices from test data
        # so each test point gets the correct nugget based on its source
        train_inputs = getattr(model, "train_inputs", None)
        if train_inputs and len(train_inputs) > 0:
            reference = train_inputs[0]
            test_x = test_x.to(device=reference.device, dtype=reference.dtype)

        if isinstance(model.likelihood, MultiLikelihood):
            model.likelihood.set_fidelity_indices(test_x, is_test=True)
        observed_pred = model.likelihood(model(test_x))

        # Get the mean, lower and upper confidence bounds
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()
        stddev = observed_pred.stddev

        logger.info("Evaluation completed.")
        return mean, lower, upper, stddev
