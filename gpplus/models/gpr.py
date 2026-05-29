import os

import gpytorch
import torch

from ..config import logger
from ..kernels import GaussianKernel, LogScaleKernel
from ..likelihoods import LogGaussianLikelihood


class GPR(gpytorch.models.ExactGP):
    """Gaussian Process model for regression using GPyTorch.

    The GPR class encapsulates:
      - A mean module (defaults to ConstantMean if None).
      - A kernel module (defaults to a Scale Gaussian kernel if None).
      - A likelihood module (defaults to LogGaussianLikelihood if None).

    Attributes:
        mean_module (gpytorch.means.Mean): The mean function of the GP.
        covar_module (gpytorch.kernels.Kernel): The covariance (kernel) function.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood = None,
        mean_module: gpytorch.means.Mean = None,
        kernel_module: gpytorch.kernels.Kernel = None,
    ):
        """Initializes GPR.

        Args:
            train_x (torch.Tensor): Training data features.
            train_y (torch.Tensor): Training data targets.
            likelihood (gpytorch.likelihoods.Likelihood, optional): The likelihood function. Defaults to \
                GaussianLikelihood if None.
            mean_module (gpytorch.means.Mean, optional): Mean function. Defaults to ConstantMean if None.
            kernel_module (gpytorch.kernels.Kernel, optional): Covariance kernel function.
                Defaults to a ScaleKernel * Gaussian combo if None.

        Raises:
            TypeError: If any of `train_x`, `train_y`, or `likelihood` are of incorrect types.
        """
        self.dtype = train_x.dtype

        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            logger.error("train_x and train_y must be torch.Tensor instances.")
            raise TypeError("train_x and train_y must be torch.Tensor instances.")

        logger.debug(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")

        if likelihood is None:
            likelihood = LogGaussianLikelihood()
            logger.warning("No likelihood provided. Using LogGaussianLikelihood as default.")

        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean()
            logger.warning("No mean_module provided. Using ConstantMean as default.")

        if kernel_module is None:
            input_dim = train_x.shape[-1]
            kernel_module = LogScaleKernel(GaussianKernel(ard_num_dims=input_dim))  # Uses one lengthscale per dimension
            logger.warning(
                "No kernel_module provided. Using LogScaleKernel(GaussianKernel(ard_num_dims=None)) "
                f"(single lengthscale; input_dim={input_dim})."
            )

        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            logger.error("train_x and train_y must be torch.Tensor instances.")
            raise TypeError("train_x and train_y must be torch.Tensor instances.")

        logger.debug(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")

        if not isinstance(likelihood, gpytorch.likelihoods.Likelihood):
            logger.error("likelihood must be an instance of gpytorch.likelihoods.Likelihood.")
            raise TypeError("likelihood must be an instance of gpytorch.likelihoods.Likelihood.")

        super().__init__(train_x, train_y, likelihood)

        self.mean_module = mean_module
        self.covar_module = kernel_module

        # Ensure all components use the same dtype as the input data
        self.mean_module = self.mean_module.to(dtype=self.dtype)
        self.covar_module = self.covar_module.to(dtype=self.dtype)
        self.likelihood = self.likelihood.to(dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Runs the forward pass of the Gaussian Process model with ensembling
            if embedding or calibration is probabilistic.

        Args:
            x (torch.Tensor): Test data features for prediction.

        Returns:
            gpytorch.distributions.MultivariateNormal:
                Multivariate normal distribution containing
                the mean and covariance of the predictions.

        Raises:
            TypeError: If `x` is not a torch.Tensor.
        """
        if not isinstance(x, torch.Tensor):
            logger.error("Input x must be a torch.Tensor instance.")
            raise TypeError("Input x must be a torch.Tensor.")

        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def save(self, filepath: str = "model_weights.pth") -> None:
        """Saves this model's state dictionary to the specified file.

        Args:
            filepath (str, optional): Path to save the state dictionary file.
                Defaults to 'model_weights.pth' in the current directory.
        """
        logger.info(f"Saving model state dict to {filepath}")
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str = "model_weights.pth") -> None:
        """Loads this model's state dictionary from the specified file.

        Args:
            filepath (str, optional): Path to the file containing the saved state dict.
                Defaults to 'model_weights.pth' in the current directory.

        Raises:
            FileNotFoundError: If no file is found at `filepath`.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model weights found at {filepath}")

        logger.info(f"Loading model state dict from {filepath}")
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict)
