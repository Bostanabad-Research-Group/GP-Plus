import os

import gpytorch
import torch

from ..config import logger
from ..kernels import GaussianKernel


class MTGPR(gpytorch.models.ExactGP):
    """Gaussian Process model for MultiTask regression using GPyTorch.

    The GPR class encapsulates:
      - A mean module (defaults to ConstantMean if None).
      - A kernel module (defaults to a Scale Gaussian kernel if None).
      - A likelihood module (defaults to GaussianLikelihood if None).

    Attributes:
        mean_module (gpytorch.means.Mean): The mean function of the GP.
        covar_module (gpytorch.kernels.Kernel): The covariance (kernel) function.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        num_tasks: int = 1,
        rank_likelihood: int = 0,
        rank_kernel: int = 1,
        likelihood: gpytorch.likelihoods.Likelihood = None,
        mean_module: gpytorch.means.MultitaskMean = None,
        kernel_module: gpytorch.kernels.MultitaskKernel = None,
    ):
        """Initializes GPR.

        Args:
            train_x (torch.Tensor): Training data features.
            train_y (torch.Tensor): Training data targets.
            likelihood (gpytorch.likelihoods.Likelihood, optional): The likelihood function. Defaults to \
                MultitaskGaussianLikelihood if None.
            mean_module (gpytorch.means.MultitaskMean, optional): Mean function. Defaults to ConstantMean if None.
            kernel_module (gpytorch.kernels.Kernel, optional): Covariance kernel function.
                Defaults to a ScaleKernel * MultitaskGaussian combo if None.

        Raises:
            TypeError: If any of `train_x`, `train_y`, or `likelihood` are of incorrect types.
        """
        # Attributes
        self.num_tasks = num_tasks
        self.rank_likelihood = rank_likelihood
        self.rank_kernel = rank_kernel

        if likelihood is None:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=num_tasks, rank=self.rank_likelihood
            )
            logger.warning("No likelihood provided. Using MultitaskGaussianLikelihood as default.")

        if mean_module is None:
            base_mean = gpytorch.means.ConstantMean()
            mean_module = gpytorch.means.MultitaskMean(base_mean, self.num_tasks)
            logger.warning("No mean_module provided. Using ConstantMean as default.")

        if kernel_module is None:
            base_kernel = gpytorch.kernels.ScaleKernel(GaussianKernel())
            kernel_module = gpytorch.kernels.MultitaskKernel(
                base_kernel, num_tasks=self.num_tasks, rank=self.rank_kernel
            )
            logger.warning("No kernel_module provided. Using Gaussian Kernel as default.")

        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            logger.error("train_x and train_y must be torch.Tensor instances.")
            raise TypeError("train_x and train_y must be torch.Tensor instances.")

        logger.debug(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")

        super().__init__(train_x, train_y, likelihood)

        self.mean_module = mean_module
        self.covar_module = kernel_module

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultitaskMultivariateNormal:
        """Runs the forward pass of the Gaussian Process model.

        Args:
            x (torch.Tensor): Test data features for prediction.

        Returns:
            gpytorch.distributions.MultitaskMultivariateNormal:
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
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)

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
