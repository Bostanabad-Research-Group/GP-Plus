import numpy as np
import torch


class StandardScaler:
    """A utility class for standardizing data by subtracting the mean and dividing by the standard deviation.

    This scaler follows a similar API to scikit-learn's StandardScaler but is implemented
    for PyTorch tensors. It computes the mean and standard deviation of the features
    along the first dimension (assuming data shape is [N, features]).

    Attributes:
        mean (torch.Tensor or None): The per-feature mean, computed in the `fit` method.
        std (torch.Tensor or None): The per-feature standard deviation, computed in the `fit` method.
    """

    def __init__(self):
        """Initializes a new instance of the StandardScaler with empty mean and std attributes."""
        self.mean = None
        self.std = None

    def fit(self, data: torch.Tensor) -> None:
        """Computes the mean and standard deviation for each feature in the given data.

        Args:
            data (torch.Tensor): The input data of shape [N, features].

        Notes:
            - If `data` has NaN or Inf values, the computed statistics may be invalid.
            - Uses `std(dim=0, correction=0)` which mimics `unbiased=False` behavior (like
              dividing by N instead of N-1 in NumPy).
        """
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, correction=0, keepdim=True)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Applies standardization to the input data using the stored mean and std.

        Args:
            data (torch.Tensor): The input data of shape [N, features] to be standardized.

        Returns:
            torch.Tensor: The standardized data, where each feature has zero mean and unit variance.

        Raises:
            ValueError: If `mean` or `std` have not been set (i.e., if `fit` has not been called).
        """
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler has not been fitted. Call `fit` first.")

        # (Optional) You may want to handle the case where self.std == 0, which can lead to NaNs.
        # For instance:
        # safe_std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
        # return (data - self.mean) / safe_std

        if isinstance(data, np.ndarray):
            tensor = torch.as_tensor(data, dtype=self.mean.dtype, device=self.mean.device)
            return ((tensor - self.mean) / self.std).detach().cpu().numpy()
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Reverts the standardization of the input data using the stored mean and std.

        Args:
            data (torch.Tensor): The standardized data of shape [N, features].

        Returns:
            torch.Tensor: The data in its original scale.

        Raises:
            ValueError: If `mean` or `std` have not been set (i.e., if `fit` has not been called).
        """
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler has not been fitted. Call `fit` first.")
        if isinstance(data, np.ndarray):
            tensor = torch.as_tensor(data, dtype=self.mean.dtype, device=self.mean.device)
            return (tensor * self.std + self.mean).detach().cpu().numpy()
        return data * self.std + self.mean


if __name__ == "__main__":
    # Example usage
    # -------------
    # Assume X_train is your training data as a torch.Tensor of shape [N, features].
    X_train = torch.randn(100, 10)  # example data

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)

    # Later, for test data or predictions:
    X_test = torch.randn(20, 10)  # example test data
    X_test_std = scaler.transform(X_test)

    # Suppose predictions are in standardized space:
    predictions_std = torch.randn(20, 10)  # dummy model predictions
    predictions_original = scaler.inverse_transform(predictions_std)

    print("Standardized predictions:", predictions_std)
    print("Original-scale predictions:", predictions_original)
