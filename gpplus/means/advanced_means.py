# import torch
from gpytorch.means import Mean

from ..utils import InputTransformNet

################################


class CompositeMean(Mean):
    """
    A mean function that applies a user-provided transformation module
    to the input. The transformation is expected to yield a single-dimensional
    output (shape (batch_size, 1) or (batch_size,)) which represents the mean.

    This can be any valid callable or nn.Module, including:
      - Polynomial feature expansions
      - Small neural networks
      - Other custom transforms

    Unlike a typical 'base_mean', this class just returns the
    output of the provided transformation.
    """

    def __init__(self, input_transform):
        """
        Args:
            input_transform (callable or nn.Module):
                A user-defined transformation that takes x of shape (batch_size, input_dim)
                and returns a single-dimensional output (batch_size,).
        """
        super().__init__()
        self.input_transform = input_transform

    def forward(self, x):
        """
        Applies the user-provided transformation and returns its output
        as the mean.

        Args:
            x (torch.Tensor): Input of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Mean predictions of shape (batch_size,).
        """
        # Ensure final shape is (batch_size,) by squeezing if needed
        return self.input_transform(x).squeeze(-1)


################################


class NeuralMean(Mean):
    """
    A mean function that builds its own MLP internally using InputTransformNet.
    The final layer of InputTransformNet must have `dims=1`, so that the
    network outputs a single scalar per data point.

    For example, a valid layer_config might look like:

        layer_config = {
            0: {"dims": 64, "activation": nn.ReLU},
            1: {"dims": 32, "activation": nn.ReLU},
            2: {"dims": 1,  "activation": nn.Identity},
        }

    This ensures the network outputs shape (batch_size, 1), which we
    then squeeze to (batch_size,).
    """

    def __init__(self, input_dim, layer_config):
        """
        Args:
            input_dim (int): Number of input features.
            layer_config (dict): Configuration for each layer, where the last layer
                                 must have 'dims' == 1.
        """
        super().__init__()
        # Build the main transform network
        self.transform_net = InputTransformNet(input_dim, layer_config)

        # Check that the last layer indeed produces 1 dimension
        last_layer_idx = max(layer_config.keys())  # highest key
        last_layer_dims = layer_config[last_layer_idx]["dims"]
        if last_layer_dims != 1:
            raise ValueError(
                f"For NeuralMean, the final layer in `layer_config` must have dims=1, but got {last_layer_dims}."
            )

    def forward(self, x):
        """
        Computes the mean by passing inputs through the internal network.

        Args:
            x (torch.Tensor): Input of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Mean predictions of shape (batch_size,).
        """
        out = self.transform_net(x)  # shape (batch_size, 1)
        return out.squeeze(-1)  # shape (batch_size,)
