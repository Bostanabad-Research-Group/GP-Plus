import torch.nn as nn


class InputTransformNet(nn.Module):
    """
    Defines a neural network for input transformation based on a layer configuration dictionary.
    """

    def __init__(self, input_dim, layer_config):
        """
        Args:
            input_dim (int): The input dimension of the data.
            layer_config (dict): A dictionary where keys are layer indices (int), and values are dicts:
                                 'dims' (int): Number of units in the layer.
                                 'activation' (callable): Activation function class (e.g., torch.nn.ReLU).
        """
        super().__init__()
        layers = []
        prev_dim = input_dim

        # Construct the network layers
        for layer_params in layer_config.values():
            layers.append(nn.Linear(prev_dim, layer_params["dims"]))  # Linear layer
            layers.append(layer_params["activation"]())  # Activation function
            prev_dim = layer_params["dims"]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Transforms the input tensor using the defined network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Transformed input tensor.
        """
        return self.network(x)
