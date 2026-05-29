import torch

softplus = torch.nn.Softplus()


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))
