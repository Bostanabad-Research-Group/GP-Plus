import torch
import numpy as np
def data_type_check(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        print(f"Warning: Data type was numpy.ndarray. GP+ made it a torch tensor to be able to continue.")
        return torch.from_numpy(data)
    else:
        print(f"Warning: Data type was {type(data)}. GP+ made it a torch tensor to be able to continue.")
        return torch.tensor(data)