import torch
import numpy as np
from gpplus.utils import data_type_check

def setlevels(X, qual_index = None, return_label = False):
    labels = []
    if qual_index == []:
        return X
    if qual_index is None:
        qual_index = list(range(X.shape[-1]))
    X=data_type_check(X)
    # Check if X is a PyTorch tensor
    if isinstance(X, torch.Tensor):
        # Move X to CPU if it's on CUDA
        if X.is_cuda:
            X = X.cpu()
        temp = X.clone()
    # Check if X is a NumPy array
    else:
        # Handle other types or raise an error
        raise TypeError("X must be a PyTorch tensor or a NumPy array.")

    # Convert PyTorch tensor to NumPy array if needed
    if isinstance(temp, torch.Tensor):
        temp = temp.numpy()

    if temp.ndim > 1:
        for j in qual_index:
            l = np.sort(np.unique(temp[..., j])).tolist()
            labels.append(l)
            #l =  torch.unique(temp[..., j], sorted = True).tolist()
            temp[..., j] = torch.tensor([*map(lambda m: l.index(m),temp[..., j])])
    else:
            l = torch.unique(temp, sorted = True)
            temp = torch.tensor([*map(lambda m: l.tolist().index(m), temp)])
    
    
    if temp.dtype == object:
        temp = temp.astype(float)
        if type(X) == np.ndarray:
            temp = torch.from_numpy(temp)
        
        if return_label:
            return temp, labels
        else:
            return temp
    else:
        if type(X) == np.ndarray:
            temp = torch.from_numpy(temp)
        if return_label:
            return temp, labels
        else:
            return temp