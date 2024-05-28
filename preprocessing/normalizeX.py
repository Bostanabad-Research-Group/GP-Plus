
import numpy as np
import torch
import warnings



def standard(Xtrain, qual_index, Xtest = None):
    quant_index=list(range(Xtrain.shape[1]))
    for item in qual_index.keys():
        quant_index.remove(item)

    if not isinstance(Xtrain, torch.Tensor):
        original_type = type(Xtrain).__name__
        Xtrain = torch.tensor(Xtrain)

    if Xtest is not None and not isinstance(Xtest, torch.Tensor):
        original_type = type(Xtest).__name__
        Xtest = torch.tensor(Xtest)
    
    
    if len(quant_index) == 0:
        return Xtrain
    
    temp = Xtrain[..., quant_index]
    if type(temp) != torch.Tensor:
        temp = temp.astype(float)

    # mean_xtrain = temp.mean(axis = 0)
    # std_xtrain = temp.std(axis = 0)
    mean_xtrain, std_xtrain = compute_mean_std(temp)
    # Check for NaN values in the original data
    if torch.isnan(temp).any():
        print("Warning: There are NaN values in the data. Mean and standard deviation were calculated excluding these values.")

    temp=(temp - mean_xtrain)/std_xtrain
    Xtrain[..., quant_index] = temp
    if type(Xtrain) == np.ndarray:
        Xtrain = torch.from_numpy(Xtrain)
    if Xtest is None:
        return Xtrain,mean_xtrain, std_xtrain
    else:
        temp2 = Xtest[..., quant_index]
        if type(temp2) != torch.Tensor:
            temp2 = temp2.astype(float)
        temp2 = (temp2 - mean_xtrain)/std_xtrain
        Xtest[..., quant_index] = temp2
        if type(Xtest) == np.ndarray:
            Xtest = torch.from_numpy(Xtest)
        return Xtrain, Xtest, mean_xtrain, std_xtrain
    

def compute_mean_std(tensor):
    means = torch.nanmean(tensor, dim=0)

    # Mask for non-NaN elements
    mask = ~torch.isnan(tensor)
    
    # Calculating the standard deviation
    diffs = tensor - means
    diffs[~mask] = 0  # Set differences to 0 where tensor is NaN
    sum_sq = torch.sum(diffs ** 2, dim=0)
    count = mask.sum(dim=0)

    # Avoid division by zero for columns with all NaN values
    zero_count_mask = count == 0
    count[zero_count_mask] = 1  # Temporarily set to 1 to avoid division by zero

    stds = torch.sqrt(sum_sq / count)
    stds[zero_count_mask] = 0  # Set std to 0 for all-NaN columns

    return means, stds

