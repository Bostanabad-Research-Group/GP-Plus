from gpplus.preprocessing import standard 
from gpplus.preprocessing import setlevels
from sklearn.model_selection import train_test_split
import torch
import warnings

def train_test_split_normalizeX(
    X,
    y,
    test_size=None,
    shuffle=True,
    stratify=None,
    qual_index_val = {},
    random_state=1,
    return_mean_std = False,
    set_levels = False
):
    # Finding the quant index from qual index
    qual_index = list(qual_index_val.keys())
    # all_index = set(range(X.shape[-1]))
    # if len(Calibration_index)>0:
    #     quant_index = list(set(quant_index).difference(Calibration_index))

    if set_levels:
        # This will assign levels to categorical evenif the levels are strings
        X = setlevels(X, qual_index = qual_index)
    # Split test and train
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, 
        test_size= test_size, shuffle= shuffle,random_state=random_state, stratify=stratify)
    # Standard
    Xtrain, Xtest, mean_train, std_train = standard(Xtrain = Xtrain, 
        qual_index = qual_index_val, Xtest = Xtest)

    variables = {'Xtrain': Xtrain, 'Xtest': Xtest, 'ytrain': ytrain, 'ytest': ytest}
    for var_name, var in variables.items():
        if not isinstance(var, torch.Tensor):
            original_type = type(var).__name__
            # warnings.warn(f"'{var_name}' was not a torch.Tensor (type: {original_type}). It is converted to torch.Tensor to proceed with the preprocessing.")
            variables[var_name] = torch.tensor(var)

    Xtrain, Xtest, ytrain, ytest = variables['Xtrain'], variables['Xtest'], variables['ytrain'], variables['ytest']



    if return_mean_std:
        return Xtrain, Xtest, ytrain, ytest, mean_train, std_train

    return Xtrain, Xtest, ytrain, ytest