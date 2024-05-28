# Copyright © 2023, Dr. Bostanabad's research group at the University of California, Irvine.
# 
# GP+ Intellectual Property Notice:
# 
# The software known as GP+ is the proprietary material of Dr. Bostanabad's research group at the University of California, Irvine. 
# Non-profit academic institutions and U.S. government agencies may utilize this software exclusively for educational and research endeavors. 
# All other entities are granted permission for evaluation purposes solely; any additional utilization demands prior written consent from the appropriate authority. 
# The direct sale or redistribution of this software, in any form, without explicit written authorization is strictly prohibited. 
# Users are permitted to make duplicate copies of the software, contingent upon the assurance that no copies are sold or redistributed and they adhere to the stipulated terms herein.
# 
# Being academic research software, GP+ is provided on an "as is" m_gp, devoid of warranties, whether explicit or implicit. 
# The act of downloading or executing any segment of this software inherently signifies compliance with these terms. 
# The developers reserve the right to modify these terms and conditions without prior intimation at any juncture.

import torch
import math
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Dict,List,Optional
from copy import deepcopy
from tqdm import tqdm
import numpy as np

def get_bounds(likobj, theta):

    dic = likobj.unpack_parameters(theta)

    minn = np.empty(0)
    maxx = np.empty(0)
    for name, values in dic.items():
        # print(name)
        for ii in range(len(likobj.model.qual_kernel_columns)):
            if name ==  str(likobj.model.qual_kernel_columns[ii]):
                minn = np.concatenate( (minn,  np.repeat(-3, values.numel()) ) )
                maxx = np.concatenate( (maxx,  np.repeat(3, values.numel()) ) )
        if name == 'likelihood.noise_covar.raw_noise':
            minn = np.concatenate( (minn,  np.repeat(-np.inf, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( np.inf, values.numel()) ) )
        elif name == 'covar_module.raw_outputscale':
            minn = np.concatenate( (minn,  np.repeat(0, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( np.inf, values.numel()) ) )
        elif 'raw_lengthscale' in name:
            minn = np.concatenate( (minn,  np.repeat(-10.0, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( 3.0, values.numel()) ) )
        elif name == 'covar_module.base_kernel.kernels.2.raw_lengthscale':
            maxx = np.concatenate( (maxx,  np.repeat( 3.0, values.numel()) ) )
        elif name == 'A_matrix.fci.[10]fci':
            minn = np.concatenate( (minn,  np.repeat(-np.inf, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( np.inf, values.numel()) ) )
            ######################################################################################### For multiple Bases ##################################
        elif name[0:4] == 'mean':
            minn = np.concatenate( (minn,  np.repeat(-1.5, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( 1.5, values.numel()) ) )
    return np.array(minn).reshape(-1,), np.array(maxx).reshape(-1,)


def fit_model_torch(
    model,
    model_param_groups:Optional[List]=None,
    lr_default:float=0.01,
    num_iter:int=100,
    num_restarts:int=0,
    break_steps:int = 50) -> float:
    '''Optimize the likelihood/posterior of a standard GP+ model using `torch.optim.Adam`.

    This is a convenience function that covers many situations for optimizing a standard GP model.
    Note that using L-BFGS through `fit_model_scipy` function is a better optimization strategy.

    :param model: A model instance derived from the `models.GPR` class. Can also pass a instance
        inherting from `gpytorch.models.ExactGP` provided that `num_restarts=0` or 
        the class implements a `.reset_parameters` method.
    :type model: models.GPR

    :param model_param_groups: list of parameters to optimizes or dicts defining parameter
        groups. If `None` is specified, then all parameters with `.requires_grad`=`True` are 
        included. Defaults to `None`.
    :type model_param_groups: list, optional

    :param lr_default: The default learning rate for all parameter groups. To use different 
        learning rates for some groups, specify them `model_param_groups`. 
    :type lr_default: float, optional

    :param num_iter: The number of optimization steps from each starting point. This is the only
        termination criterion for the optimizer.
    :type num_iter: float, optional

    :param num_restarts: The number of times to restart the local optimization from a 
        new starting point. Defaults to 5
    :type num_restarts: int, optional

    :returns: the best (negative) log-likelihood/log-posterior found
    :rtype: float
    '''  
    model.train()
    
    # objective
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    f_inc = math.inf
    current_state_dict = model.state_dict()


    loss_hist_total = []
    # ############################  ADAM ################################

    for i in range(num_restarts+1):
        optimizer = torch.optim.Adam(
            model.parameters() if model_param_groups is None else model_param_groups, 
            lr=lr_default)
        loss_hist = []
        epochs_iter = tqdm(range(num_iter),desc='Epoch',position=0,leave=True)
        for j in epochs_iter:
            # zero gradients from previous iteration
            optimizer.zero_grad()
            # output from model
            output = model(*model.train_inputs)
            # calculate loss and backprop gradients
            loss = -mll(output,model.train_targets)
            loss.backward()
            optimizer.step()

            acc_loss = loss.item()
            desc = f'Epoch {j} - loss {acc_loss:.4f}'
            epochs_iter.set_description(desc)
            epochs_iter.update(1)
            loss_hist.append(acc_loss)

            if j > break_steps and j%break_steps == 0:
                if ( (torch.mean(torch.Tensor(loss_hist)[j-break_steps:j]) - loss_hist[j]) <= 0 ):
                    break
        
        loss_hist_total.append(loss_hist)

        if loss.item()<f_inc:
            current_state_dict = deepcopy(model.state_dict())
            f_inc = loss.item()
        
        if i < num_restarts:
            model.reset_parameters()
    
    model.load_state_dict(current_state_dict)

    return f_inc, loss_hist_total

