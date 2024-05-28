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


import math
import numpy as np
import torch
from warnings import simplefilter,catch_warnings
from gpytorch import settings as gptsettings
from gpytorch.utils.warnings import GPInputWarning
from scipy.spatial import distance_matrix
from .mll_scipy import fit_model_scipy
from ..models import GPR
from copy import deepcopy
from typing import Tuple,Dict

def loocv_rrmse(model:GPR):
    # generate prediction caches (using predictions)
    model.eval()
    if model.prediction_strategy is None:
        with torch.no_grad(), gptsettings.fast_computations(log_prob=False):
            with catch_warnings():
                simplefilter('ignore',category=GPInputWarning)
                _ = model(*model.train_inputs).mean

    Kinv_y = model.prediction_strategy.mean_cache
    Rinv = model.prediction_strategy.covar_cache # RinvRinv^T= Kinv
    Kinv_diag = (Rinv**2).sum(dim=-1)

    loo_error = Kinv_y/Kinv_diag
    return (loo_error**2).mean().sqrt().item()

# NLL is negative loglikelihood
def fit_model_continuation(
    model:GPR,add_prior:bool=True,
    num_restarts:int=32,criterion:str='NLL',
    initial_noise_var:float=1,
    red_factor:float=math.sqrt(10),
    options:Dict={},
    n_jobs:int= -1,
    accuracy = 1e-2,
    method = 'L-BFGS-B',
    constraint=False,
    regularization_parameter=[0, 0],
    bounds=False
)-> Tuple[float,Dict]:
    """Sequentially optimize the log-likelihood of a standard GP model for a decreasing
    sequence of noise variances.

    This function, based on the work of `Bostanabad et al. (2018)`_, leverages the smoothing 
    effect of the noise variance on the log-likelihood profile. At each iteration, the function 
    calls `fit_model_scipy` to optimize the log-likelihood while holding the noise variance fixed. 
    The noise variance is then halved for the next iteration. The iteration is terminated when one
    of the following occurs:
        1. criterion begins to increase
        2. a lower bound on the noise variance is reached
        3. cholesky matrix is singular

    If `num_restarts` > 0, multistart optimization is used at each step. In the first iteration,
    the current state of the model and `num_restarts` samples drawn from the prior distribution are 
    used as starting points. In each subsequent iteration, (distinct) optima from the previous iteration
    are used as starting points.
    
    Unlike `Bostanabad et al. (2018)`_, this function uses negative log-likelihood in place of leave
    one-out cross-validation (LOOCV) RMSE as one of the termination criterion. We find that LOOCV is
    unreliable in the case of LVGPs. 

    The initial noise variance should be specifed before passing the model. A value of 1 works for
    many situations. A higher value may be needed when there are many observations and many 
    hyperparameters to estimate. 

    :param model: A model instance derived from the `models.GPR` class. Can also pass a instance
        inherting from `gpytorch.models.ExactGP` provided that `num_restarts=0` or 
        the class implements a `.reset_parameters` method.
    :type model: models.GPR

    :param add_prior: Whether to add the hyperparameter priors to the log-likelihood to optimize the 
        posterior. Optimizing the log-posterior is some what more robust than optimizing the log-likelihood
        when there are few training data. Defaults to True
    :type num_restarts: bool, optional

    :param num_restarts: The number of times to restart the local optimization from a new starting 
        point at the initial noise variance. Subsequent optimizations steps are initialized from
        from the (distinct) optima at the previous step. Defaults to 5.
    :type num_restarts: int, optional

    :param criterion: The criterion used for termination. Can either be 'NLL' or 'LOOCV'. Defaults
        to 'NLL'
    :type criterion: str, optional

    :param initial_noise_var: The initial noise variance. Defaults to 1.
    :type initial_noise_var: float, optional

    :param red_factor: Factor to reduce the noise variance by at each step. This needs to be > 2. 
        Defaults to sqrt(10).
    :type red_factor: float,optional

    :param options: A dictionary of `L-BFGS-B` options to be passed to `scipy.optimize.minimize`.
    :type options: dict,optional

    :param n_jobs: Number of jobs to run in parallel. Uses `joblib` for parallelization. Deafults to 1. 
    :type n_jobs: int,optional

    .. _Bostanabad et al. (2018):
        https://doi.org/10.1002/nme.5751

    Returns:
        A two element tuple with the following elements
            - the value of the criterion at termination
            - a dictionary with the following entries
                - 'noise_history': list of noise variances tried
                - 'nll_history': negative log-likelihoods of the optimal hyperparameters at each noise variance
                - 'loocv_history': LOOCV RRMSEs of the optimal hyperparameters at each noise variance
                - 'optimization_history': list of list of optimization result objects
    """
    if criterion.upper() not in ['NLL','LOOCV']:
        raise AttributeError('criterion must be one of NLL or LOOCV')

    if red_factor < 2:
        raise RuntimeError('Reduction factor for noise variance needs to be greater then 2')

    if model.likelihood.raw_noise.requires_grad:
        model.likelihood.raw_noise.requires_grad_(False)
    
    t = 0    
    theta0_list = None
    while True:

        t += 1
        initial_noise_var_new = initial_noise_var
        if t == 1:
            noises = [initial_noise_var_new/(10**i) for i in range(int(10/t))]
            # noises = [initial_noise_var_new/(np.sqrt(10)**i) for i in range(int(100/t))]

        else:
            if index >= 2 and index < len(history['noise_history'])-2:
                noises = np.linspace(history['noise_history'][index-1], history['noise_history'][index+1], 10)
                initial_noise_var = history['noise_history'][index-1]
                model.load_state_dict(old_state_dict[index-1])
            elif index >= 1 and index < len(history['noise_history'])-1:
                noises = np.linspace(history['noise_history'][index-1], history['noise_history'][index+1], 10)
                initial_noise_var = history['noise_history'][index-1]
                model.load_state_dict(old_state_dict[index-1])
            else:
                model.load_state_dict(old_state_dict[index])
                nllll=history['nll_history'][index]
                print(f'Negative log likelihood={nllll}')
                return history['nll_history'][index],history
                

        noise_list = []
        nll_list = []
        loocv_list = []
        reslist_list = []
        t += 1
        old_crit = math.inf

        old_state_dict = {}


        for i in range(len(noises)):
            model.train()
            model.likelihood.initialize(**{'noise':noises[i]})
            old_state_dict[i] = deepcopy(model.state_dict())
            
            reslist,nll = fit_model_scipy(
                model,add_prior,num_restarts=num_restarts,theta0_list=theta0_list,options=options, n_jobs= n_jobs, method=method, constraint=constraint, regularization_parameter=regularization_parameter, bounds=bounds
            )
            
            if all([isinstance(res,RuntimeError) or isinstance(res,TypeError) for res in reslist]):
                # some error
                break
            
            noise_list.append(model.likelihood.noise.data)
            nll_list.append(nll)
            loocv_list.append('NLL') ## Amin Changend loocv_list.append(loocv_rrmse(model))
            reslist_list.append(reslist)
            crit = nll if criterion.upper()=='NLL' else loocv_list[-1]   #Amin changed
            
            '''
            if crit >= old_crit:
                model.load_state_dict(old_state_dict)
                best_crit = old_crit
                if red_factor > 2:
                    red_factor = 2
                    model.likelihood.initialize(**{'noise':noise_list[-2]/red_factor})
                    continue
                break
            '''

            theta0_list = []
            for res in reslist:
                if isinstance(res,Exception):
                    continue
                if len(theta0_list) > 0:
                    dists = distance_matrix(res.x.reshape(1,-1),np.row_stack(theta0_list)).ravel()
                    if np.any(dists < 1e-2*res.x.shape[0]):
                        continue

                theta0_list.append(res.x)
            
            try:
                model.likelihood.initialize(**{'noise':noise_list[-1]/red_factor})
            except:
                # violates constraints
                try:
                    model.likelihood.initialize(**{'noise':noise_list[-1]/red_factor+1e-10})
                except:
                    break
            old_crit =  math.inf #crit   Amin changed

        history = {
        'noise_history':noise_list,
        'nll_history':nll_list,
        'loocv_history':loocv_list,
        'optimization_history':reslist_list
        }

        index = np.argmin(history['nll_history'])


        print('Finished for loop')


        print(history['nll_history'])

        if np.abs(initial_noise_var_new - history['noise_history'][index]) < accuracy:
            model.load_state_dict(old_state_dict[index])
            break

        

    return history['nll_history'][index],history
