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
import numpy as np
from gpytorch import settings as gptsettings
from gpytorch.utils.errors import NanError,NotPSDError
from scipy.optimize import minimize,OptimizeResult
from collections import OrderedDict
from functools import reduce
from joblib import Parallel,delayed
from joblib.externals.loky import set_loky_pickler
from typing import Dict,List,Tuple,Optional,Union
from copy import deepcopy
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
#######################################################

from gpplus.utils.interval_score import interval_score_function
tkwargs = {
    "dtype": torch.float,
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
}

def marginal_log_likelihood(model,add_prior:bool,regularization_parameter=[0,0]):
    output = model(*model.train_inputs)
    out = model.likelihood(output).log_prob(model.train_targets)
    if add_prior:
        # add priors
        for _, module, prior, closure, _ in model.named_priors():
            out.add_(prior.log_prob(closure(module)).sum())
    temp = 0
    temp_1=0
    for name, param in model.named_parameters():
        string_list = ['fci', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h8','h9', 'h10', 'h11', 'h12','fce']
        if name in string_list:
            temp += torch.norm(param)
            temp_1 += torch.sum(torch.abs(param))
        elif name in ['nn_model.' + str + '.bias' for str in string_list]:
            temp += torch.norm(param)
            temp_1 += torch.sum(torch.abs(param))

    out -= regularization_parameter[0]*temp_1 + regularization_parameter[1]* temp    
    ## Interval Score if neede for BO
    if model.interval_score is True:
        score, accuracy = interval_score_function(output.mean + 1.96 * output.variance.sqrt(), output.mean - 1.96 * output.variance.sqrt(), model.y_scaled)
        return out - 0.08*torch.abs(out) * score#- torch.exp(model.interval_alpha) * score
    return out 


class MLLObjective:

    def __init__(self,model,add_prior,regularization_parameter):
        self.model = model 
        self.add_prior = add_prior
        self.regularization_parameter=regularization_parameter

        parameters = OrderedDict([
            (n,p) for n,p in self.model.named_parameters() if p.requires_grad
        ])
        self.param_shapes = OrderedDict()
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                if len(parameters[n].size()) > 0:
                    self.param_shapes[n] = parameters[n].size()
                else:
                    self.param_shapes[n] = torch.Size([1])
    
    def pack_parameters(self) -> np.ndarray:
        parameters = OrderedDict([
            (n,p) for n,p in self.model.named_parameters() if p.requires_grad
        ])
        
        return np.concatenate([parameters[n].cpu().data.numpy().ravel() for n in parameters])
    
    def unpack_parameters(self, x:np.ndarray) -> torch.Tensor:
        i = 0
        named_parameters = OrderedDict()
        for n in self.param_shapes:
            param_len = reduce(lambda x,y: x*y, self.param_shapes[n])
            # slice out a section of this length
            param = x[i:i+param_len]
            # reshape according to this size, and cast to torch
            param = param.reshape(*self.param_shapes[n])
            named_parameters[n] = torch.from_numpy(param).to(**tkwargs)
            # update index
            i += param_len
        return named_parameters

    def pack_grads(self) -> None:
        """Concatenate gradients from the parameters to 1D numpy array
        """
        grads = []
        for name,p in self.model.named_parameters():
            if p.requires_grad:
                grad = p.grad.cpu().data.numpy()
                grads.append(grad.ravel())
        return np.concatenate(grads).astype(np.float64)

    def fun(self, x:np.ndarray,return_grad=True) -> Union[float,Tuple[float,np.ndarray]]:
        # unpack x and load into module 
        state_dict = self.unpack_parameters(x)
        old_dict = self.model.state_dict()
        old_dict.update(state_dict)
        self.model.load_state_dict(old_dict)

        self.model.zero_grad()
        obj = -marginal_log_likelihood(self.model, self.add_prior,self.regularization_parameter) # negative sign to minimize
        
        if return_grad:
            obj.backward()
            
            return obj.item(),self.pack_grads()
        
        return obj.item()


def _sample_from_prior(model) -> np.ndarray:
    out = []
    for _,module,prior,closure,_ in model.named_priors():
        if not closure(module).requires_grad:
            continue
            
        out.append(prior.expand(closure(module).shape).sample().cpu().numpy().ravel())
    
    return np.concatenate(out)


def cons_f(x,likobj):
    zeta = torch.tensor(likobj.model.zeta, dtype = torch.float64)
    A = likobj.unpack_parameters(x)['fci']
    likobj.model.nn_model.fci.weight.data = A
    positions = likobj.model.nn_model(zeta)
    out_constraint=positions.detach().numpy().reshape(-1,)
    return out_constraint[0:8]

def get_bounds(likobj, theta):

    dic = likobj.unpack_parameters(theta)

    minn = np.empty(0)
    maxx = np.empty(0)
    for name, values in dic.items():
        for ii in range(len(likobj.model.qual_kernel_columns)):
            if name ==  str(likobj.model.qual_kernel_columns[ii]):
                minn = np.concatenate( (minn,  np.repeat(-3, values.numel()) ) )
                maxx = np.concatenate( (maxx,  np.repeat(3, values.numel()) ) )
        if name == 'likelihood.noise_covar.raw_noise' or name.startswith('[') or name.startswith('latent['):
            minn = np.concatenate( (minn,  np.repeat(-np.inf, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( np.inf, values.numel()) ) )
        if 'raw_lengthscale' in name:
            minn = np.concatenate( (minn,  np.repeat(-10.0, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( 3.0, values.numel()) ) )
        elif name.startswith('covar_module'):
            minn = np.concatenate( (minn,  np.repeat(-10.0, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( 3.0, values.numel()) ) )
            ######################################################################################### For multiple Bases ##################################
        elif name.startswith('mean'):
            minn = np.concatenate( (minn,  np.repeat(-1.5, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( 1.5, values.numel()) ) )
        elif name.startswith('Theta_'):
            minn = np.concatenate( (minn,  np.repeat(-15, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( 15, values.numel()) ) )
        elif name.startswith('encoder'):
            minn = np.concatenate( (minn,  np.repeat(-15, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( 15, values.numel()) ) )
            ######################################################################################### For A_matrix and Variationa encoder  #################
        elif name.startswith('A_matrix'):
            minn = np.concatenate( (minn,  np.repeat(-10, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( 10, values.numel()) ) )
    return np.array(minn).reshape(-1,), np.array(maxx).reshape(-1,)



def _fit_model_from_state(likobj,theta0,jac,options, method = 'trust-constr',constraint=False,bounds=False):
    
    min, max = get_bounds(likobj, theta0)
    bounds_acts = Bounds(min, max)
    nonlinear_constraint = NonlinearConstraint(lambda x: cons_f(x, likobj), [0,0,0,0,-5,0,-5,-5],[0,0,5,0,5,5,5,5], jac='2-point', hess=BFGS())
    '''
    if constraint==True:
        nonlinear_constraint = NonlinearConstraint(lambda x: cons_f(x, likobj), [0,0,0,0,-inf,0],[0,0,inf,0,inf,inf], jac='2-point', hess=BFGS())
    
    else:
        nonlinear_constraint = NonlinearConstraint(lambda x: cons_f(x, likobj), [-inf,-inf,-inf,-inf,-inf,-inf],[inf,inf,inf,inf,inf,inf], jac='2-point', hess=BFGS())

    '''
    eq_cons = {'type': 'eq',
                'fun' : lambda x: np.array([cons_f(x, likobj)[0],cons_f(x, likobj)[1],cons_f(x, likobj)[3]])}
    ineq_cons = {'type': 'ineq',
                'fun' : lambda x: np.array([cons_f(x, likobj)[2],cons_f(x, likobj)[5]])}
    if constraint==True:
        nonlinear_constraint=[nonlinear_constraint]
    else:
        nonlinear_constraint=[]


    if bounds==True:
        bounds=bounds_acts
    else:
        bounds=None


    try:
        with gptsettings.fast_computations(log_prob=False):
            return minimize(
                fun = likobj.fun,
                x0 = theta0,
                args=(True) if jac else (False),

                method = method,
                jac=jac,
                bounds=bounds,
                constraints= nonlinear_constraint,
                #constraints=[eq_cons, ineq_cons],
                options= options 
            )


    except Exception as e:
        if isinstance(e,NotPSDError) or isinstance(e, NanError):
            # Unstable hyperparameter configuration. This can happen if the 
            # initial starting point is bad. 
            return e
        else:
            # There is some other issue, most likely with the inputs supplied
            # by the user. Raise error to indicate the problematic part.
            raise



def fit_model_scipy(
    model,
    add_prior:bool=True,
    num_restarts:int=1,
    theta0_list:Optional[List[np.ndarray]]=None,
    jac:bool=True, 
    options:Dict={},
    n_jobs:int=-1,
    method = 'L-BFGS-B',
    constraint=False,
    bounds=False,
    regularization_parameter:List[int]=[0,0]
    ) -> Tuple[List[OptimizeResult],float]:

    defaults = {}

    if method == 'L-BFGS-B':
        defaults = {'ftol':1e-6,'gtol':1e-5,'maxfun':5000,'maxiter':2000}
    elif method == 'trust-constr':
        defaults = {'verbose': 1}
    elif method == 'BFGS':
        defaults = {'gtol': 1e-07, 'norm': np.inf, 'eps': 1.4901161193847656e-08,'maxiter': None, 'disp': False, 'return_all': False, 'finite_diff_rel_step': None}#
    elif method == 'SLSQP':
        defaults = { 'maxiter': 100, 'ftol': 1e-06, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None}#
    elif method == 'Newton-CG':
        defaults={'xtol': 1e-05, 'eps': 1.4901161193847656e-08, 'maxiter': None, 'disp': False, 'return_all': False}
    else:
        raise ValueError('Wrong method')

    if len(options) > 0:
        for key in options.keys():
            if key not in defaults.keys():
                raise RuntimeError('Unknown option %s!'%key)
            defaults[key] = options[key]

    likobj = MLLObjective(model,add_prior,regularization_parameter)

    if theta0_list is None:
        theta0_list = [likobj.pack_parameters()]
        if num_restarts > -1:
            theta0_list.extend([_sample_from_prior(model) for _ in range(num_restarts+1)])
            theta0_list.pop(0)                                                                     
    
    set_loky_pickler("dill") 

    out = Parallel(n_jobs=n_jobs,verbose=0)(
        delayed(_fit_model_from_state)(likobj,theta0,jac,defaults, method,constraint,bounds) \
            for theta0 in theta0_list
    )
    set_loky_pickler("pickle")

    nlls_opt = [np.inf if isinstance(res,Exception) else res.fun for res in out]
    best_idx = np.argmin(nlls_opt)
    try:
        theta_best = out[best_idx].x
        old_dict = deepcopy(model.state_dict())
        old_dict.update(likobj.unpack_parameters(theta_best))
        model.load_state_dict(old_dict)

        if 'fci' in [name for name,p in model.named_parameters()]:
            model.nn_model.fci.weight.data = old_dict['fci']       
    except:
        pass
    return out,nlls_opt[best_idx]