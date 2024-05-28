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
import numpy as np
from typing import Dict,List,Optional
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from tabulate import tabulate
import sobol_seq
import warnings

import torch
from torch import nn
from torch import Tensor 
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F 

import gpytorch
from gpytorch.constraints import Positive
from gpytorch.means import Mean
from gpytorch.priors import NormalPrior
from gpytorch.distributions import MultivariateNormal

from gpplus.visual.plot_latenth import plot_sep
from gpplus.models.gpregression import GPR
from gpplus import kernels
from gpplus.priors import MollifiedUniformPrior
from gpplus.preprocessing import setlevels
from gpplus.utils import set_seed, data_type_check
from gpplus.optim import fit_model_scipy, fit_model_continuation, fit_model_torch


class GP_Plus(GPR):
    """The GP_Plus class extends Gaussian Processes (GPs) to learn nonlinear and probabilistic manifolds, handle categorical inputs, and more.

    :note: Binary categorical variables should not be treated as qualitative inputs. There is no 
        benefit from applying a latent variable treatment for such variables. Instead, treat them
        as numerical inputs.
    
    :param train_x: The training inputs. This represents the input data.
    :param train_y: The training outputs. This represents the output data.
    :param dtype: The data type of the model and data, which could be float32 or float64 torch tensor.
    :param device: Specifies whether the model will be built on CPU or CUDA if available. Otherwise, it should always be CPU.
    :param qual_index: A dictionary that indicates categorical inputs. The keys represent the number of columns that are categorical, starting from 0, and the values indicate the number of levels for each specific categorical input.
    :param multiple_noise: Used in cases where there are multiple sources of data (multiple fidelity emulation), allowing for different noise estimation for each fidelity data source.
    :param lb_noise: The lower bound for noise estimation.
    :param fix_noise: Determines whether to fix noise to a specific value. It can be True or False. If True, it indicates that noise should be fixed.
    :param fix_noise_val: The value to which noise is fixed when `fix_noise` is True.
    :param quant_correlation_class: The class of correlation used in the model.
    :param fixed_length_scale: Indicates whether the length scale should be fixed. It can be True or False. If True, the length scale will be fixed.
    :param fixed_length_scale_val: The value of the length scale in fixed scenarios.
    :param encoding_type: The type of encoding used for categorical inputs.
    :param embedding_dim: The dimension of the encoding (dimension of embedding).
    :param separate_embedding: Specifies the layers in the embedding used for mapping.
    :param embedding_type: The type of embedding, could be 'deterministic' or 'probabilistic'.
    :param NN_layers_embedding: A list that shows the architecture of neural network layers for the embedding.
    :param m_gp: Specifies the type of mean function considered for the GP model.
    :param m_gp_ref: The mean function for the reference source, which has an ID equal to 0.
    :param NN_layers_m_gp: A list that shows the architecture of the neural network used in the GP model for cases where a neural network is used as the mean function.
    :param calibration_type: The type of calibration used, could be 'deterministic' or 'probabilistic'.
    :param calibration_id: A list that indicates the inputs that have calibration parameters.
    :param mean_prior_cal: The mean prior used for calibration, which should have the same length as `calibration_id`.
    :param std_prior_cal: The standard deviation prior used for calibration, which should have the same length as `calibration_id`.
    :param interval_score: Indicates whether to add interval scoring during optimization.It can be True or False. If True, it is added by default with a coefficient of 0.08.
    :param seed_number: The seed number for random number generation to ensure reproducible results.
    """
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        dtype= torch.float,
        device="cpu",
        qual_index = {},
        multiple_noise = False,
        lb_noise:float=1e-8,
        fix_noise:bool=False,
        fix_noise_val:float=1e-5,
        quant_correlation_class:str='Rough_RBF',
        fixed_length_scale:bool=False,
        fixed_length_scale_val=torch.tensor([1.0]),
        encoding_type = 'one-hot',
        embedding_dim:int=2,
        separate_embedding = [] , #max=2
        embedding_type='deterministic',
        NN_layers_embedding:list = [],
        m_gp='single_constant',
        m_gp_ref='zero',
        NN_layers_m_gp=[],
        calibration_type='deterministic',
        calibration_id=[],
        mean_prior_cal=None,  
        std_prior_cal=None,  
        interval_score=False,
        num_pass_train=1,
        num_pass_pred=1,
        seed_number=1
    ) -> None:
        if calibration_type=='probabelistic' or  calibration_type=='probabelistic':
            num_pass_train=20
            num_pass_pred=30
        
        if mean_prior_cal is None:
            mean_prior_cal = [0 for _ in calibration_id]
        if std_prior_cal is None:
            std_prior_cal = [1 for _ in calibration_id]
        
        self.mean_prior_cal=mean_prior_cal
        self.std_prior_cal=std_prior_cal
        
        self.interval_score=interval_score
        tkwargs = {}  # or dict()
        tkwargs['dtype'] = dtype
        tkwargs['device'] =torch.device(device)
        self.tkwargs=tkwargs


        self.mean_prior_cal=mean_prior_cal
        self.std_prior_cal=std_prior_cal
        if fixed_length_scale:
            self.fixed_length_scale_val=fixed_length_scale_val.to(**self.tkwargs)
        else:
            self.fixed_length_scale_val=None
            
        # Checking inputs & output
        train_x=data_type_check(train_x)
        train_y=data_type_check(train_y)

        if not isinstance(qual_index, dict):
            raise ValueError("qual_index should be a dictionary.") 

        if multiple_noise not in [True, False]:
            raise ValueError("multiple_noise should be either True or False.")

        if not isinstance(embedding_dim, int):
            raise ValueError("embedding_dim should be an integer.")

        if quant_correlation_class not in ['Rough_RBF', 'RBFKernel', 'Matern32Kernel', 'Matern12Kernel','Matern52Kernel']:
            raise ValueError("quant_correlation_class should be 'Rough_RBF', 'RBFKernel', 'Matern32Kernel', 'Matern12Kernel','Matern52Kernel'.")

        if fix_noise not in [True, False]:
            raise ValueError("fix_noise should be either True or False.")

        if not isinstance(NN_layers_embedding, list) or not all(isinstance(i, int) for i in NN_layers_embedding):
            raise ValueError("NN_layers_embedding should be a list of integers representing the number of neurons in each layer.")

        if encoding_type != 'one-hot':
            raise ValueError("encoding_type should be 'one-hot'.")

        if embedding_type not in ['deterministic', 'probabilistic']:
            raise ValueError("embedding_type should be either 'deterministic' or 'probabilistic'.")

        if not isinstance(separate_embedding, list) or not all(isinstance(i, int) for i in separate_embedding):
            raise ValueError("separate_embedding should be a list with integers showing the number of categorical inputs to be considered in a separate manifold in each layer.")

    
        supported_singl_functions = ['single_sin', 'single_cos', 'single_exp', 'single_log', 'single_tan', 'single_asin', 'single_acos', 'single_atan', 
                                    'single_sinh', 'single_cosh', 'single_tanh', 'single_asinh', 'single_acosh', 'single_atanh', 'single_sqrt', 
                                    'single_abs', 'single_ceil', 'single_floor', 'single_round'] ########## single_functions!!!!

        self.supported_singl_m_gp_functions=supported_singl_functions
        
        supported_multi_m_gp_functions=['single_zero', 'single_polynomial', 'single_constant', 'multiple_polynomial_2d', 'multiple_constant', 'neural_network']


        if not isinstance(NN_layers_m_gp, list) or not all(isinstance(i, int) for i in NN_layers_m_gp):
            raise ValueError("NN_layers_m_gp should be a list with integers representing the number of neurons in each layer for the mean function.")

        if not isinstance(calibration_id, list) or not all(isinstance(i, int) for i in calibration_id):
            raise ValueError("calibration_id should be a list where each entry shows the column number in the dataset that the calibration parameters are assigned to.")
        
        train_x=self.fill_nan_with_mean(train_x,calibration_id)
        ###############################################################################################
        ###############################################################################################
        self.seed=seed_number
        self.calibration_id=calibration_id
        self.calibration_source_index=0    ## It is supposed the calibration parameter is for high fidelity needs
        qual_index_list = list(qual_index.keys())
        all_index = set(range(train_x.shape[-1]))
        quant_index = list(all_index.difference(qual_index_list))
        num_levels_per_var = list(qual_index.values())
        #------------------- lm columns --------------------------
        lm_columns = list(set(qual_index_list).difference(separate_embedding))
        if len(lm_columns) > 0:
            qual_kernel_columns = [*separate_embedding, lm_columns]
        else:
            qual_kernel_columns = separate_embedding
        #########################
        train_y=train_y.reshape(-1)
        if multiple_noise:
            noise_indices = list(range(0,num_levels_per_var[-1]))
        else:
            noise_indices = []

        if len(qual_index_list) == 1 and num_levels_per_var[0] < 2:
            temp = quant_index.copy()
            temp.append(qual_index_list[0])
            quant_index = temp.copy()
            qual_index_list = []
            embedding_dim = 0
        elif len(qual_index_list) == 0:
            embedding_dim = 0

        if len(qual_index_list) == 0:
            embedding_dim = 0

        if len(qual_index_list) > 0:
            ####################### Defined multiple kernels for seperate variables ###################
            qual_kernels = []
            for i in range(len(qual_kernel_columns)):
                qual_kernels.append(kernels.RBFKernel(
                    active_dims=torch.arange(embedding_dim) + embedding_dim * i) )
                qual_kernels[i].initialize(**{'lengthscale':1.0})
                qual_kernels[i].raw_lengthscale.requires_grad_(False)
                
        quant_correlation_class_name = quant_correlation_class  
        if quant_correlation_class_name == 'Rough_RBF':
            quant_correlation_class = 'RBFKernel'
        if len(quant_index) == 0:
            correlation_kernel = qual_kernels[0]
            for i in range(1, len(qual_kernels)):
                correlation_kernel *= qual_kernels[i]
        else:
            try:
                quant_correlation_class = getattr(kernels,quant_correlation_class)
            except:
                raise RuntimeError(
                    "%s not an allowed kernel" % quant_correlation_class
                )
            if quant_correlation_class_name == 'RBFKernel':
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=len(qual_kernel_columns) * embedding_dim+torch.arange(len(quant_index)),
                    lengthscale_constraint= Positive(transform= torch.exp,inv_transform= torch.log)
                )
            elif quant_correlation_class_name == 'Rough_RBF':
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=len(qual_kernel_columns)*embedding_dim+torch.arange(len(quant_index)),
                    lengthscale_constraint= Positive(transform= lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))
                )
            elif quant_correlation_class_name == 'Matern12Kernel':
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=len(qual_kernel_columns)*embedding_dim+torch.arange(len(quant_index)),
                    lengthscale_constraint= Positive(transform= lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))
                )
            
            elif quant_correlation_class_name == 'Matern32Kernel':
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=len(qual_kernel_columns)*embedding_dim+torch.arange(len(quant_index)),
                    lengthscale_constraint= Positive(transform= lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))             
                )
            elif quant_correlation_class_name == 'Matern52Kernel':
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=len(qual_kernel_columns)*embedding_dim+torch.arange(len(quant_index)),
                    lengthscale_constraint= Positive(transform= lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))       
                )
                #####################
            if quant_correlation_class_name == 'RBFKernel':
                quant_kernel.register_prior(
                    'lengthscale_prior', MollifiedUniformPrior(math.log(0.1),math.log(10)),'raw_lengthscale'
                )
                
            elif quant_correlation_class_name == 'Rough_RBF':
                quant_kernel.register_prior(
                    'lengthscale_prior',NormalPrior(-3.0,3.0),'raw_lengthscale'
                )
            elif quant_correlation_class_name == 'Matern12Kernel':
                quant_kernel.register_prior(
                    'lengthscale_prior',NormalPrior(-3.0,3.0),'raw_lengthscale'
                )

            elif quant_correlation_class_name == 'Matern32Kernel':
                quant_kernel.register_prior(
                    'lengthscale_prior',NormalPrior(-3.0,3.0),'raw_lengthscale'
                )

            elif quant_correlation_class_name == 'Matern52Kernel':
                quant_kernel.register_prior(
                    'lengthscale_prior',NormalPrior(-3.0,3.0),'raw_lengthscale'
                )
            if len(qual_index_list) > 0:
                temp = qual_kernels[0]
                for i in range(1, len(qual_kernels)):
                    temp *= qual_kernels[i]
                correlation_kernel = temp*quant_kernel #+ qual_kernel + quant_kernel
            else:
                correlation_kernel = quant_kernel
            #####################
        super(GP_Plus,self).__init__(
            train_x=train_x,train_y=train_y,noise_indices=noise_indices,
            correlation_kernel=correlation_kernel,
            fix_noise=fix_noise,fix_noise_val=fix_noise_val,lb_noise=lb_noise
        )

        self.calibration_type=calibration_type
        for n, mean_prior, std_prior in zip(self.calibration_id, self.mean_prior_cal, self.std_prior_cal):
            if self.calibration_type == 'probabilistic':
                setattr(self, 'Theta_' + str(n), LinearVariational(batch_shape=torch.Size([]), mean_prior=mean_prior, std_prior=0*std_prior).to(**tkwargs))
                setattr(self, 'calibration_element' + str(n), torch.where(train_x[:, -1] == self.calibration_source_index)[0])
            else:
                setattr(self, 'Theta_' + str(n), gpytorch.means.ConstantMean(prior=NormalPrior(mean_prior, std_prior)))
                setattr(self, 'calibration_element' + str(n), torch.where(train_x[:, -1] == self.calibration_source_index)[0])

            train_x[getattr(self, 'calibration_element' + str(n)), n] = torch.zeros_like(train_x[getattr(self, 'calibration_element' + str(n)), n])

        # register index and transforms
        self.register_buffer('quant_index',torch.tensor(quant_index))
        self.register_buffer('qual_index_list',torch.tensor(qual_index_list))

        self.qual_kernel_columns = qual_kernel_columns
        # latent variable mapping
        self.num_levels_per_var = num_levels_per_var
        self.embedding_dim = embedding_dim
        self.encoding_type = encoding_type
        self.embedding_type=embedding_type
        self.perm =[]
        self.zeta = []
        self.random_zeta=[]
        self.perm_dict = []
        self.A_matrix = []
        self.epsilon=None
        self.epsilon_f=None
        self.embeddings_Dtrain=[]
        self.count=train_x.size()[0]
        self.num_pass_train=num_pass_train
        self.num_pass_pred=num_pass_pred
        if len(qual_kernel_columns) > 0:
            for i in range(len(qual_kernel_columns)):
                if type(qual_kernel_columns[i]) == int:
                    num = self.num_levels_per_var[qual_index_list.index(qual_kernel_columns[i])]
                    cat = [num]
                else:
                    cat = [self.num_levels_per_var[qual_index_list.index(k)] for k in qual_kernel_columns[i]]
                    num = sum(cat)

                zeta, perm, perm_dict = self.zeta_matrix(num_levels=cat, embedding_dim = self.embedding_dim)
                self.zeta.append(zeta)
                self.perm.append(perm)
                self.perm_dict.append(perm_dict)       
                ###################################  latent map (manifold) #################################   
                if self.embedding_type=='probabilistic':
                    setattr(self,'A_matrix', Variational_Encoder(self, input_size= num, num_classes=5, 
                        layers =NN_layers_embedding, name = str(qual_kernel_columns[i])).to(**tkwargs))
                else:
                    model_temp = FFNN(self, input_size= num, num_classes=embedding_dim, 
                        layers = NN_layers_embedding, name ='latent'+ str(qual_kernel_columns[i])).to(**self.tkwargs)
                    self.A_matrix.append(model_temp)

        ##################################################################################
        if fixed_length_scale == True:
            self.covar_module.m_gp_kernel.raw_lengthscale.data = self.fixed_length_scale_val 
            self.covar_module.m_gp_kernel.raw_lengthscale.requires_grad = False  
        ###################################  Mean Function #################################   
        i=0
        self.m_gp=m_gp
        self.m_gp_ref=m_gp_ref
        self.num_sources=int(torch.max(train_x[:,-1]))
        size=train_x.shape[1]
        if self.m_gp.startswith('single'):
            self.single_m_gp_register(size,m_gp_type=self.m_gp,wm='mean_module')
        elif self.m_gp.startswith('multi'):
            self.multi_m_gp_register (train_x,supported_multi_m_gp_functions,self.m_gp_ref)
        elif self.m_gp=='neural_network': 
            setattr(self,'mean_module_NN_All', FFNN_as_Mean(self, input_size= train_x.shape[1]+2*len(qual_index_list)-len(qual_index_list), num_classes=1,layers =NN_layers_m_gp, name = str('mean_module_'+str(i)+'_')).to(**tkwargs)) 
        else: 
            raise ValueError('The "m_gp" argument must start with "multi", "single", or "neural_network".')



    def forward(self,x:torch.Tensor) -> MultivariateNormal:
        if self.embedding_type=='probabilistic' or self.calibration_type=='probabilistic':
            set_seed(self.seed)
            if self.training:
                Numper_of_pass=self.num_pass_train#5 
            else:
                Numper_of_pass=self.num_pass_pred#10 
        else:
            Numper_of_pass=1
        
        Sigma_sum=torch.zeros(x.size(0),x.size(0), dtype=torch.float64).to(self.tkwargs['device'])
        mean_x_sum=torch.zeros(x.size(0), dtype=torch.float64).to(self.tkwargs['device'])

        for NP in range(Numper_of_pass):
            x_forward_raw=x.clone()
            nd_flag = 0
            if x.dim() > 2:
                xsize = x.shape
                x = x.reshape(-1, x.shape[-1])
                nd_flag = 1
            
            x_new= x
            if len(self.qual_kernel_columns) > 0:
                embeddings = []
                for i in range(len(self.qual_kernel_columns)):
                    temp= self.transform_categorical(x=x[:,self.qual_kernel_columns[i]].clone().type(torch.int64).to(self.tkwargs['device']), 
                        perm_dict = self.perm_dict[i], zeta = self.zeta[i])
                dimm=x_forward_raw.size()[0]
                if self.embedding_type=='probabilistic': 
                    # Convert to list of tuples
                    x_raw=torch.zeros(temp.size(0),2)
                    # Find unique rows
                    unique_rows, indices = torch.unique(temp, dim=0, return_inverse=True)
                    temp= unique_rows
                    dimm=unique_rows.size()[0]
                    if self.training:
                        epsilon=torch.normal(mean=0,std=1,size=[dimm,2])
                        embeddings.append(getattr(self,'A_matrix')(x=temp.float().to(**self.tkwargs),epsilon=epsilon))
                    else:
                        if x.size()[0]==self.count:
                            epsilon=torch.normal(mean=0,std=1,size=[dimm,2])
                            embeddings.append(getattr(self,'A_matrix')(x=temp.float().to(**self.tkwargs),epsilon=epsilon))
                            self.embeddings_Dtrain.append(embeddings[0])
                        else:
                            embeddings.append(self.embeddings_Dtrain[NP])
                    for i, index in enumerate(indices):
                        x_raw[i] = embeddings[0][index]
                    embeddings=x_raw
                    x_new= torch.cat([embeddings,x[...,self.quant_index.long()]],dim=-1)
                else:
                    embeddings.append(self.A_matrix[i](temp.float().to(**self.tkwargs)))
                    x_new= torch.cat([embeddings[0],x[...,self.quant_index.long()].to(**self.tkwargs)],dim=-1)
                
                ## For Calibration
                if len(self.calibration_id)>0:
                    if self.training:
                        for n in self.calibration_id:
                            if self.calibration_type=='probabilistic':
                                s=torch.ones_like(x_new[getattr(self,'calibration_element'+str(n)),embeddings[0].size(1)+n]).shape
                                epsilon=torch.normal(mean=0,std=1,size=[s[0],1])
                                Theta=(getattr(self,'Theta_'+str(n))(epsilon.clone().reshape(-1,1)))
                                x_new[getattr(self,'calibration_element'+str(n)),embeddings[0].size(1)+n]=\
                                    torch.ones_like(x_new[getattr(self,'calibration_element'+str(n)),embeddings[0].size(1)+n])*(Theta.reshape(-1))
                                x_new[getattr(self,'calibration_element'+str(n)),embeddings[0].size(1)+n]=\
                                    torch.ones_like(x_new[getattr(self,'calibration_element'+str(n)),embeddings[0].size(1)+n])*(getattr(self,'Theta_'+str(n))(x[i,-1].clone().flatten().reshape(-1,1)))
                    else:
                        for n in self.calibration_id:
                            if self.calibration_type=='probabilistic':
                                epsilon=torch.normal(mean=0,std=1,size=[1,1])
                                calibration_element=torch.where(x[:, -1]==self.calibration_source_index)[0]
                                x_new[calibration_element,embeddings[0].size(1)+n]=\
                                        torch.ones_like(x_new[calibration_element,embeddings[0].size(1)+n])*(getattr(self,'Theta_'+str(n))(epsilon.clone().reshape(-1,1)))
                            else:
                                calibration_element=torch.where(x[:, -1]==self.calibration_source_index)[0]
                                x_new[calibration_element,embeddings[0].size(1)+n]=\
                                    torch.ones_like(x_new[calibration_element,embeddings[0].size(1)+n])*(getattr(self,'Theta_'+str(n))(x[i,-1].clone().reshape(-1,1)))               
            if nd_flag == 1:
                x_new = x_new.reshape(*xsize[:-1], -1)
        #################### Multiple baises (General Case) ####################################  
            if self.m_gp.startswith('multi'):
                mean_x = self.multi_mean(x_new,x_forward_raw).to(**self.tkwargs) 
            elif self.m_gp.startswith('neural_network'):
                mean_x = getattr(self, 'mean_module_NN_All')(x_new.clone().detach()).reshape(-1)
            else:
                mean_x = self.single_mean(x_new).to(**self.tkwargs) 

            covar_x = self.covar_module(x_new).to(**self.tkwargs)
            mean_x_sum+=mean_x
            Sigma_sum += covar_x.evaluate()+ torch.outer(mean_x, mean_x)

        # End of the loop for forward pasess ----> Compute ensemble mean and covariance
        k = Numper_of_pass
        ensemble_mean = mean_x_sum/k
        ensemble_covar = torch.zeros_like(Sigma_sum) 
        ensemble_covar= Sigma_sum/k
        ensemble_covar -= torch.outer(ensemble_mean, ensemble_mean)
        ensemble_covar=gpytorch.lazy.NonLazyTensor(ensemble_covar)
        Sigma_sum=0
        return MultivariateNormal(ensemble_mean,ensemble_covar)
    
    ################################################################ Mean Functions #####################################################################################
    
    def single_m_gp_register (self,size=1,m_gp_type='single_zero',wm='mean_module'):
        if m_gp_type in self.supported_singl_m_gp_functions:
            setattr(self,wm, LinearMean_with_prior(input_size=size, batch_shape=torch.Size([]), bias=False)) 
        elif m_gp_type.startswith('single_polynomial'):
            degree = int(m_gp_type.split('d')[-1])
            setattr(self,wm, LinearMean_with_prior(input_size=degree*(size), batch_shape=torch.Size([]), bias=True)) 
        elif m_gp_type=='single_constant':
            setattr(self,wm, gpytorch.means.ConstantMean(prior=NormalPrior(0.,1)) )
        elif m_gp_type=='single_zero':
            setattr(self,wm, gpytorch.means.ZeroMean())  
    
    def multi_m_gp_register(self,train_x,supported_multi_m_gp_functions,m_gp_ref):
        size=train_x.shape[1]
        if self.m_gp in supported_multi_m_gp_functions:
            for i in range(self.num_sources +1):
                if i==0:
                    m_gp_type= 'single_'+m_gp_ref
                else:
                    m_gp_type='single'+self.m_gp[8:]
                self.single_m_gp_register(size,m_gp_type=m_gp_type,wm='mean_module_'+str(i))

    def single_mean(self, x):
        m_gp_type=self.m_gp
        supported_functions = ['sin', 'cos', 'exp', 'log', 'tan', 'asin', 'acos', 'atan', 
                               'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'sqrt', 
                               'abs', 'ceil', 'floor', 'round']
        
        if m_gp_type in supported_functions:
            # Dynamically call the PyTorch function m_gpd on m_gp_type
            transformed_x = getattr(torch, m_gp_type)(x.clone()).float()
        elif m_gp_type.startswith('polynomial-d'):
            degree = int(m_gp_type.split('d')[-1])
            transformed_x = x.clone().double()
            polynomial_terms = [transformed_x.pow(n).float() for n in range(1, degree + 1)]
            transformed_x = torch.cat(polynomial_terms, dim=1)
        else:
            # Default case
            transformed_x = x.float().clone()
        mean_x = getattr(self, 'mean_module')(transformed_x)
        return mean_x

    def multi_mean(self,x,x_forward_raw):
        mean_x=torch.zeros_like(x[:,-1])
        if self.m_gp=='multiple_constant':
            for i in range(len(mean_x)):
                qq=int(x_forward_raw[i,-1])                        
                mean_x[i] = getattr(self, 'mean_module_' + str(qq))(x_forward_raw[i,:].clone().float().reshape(1,-1))
        elif self.m_gp=='multiple_polynomial':
            for i in range(len(mean_x)):
                qq=int(x_forward_raw[i,-1])
                # mean_x[i]=getattr(self,'mean_module_'+str(qq))(torch.cat((torch.tensor((x[i,-1].clone().double().reshape(-1,1).float())**2),torch.tensor(x[i,-1].clone().double()).reshape(-1,1).float()),1))
                mean_x[i] = getattr(self, 'mean_module_' + str(qq))(torch.cat((x_forward_raw.clone().detach().double().reshape(-1, 1).float() ** 2,
                        x[i, -1].clone().detach().double().reshape(-1, 1).float()),1))
        
        elif self.m_gp=='neural_network':
            mean_x = getattr(self, 'mean_module_NN_All')(x.clone()).reshape(-1)
        return mean_x 

    ################################################################ Fit #####################################################################################
    def fit(self,add_prior:bool=True,num_restarts:int=64,theta0_list:Optional[List[np.ndarray]]=None,jac:bool=True,
            options:Dict={},n_jobs:int=-1,method = 'L-BFGS-B',constraint=False,bounds=False,regularization_parameter:List[int]=[0,0],optim_type='scipy'):
        print("## Learning the model's parameters has started ##")

        if self.tkwargs['device'].type == 'cuda':
            if optim_type == 'adam_torch':
                fit_model_torch(model=self,
                                model_param_groups=None,
                                lr_default=0.01,
                                num_iter=100,
                                num_restarts=64,
                                break_steps=50)
            else:
                # Issue a warning and proceed with the 'adam_torch' optimizer
                warnings.warn('The model is built to run on CUDA (GPU), but the current optimization type is invalid for this configuration. So, the optimizer is now using adam_torch to train the model.')
                fit_model_torch(model=self.to(**self.tkwargs),
                                model_param_groups=None,
                                lr_default=0.01,
                                num_iter=100,
                                num_restarts=4,
                                break_steps=50)
        else:
            if optim_type=='scipy':
                fit_model_scipy (self,add_prior,num_restarts,theta0_list,jac, options,n_jobs,method ,constraint,bounds,regularization_parameter)
            elif optim_type=='continuation':
                fit_model_continuation(model=self,add_prior=add_prior,
                num_restarts=num_restarts,criterion='NLL',
                initial_noise_var=1,
                red_factor=math.sqrt(10),
                options=options,
                n_jobs= n_jobs,
                accuracy = 1e-2,
                method = method,
                constraint=constraint,
                regularization_parameter=regularization_parameter,
                bounds=bounds
                )
            elif optim_type=='adam_torch':
                fit_model_torch (model=self,
                    model_param_groups=None,
                    lr_default=0.01,
                    num_iter=100,
                    num_restarts=num_restarts,
                    break_steps= 50)
            else:
                raise ValueError(
                    'Invalid optim_type. You must choose one of the following: '
                    '"scipy" (default), "continuation", or "adam_torch".\n'
                    '- "scipy": Uses the SciPy library for optimization, suitable for most CPU-m_gpd computations.\n'
                    '- "continuation": A method designed for more complex optimization scenarios, potentially offering better results in most cases with higher computational cost.\n'
                    '- "adam_torch": Employs the Adam optimizer from the PyTorch library, optimized for GPU-m_gpd computations and large datasets.'
                )
        print("## Learning the model's parameters is successfully finished ##")




    def fill_nan_with_mean(self,train_x,cal_ID):
        # Check if there are any NaNs in the tensor
        if torch.isnan(train_x).any():
            if len(cal_ID)==0:
                print("There are NaN values in the data, which will be filled with column-wise mean values.")
            else:
                print("There are NaN values in the data, which will be estimated in calibration process")
            # Compute the mean of non-NaN elements column-wise
            col_means = torch.nanmean(train_x, dim=0)
            # Find indices where NaNs are located
            nan_indices = torch.isnan(train_x)
            # Replace NaNs with the corresponding column-wise mean
            train_x[nan_indices] = col_means.repeat(train_x.shape[0], 1)[nan_indices]

        return train_x
    ############################  Prediction and Visualization  ###############################
    
    def predict(self, Xtest,return_std=True, include_noise = True):
        Xtest=data_type_check(Xtest)
        with torch.no_grad():
            return super().predict(Xtest, return_std = return_std, include_noise= include_noise)
    
    def predict_with_grad(self, Xtest,return_std=True, include_noise = True):
        Xtest=data_type_check(Xtest)
        return super().predict(Xtest, return_std = return_std, include_noise= include_noise)
    
    def noise_value(self):
        noise = self.likelihood.noise_covar.noise.detach() * self.y_std**2
        return noise

    def score(self, Xtest, ytest, plot_MSE = True, title = None, seperate_levels = False):
        Xtest=data_type_check(Xtest)
        ytest=data_type_check(ytest)
        ytest=ytest.reshape(-1).to(self.tkwargs['device'])
        Xtest=Xtest.to(self.tkwargs['device'])
        plt.rcParams.update({'font.size': 14})
        ypred = self.predict(Xtest, return_std=False)
        mse = ((ytest.reshape(-1)-ypred)**2).mean()
        print('################MSE######################')
        print(f'MSE = {mse:.5f}')
        print('################Noise####################')
        noise = self.likelihood.noise_covar.noise.detach() * self.y_std**2
        
        print(f'The estimated noise parameter (varaince) is {noise}')
        print(f'The estimated noise std is {np.sqrt(noise.cpu())}')
        print('#########################################')

        if plot_MSE:
            _ = plt.figure(figsize=(8,6))
            _ = plt.plot(ytest.cpu().numpy(), ypred.cpu().numpy(), 'ro', label = 'Data')
            _ = plt.plot(ytest.cpu().numpy(), ytest.cpu().numpy(), 'b', label = 'MSE = ' + str(np.round(mse.detach().item(),3)))
            _ = plt.xlabel(r'Y_True')
            _ = plt.ylabel(r'Y_predict')
            _ = plt.legend()
            if title is not None:
                _ = plt.title(title)

        if seperate_levels and len(self.qual_index_list) > 0:
            for i in range(self.num_levels_per_var[0]):
                index = torch.where(Xtest[:,self.qual_index_list] == i)[0]
                _ = self.score(Xtest[index,...], ytest[index], 
                    plot_MSE=True, title = 'results' + ' Only Source ' + str(i), seperate_levels=False)
        return ypred

    def plot_xy(self, Xtest, ytest, input_column):
        Xtest=data_type_check(Xtest)
        ytest=data_type_check(ytest)
        
        ytest=ytest.reshape(-1).to(**self.tkwargs)
        Xtest=Xtest.to(**self.tkwargs)
        if len(input_column) > 2:
            raise ValueError("Visualization can only be done for one or two input versions.")

        plt.rcParams.update({'font.size': 14})
        ypred = self.predict(Xtest, return_std=False)

        if len(input_column) == 1:
            plt.scatter(Xtest[:,input_column[0]].cpu().numpy(), ypred.cpu().numpy(),marker= 'o', label='Prediction')
            plt.xlabel('Input: ' + str(input_column[0]))
            plt.ylabel('Output')
        elif len(input_column) == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(Xtest[:,input_column[0]].cpu().numpy(), Xtest[input_column[1]].cpu().numpy(), ytest.cpu().numpy(), c='r', marker='o', label='Ground Truth')
            ax.scatter(Xtest[:,input_column[0]].cpu().numpy(), Xtest[input_column[1]].cpu().numpy(), ypred.cpu().numpy(), c='b', marker='^', label='Prediction')
            ax.set_xlabel('Input: ' + str(input_column[0]))
            ax.set_ylabel('Input: ' + str(input_column[1]))
            ax.set_zlabel('Output')

        plt.title('Input(s): ' + str(input_column) + ' versus Output')
        plt.legend()
        plt.show()



    def plot_predict_xy(self, var_input, value_range, val_fixed_inputs, num_points=100):
        """
        Plots the prediction of the model for a specified input variable while keeping other inputs fixed.

        Parameters:
        - var_input (int): The index of the input variable to vary.
        - value_range (list or tuple): The range [min, max] over which to vary var_input.
        - val_fixed_inputs (dict): Dictionary with column indices as keys and fixed values as values for all input variables except var_input.
        - num_points (int): The number of points in the linspace for var_input. Default is 100.

        Raises:
        - ValueError: If var_input is not a valid index for the input variables.
        """

        # Check if var_input is a valid index
        if not (0 <= var_input < len(val_fixed_inputs)+1):
            raise ValueError("var_input must be a valid index of the input variables.")

        # Define the number of points in the linspace
        var_input_values = np.linspace(value_range[0], value_range[1], num_points)
        
        # Initialize Xtest with the appropriate shape
        Xtest = np.zeros((num_points, len(val_fixed_inputs)+1))

        # Fill Xtest with fixed values
        for i in range(Xtest.shape[1]):
            if i == var_input:
                Xtest[:, i] = var_input_values
            elif i in val_fixed_inputs:
                Xtest[:, i] = val_fixed_inputs[i]
            else:
                raise ValueError(f"Column index {i} is not specified in val_fixed_inputs and is not the variable input index.")

        Xtest=torch.tensor(Xtest)        
        # Make predictions
        ypred = self.predict(Xtest, return_std=False)
        # Plotting
        plt.plot(Xtest[:, var_input].cpu().numpy(), ypred.cpu().numpy(), label='Prediction')
        plt.xlabel('Input ' + str(var_input))
        plt.ylabel('Output')
        plt.title('Input: ' + str(var_input) + ' versus Output')
        plt.legend()
        plt.show()

    def plot_xy_print_params(self, Xtest, ytest, Xtrain, ytrain, model):
            Xtest=data_type_check(Xtest)
            ytest=data_type_check(ytest)
            ytest=ytest.reshape(-1).to(self.tkwargs['device'])
            Xtest=Xtest.to(self.tkwargs['device'])
            Xtest, indices = torch.sort(Xtest, dim = 0)
            ytest = ytest[indices]

            mean_pred, std_pred = model.predict(Xtest.to(**self.tkwargs), return_std=True)
            confidence_interval = 1.96 * std_pred

            plt.rcParams.update({'font.size': 14})
            plt.scatter(Xtrain, ytrain, marker='x', color='red')
            plt.plot(Xtest, ytest,color='black', linewidth=4.0, label='Exact')
            plt.plot(Xtest, mean_pred.cpu(), color='green', linestyle='dashed', linewidth=4.0, label = 'Predicted')
            plt.fill_between(Xtest.squeeze(), mean_pred.cpu() - confidence_interval.cpu(), mean_pred.cpu() + confidence_interval.cpu(), color='lightblue', alpha=0.7, label='95% CI')
            plt.scatter(Xtrain, ytrain, s=100, marker='x', color='red', label='Training data')
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
            title = r"$\hat{\beta}$ = " + f"{model.mean_module.constant.item():.3f}" + r", $\hat{\omega}$ = " + f"{model.covar_module.m_gp_kernel.raw_lengthscale.data.item():.3e}"+ r", $\hat{\delta}$ = " + f"{model.noise_value().item():.3e}"
            plt.title(title, fontsize = 15, loc="center")
            plt.legend()
            plt.show()
            
    def evaluation_2(self,Xtest,ytest,n_FP=1):
        Xtest=data_type_check(Xtest)
        ytest=data_type_check(ytest)
        
        ytest=ytest.reshape(-1).to(self.tkwargs['device'])
        Xtest=Xtest.to(self.tkwargs['device'])
        self.eval()
        likelihood=self.likelihood
        likelihood.fidel_indices=self.train_inputs[0][:,-1]
        output=self(Xtest)
        likelihood.fidel_indices=Xtest[:,-1]
        ytest_sc = (ytest-self.y_min)/self.y_std
        mean_temp=[]
        var_temp=[]
        for i in range (n_FP):
            with torch.no_grad():
                trained_pred_dist = likelihood(output)
                mean_temp.append(trained_pred_dist.mean)
                var_temp.append(trained_pred_dist.variance)
            
        sum_list = [mean**2 + var for mean, var in zip(mean_temp, var_temp)]
        sum_tensors = sum(sum_list)/n_FP
        mean_ensamble=sum(mean_temp)/n_FP
        var_ensamble=sum_tensors -mean_ensamble**2
        std_ensamble=var_ensamble.sqrt()
        mu_low, mu_up=mean_ensamble-1.96*std_ensamble, mean_ensamble+1.96*std_ensamble
        final_mse=((ytest_sc.reshape(-1)-mean_ensamble)**2).mean()
        def interval_score(y_true,mu_low, mu_up, alpha = 0.05):
            out = mu_up - mu_low
            out += (y_true > mu_up)* 2/alpha * (y_true - mu_up)
            out += (y_true <mu_low)* 2/alpha * (mu_low - y_true)
            return out
        IS=interval_score(ytest_sc,mu_low, mu_up, alpha = 0.05).mean()
        NIS=IS*torch.abs(self.y_std)/ytest.std()
        NRMSE=torch.sqrt((final_mse*(self.y_std)**2)/ytest.std()**2)    
        table_data = [
        ['NRMSE', NRMSE],
        ['NIS', NIS],
        ]
        # Print the table
        table = tabulate(table_data, headers=['Metric', 'Value'], tablefmt='fancy_grid', colalign=("left", "left"))
        print(table)
        # return NIS, NRMSE


    def rearrange_one_hot(self,tensor):
        # Find the indices that sort each row
        sorted_indices = torch.argsort(tensor, dim=1, descending=True)
        # Generate a new tensor of zeros with the same shape
        new_tensor = torch.zeros_like(tensor)
        # Place '1's in the appropriate positions m_gpd on the sorted indices
        for i in range(tensor.size(0)):
            new_tensor[i, sorted_indices[i, 0]] = 1

        return torch.flip(new_tensor, dims=[0])
    def visualize_latent(self,type='cat',rpearts=500):
        if self.embedding_type=='deterministic':
            if len(self.qual_kernel_columns) > 0:
                for i in range(len(self.qual_kernel_columns)):
                    zeta = self.zeta[i]
                    dimm=zeta.size()[0]
                    zeta_epsilon=torch.normal(mean=0,std=1,size=[dimm,2])

                    A = getattr(self,'A_matrix')
                    positions = A[i](x=zeta.float().to(**self.tkwargs))
                    level = torch.max(self.perm[i]+1, axis = 0)[0].tolist()

                    perm = self.perm[i]
                    plot_sep(type=type,positions = positions, levels = level, perm = perm, constraints_flag=False)
        elif self.embedding_type=='probabilistic':
            for i in range(len(self.qual_kernel_columns)):
                temp= self.transform_categorical(x=self.train_inputs[0][:,self.qual_kernel_columns[i]].clone().type(torch.int64).to(self.tkwargs['device']), perm_dict = self.perm_dict[i], zeta = self.zeta[i])
            unique_rows, indices = torch.unique(temp, dim=0, return_inverse=True)
            xp=self.rearrange_one_hot(unique_rows)
            z_p_list =[]
            label=[]
            epsilon=torch.normal(mean=0,std=1,size=[rpearts,2])
            for i in range(self.num_levels_per_var[0]):
                x_0=xp[i]
                x_0=x_0.repeat(rpearts, 1)
                z_p = getattr(self, 'A_matrix')(x=x_0, epsilon=epsilon)
                z_p_list.append(z_p)
                label.append(i*torch.ones_like(z_p))
            z_p_all = torch.cat(z_p_list, dim=0)
            label_ground_truth=torch.cat(label, dim=0)
            #########################
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 25
            # plt.rcParams['figure.dpi']=150
            tab20 = plt.get_cmap('tab10')
            colors = tab20.colors
            colors=['deeppink','gold','darkorange','gray','orangered']
            plt.figure(figsize=(8,6))

            # Assuming z_p_all is a torch.Tensor
            z_p_all_np = z_p_all.detach().numpy()
            unique_labels = np.unique(label_ground_truth)
            markers = ['X','o','s',"v", 'p']

            for idx, label in enumerate(unique_labels):
                mask = (label_ground_truth == label)
                plt.scatter(z_p_all_np[mask[:,0], 0], z_p_all_np[mask[:,0], 1], 
                            c=colors[idx], 
                            marker=markers[idx], 
                            alpha=.6,
                            s=250, 
                            label=f'Label {label}')

            # Create the legend and get the legend handles and labels
            legend=[ 'HF', 'LF1','LF2','LF3']
            plt.xlabel(r'$z_1$',labelpad=0,rotation=0,usetex=True)
            plt.ylabel(r'$z_2$',labelpad=14,rotation=0,usetex=True)
            plt.tight_layout()
            plt.show()

    
    def evaluation(self,Xtest,ytest):
        Xtest=data_type_check(Xtest)
        ytest=data_type_check(ytest)
        self.eval()
        ytest=ytest.reshape(-1).to(self.tkwargs['device'])
        Xtest=Xtest.to(self.tkwargs['device'])
        ytest=ytest.reshape(-1)
        Xtest=Xtest
        likelihood=self.likelihood
        ytest_sc = (ytest-self.y_min)/self.y_std

        with torch.no_grad():
            trained_pred_dist = likelihood(self(Xtest))
        # Negative Log Predictive Density (NLPD)
        final_nlpd = gpytorch.metrics.negative_log_predictive_density(trained_pred_dist,ytest_sc)
        
        # Mean Squared Error (MSE)
        final_mse = gpytorch.metrics.mean_squared_error(trained_pred_dist, ytest_sc, squared=True)
        # Mean Absolute Error (MAE)
        final_mae = gpytorch.metrics.mean_absolute_error(trained_pred_dist, ytest_sc)
        def interval_score(y_true, trained_pred_dist, alpha = 0.05):
            mu_low, mu_up = trained_pred_dist.confidence_region()
            out = mu_up - mu_low
            out += (y_true > mu_up)* 2/alpha * (y_true - mu_up)
            out += (y_true <mu_low)* 2/alpha * (mu_low - y_true)
            return out
        IS=interval_score(ytest_sc, trained_pred_dist, alpha = 0.05).mean()

        ## back to the original scale:
        final_mse=final_mse*(self.y_std)**2
        final_mae=final_mae*torch.abs(self.y_std)
        IS=IS*torch.abs(self.y_std)
        ###    
        RRMSE=torch.sqrt(final_mse/torch.var(ytest))
        table_data = [
            ['Negative Log-Likelihood (NLL)', final_nlpd],
            ['Mean Squared Error (MSE)', final_mse],
            ['Mean Absolute Error  (MAE)', final_mae],
            ['Relative Root Mean Square Error (RRMSE)', RRMSE],
            ['Interval Score (IS)', IS]
        ]
        # Print the table
        table = tabulate(table_data, headers=['Metric', 'Value'], tablefmt='fancy_grid', colalign=("left", "left"))
        print(table)

    def calibration_result(self,mean_train,std_train):
        self.calibration_id
        n_s=0
        for n in self.calibration_id:
            n_s+=1
            if self.calibration_type=='probabilistic':
                mean= (getattr(self,'Theta_'+str(n)).weights*std_train[n] +mean_train[n])[0].detach().numpy()
                STD= torch.abs((getattr(self,'Theta_'+str(n)).bias)*std_train[n])[0].detach().numpy()
                print("For Calibration parameter Theta_"+str(n)+ " Estimated Mean is "  + str(mean)+" and Estimated STD is "  + str(STD))
                x = np.linspace(mean-5*STD,mean+5*STD, 1000)
                pdf_values = norm.pdf(x, mean, STD).squeeze()
                plt.figure()
                plt.plot(x, pdf_values, label='Zeta_'+str(n))
                plt.title(r'$\mathit{\hat{\Theta}}_{' + str(n) + '}$')  # Italic LaTeX styled title with hat only on Theta
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.grid(True) 

            else:
                xx=torch.where(self.train_inputs[0][:,-1]==0)[0][0]
                GT=self.train_inputs[0][xx,n]*std_train[n] +mean_train[n]
                print("=================GP + Results===================")
                print("Estimated Calibration parameter for Zeta_"+str(n_s)+ " is "  + str((getattr(self,'Theta_'+str(n)).constant.detach()*std_train[n] +mean_train[n])))

    @classmethod
    def show(cls):
        plt.show()
        
    def get_params(self, name = None):
        params = {}
        print('###################Parameters###########################')
        for n, value in self.named_parameters():
             params[n] = value
        if name is None:
            print(params)
            return params
        else:
            if name == 'Mean':
                key = 'mean_module.constant'
            elif name == 'Sigma':
                key = 'covar_module.raw_outputscale'
            elif name == 'Noise':
                key = 'likelihood.noise_covar.raw_noise'
            elif name == 'Omega':
                for n in params.keys():
                    if 'raw_lengthscale' in n and params[n].numel() > 1:
                        key = n
            print(params[key])
            return params[key]
    

    def sample_y(self, size = 1, X = None, plot = False):
        if X == None:
            X = self.train_inputs[0]
        
        self.eval()
        out = self.likelihood(self(X))
        draws = out.sample(sample_shape = torch.Size([size]))
        index = np.argsort(out.loc.detach().numpy())
        if plot:
            _ = plt.figure(figsize=(12,6))
            _ = plt.scatter(list(range(len(X))), out.loc.detach().numpy()[index], color = 'red', s = 20, marker = 'o')
            _ = plt.scatter(np.repeat(np.arange(len(X)).reshape(1,-1), size, axis = 0), 
                draws.detach().numpy()[:,index], color = 'blue', s = 1, alpha = 0.5, marker = '.')
        return draws

    def get_latent_space(self):
        if len(self.qual_index_list) > 0:
            zeta = torch.tensor(self.zeta, dtype = torch.float64).to(**self.tkwargs)
            positions = self.nn_model(zeta)
            return positions.detach()
        else:
            print('No categorical Variable, No latent positions')
            return None



    def LMMAPPING(self, num_features:int, type = 'Linear',embedding_dim = 2):

        if type == 'Linear':
            in_feature = num_features
            out_feature = embedding_dim
            lm = torch.nn.Linear(in_feature, out_feature, bias = False)
            return lm

        else:
            raise ValueError('Only Linear type for now')    

    def zeta_matrix(self,
        num_levels:int,
        embedding_dim:int,
        batch_shape=torch.Size()
    ) -> None:

        if any([i == 1 for i in num_levels]):
            raise ValueError('Categorical variable has only one level!')

        if embedding_dim == 1:
            raise RuntimeWarning('1D latent variables are difficult to optimize!')
        
        for level in num_levels:
            if embedding_dim > level - 0:
                embedding_dim = min(embedding_dim, level-1)
                raise RuntimeWarning(
                    'The LV dimension can atmost be num_levels-1. '
                    'Setting it to %s in place of %s' %(level-1,embedding_dim)
                )
    
        from itertools import product
        levels = []
        for l in num_levels:
            levels.append(torch.arange(l))

        perm = list(product(*levels))
        perm = torch.tensor(perm, dtype=torch.int64)

        #-------------Mapping-------------------------
        perm_dic = {}
        for i, row in enumerate(perm):
            temp = str(row.tolist())
            if temp not in perm_dic.keys():
                perm_dic[temp] = i

        #-------------One_hot_encoding------------------
        for ii in range(perm.shape[-1]):
            if perm[...,ii].min() != 0:
                perm[...,ii] -= perm[...,ii].min()
            
        perm_one_hot = []
        for i in range(perm.size()[1]):
            perm_one_hot.append( torch.nn.functional.one_hot(perm[:,i]) )

        perm_one_hot = torch.concat(perm_one_hot, axis=1)

        return perm_one_hot, perm, perm_dic

    #################################### transformation functions####################################

    def transform_categorical(self, x:torch.Tensor,perm_dict = [], zeta = []) -> None:
        if x.dim() == 1:
            x = x.reshape(-1,1)
        # categorical should start from 0
        if self.training == False:
            x = torch.tensor(setlevels(x))
        if self.encoding_type == 'one-hot':
            try:
                index = [perm_dict[str(row.tolist())] for row in x]
            except KeyError:
                raise ValueError(
                    "The categorical input (or source indices) are not defined properly. "
                    "They should be integer values starting from zero. To solve the issue, "
                    "you can use the 'setlevels' function, which is a preprocessing function."
                )
            if x.dim() == 1:
                x = x.reshape(len(x),)

            return zeta[index,:]  

    def transform_categorical_random_varible_for_latent(self,x_raw, x:torch.Tensor,perm_dict = [], zeta = []) -> None:
        
        dimm=zeta.size()[0]
        # zeta=torch.normal(0,1,size=[dimm,2])

        self.random_zeta.append(torch.normal(mean=0,std=1,size=[dimm,2]))
        
        if x_raw.requires_grad:
            random_zeta_appy=self.random_zeta[-1]
        else:
            random_zeta_appy=self.random_zeta[0]
        
        if x.dim() == 1:
            x = x.reshape(-1,1)
        # categorical should start from 0
        if self.training == False:
            x = setlevels(x)
        if self.encoding_type == 'one-hot':
            index = [perm_dict[str(row.tolist())] for row in x]

            if x.dim() == 1:
                x = x.reshape(len(x),)

            return random_zeta_appy[index,:]  
        
    def final_transform_categorical_random_varible_for_latent(self,x_raw, x:torch.Tensor,perm_dict = [], zeta = []) -> None:
        
        if x.dim() == 1:
            x = x.reshape(-1,1)
        if self.training == False:
            x = setlevels(x)
        if self.encoding_type == 'one-hot':
            index = [perm_dict[str(row.tolist())] for row in x]

            if x.dim() == 1:
                x = x.reshape(len(x),)

        dimm=zeta.size()[0]

        self.random_zeta.append(torch.normal(mean=0,std=1,size=[dimm,2]))
        
        if x_raw.requires_grad:
            random_zeta_appy=self.random_zeta[-1]
        else:
            if x_raw.size()[0]==400:
                random_zeta_appy=0*self.random_zeta[0]
            else: 
                random_zeta_appy=self.random_zeta[0]
        return random_zeta_appy[index,:]       
    

    def Sobol(self, N=10000):
        """

        This function calculates the sensitivity indecies for a function
        Inputs:
            self (GP_Model): The GP model (fitted by GP+) with p inputs and dy outputs.

            N: is the size of the Sobol sequence used for evaluating the indecies. Should be larger than 1e5 for accuracy.

        Outputs:
            S: Matrix of size dy-by-p of main sensitivity indecies. 
            ST: Matrix of size dy-by-p of total sensitivity indecies.
        """
        if N<1e5:
            warnings.warn('Increase N for accuracy!')

        p = self.train_inputs[0].shape[1] 
        dy = 1# self.train_targets.shape[1] 

        self.qual_index_list
        self.num_levels_per_var
        # sequence = torch.from_numpy( sobol_seq.i4_sobol_generate(2*p, N)).to(**self.tkwargs)
        sequence = torch.from_numpy( sobol_seq.i4_sobol_generate(2*p, N))
        def normalize_sobol_sequence(sequence, train_inputs,p):
            
            temp_1 = sequence[:,p:]
            temp_2 = sequence[:,:p]
            
            # Normalize the sequence
            mins = train_inputs.min(dim=0)[0]
            maxs = train_inputs.max(dim=0)[0]

            sequence_1= mins + (maxs - mins) * temp_1
            sequence_2= mins + (maxs - mins) * temp_2
            # Take care of categotrical inputes
            j=0
            for i in self.qual_index_list:
                temp_1[:,i]= temp_1[:,i]*(self.num_levels_per_var[j]-1)
                temp_2[:,i]=temp_2[:,i]*(self.num_levels_per_var[j]-1)
                sequence_1[:,i]=temp_1[:,i].round()
                sequence_2[:,i]=temp_2[:,i].round()
                j+=1
                return sequence_1,sequence_2
        A,B = normalize_sobol_sequence(sequence, self.train_inputs[0],p)

        # # A = A * (self.Y.max(axis=0) - self.Y.min(axis=0)) + self.Y.min(axis=0) ## Normalize genrated data

        # B = A[:,p:]
        # A = A[:,:p]
        
        AB = torch.zeros((N,p,p))
        for i in range(p):
            AB[:,:,i] = A
            AB[:,i,i] = B[:,i]
            
        FA = self.predict(A,return_std=False).detach().cpu().numpy().reshape(-1,1)

        FB = self.predict(B,return_std=False).detach().cpu().numpy().reshape(-1,1)

        FAB = np.zeros((N, p, dy))
        for i in range(p):
            temp = self.predict(AB[:, :, i],return_std=False).detach().cpu().numpy()
            FAB[:, i, :] = temp.reshape(-1,1)

        S = np.zeros((p, dy))
        ST = np.zeros((p, dy))

        for i in range(p):
            temp = FAB[:, i, :]
            S[i, :] = np.sum(FB * (temp - FA), axis=0) / N
            ST[i, :] = np.sum((FA - temp)**2, axis=0) / (2 * N)
            
        varY = np.var(np.concatenate([FA,FB]), axis=0)
        S = (S / varY).T
        ST = (ST / varY).T

        return S, ST

######################################################################## Other Classes Used in GP_Pluse  #####################################################
class FFNN(nn.Module):
    def __init__(self, GP_Plus, input_size, num_classes, layers,name):
        super(FFNN, self).__init__()
        self.hidden_num = len(layers)
        if self.hidden_num > 0:
            self.fci = nn.Linear(input_size, layers[0], bias=False) 
            GP_Plus.register_parameter(str(name)+'fci', self.fci.weight)
            GP_Plus.register_prior(name = 'latent_prior_fci', prior=gpytorch.priors.NormalPrior(0.,1), param_or_closure=str(name)+'fci')

            for i in range(1,self.hidden_num):
                setattr(self, 'h' + str(i), nn.Linear(layers[i-1], layers[i], bias=False))
                GP_Plus.register_parameter(str(name)+'h'+str(i), getattr(self, 'h' + str(i)).weight )
                GP_Plus.register_prior(name = 'latent_prior'+str(i), prior=gpytorch.priors.NormalPrior(0.,1), param_or_closure=str(name)+'h'+str(i))
            
            self.fce = nn.Linear(layers[-1], num_classes, bias= False)
            GP_Plus.register_parameter(str(name)+'fce', self.fce.weight)
            GP_Plus.register_prior(name = 'latent_prior_fce', prior=gpytorch.priors.NormalPrior(0.,1), param_or_closure=str(name)+'fce')
        else:
            self.fci = Linear_MAP(input_size, num_classes, bias = False)
            GP_Plus.register_parameter(name, self.fci.weight)
            GP_Plus.register_prior(name = 'latent_prior_'+name, prior=gpytorch.priors.NormalPrior(0,1) , param_or_closure=name)

    def forward(self, x, transform = lambda x: x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between and for that (since it has no parameters)
        I recommend using nn.functional (F)
        """
        if self.hidden_num > 0:
            x = torch.tanh(self.fci(x))
            for i in range(1,self.hidden_num):
                #x = F.relu(self.h(x))
                x = torch.tanh( getattr(self, 'h' + str(i))(x) )
            
            x = self.fce(x)
        else:
            #self.fci.weight.data = torch.sinh(self.fci.weight.data)
            x = self.fci(x, transform)
        return x
    
############################################
class FFNN_as_Mean(gpytorch.Module):
    def __init__(self, GP_Plus, input_size, num_classes, layers,name):
        super(FFNN_as_Mean, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.hidden_num = len(layers)
        if self.hidden_num > 0:
            self.fci = Linear_class(input_size, layers[0], bias=True, name='fci') 
            for i in range(1,self.hidden_num):
                setattr(self, 'h' + str(i), Linear_class(layers[i-1], layers[i], bias=True,name='h' + str(i)))
            
            self.fce = Linear_class(layers[-1], num_classes, bias=True,name='fce')
        else:
            self.fci = Linear_class(input_size, num_classes, bias=True, dtype = torch.float32,name='fci') #Linear_MAP(input_size, num_classes, bias = True)

    def forward(self, x, transform = lambda x: x):

        if self.hidden_num > 0:
            
            x = torch.tanh(self.fci(x))
            # x = self.dropout(x)
            # x = self.fci(x)
            for i in range(1,self.hidden_num):
                # x = torch.sigmoid( getattr(self, 'h' + str(i))(x) )
                # x =  getattr(self, 'h' + str(i))(x) 
                x = torch.tanh( getattr(self, 'h' + str(i))(x) )
                x = self.dropout(x)
            x = self.fce(x)
        else:
            #self.fci.weight.data = torch.sinh(self.fci.weight.data)
            x = self.fci(x)

        return x
    
############################################
class Linear_VAE(Mean):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    def __init__(self, in_features: int, out_features: int, bias: bool = True, name=None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear_VAE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name=str(name)
        self.register_parameter(name=str(self.name)+'weight',  parameter= Parameter(torch.empty((out_features, in_features), **factory_kwargs)))
        self.register_prior(name =str(self.name)+ 'prior_m_weight_fci', prior=gpytorch.priors.NormalPrior(0.,.2), param_or_closure=str(self.name)+'weight')

        if bias:

            self.register_parameter(name=str(self.name)+'bias',  parameter=Parameter(torch.empty(out_features, **factory_kwargs)))
            self.register_prior(name= str(self.name)+'prior_m_bias_fci', prior=gpytorch.priors.NormalPrior(0.,.05), param_or_closure=str(self.name)+'bias')
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:                                             

        init.kaiming_uniform_( getattr(self,str(self.name)+'weight'), a=math.sqrt(5))
        if getattr(self,str(self.name)+'bias') is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(getattr(self,str(self.name)+'weight'))
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(getattr(self,str(self.name)+'bias'), -bound, bound)

    def forward(self, input) -> Tensor:

        return F.linear(input.double(), getattr(self,str(self.name)+'weight').double(), getattr(self,str(self.name)+'bias').double())      ### Forced to Add .double() for NN in mean function

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

############################################
class LinearVariational(Mean):
    def __init__(self, batch_shape=torch.Size(),mean_prior=1,std_prior=1):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape,1,1)))
        self.register_prior(name = 'weights_prior', prior=gpytorch.priors.NormalPrior(mean_prior,1), param_or_closure='weights')
        self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1,1)))
        self.register_prior(name = 'bias_prior', prior=gpytorch.priors.NormalPrior(std_prior,1), param_or_closure='bias')

    def forward(self, epsilon):
        res = self.weights + (torch.abs(self.bias)) *epsilon
        return res

##########################################
class LinearMean_with_prior(Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        self.register_prior(name = 'weights_prior', prior=gpytorch.priors.NormalPrior(0.,.5), param_or_closure='weights')
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
            self.register_prior(name = 'bias_prior', prior=gpytorch.priors.NormalPrior(0.,.5), param_or_closure='bias')
        else:
            self.bias = None
    def forward(self, x):
        res = x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res
    
############################################
class Variational_Encoder(gpytorch.Module):
    def __init__(self, GP_Plus, input_size, num_classes, layers,name):
        super(Variational_Encoder, self).__init__() 
        self.dropout = nn.Dropout(0.2)
        self.hidden_num = len(layers)
        if self.hidden_num > 0:
            self.fci = Linear_VAE(input_size, layers[0], bias=True, name='fci') 
            for i in range(1,self.hidden_num):
                #self.h = nn.Linear(neuran[i-1], neuran[i])
                setattr(self, 'h' + str(i), Linear_VAE(layers[i-1], layers[i], bias=True,name='h' + str(i)))
            self.fce = Linear_VAE(layers[-1], num_classes, bias=True,name='fce')
        else:
            self.fci = Linear_VAE(input_size, num_classes, bias=True, dtype = torch.float32,name='fci') 

    def forward(self, x,epsilon, transform = lambda x: x):
        if self.hidden_num > 0:
            # x = torch.tanh(self.fci(x))
            x =self.fci(x)
            for i in range(1,self.hidden_num):
                # x = F.relu(self.h(x))
                x = torch.tanh( getattr(self, 'h' + str(i))(x) )
                # x = self.dropout(x)
            output = self.fce(x)

            epsilon_1, epsilon_2 = epsilon[:, 0:1], epsilon[:, 1:2]
            L22, L21, L11, Mu_2, Mu_1 = output[:, 0:1], output[:, 1:2], output[:, 2:3], output[:, 3:4], output[:, 4:5]
            # Optimized calculation using matrix operations
            X_1 = Mu_1 + 1*torch.abs(L11) * epsilon_1
            X_2 = Mu_2 + 1*L21 * epsilon_1 + 1*torch.abs(L22) * epsilon_2
            x = torch.cat((X_1,X_2),1)

        else:  
            output = self.fci(x)
            epsilon_1, epsilon_2 = epsilon[:, 0:1], epsilon[:, 1:2]
            L22, L21, L11, Mu_2, Mu_1 = output[:, 0:1], output[:, 1:2], output[:, 2:3], output[:, 3:4], output[:, 4:5]
            # calculation using matrix operations
            X_1 = Mu_1 + 1*torch.abs(L11) * epsilon_1
            X_2 = Mu_2 + 1*L21 * epsilon_1 + 1*torch.abs(L22) * epsilon_2
            x = torch.cat((X_1,X_2),1)
        return x 
    
############################################
class Linear_class(Mean):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, name=None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear_class, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.name=str(name)
        self.register_parameter(name=str(self.name)+'weight',  parameter= Parameter(torch.empty((out_features, in_features), **factory_kwargs)))
        self.register_prior(name =str(self.name)+ 'prior_m_weight_fci', prior=gpytorch.priors.NormalPrior(0.,0.01), param_or_closure=str(self.name)+'weight')
        if bias:
            self.register_parameter(name=str(self.name)+'bias',  parameter=Parameter(torch.empty(out_features, **factory_kwargs)))
            self.register_prior(name= str(self.name)+'prior_m_bias_fci', prior=gpytorch.priors.NormalPrior(0.,.001), param_or_closure=str(self.name)+'bias')
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:                                         
        init.kaiming_uniform_( getattr(self,str(self.name)+'weight'), a=math.sqrt(5))
        if getattr(self,str(self.name)+'bias') is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(getattr(self,str(self.name)+'weight'))
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(getattr(self,str(self.name)+'bias'), -bound, bound)

    def forward(self, input) -> Tensor:

        # return F.linear(input, getattr(self,str(self.name)+'weight').double(), getattr(self,str(self.name)+'bias').double())      ### Forced to Add .double() for NN in mean function
        return F.linear(input, getattr(self,str(self.name)+'weight'), getattr(self,str(self.name)+'bias'))      ### Forced to Add .double() for NN in mean function

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
############################################
class Linear_MAP(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        
    def forward(self, input, transform = lambda x: x):
        return F.linear(input,transform(self.weight), self.bias)