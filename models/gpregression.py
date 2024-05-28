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
import gpytorch
import math
from gpytorch.models import ExactGP
from gpytorch import settings as gptsettings
from gpytorch.priors import NormalPrior,LogNormalPrior
from gpytorch.constraints import GreaterThan,Positive
from gpytorch.distributions import MultivariateNormal
from .. import kernels
from ..priors import LogHalfHorseshoePrior,MollifiedUniformPrior
from ..utils.transforms import softplus,inv_softplus
from typing import List,Tuple,Union
from gpplus.likelihoods_noise.multifidelity import Multifidelity_likelihood
from botorch.models.utils import gpt_posterior_settings
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel, GPyTorchModel
from botorch import settings
from botorch.models.utils import fantasize as fantasize_flag, validate_input_scaling
from botorch.sampling.samplers import MCSampler
from torch import Tensor
from typing import Any, Dict, List, Optional, Union


class GPR(ExactGP, GPyTorchModel):
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        correlation_kernel,
        noise_indices:List[int],
        fix_noise:bool=False,
        fix_noise_val:float=1e-5,
        lb_noise:float=1e-12,
    ) -> None:
        # check inputs
        if not torch.is_tensor(train_x):
            raise RuntimeError("'train_x' must be a tensor")
        if not torch.is_tensor(train_y):
            raise RuntimeError("'train_y' must be a tensor")

        if train_x.shape[0] != train_y.shape[0]:
            raise RuntimeError("Inputs and output have different number of observations")
        
        # initializing likelihood
        noise_constraint=GreaterThan(lb_noise,transform=torch.exp,inv_transform=torch.log)
        
        if len(noise_indices) == 0:

            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
        else:

            likelihood = Multifidelity_likelihood(noise_constraint=noise_constraint, noise_indices=noise_indices, fidel_indices=train_x[:,-1])
        y_min= train_y.min()
        y_std= train_y.max()-train_y.min()
        train_y_sc = (train_y-y_min)/y_std

        ExactGP.__init__(self, train_x,train_y_sc, likelihood)
        
        # registering mean and std of the raw response
        self.register_buffer('y_min',y_min)
        self.register_buffer('y_std',y_std)
        self.register_buffer('y_scaled',train_y_sc)

        self._num_outputs = 1

        # initializing and fixing noise
        # if noise is not None:
        #     self.likelihood.initialize(noise=noise)
        
        self.likelihood.register_prior('noise_prior',LogHalfHorseshoePrior(0.01,lb_noise),'raw_noise')
        if fix_noise:
            self.likelihood.raw_noise.requires_grad_(False)
            self.likelihood.noise_covar.noise =torch.tensor(fix_noise_val)

        if isinstance(correlation_kernel,str):
            try:
                correlation_kernel_class = getattr(kernels,correlation_kernel)
                correlation_kernel = correlation_kernel_class(
                    ard_num_dims = self.train_inputs[0].size(1),
                    lengthscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log),
                )
                correlation_kernel.register_prior(
                    'lengthscale_prior',MollifiedUniformPrior(math.log(0.1),math.log(10)),'raw_lengthscale'
                )
            except:
                raise RuntimeError(
                    "%s not an allowed kernel" % correlation_kernel
                )
        elif not isinstance(correlation_kernel,gpytorch.kernels.Kernel):
            raise RuntimeError(
                "specified correlation kernel is not a `gpytorch.kernels.Kernel` instance"
            )

        self.covar_module = kernels.ScaleKernel(
            base_kernel = correlation_kernel,
            outputscale_constraint=Positive(transform=softplus,inv_transform=inv_softplus),
        )
        # register priors
        self.covar_module.register_prior(
            'outputscale_prior',LogNormalPrior(1e-6,1.),'outputscale'
        )
    
    def forward(self,x:torch.Tensor)->MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x,covar_x)
    
    def predict(
        self,x:torch.Tensor,return_std:bool=False,include_noise:bool=False
    )-> Union[torch.Tensor,Tuple[torch.Tensor]]:

        self.eval()
        with gptsettings.fast_computations(log_prob=False):
            # determine if batched or not
            ndim = self.train_targets.ndim
            if ndim == 1:
                output = self(x)
            else:
                # for batched GPs 
                num_samples = self.train_targets.shape[0]
                output = self(x.unsqueeze(0).repeat(num_samples,1,1))
            self.fidel_indices=x[:,-1]
            if return_std and include_noise:
                # x=self.fidel_indices
                self.likelihood.fidel_indices=x[:,-1]   ### the fidelity indaces are alrrady set for training data, here we update them test data since 
                output = self.likelihood(output)

            out_mean = self.y_min + self.y_std*output.mean
            
            # standard deviation may not always be needed
            if return_std:
                out_std = output.variance.sqrt()*self.y_std
                return out_mean,out_std

            return out_mean

    def posterior(
        self,
        X,
        output_indices = None,
        observation_noise= True,
        posterior_transform= None,
        **kwargs,
    ):

        self.eval()
        with gpt_posterior_settings() and gptsettings.fast_computations(log_prob=False):
    
            if observation_noise:
                return GPyTorchPosterior(mvn = self.likelihood(self(X.double())))
            else:
                return GPyTorchPosterior(mvn = self(X.double()))
    
    def reset_parameters(self) -> None:
        """Reset parameters by sampling from prior
        """
        for _,module,prior,closure,setting_closure in self.named_priors():
            if not closure(module).requires_grad:
                continue
            setting_closure(module,prior.expand(closure(module).shape).sample().to(**self.tkwargs))


    def fantasize(
            self,
            X: Tensor,
            sampler: MCSampler,
            observation_noise: Union[bool, Tensor] = True,
            **kwargs: Any,
        ):
            r"""Constructs a fantasy model using a specified procedure.

            This method constructs a fantasy model by following these steps:
            1. Compute the model's posterior at `X`. If `observation_noise=True`, the posterior
            includes observation noise, which is determined as the mean of the observation
            noise in the training data. If `observation_noise` is a Tensor, it is used directly
            as the observation noise.
            2. Sample from this posterior using the provided `sampler` to create "fake" observations.
            3. Update (condition) the model with these new fake observations.

            Args:
                X: A Tensor of dimensions `batch_shape x n' x d`, where `d` represents the feature
                space dimension, `n'` is the number of points per batch, and `batch_shape`
                is the batch shape. This batch shape must be compatible with the model's
                existing batch shape.
                sampler: A sampler used for drawing samples from the model's posterior at `X`.
                observation_noise: A boolean or a Tensor. If True, the mean of the observation
                                noise from the training data is used in the posterior. If a
                                Tensor, it specifies the observation noise directly.

            Returns:
                A fantasy model, updated based on the sampled fake observations.
            """
            propagate_grads = kwargs.pop("propagate_grads", False)
            with fantasize_flag():
                with settings.propagate_grads(propagate_grads):
                    post_X = self.posterior(
                        X, observation_noise=observation_noise, **kwargs
                    )
                Y_fantasized = sampler(post_X)  # num_fantasies x batch_shape x n' x m
                # Use the mean of the previous noise values (TODO: be smarter here).
                # noise should be batch_shape x q x m when X is batch_shape x q x d, and
                # Y_fantasized is num_fantasies x batch_shape x q x m.
                noise_shape = Y_fantasized.shape[1:]
                noise = self.likelihood.noise.mean().expand(noise_shape)
                return self.condition_on_observations(
                    X=self.transform_inputs(X), Y=Y_fantasized, noise=noise
                )
