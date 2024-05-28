# Copyright © 2021 by Northwestern University.
# 
# LVGP-PyTorch is copyrighted by Northwestern University. It may be freely used 
# for educational and research purposes by  non-profit institutions and US government 
# agencies only. All other organizations may use LVGP-PyTorch for evaluation purposes 
# only, and any further uses will require prior written approval. This software may 
# not be sold or redistributed without prior written approval. Copies of the software 
# may be made by a user provided that copies are not sold or distributed, and provided 
# that copies are used under the same terms and conditions as agreed to in this 
# paragraph.
# 
# As research software, this code is provided on an "as is'' basis without warranty of 
# any kind, either expressed or implied. The downloading, or executing any part of this 
# software constitutes an implicit agreement to these terms. These terms and conditions 
# are subject to change at any time without prior notice.

import math
import torch
from gpytorch.priors import Prior
from torch.distributions import HalfCauchy,HalfNormal,constraints
from torch.distributions.utils import broadcast_all
from numbers import Number

class LogHalfHorseshoePrior(Prior):
    """Prior for the log-noise variance hyperparameter for GPs. 

    This is parameterized by `scale` and `lb`. `lb` is the lower bound on the noise variance. 
    The `scale` parameter is more important. The default value for `scale` - 0.01 - works well 
    for deterministic and low-noise situations. A larger value may be need in noisy sitations 
    and small training datasets. A larger scale implies more noisy data as the prior.

    To change the scale for this prior for a model to say 0.1,
        >>> model.likelihood.register(
        >>>     'noise_prior',LogHalfHorseshoePrior(0.1,model.likelihood.noise_prior.lb),
        >>>     'raw_noise'
        >>> )

    .. note::
        The `log_prob` method is only approximate and unnormalized. There is no closed form
        expression for the underlying horseshoe distribution. The lower and upper bounds on
        its' density are, however, known. Here, we use the same approximate density value that 
        the spearmint package uses.
    
    :param scale: scale parameter of the Horseshoe distribution
    :type scale: float or torch.Tensor

    :param lb: lower bound on the original scale. Defaults to 1e-6
    :type lb: float or torch.Tensor, optional
    """
    arg_constraints = {"scale": constraints.positive,"lb":constraints.positive}
    support = constraints.real
    def __init__(self, scale, lb=1e-6,validate_args=None):
        self.scale,self.lb = broadcast_all(scale,lb)
        if isinstance(scale,Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.scale.size()
        super().__init__(batch_shape,validate_args=validate_args)
    
    def transform(self, x):
        return self.lb + torch.exp(x)

    def log_prob(self, X):
        # first term is the density in the original scale
        # the second term is for the transformation
        return torch.log(torch.log(1+3*(self.scale / self.transform(X)) ** 2))+ X

    def rsample(self, sample_shape=torch.Size([])):
        local_shrinkage = HalfCauchy(1).rsample(self.scale.shape).to(self.lb)
        param_sample = HalfNormal(local_shrinkage * self.scale).rsample(sample_shape).to(self.lb)
        if len(self.lb) > 1:
            param_sample[param_sample<self.lb[0]] = self.lb[0]
        else:
            param_sample[param_sample<self.lb] = self.lb
        return param_sample.log()

    def expand(self,expand_shape, _instance=None):
        batch_shape = torch.Size(expand_shape)
        return LogHalfHorseshoePrior(self.scale.expand(batch_shape))