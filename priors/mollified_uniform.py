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
from torch.distributions import constraints,Uniform,Normal
from torch.distributions.utils import broadcast_all
from numbers import Number

class MollifiedUniformPrior(Prior):
    r"""Uniform distribution that is differentiable everywhere 

    This is an approximation to the Uniform distribution which maintains differentiability by placing a 
    Gaussian distribution over points away from the original support. The density for a single dimension is

    .. math::

        p(x) &= \frac{M}{\sqrt{2\pi\sigma^2}}\exp\left[-\frac{d(x)^2}{2\sigma^2}\right], \\
        d(x) &= \begin{cases}
            a-x & x < a \\
            0 & a\leq x < b\\
            x-b & x \geq b
        \end{cases}

    :param a: lower range (inclusive)
    :type a: float or torch.Tensor

    :param b: upper range (exclusive)
    :type b: float or torch.Tensor

    :param tail_sigma: Standard deviation of the Gaussian distributions on the tails. Lower values make the 
        approximation closer to Uniform, but may increase the optimization effort. Defaults to 0.1
    :type tail_sigma: float or torch.Tensor, optional

    .. note::
        The `rsample` method for this distribution returns uniformly distributed samples from the interval `[a,b)`, 
        and **not** from the Mollified distribution. The `log_prob` method, however, returns the correct probability.
        The purpose of the priors in this package is to generate initial starting points for optimizing
        hyperparameters and for MAP estimation. 
    """
    arg_constraints = {'a':constraints.real,'b':constraints.real,'tail_sigma':constraints.positive}
    support = constraints.real
    def __init__(self,a,b,tail_sigma=0.1):
        self.a,self.b,self.tail_sigma = broadcast_all(a,b,tail_sigma)
        
        if isinstance(a,Number) or isinstance(b,Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()

        super().__init__(batch_shape)
    
    @property
    def mean(self):
        return (self.a+self.b)/2
    
    @property
    def _half_range(self):
        return (self.b-self.a)/2

    @property
    def _log_normalization_constant(self):
        return -torch.log(1+(self.b-self.a)/(math.sqrt(2*math.pi)*self.tail_sigma))

    def log_prob(self,X):
        # expression preserving gradients under automatic differentiation
        tail_dist = ((X-self.mean).abs()-self._half_range).clamp(min=0)
        return Normal(loc=torch.zeros_like(self.a),scale=self.tail_sigma).log_prob(tail_dist)+self._log_normalization_constant
    
    def rsample(self,sample_shape=torch.Size([])):
        return Uniform(self.a,self.b).rsample(sample_shape).to(self.a)

    def expand(self,expand_shape):
        batch_shape = torch.Size(expand_shape)
        return MollifiedUniformPrior(
            self.a.expand(batch_shape),
            self.b.expand(batch_shape),
            self.tail_sigma.expand(batch_shape)
        )
