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

from distutils.log import error
from turtle import forward
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import _HomoskedasticNoiseBase
from sklearn.covariance import log_likelihood
import torch
from typing import Any, Optional
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor, ConstantDiagLazyTensor
from torch import Tensor

class Multifidelity_likelihood(_GaussianLikelihoodBase):

    def __init__(self, fidel_indices: Tensor,noise_indices: list = [1], noise_prior = None, noise_constraint = None, 
        learn_additional_noise = False, batch_shape = torch.Size(), **kwargs) -> None:
        # num_noises = len(noise_indices)
        num_noises = len(noise_indices)
        noise_covar = Multifidelity_noise(noise_prior=noise_prior, 
        noise_constraint=noise_constraint, batch_shape=batch_shape, num_noises = num_noises)
        super().__init__(noise_covar = noise_covar)


        self.fidel_indices = fidel_indices
        self.noise_indices = noise_indices

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)


    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        # This runs the forward method in noise class
        # Shape is not used any more.
        return self.noise_covar(*params, fidel_indices = self.fidel_indices, noise_indices = self.noise_indices)
    

    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)





class Multifidelity_noise(_HomoskedasticNoiseBase):
    def __init__(self, noise_prior=None, noise_constraint=None, batch_shape= torch.Size(), num_noises=1):
        super().__init__(noise_prior, noise_constraint, batch_shape, num_tasks= num_noises)


    def forward(self, *params: Any, shape: Optional[torch.Size] = None, 
        fidel_indices: Tensor, noise_indices: list, **kwargs: Any) -> DiagLazyTensor:
        
        """_summary_
        
        Indices are very important here and is coming from input and shows the multifidelity level of each input data

        Raises:
            ValueError: _description_
        """

        if len(fidel_indices) ==0 or fidel_indices is None:
            raise ValueError('You need to specify a list of indices for noise such as [1,3]')
        # This contains a list of diagonal matrices with defined noise. Crates [batch * 1 * noise_size * n * n]
        covar = super().forward(*params, shape= fidel_indices.shape, **kwargs)

        if covar.dim() > 2:
            if covar.shape[1] is not len(noise_indices):
                raise ValueError('Something is wrong, number of noise and indices are not the same')

        if covar.dim() == 4: # no batch
            covar = covar.squeeze(0)
        elif covar.dim() == 5: # for batch
            covar = covar.squeeze(1) 


        # This part is for categorical_indices
        temp = ConstantDiagLazyTensor(torch.tensor([0.0]), len(fidel_indices))
        temp = temp.to(dtype=covar.dtype, device=covar.device)
        for i in range(len(noise_indices)):
            if i ==0:
                diag = DiagLazyTensor( (fidel_indices == noise_indices[i]) )#.type(torch.int32)
                if covar.dim() == 4: # batch
                    temp += diag * covar[:,i,...]
                
                elif covar.dim() == 3: # 
                    temp += diag * covar[i,...]

                elif covar.dim() == 2:
                    temp += diag * covar
                else:
                    raise ValueError('Covar is 1D? why?')
            else :
                diag = DiagLazyTensor( (fidel_indices == noise_indices[i]).type(torch.int32) )
                if covar.dim() == 4: # batch
                    temp += diag * covar[:,i,...]
                
                elif covar.dim() == 3: # 
                    temp += diag * covar[i,...]

                elif covar.dim() == 2:
                    temp += diag * covar
                else:
                    raise ValueError('Covar is 1D? why?')
        

        

        return temp


'''
if __name__ == '__main__':
    multi_likelihood = Multifidelity_likelihood(num_noises=3)
    multi_noise = Multifidelity_noise(num_noises=2)
    print(multi_likelihood)
    fidel_indices = torch.tensor([1, 3, 2, 3, 2, 1, 3])
    noise_indices = [1, 3]
    covar = multi_noise(fidel_indices = fidel_indices, noise_indices = noise_indices)
    aa = 1
'''