from gpytorch.kernels import LinearKernel, RBFKernel, ScaleKernel

from .matern import Matern32Kernel, Matern52Kernel
from .Rough_RBF import Rough_RBF

#### Amin Added  To Test wighted_RBF ####
from .wighted_RBF import wighted_RBF
from .wighted_RBF_Z import wighted_RBF_Z
