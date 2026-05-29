from functools import partialmethod

from gpytorch.kernels import MaternKernel


class Matern32Kernel(MaternKernel):
    __init__ = partialmethod(MaternKernel.__init__,nu=1.5)

class Matern52Kernel(MaternKernel):
    __init__ = partialmethod(MaternKernel.__init__,nu=2.5)