from gpytorch.kernels import RBFKernel
from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode
import torch

def postprocess_rbf(dist_mat):
    return dist_mat.div_(-1).exp_()


class wighted_RBF_Z(RBFKernel):
    """_summary_

    Args:
        RBFKernel (_type_): _description_
    """
    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        # weight_value=torch.min(torch.tensor(1/self.lengthscale),torch.tensor(1)).item()
        weight_value=1/self.lengthscale
        # weight_value=1
        # print(f'weight_value ={weight_value}')

        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):

            ten_power_omega_sqrt = self.lengthscale.sqrt()
            
            # weight_value=self.lengthscale
            # x1=x1[16,:]* torch.ones_like(x1)
            # x1_ = x1[:,1:].mul(ten_power_omega_sqrt)
            # #### Amin Added this
            # x2_ = x2[:,1:].mul(ten_power_omega_sqrt)


            # return 0*torch.eye(COV_old.size()[0])
            # return self.covar_dist(
            #     x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rbf, postprocess=True, **params
            # )
        # return RBFCovariance.apply(
        #     x1,
        #     x2,
        #     self.lengthscale,
        #     lambda x1, x2: self.covar_dist(
        #         x1, x2, square_dist=True, diag=False, dist_postprocess_func=postprocess_rbf, postprocess=False, **params
        #     ),
        # )
            x1_ = x1[:,1:].div(self.lengthscale)
            x2_ = x2[:,1:].div(self.lengthscale)
            x_INDEC=x1[:,0]
            COV_old=self.covar_dist(
                x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rbf, postprocess=True, **params
            )
            W=(x_INDEC==0).nonzero()
            H_value=x2_[W[0]]
            H_value_vectro=H_value*torch.ones_like(x2_)
            # COV_new=torch.diag(torch.norm(x1_-H_value, dim=1))

            Temp=self.covar_dist(
                x1_, H_value_vectro, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rbf, postprocess=True, **params
            )
            COV_new=torch.diag(torch.diagonal(Temp, 0))


            
            weight=torch.ones_like(COV_old)
            weight[W,:]=weight[W,:]/weight_value
            weight[:,W]=weight[:,W]/weight_value

            # weight[W,W]=weight_value*weight[W,W]
            # return  torch.mul(COV_old,weight_value*weight)
            return  weight_value*weight
        else:
            COV_old=RBFCovariance.apply(
                x1,
                x2,
                self.lengthscale,
                lambda x1, x2: self.covar_dist(
                    x1[:,1:], x2[:,1:], square_dist=True, diag=False, dist_postprocess_func=postprocess_rbf, postprocess=False, **params
                ),
            )
            x_INDEC_1=x1[:,0]
            x_INDEC_2=x2[:,0]
            W_1=(x_INDEC_1==0).nonzero()
            W_2=(x_INDEC_2==0).nonzero()

            # weight_value=self.lengthscale
            
            weight=torch.ones_like(COV_old)
            # for i in W_2:
            #     weight[W_1,i]=weight[W_1,i]/weight_value
            weight[W_1,:]=weight[W_1,:]/weight_value
            weight[:,W_2]=weight[:,W_2]/weight_value

            # for i in W_2:
            #     weight[W_1,i]=weight[W_1,i]*weight_value

            # return torch.mul(COV_old,weight_value*weight)
            
            return weight_value*weight

        #     return torch.ones_like(COV_old)
        
        # return torch.ones_like(RBFCovariance.apply(
        #     x1,
        #     x2,
        #     self.lengthscale,
        #     lambda x1, x2: self.covar_dist(
        #         x1, x2, square_dist=True, diag=False, dist_postprocess_func=postprocess_rbf, postprocess=False, **params
        #     ),
        # ))
    


        #     return torch.eye(torch.zeros_like(COV_old).size()[0])
        
        # return torch.eye(torch.zeros_like(RBFCovariance.apply(
        #     x1,
        #     x2,
        #     self.lengthscale,
        #     lambda x1, x2: self.covar_dist(
        #         x1, x2, square_dist=True, diag=False, dist_postprocess_func=postprocess_rbf, postprocess=False, **params
        #     ),
        # )).size()[0])
        
        # return RBFCovariance.apply(
        #     x1,
        #     x2,
        #     self.lengthscale,
        #     lambda x1, x2: self.covar_dist(
        #         x1, x2, square_dist=True, diag=False, dist_postprocess_func=postprocess_rbf, postprocess=False, **params
        #     ),
        # )