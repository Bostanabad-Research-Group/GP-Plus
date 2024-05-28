# LMGP Visualization
# 
#
#

from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import torch


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
}


def plot_ls(model, constraints_flag = True):
    # 
    # plot latent values

    zeta = torch.tensor(model.zeta, dtype = torch.float64).to(**tkwargs)
    #A = model.nn_model.weight.detach()
    perm = model.perm
    levels = model.num_levels_per_var
    #positions = torch.matmul(zeta, A.T)   # this gives the position of each combination in latent space

    positions = model.nn_model(zeta)
    positions = positions.detach()

    # applying the constrains
    if constraints_flag:
        positions = constrains(positions.to(**tkwargs))


    fig,axs = plt.subplots(1, len(levels),figsize=(12,6))
    #colors = {0:'blue', 1:'r', 2:'g', 3:'c', 4:'m', 5:'k', 6:'y'}
    tab20 = plt.get_cmap('tab10')
    colors = tab20.colors

    # loop over the number of variables
    for j in range(len(levels)):

        for i in range(levels[j]):
            index = torch.where(perm[:,j] == i) 
            #col = list(map(lambda x: colors[x], np.ones(index[0].numpy().shape) * i))
            axs[j].scatter(positions.cpu().numpy()[index][:,0], positions.cpu().numpy()[index][:,1], label = 'level' + str(i+1), color = colors[i%10], s = (i+1)*50/levels[j])
            axs[j].set_title('Variable ' + str(j), fontsize = 15)
            axs[j].set_xlabel(r'$z_1$', fontsize = 15)
            axs[j].set_ylabel(r'$z_2$', fontsize = 15)
            axs[j].legend()

        fig.tight_layout()


def constrains(z):
    n = z.shape[0]
    z = z - z[0,:]

    if z[1,0] < 0:
        z[:, 0] *= -1
    
    rot = torch.atan(-1 * z[1,1]/z[1,0])
    R = torch.tensor([ [torch.cos(rot), -1 * torch.sin(rot)], [torch.sin(rot), torch.cos(rot)]]).to(**tkwargs)

    z = torch.matmul(R, z.T)
    z = z.T
    if z.shape[1] > 2 and z[2,1] < 0:
        z[:, 1] *= -1
    
    return z