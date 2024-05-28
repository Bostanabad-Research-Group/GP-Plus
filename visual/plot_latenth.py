import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.cm as cm


def plot_sep(type ,positions, levels, perm, constraints_flag = True, suptitle= None):
    if positions.ndim > 2:
        positions = positions.mean(axis = 0)
    else:
        positions = positions.detach()

    if positions.ndim > 2:
        positions = positions.mean(axis = 0) 

    # applying the constrains
    if constraints_flag:
        positions = constrains(positions)


    positions = positions.cpu().detach().numpy()
    legend=[ 'HF', 'LF1','LF2','LF3','LF4','LF5','LF6','LF7','LF8','LF9','LF10','LF11','LF12','LF13','LF14','LF15','LF16','LF17','LF18','LF19','LF20']

    # plt.rcParams.update({'font.size': 14})
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    plt.rcParams['figure.dpi']=150
    #colors = {0:'blue', 1:'r', 2:'g', 3:'c', 4:'m', 5:'k', 6:'y'}
    # tab20 = plt.get_cmap('tab10')
    colors_base = ['deeppink', 'gold', 'darkorange', 'gray', 'orangered', 'blue']
    cmap = cm.get_cmap('tab20', 20)  # Get a colormap from matplotlib, 'tab20' has nice distinct colors
    colors = colors_base  + [cmap(i) for i in range(len(colors_base), 20)]
    marker = ['X','o','s',"v", 'p', 'v', '^', '<', '>','*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_','1', '2', '3', '4',]
    # loop over the number of variables
    if len(levels) > 1:
        fig,axs = plt.subplots(1, len(levels),figsize=(10,4))
        for j in range(len(levels)):
            for i in range(levels[j]):
                index = torch.where(perm[:,j] == i) 
                if i<=10:
                    fontsize=5
                    s_size=100
                else:
                    fontsize=5
                    s_size=100
                if  type=='mf':
                    axs[j].scatter(positions[index][...,0], positions[index][...,1], label = legend[i], color = colors[i], marker=marker[i],s=s_size,alpha=.6)#marker=r'$\clubsuit$'
                    plt.xlabel(r'$z_1$',labelpad=0,rotation=0,fontsize=20)
                    plt.ylabel(r'$z_2$',labelpad=10,rotation=0, fontsize=20)
                
                elif type=='cat':
                    axs[j].scatter(positions[index][...,0], positions[index][...,1], label = 'level' + str(i+1), color = colors[i], marker=marker[i],s=s_size,alpha=.6)#marker=r'$\clubsuit$'
                    axs[j].set_xlabel(r'$h_1$', labelpad=0, fontsize=20)
                    axs[j].set_ylabel(r'$h_2$', labelpad=5,fontsize=20)
                else:
                    raise ValueError("type should be either 'mf' or cat:")
                # plt.tight_layout()
                # axs[j].legend()
                axs[j].legend(loc='upper right', fontsize='xx-small')

                tempxi = np.min(positions[...,0])-0.2 * (np.abs(np.min(positions[...,0])) +5)
                tempxx = np.max(positions[...,0]) + 0.2 * (np.abs(np.max(positions[...,0])) +5)
                tempyi = np.min(positions[...,1])-0.2 * (np.abs(np.min(positions[...,1])) +5)
                tempyx = np.max(positions[...,1]) + 0.2 * (np.abs(np.max(positions[...,1])) +5)
                axs[j].set_xlim(tempxi, tempxx)
                axs[j].set_ylim(tempyi, tempyx)
    else:
        fig,axs = plt.subplots(1, len(levels),figsize=(6,4))
        for j in range(len(levels)):
            for i in range(levels[j]):
                index = torch.where(perm[:,j] == i) 
                if i<=7:
                    fontsize=20
                    s_size=200
                else:
                    fontsize=20
                    s_size=100
                if  type=='mf':
                    axs.scatter(positions[index][...,0], positions[index][...,1], label = legend[i], color = colors[i], marker=marker[i],s=s_size,alpha=.6)#marker=r'$\clubsuit$'
                    plt.xlabel(r'$z_1$',labelpad=0,rotation=0, fontsize=fontsize)
                    plt.ylabel(r'$z_2$',labelpad=10,rotation=0, fontsize=fontsize)
                
                elif type=='cat':
                    axs.scatter(positions[index][...,0], positions[index][...,1], label = 'level' + str(i+1), color = colors[i], marker=marker[i],s=s_size,alpha=.6)#marker=r'$\clubsuit$'
                    axs.set_xlabel(r'$h_1$', labelpad=0, fontsize=fontsize)
                    axs.set_ylabel(r'$h_2$', labelpad=5,fontsize=fontsize)
                else:
                    raise ValueError("type should be either 'mf' or cat:")
                # plt.tight_layout()
                # axs[j].legend()
                axs.legend(loc='upper right', fontsize='xx-small')

                tempxi = np.min(positions[...,0])-0.2 * (np.abs(np.min(positions[...,0])) +5)
                tempxx = np.max(positions[...,0]) + 0.2 * (np.abs(np.max(positions[...,0])) +5)
                tempyi = np.min(positions[...,1])-0.2 * (np.abs(np.min(positions[...,1])) +5)
                tempyx = np.max(positions[...,1]) + 0.2 * (np.abs(np.max(positions[...,1])) +5)
                axs.set_xlim(tempxi, tempxx)
                axs.set_ylim(tempyi, tempyx)
    fig.tight_layout()

    plt.show()
    if suptitle is not None:
        plt.suptitle(suptitle,fontsize=18)


def constrains(z):
    n = z.shape[0]
    z = z - z[0,:]

    if z[1,0] < 0:
        z[:, 0] *= -1
    
    rot = torch.atan(-1 * z[1,1]/z[1,0])
    R = torch.tensor([ [torch.cos(rot), -1 * torch.sin(rot)], [torch.sin(rot), torch.cos(rot)]])

    z = torch.matmul(R, z.T)
    z = z.T
    if z.shape[1] > 2 and z[2,1] < 0:
        z[:, 1] *= -1
    
    return z
    


