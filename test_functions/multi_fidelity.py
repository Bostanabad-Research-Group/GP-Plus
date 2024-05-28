import numpy as np
from scipy.stats.qmc import Sobol, scale
import math
import torch
from pyDOE import lhs

################################## Wing #########################################
def wing(n=100, X = None, 
    fidelity = 0,noise_std = 0.0, random_state = None, shuffle = True):
    if random_state is not None:
        np.random.seed(random_state)

    dx = 10
    l_bound = [150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025]
    u_bound = [200, 300, 10, 10, 45, 1, 0.18, 6, 2500, 0.08]
    out_flag = 0
    if X is None:
        sobolset = Sobol(d=dx, seed = random_state)
        X = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n, :]
        X = scale(X, l_bounds=l_bound, u_bounds=u_bound)
        out_flag = 1
    if type(X) != np.ndarray:
        X = np.array(X)
    Sw = X[..., 0]
    Wfw = X[..., 1]
    A = X[..., 2]
    Gama = X[..., 3] * (np.pi/180.0)
    q = X[..., 4]
    lamb = X[..., 5]
    tc = X[..., 6]
    Nz = X[..., 7]
    Wdg = X[..., 8]
    Wp = X[..., 9]
    # This is the output

    if fidelity == 0:
        y = 0.036 * Sw**0.758 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
            q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3) *\
            (Nz * Wdg) ** 0.49 + Sw * Wp
    elif fidelity == 1:
        y = 0.036 * Sw**0.758 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
            q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3)\
            * (Nz * Wdg) ** 0.49 + 1 * Wp 

    elif fidelity == 2:
        y = 0.036 * Sw**0.8 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
            q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3)\
            * (Nz * Wdg) ** 0.49 + 1 * Wp


    elif fidelity == 3:
        y = 0.036 * Sw**0.9 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
            q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3)\
            * (Nz * Wdg) ** 0.49 + 0 * Wp
    else:
        raise ValueError('only 4 fidelities of 0,1,2,3 have been implemented ')

    if X is None:
        if shuffle:
            index = np.random.randint(0, len(y), size = len(y))
            X = X[index,...]
            y = y[index]

    if noise_std > 0.0:
        if out_flag == 1:
            return X, y + np.random.randn(*y.shape) * noise_std
        else:
            return y       
    else:
        if out_flag == 1:
            return X, y
        else:
            return y


################################# Multi-fidelity ####################################
def multi_fidelity_wing(X = None,
    n={'0': 50, '1': 100, '2': 100, '3': 100},
    noise_std={'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0},
    random_state = None, shuffle = True):
    if X is None:
        X_list = []
        y_list = []
        for level, num in n.items():
            if level  in ['0','1','2','3'] and num > 0:
                X, y = wing(n=num, fidelity = int(level), 
                    noise_std= noise_std[level], random_state = random_state)
                X = np.hstack([X, np.ones(num).reshape(-1, 1) * float(level)])
                X_list.append(X)
                y_list.append(y)
            else:
                raise ValueError('Wrong label, should be h, l1, l2 or l3')
        return np.vstack([*X_list]), np.hstack(y_list)
    else:
        fidelity = [i for i in n.keys()]
        y_list = []
        if type(X) == np.ndarray:
            X = torch.tensor(X)
        for f in fidelity:
            # Find the index for each fidelity then call the wing
            index = [i[0] for i in torch.argwhere(X[...,-1]== int(f))]
            y_list.append(wing(X=X[index, 0:-1], fidelity=int(f),  
                noise_std= noise_std[f]))
        return torch.tensor(np.hstack(y_list))



def multi_fidelity_wing_value(input):
    y_list = []
    for value in input:
        if value[-1] == 0.0:
            y_list.append(wing(X=value))
        elif value[-1] == 1.0:
            y_list.append(wing(X=value))
        elif value[-1] == 2.0:
            y_list.append(wing(X=value))
        elif value[-1] == 3.0:
            y_list.append(wing(X=value))
        else:
            raise ValueError('Wrong label, should be 0, 1, 2 or 3')
    return torch.tensor(np.hstack(y_list))


def Augmented_branin(input, negate = True, mapping = None):

    X = input.clone()

    if mapping is not None:
        X[..., 2] = torch.tensor(list(map(lambda x: mapping[str(float(x))], X[..., 2]))).to(X)

    t1 = (
        X[..., 1]
        - (5.1 / (4 * math.pi ** 2) - 0.1 * (1 - X[..., 2])) * X[..., 0] ** 2
        + 5 / math.pi * X[..., 0]
        - 6
    )
    t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
    return -(t1 ** 2 + t2 + 10) if negate else (t1 ** 2 + t2 + 10)


def Borehole_MF_BO(init_data,x,var = (0, 0, 0, 0,0)):
    # Define functions
    def y_h(x,noise_std=4):
        Tu, Hu, Hl, r, rw, Tl, L, Kw = [x[:, i] for i in range(8)]
        Numer = 2*np.pi*Tu
        y= (Numer*(Hu-Hl))/(np.log(r/rw)*(1+((2*L*Tu)/(np.log(r/rw)*(rw**2)*Kw))+(Tu/Tl)))
        if noise_std>0:
            return y + np.random.randn(*y.shape) * noise_std
        else:
            return y
    def y_l1(x):
        Tu, Hu, Hl, r, rw, Tl, L, Kw = [x[:, i] for i in range(8)]
        Numer = 2*np.pi*Tu
        return (Numer*(Hu-.8*Hl))/(np.log(r/rw)*(1+((1*L*Tu)/(np.log(r/rw)*(rw**2)*Kw))+(Tu/Tl)))
    def y_l2(x):
        Tu, Hu, Hl, r, rw, Tl, L, Kw = [x[:, i] for i in range(8)]
        Numer = 2*np.pi*Tu
        return (Numer*(Hu-Hl))/(np.log(r/rw)*(1+((8*L*Tu)/(np.log(r/rw)*(rw**2)*Kw))+.75*(Tu/Tl)))
    def y_l3(x):
        Tu, Hu, Hl, r, rw, Tl, L, Kw = [x[:, i] for i in range(8)]
        Numer = 2*np.pi*Tu
        return (Numer*(1.09*Hu-Hl))/(np.log(4*r/rw)*(1+((3*L*Tu)/(np.log(r/rw)*(rw**2)*Kw))+(Tu/Tl)))
    def y_l4(x):
        Tu, Hu, Hl, r, rw, Tl, L, Kw = [x[:, i] for i in range(8)]
        Numer = 2*np.pi*Tu
        return (Numer*(1.05*Hu-Hl))/(np.log(2*r/rw)*(1+((3*L*Tu)/(np.log(r/rw)*(rw**2)*Kw))+(Tu/Tl)))
    if init_data:
        # Set parameters
        n_train = tuple(x.values())
        dx = 9
        dnum = 8
        dy = 1
        # dt is a list whose indices correspond to the categorical variable
        # and whose entries correspond to the number of levels for that variable
        dt = (5,)
        dsource = (4,)
        num_idx = np.array(list(range(0, 8)))
        source_idx = np.array([8])
        MIN = (100, 990, 700, 100, .05, 10, 1000, 6000)
        MAX = (1000, 1110, 820, 10000, .15, 500, 2000, 12000)
        RANGE = [MAX[i] - MIN[i] for i in range(dnum)]
        y_list = [y_h, y_l1, y_l2, y_l3,y_l4]
        y = lambda x, t: y_list[t](x)
        # Preallocate data arrays and indices
        # First, get the total number of input points for each set
        train_tot = sum(n_train)
        x_train = np.empty([train_tot, dx])
        y_train = np.empty([train_tot, dy])
        # Initialize the indices for training
        train_start = 0
        # Generate data
        for i in range(dt[0]):
            # Set end indices
            train_end = train_start + n_train[i]
            n_train_temp = n_train[i]
            x_train_temp = lhs(dnum, samples = n_train_temp) * RANGE + MIN
            y_train_temp = y(x_train_temp, i).reshape((n_train_temp, dy))
            y_train_temp = y_train_temp + np.sqrt(var[i])*np.random.standard_normal(size=y_train_temp.shape)
            # Store the data
            x_train[train_start : train_end, 0:dnum] = x_train_temp
            x_train[train_start : train_end, dnum:dnum+source_idx[0]] = i*np.ones((np.shape(x_train_temp)[0], 1))
            y_train[train_start : train_end, :] = y_train_temp
            # Update indices
            train_start = train_end
    #(dx, dnum, dsource, dt, dy, num_idx, source_idx, min_x, max_x, min_y, max_y, x_train, y_train, x_val, y_val, x_test, y_test)
        return x_train , y_train
    else:
        if type(x) == torch.Tensor:
            input_copy = x.clone()
        elif type(x) == np.ndarray:
            input_copy = np.copy(x)
        y_list = []
        for X in input_copy:
            if X[-1].numpy() == 0.0:
                y_list.append(y_h(X[0:-1].unsqueeze(0), noise_std=2)) #, noise_std=3.31
            elif X[-1].numpy() == 1.0:
                y_list.append(y_l1(X[0:-1].unsqueeze(0)))
            elif X[-1].numpy() == 2.0:
                y_list.append(y_l2(X[0:-1].unsqueeze(0)))
            elif X[-1] == 3.0:
                y_list.append(y_l3(X[0:-1].unsqueeze(0)))
            elif X[-1]==4.0:
                y_list.append(y_l4(X[0:-1].unsqueeze(0)))
            else:
                raise ValueError('Wrong label, should be h, l1, l2 or l3')
        return torch.tensor(np.hstack(y_list))
