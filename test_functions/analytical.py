import numpy as np
from scipy.stats.qmc import Sobol, scale
from gpplus.preprocessing import setlevels

####################################Wing Function################################################
def wing(n=100, X = None, noise_std = 0.0, random_state = None, shuffle = True):

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
    y = 0.036 * Sw**0.758 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
        q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3) *\
        (Nz * Wdg) ** 0.49 + Sw * Wp


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


####################################Borehole Function################################################
def borehole(n=100, X = None, noise_std = 0.0, random_state = None, shuffle = True):

    if random_state is not None:
        np.random.seed(random_state)

    dx = 8
    l_bound = [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855]
    u_bound = [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]
    out_flag = 0
    if X is None:
        sobolset = Sobol(d=dx, seed = random_state)
        X = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n, :]
        X = scale(X, l_bounds=l_bound, u_bounds=u_bound)
        out_flag = 1
    if type(X) != np.ndarray:
        X = np.array(X)
    rw = X[..., 0]
    r = X[..., 1]
    Tu = X[..., 2]
    Hu = X[..., 3] 
    Tl = X[..., 4]
    Hl = X[..., 5]
    L = X[..., 6]
    Kw = X[..., 7]

    frac1 = 2 * np.pi * Tu * (Hu-Hl)

    frac2a = 2*L*Tu / (np.log(r/rw)*rw**2*Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r/rw) * (1+frac2a+frac2b)

    y = frac1 / frac2


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

####################################Borehole Function################################################

def borehole_mixed_variables(n=100, X = None, qual_dict = {0:5, 6:3}, 
    noise_std = 0.0, random_state = None, shuffle = True):

    labels = {'rw':0, 'r':1, 'Tu':2, 'Hu':3, "Tl":4, 'Hl':5, 'L':6, 'Kw':7}

    dx = 8
    l_bound = [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855]
    u_bound = [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]
    out_flag = 0
    data = {}
    if X is None:
        sobolset = Sobol(d=dx, seed = random_state)
        X = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n, :]
        X = scale(X, l_bounds=l_bound, u_bounds=u_bound)
        # for categorical variables we select t1 levels from the boundary
        for key, value in qual_dict.items():
            levels = np.random.uniform(l_bound[key], u_bound[key], size = value)
            X[...,key] = np.random.choice(levels, size = len(X), replace=True)

        out_flag = 1
    if type(X) != np.ndarray:
        X = np.array(X)
    data['rw'] = X[..., 0]
    data['r'] = X[..., 1]
    data['Tu'] = X[..., 2]
    data['Hu'] = X[..., 3] 
    data['Tl'] = X[..., 4]
    data['Hl'] = X[..., 5]
    data['L'] = X[..., 6]
    data['Kw'] = X[..., 7]

    frac1 = 2 * np.pi * data['Tu'] * (data['Hu']-data['Hl'])

    frac2a = 2*data['L']*data['Tu'] / (np.log(data['r']/data['rw'])*data['rw']**2*data['Kw'])
    frac2b = data['Tu'] / data['Tl']
    frac2 = np.log(data['r']/data['rw']) * (1+frac2a+frac2b)

    y = frac1 / frac2

    if shuffle:
        index = np.random.randint(0, len(y), size = len(y))
        X = X[index,...]
        y = y[index]

    # This will assign levels to categorical evenif the levels are strings
    X = setlevels(X, qual_index = list(qual_dict.keys()))

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


####################################### Wing_mixed #######################################
def wing_mixed_variables(n=100, X = None, qual_dict = {0:5, 6:3}, noise_std = 0.0, random_state = None, shuffle = True):

    if random_state is not None:
        np.random.seed(random_state)

    labels = {'Sw':0, 'Wfw':1, 'A':2, 'Gama':3, "q":4, 'lamb':5, 'tc':6, 'Nz':7, 'Wdg':8, 'Wp':9}

    dx = 10
    l_bound = [150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025]
    u_bound = [200, 300, 10, 10, 45, 1, 0.18, 6, 2500, 0.08]
    out_flag = 0
    if X is None:
        sobolset = Sobol(d=dx, seed = random_state)
        X = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n, :]
        X = scale(X, l_bounds=l_bound, u_bounds=u_bound)
        for key, value in qual_dict.items():
            levels = np.random.uniform(l_bound[key], u_bound[key], size = value)
            X[...,key] = np.random.choice(levels, size = len(X), replace=True)
            
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
    y = 0.036 * Sw**0.758 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
        q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3) *\
        (Nz * Wdg) ** 0.49 + Sw * Wp

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


#################################### Sine Function ################################################
def sine_1D(n=100, X = None, noise_std = 0.0, frequency = 1.0, absolute_value_flag = False, random_state = None, shuffle = True):
    if random_state is not None:
        np.random.seed(random_state)
    dx = 1
    l_bound = [-1.0]
    u_bound = [1.0]
    out_flag = 0
    if X is None:
        sobolset = Sobol(d=dx, seed = random_state)
        X = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n, :]
        X = scale(X, l_bounds=l_bound, u_bounds=u_bound)
        out_flag = 1
    if type(X) != np.ndarray:
        X = np.array(X)
    
    y = np.sin(2*np.pi*frequency*X[:,0])

    if absolute_value_flag == True:
        y = np.abs(y)

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
