# GP+
---
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Conda](https://img.shields.io/conda/v/gpytorch/gpytorch.svg)](https://anaconda.org/gpytorch/gpytorch)
[![PyPI](https://img.shields.io/pypi/v/gpytorch.svg)](https://pypi.org/project/gpytorch)

Python Library for Generalized Gaussian Process Modeling,


## Installation

**Requirements**:
- Python >= 3.8
- PyTorch >= 1.11

Install GP+ using pip:

```bash
pip install gpplus
```
## Emulation  
Begin with a simple emulation scenario using GP+. This example uses the borehole function for demonstration purposes.
```bash
# Import necessary modules
from gpplus.models import GP_Plus
from gpplus.test_functions.analytical import borehole
from gpplus.preprocessing import train_test_split_normalizeX
from gpplus.utils import set_seed

# Generate and split the data
set_seed(1245)
X, y = borehole(n=10500, random_state=12345)
Xtrain, Xtest, ytrain, ytest = train_test_split_normalizeX(X, y, test_size=0.95)

# Create and fit the GP+ model
model = GP_Plus(Xtrain, ytrain, device='cuda')
model.fit(n_jobs=-1, num_restarts=32)

# Evaluate the model
model.evaluation(Xtest, ytest)
```
## Emulation with Mixed Data
GP+ also supports emulation with mixed data types (numerical and categorical). Here's how you can handle such a scenario:
```bash
# Import modules and set random seed
from gpplus.models import GP_Plus
from gpplus.test_functions.analytical import borehole_mixed_variables
from gpplus.preprocessing import train_test_split_normalizeX
from gpplus.utils import set_seed

set_seed(4)

# Generate mixed data
qual_index = {0: 5, 5: 5}
U, y = borehole_mixed_variables(n=10000, qual_ind_val=qual_index, random_state=4)
Utrain, Utest, ytrain, ytest = train_test_split_normalizeX(U, y, test_size=0.99, qual_index_val=qual_index)

# Create, train, and evaluate the model
model = GP_Plus(Utrain, ytrain, qual_ind_lev=qual_index)
model.fit(bounds=True)
model.visualize_latent()
model.evaluation(Utest, ytest)
```
## Multi-Fidelity Emulation
GP+ excels in multi-fidelity modeling scenarios, effectively leveraging data from various fidelity levels.
```bash
# Import necessary libraries
from gpplus.models import GP_Plus
from gpplus.test_functions.multi_fidelity import multi_fidelity_wing
from gpplus.preprocessing import train_test_split_normalizeX
from gpplus.utils import set_seed

# Set parameters and generate data
set_seed(4)
qual_index = {10: 4}
num = {'0': 5000, '1': 10000, '2': 10000, '3': 10000}
noise_std = {'0': 0.5, '1': 1.0, '2': 1.5, '3': 2.0}
X, y = multi_fidelity_wing(n=num, noise_std=noise_std, random_state=4)

# Split and normalize data
Xtrain, Xtest, ytrain, ytest = train_test_split_normalizeX(X, y, test_size=0.99, qual_index_val=qual_index, stratify=X[..., list(qual_index.keys())])

# Initialize and fit the GP+ model
model = GP_Plus(Xtrain, ytrain, qual_ind_lev=qual_index, multiple_noise=True, base='multiple_constant')
model.fit(n_jobs=-1)

# Evaluate and visualize results
model.score(Xtest, ytest, plot_MSE=True, title='Multiple Noise & Mixed Base', seperate_levels=True)
model.visualize_latent(type='MF')
```



## Bayesian Optimization (BO)
Here is a quick rundown of the main components of the multi-fidelity BO loop for Borehole example.
  1. Importing required packages
```python
from gpplus.test_functions.multi_fidelity import Borehole_MF_BO
from gpplus.utils import set_seed
from gpplus.preprocessing.normalizeX import standard
from gpplus.bayesian_optimizations.BO_GP_plus import BO, Visualize_BO
```
  2. Define the problem-specific parameters: Here we define the index of categorical variables and lower and upper bounds of the problem for optimization
```python
qual_index={8:5}
l_bound = [100,990, 700,100,0.05,10,1000,6000]            
u_bound = [1000,1110,820,10000,0.15,500,2000,12000]
```
  3. Initialization: Number of initial samples from each source and the corresponding sampling cost of each source.
```python
n_train_init = {'0': 5, '1': 5, '2': 50, '3': 5, '4': 50}
costs = {'0': 1000, '1': 100, '2': 10, '3':100, '4':10} 
```
  4. Data generation and standardization
```python
U_init, y_init = Borehole_MF_BO(True,n_train_init)           
U_init,umean, ustd = standard(U_init,qual_index)
```
  5. Starting BO loop: All the components of BO, including emulator and acquisition functions are embedded in the BO function. Lots of options are defined as the inputs of BO function to enable flexible optimization. These options are detailed in the paper. Here, we stick to the default setting.
```python
bestf, cost = BO(U_init,y_init,costs,l_bound,u_bound,umean,ustd,qual_index,Borehole_MF)
```

  6. Visualizing the performance of BO: The Visualize_BO function plots the best converged value ($y^*$) vs the cumulative convergence cost.
```python
Visualize_BO(bestf,cost)
```


## Citing Us


## The Team
Amin Yousefpour
Zahra Zanjani Foumani
Mehdi Shishehbor
Carlos Mora 
Ramin Bostanabad

