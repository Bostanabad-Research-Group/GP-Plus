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
## Bayesian Optimization (BO)
Here is a quick rundown of the main components of the multi-fidelity BO loop for Borehole example.
  1. Importing required packages
```python
import ...
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


