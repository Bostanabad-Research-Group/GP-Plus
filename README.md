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
  1. Define the problem-specific parameters: Here we define the index of categorical variables, and lower and upper bounds of the problem for optimization
```python
qual_index={8:5}
l_bound = [100,990, 700,100,0.05,10,1000,6000]            
u_bound = [1000,1110,820,10000,0.15,500,2000,12000]
```


## Citing Us


## The Team


