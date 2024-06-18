# GP+
---
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![Conda](https://img.shields.io/conda/v/gpytorch/gpytorch.svg)](https://anaconda.org/gpytorch/gpytorch)
[![PyPI](https://img.shields.io/pypi/v/gpytorch.svg)](https://pypi.org/project/gpytorch)

Python Library for Generalized Gaussian Process Modeling


# Installation

**Requirements**:
- Python == 3.9
- CUDA >= 11.6 (if using GPU)

To use GP+, you first need to install the specific versions of PyTorch. The installation process involves two steps: (1) installing the specific version of PyTorch based on your system, and (2) installing GP+.

## (1) Install PyTorch

### For macOS
To install PyTorch for macOS, use:

```bash
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
```

### For Linux and Windows
To install PyTorch for Linux and Windows, follow the steps below based on whether you have CUDA support or not.

#### For GPU Support (with CUDA)
If you have a compatible GPU and want to leverage GPU acceleration, install PyTorch with CUDA support:

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

#### For CPU Only
If you do not have a compatible GPU, install the CPU-only version of PyTorch:

```bash
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

## (2) Install GP+
Once you have installed the appropriate version of PyTorch, install GP+ using pip:

```bash
pip install gpplus
```


# More About GP+

GP+ is an open-source library for kernel-based learning via Gaussian processes (GPs). It systematically integrates nonlinear manifold learning techniques with GPs for single and multi-fidelity emulation, calibration of computer models, sensitivity analysis, and Bayesian optimization. GP+ is built on PyTorch and provides a user-friendly and object-oriented tool for probabilistic learning and inference. 

For more detailed information, refer to our paper: ["GP+: A Python Library for Kernel-based Learning via Gaussian Processes"](https://www.sciencedirect.com/science/article/pii/S0965997824000930?dgcid=author).


## The Team
Amin Yousefpour\
Zahra Zanjani Foumani\
Mehdi Shishehbor\
Carlos Mora\
Ramin Bostanabad


## Citing Us
To reference GP+ in your academic work, please use the following citation, now available on arXiv:

Yousefpour, Amin; Zanjani Foumani, Zahra; Shishehbor, Mehdi; Mora, Carlos; Bostanabad, Ramin. "GP+: A Python Library for Kernel-based Learning via Gaussian Processes." Advances in Engineering Software (2024). https://doi.org/10.1016/j.advengsoft.2024.103686.



## Assistance and Support
Need help with GP+? Feel free to open an issue on our GitHub page and label it according to the module or feature in question for quicker assistance.
