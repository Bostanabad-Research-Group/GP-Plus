# Basic Examples

Introductory GP+ scripts covering simple 1D regression workflows (Adam vs LBFGS, CPU vs GPU) and a single-fidelity Wing function example.

| File | What it shows |
| --- | --- |
| `GP_regression_adam.py` | Fit a basic GP with Adam on a 1D noisy sine |
| `GP_regression_LBFGS.py` | Fit a basic GP with LBFGS (`LBFGSScipy`) on a 1D noisy sine |
| `GP_regression_adam_vs_LBFGS.py` | Side-by-side Adam vs LBFGS comparison |
| `cpu_vs_gpu_adam.py` | Adam training timed on CPU vs GPU (if CUDA is available) |
| `cpu_vs_gpu_LBFGS.py` | LBFGS training timed on CPU vs GPU (if CUDA is available) |
| `Wing_SF_GP.py` | Single-fidelity GP on the Wing function (HF only; 10 continuous features) |

How to run: From the repo root, run a script with your Python environment that has `gpplus` installed (e.g. `python examples/basic/GP_regression_LBFGS.py`). Adjust seed, optimizer settings, and device as needed.

Expected output: Training loss progress and/or timing comparisons for the 1D scripts (those also plot the training data). `Wing_SF_GP.py` prints model metrics and training time.
