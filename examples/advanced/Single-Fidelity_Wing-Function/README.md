# Single-Fidelity Wing Function

Single-fidelity GP regression on the Wing function using only the high-fidelity source (`s0`). The input has 10 continuous numerical features (no categorical variables, no source columns).

Files:
- `Wing_SF_GP.py` — fits a GP on Sobol-sampled Wing data from `wing_mixed_variables`

Problem: Wing — single fidelity (HF only)

How to run: Run the `.py` file. Feel free to adjust the seed, data generation, and model inputs such as the combined kernel.

Expected output: Model metrics and training time
