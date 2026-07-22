# Single-Fidelity Buckling Function

Single-fidelity GP regression on the Buckling function using only the high-fidelity source (`s0`). Inputs mix continuous features with categorical one-hot blocks (E / K / I).

Files:
- `Buckling_SF_GP_Grouped_Cat.py` — categorical columns passed as one flat group
- `Buckling_SF_GP_Separate_Cat.py` — categorical columns passed as separate E / K / I groups

Problem: Buckling — single fidelity (HF only)

How to run: Run either `.py` file. Feel free to adjust the seed, data generation, and model inputs such as the combined kernel and its encoders.

Expected output: Model metrics, training time, latent embeddings, and encoder plots
