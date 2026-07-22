# Single-Fidelity Buckling Function

Single-fidelity GP regression on the Buckling function using only the high-fidelity source (`s0`). Inputs mix continuous features with categorical one-hot blocks (E / K / I).

## Encoder runs

Each script trains twice on the same data, with encoders passed explicitly:

1. **MatrixEncoder (A)** — `MatrixEncoder(...)` for the categorical group(s)
2. **NeuralEncoder** — `NeuralEncoder(...)` for the categorical group(s)

Files:
- `Buckling_SF_GP_Grouped_Cat_Encoder.py` — flat cat group; Matrix then Neural
- `Buckling_SF_GP_Separate_Cat_Encoder.py` — separate E / K / I cat groups; Matrix then Neural

Problem: Buckling — single fidelity (HF only)

How to run: Run either `.py` file. Feel free to adjust the seed, data generation, and model inputs.

Expected output: Model metrics for Matrix and Neural encoders, training time, latent embeddings, and encoder plots
